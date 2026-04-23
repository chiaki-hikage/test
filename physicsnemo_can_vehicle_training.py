from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. Expected raw columns
# =========================================================
EXPECTED_COLUMNS = {
    "time": "Time (sec)",
    "lat": " Latitude (deg)",
    "lon": " Longitude (deg)",
    "speed": " Vehicle speed (MPH)",
    "gps_speed": " GPS Speed (MPH)",
    "acceleration": " Acceleration (ft/s²)",
    "accel_x": " Accel X (ft/s²)",
    "accel_y": " Accel Y (ft/s²)",
    "accel_z": " Accel Z (ft/s²)",
    "accel_grav_x": " Accel (Grav) X (ft/s²)",
    "accel_grav_y": " Accel (Grav) Y (ft/s²)",
    "accel_grav_z": " Accel (Grav) Z (ft/s²)",
    "rot_x": " Rotation Rate X (deg/s)",
    "rot_y": " Rotation Rate Y (deg/s)",
    "rot_z": " Rotation Rate Z (deg/s)",   # raw sensor Z rotation, not trusted as yaw truth
    "roll": " Roll (deg)",
    "pitch": " Pitch (deg)",
    "rpm": " Engine RPM (RPM)",
    "throttle": " Absolute throttle position (%)",
    "engine_torque": " Engine Torque (lb•ft)",
    "engine_power": " Engine Power (hp)",
    "maf": " Mass air flow rate (lb/min)",
    "fuel_rate": " Fuel rate (gal/hr)",
    "calc_load": " Calculated load value (%)",
    "map_inhg": " Intake manifold absolute pressure (inHg)",
    "boost": " Boost (psi)",
    "intake_air_temp_f": " Intake air temperature (°F)",
    "coolant_temp_f": " Engine coolant temperature (°F)",
    "altitude": " Altitude (ft)",
    "bearing": " Bearing (deg)",
    "horz_accuracy": " Horz Accuracy (ft)",
}


# =========================================================
# 2. Small helpers
# =========================================================
def mph_to_mps(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) * 0.44704


def ftps2_to_mps2(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) * 0.3048


def deg_to_rad(x: np.ndarray) -> np.ndarray:
    return np.deg2rad(np.asarray(x, dtype=float))


def lbft_to_nm(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) * 1.3558179483314004


def lbmin_to_kgs(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) * 0.45359237 / 60.0


def clamp_np(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), lo, hi)


def wrap_to_pi_np(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:
    two_pi = 2.0 * torch.pi
    return torch.remainder(x + torch.pi, two_pi) - torch.pi


def latlon_to_local_xy_m(lat_deg: np.ndarray, lon_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)

    valid = np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() == 0:
        return np.full_like(lat, np.nan), np.full_like(lon, np.nan)

    lat0 = lat[valid][0]
    lon0 = lon[valid][0]

    R = 6371000.0
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    x = R * np.cos(lat0_rad) * (lon_rad - lon0_rad)
    y = R * (lat_rad - lat0_rad)
    return x, y


def inverse_sigmoid_from_bounded_value(v: float, lo: float, hi: float) -> float:
    eps = 1e-6
    v = min(max(v, lo + eps), hi - eps)
    z = (v - lo) / (hi - lo)
    return float(np.log(z / (1.0 - z)))


def bounded_value(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(raw)


def set_requires_grad(params, flag: bool) -> None:
    if isinstance(params, torch.Tensor):
        params.requires_grad = flag
        return
    for p in params:
        p.requires_grad = flag


def get_column(df: pd.DataFrame, key: str) -> pd.Series:
    col = EXPECTED_COLUMNS[key]
    if col in df.columns:
        return df[col]
    stripped = {c.strip(): c for c in df.columns}
    if col.strip() in stripped:
        return df[stripped[col.strip()]]
    raise KeyError(f"Missing column for key='{key}': expected '{col}'")


def robust_corr(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        m &= mask
    if m.sum() < 10:
        return np.nan
    if np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


# =========================================================
# 3. Config / priors
# =========================================================
@dataclass
class VehicleSpecPriors:
    mass_kg_init: float = 1800.0
    wheelbase_m: float = 2.9
    front_weight_fraction: float = 0.55  # fraction on front axle
    cf_n_per_rad_init: float = 80000.0
    cr_n_per_rad_init: float = 90000.0

    # learnable parameter bounds
    mu_bounds: Tuple[float, float] = (0.35, 1.30)
    cf_scale_bounds: Tuple[float, float] = (0.5, 1.8)
    cr_scale_bounds: Tuple[float, float] = (0.5, 1.8)
    iz_scale_bounds: Tuple[float, float] = (0.5, 1.8)
    mass_scale_bounds: Tuple[float, float] = (0.85, 1.15)

    @property
    def lf_m(self) -> float:
        # CG to front axle
        return (1.0 - self.front_weight_fraction) * self.wheelbase_m

    @property
    def lr_m(self) -> float:
        # CG to rear axle
        return self.front_weight_fraction * self.wheelbase_m

    @property
    def iz_init(self) -> float:
        return self.mass_kg_init * self.lf_m * self.lr_m


@dataclass
class TrainConfig:
    device: str = "cpu"
    seq_len: int = 25
    batch_size: int = 64
    num_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0

    hidden_dim: int = 64
    residual_scale: float = 0.08
    max_steer_rad: float = 0.35

    # 変更
    w_xy: float = 5.0
    w_v: float = 1.0
    w_r: float = 3.0
    w_psi: float = 1.0
    w_phys: float = 0.02
    w_param: float = 0.005
    w_reg: float = 1e-4

    mu_prior_sigma_rel: float = 0.25
    cf_prior_sigma_rel: float = 0.30
    cr_prior_sigma_rel: float = 0.30
    iz_prior_sigma_rel: float = 0.25
    mass_prior_sigma_rel: float = 0.08

    history_steps: int = 3
    eval_stride_div: int = 4


# =========================================================
# 4. Preprocess
# =========================================================
def preprocess_can_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    out["t_s"] = pd.to_numeric(get_column(df, "time"), errors="coerce")

    # Speed / acceleration
    out["vx_mps"] = mph_to_mps(pd.to_numeric(get_column(df, "speed"), errors="coerce").to_numpy())
    out["gps_speed_mps"] = mph_to_mps(pd.to_numeric(get_column(df, "gps_speed"), errors="coerce").to_numpy())
    out["ax_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "acceleration"), errors="coerce").to_numpy())

    # Raw accel axes (diagnostics)
    out["accel_x_sensor_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "accel_x"), errors="coerce").to_numpy())
    out["ay_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "accel_y"), errors="coerce").to_numpy())
    out["az_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "accel_z"), errors="coerce").to_numpy())

    # Gravity accel axes
    out["accel_grav_x_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "accel_grav_x"), errors="coerce").to_numpy())
    out["accel_grav_y_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "accel_grav_y"), errors="coerce").to_numpy())
    out["accel_grav_z_mps2"] = ftps2_to_mps2(pd.to_numeric(get_column(df, "accel_grav_z"), errors="coerce").to_numpy())

    # Rotation / attitude
    out["rot_x_radps"] = deg_to_rad(pd.to_numeric(get_column(df, "rot_x"), errors="coerce").to_numpy())
    out["rot_y_radps"] = deg_to_rad(pd.to_numeric(get_column(df, "rot_y"), errors="coerce").to_numpy())
    out["raw_z_radps"] = deg_to_rad(pd.to_numeric(get_column(df, "rot_z"), errors="coerce").to_numpy())
    out["roll_rad"] = deg_to_rad(pd.to_numeric(get_column(df, "roll"), errors="coerce").to_numpy())
    out["pitch_rad"] = deg_to_rad(pd.to_numeric(get_column(df, "pitch"), errors="coerce").to_numpy())

    # Powertrain
    out["rpm"] = pd.to_numeric(get_column(df, "rpm"), errors="coerce")
    out["throttle_pct"] = pd.to_numeric(get_column(df, "throttle"), errors="coerce")
    out["engine_torque_nm"] = lbft_to_nm(pd.to_numeric(get_column(df, "engine_torque"), errors="coerce").to_numpy())
    out["engine_power_hp"] = pd.to_numeric(get_column(df, "engine_power"), errors="coerce")
    out["maf_kgs"] = lbmin_to_kgs(pd.to_numeric(get_column(df, "maf"), errors="coerce").to_numpy())
    out["fuel_rate_gph"] = pd.to_numeric(get_column(df, "fuel_rate"), errors="coerce")
    out["calc_load_pct"] = pd.to_numeric(get_column(df, "calc_load"), errors="coerce")
    out["map_inhg"] = pd.to_numeric(get_column(df, "map_inhg"), errors="coerce")
    out["boost_psi"] = pd.to_numeric(get_column(df, "boost"), errors="coerce")
    out["intake_air_temp_f"] = pd.to_numeric(get_column(df, "intake_air_temp_f"), errors="coerce")
    out["coolant_temp_f"] = pd.to_numeric(get_column(df, "coolant_temp_f"), errors="coerce")

    # Position / heading
    out["altitude_m"] = pd.to_numeric(get_column(df, "altitude"), errors="coerce") * 0.3048
    out["bearing_rad"] = deg_to_rad(pd.to_numeric(get_column(df, "bearing"), errors="coerce").to_numpy())
    out["horz_accuracy_m"] = pd.to_numeric(get_column(df, "horz_accuracy"), errors="coerce") * 0.3048

    lat = pd.to_numeric(get_column(df, "lat"), errors="coerce").to_numpy()
    lon = pd.to_numeric(get_column(df, "lon"), errors="coerce").to_numpy()
    out["lat_deg"] = lat
    out["lon_deg"] = lon

    x_m, y_m = latlon_to_local_xy_m(lat, lon)
    out["x_meas_m"] = x_m
    out["y_meas_m"] = y_m

    out = out.replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both")
    out = out.sort_values("t_s").reset_index(drop=True)

    dt = np.gradient(out["t_s"].to_numpy())
    dt = clamp_np(dt, 1e-2, 10.0)
    out["dt_s"] = dt

    # GPS-based course
    out["xdot_meas_mps"] = np.gradient(out["x_meas_m"].to_numpy(), out["t_s"].to_numpy())
    out["ydot_meas_mps"] = np.gradient(out["y_meas_m"].to_numpy(), out["t_s"].to_numpy())

    xdot_s = pd.Series(out["xdot_meas_mps"]).rolling(7, center=True, min_periods=1).mean().to_numpy()
    ydot_s = pd.Series(out["ydot_meas_mps"]).rolling(7, center=True, min_periods=1).mean().to_numpy()

    course = np.unwrap(np.arctan2(ydot_s, xdot_s))
    out["course_rad"] = course
    out["xy_speed_mps"] = np.sqrt(xdot_s**2 + ydot_s**2)

    r_xy = np.gradient(out["course_rad"].to_numpy(), out["t_s"].to_numpy())
    r_xy = pd.Series(r_xy).rolling(7, center=True, min_periods=1).mean().to_numpy()
    r_xy = clamp_np(r_xy, -1.5, 1.5)
    out["r_xy_radps"] = r_xy

    out["psi_meas_rad"] = np.unwrap(out["bearing_rad"].to_numpy())
    beta_proxy = np.unwrap(out["course_rad"].to_numpy() - out["psi_meas_rad"].to_numpy())
    out["beta_proxy_rad"] = clamp_np(beta_proxy, -0.35, 0.35)

    # XY reliability
    out["is_xy_reliable"] = (
        (out["vx_mps"].to_numpy() >= 5.0) &
        (out["horz_accuracy_m"].to_numpy() <= 15.0)
    ).astype(float)

    # Gravity norm + yaw-like rate
    gx = out["accel_grav_x_mps2"].to_numpy()
    gy = out["accel_grav_y_mps2"].to_numpy()
    gz = out["accel_grav_z_mps2"].to_numpy()
    gnorm = np.sqrt(gx**2 + gy**2 + gz**2)
    out["accel_grav_norm_mps2"] = gnorm

    gnorm_safe = np.where(gnorm > 1e-6, gnorm, np.nan)
    ux = gx / gnorm_safe
    uy = gy / gnorm_safe
    uz = gz / gnorm_safe

    wx = out["rot_x_radps"].to_numpy()
    wy = out["rot_y_radps"].to_numpy()
    wz = out["raw_z_radps"].to_numpy()

    yaw_like = -(wx * ux + wy * uy + wz * uz)
    yaw_like = pd.Series(yaw_like).rolling(5, center=True, min_periods=1).mean().to_numpy()
    yaw_like = clamp_np(yaw_like, -1.5, 1.5)
    out["yaw_like_rate_radps"] = yaw_like

    out["is_yaw_reliable"] = (
        (out["vx_mps"].to_numpy() >= 5.0) &
        np.isfinite(out["yaw_like_rate_radps"].to_numpy()) &
        np.isfinite(out["r_xy_radps"].to_numpy())
    ).astype(float)

    # Supervised delta targets
    x_arr = out["x_meas_m"].to_numpy()
    y_arr = out["y_meas_m"].to_numpy()
    vx_arr = out["vx_mps"].to_numpy()
    psi_arr = out["course_rad"].to_numpy()

    out["dx_true_m"] = np.concatenate([[0.0], np.diff(x_arr)])
    out["dy_true_m"] = np.concatenate([[0.0], np.diff(y_arr)])
    out["dpsi_true_rad"] = np.concatenate([[0.0], wrap_to_pi_np(np.diff(psi_arr))])
    out["dvx_true_mps"] = np.concatenate([[0.0], np.diff(vx_arr)])

    return out


# =========================================================
# 5. Diagnostics
# =========================================================
def validate_can_training_signals(df: pd.DataFrame) -> Dict[str, float]:
    stats = {}

    vx = df["vx_mps"].to_numpy()
    gps = df["gps_speed_mps"].to_numpy()
    ax = df["ax_mps2"].to_numpy()
    t = df["t_s"].to_numpy()
    dvx_dt = np.gradient(vx, t)
    dvx_dt = pd.Series(dvx_dt).rolling(5, center=True, min_periods=1).mean().to_numpy()

    yaw_like = df["yaw_like_rate_radps"].to_numpy()
    r_xy = df["r_xy_radps"].to_numpy()
    raw_z = df["raw_z_radps"].to_numpy()

    low_speed_mask = vx >= 5.0

    stats["corr_speed_gps"] = robust_corr(vx, gps)
    stats["corr_ax_dvxdt"] = robust_corr(ax, dvx_dt)
    stats["corr_rawz_rxy_all"] = robust_corr(raw_z, r_xy)
    stats["corr_yawlike_rxy_all"] = robust_corr(yaw_like, r_xy)
    stats["corr_yawlike_rxy_vxge5"] = robust_corr(yaw_like, r_xy, low_speed_mask)
    stats["grav_norm_mean_mps2"] = float(np.nanmean(df["accel_grav_norm_mps2"].to_numpy()))
    stats["yaw_reliable_ratio"] = float(np.mean(df["is_yaw_reliable"].to_numpy()))

    print("[CAN validation]", stats)
    return stats


# =========================================================
# 6. Arrays / dataset
# =========================================================
def make_history_features(df: pd.DataFrame, history_steps: int = 3) -> np.ndarray:
    """
    Base channels:
      0 ax_mps2
      1 throttle_pct
      2 rpm
      3 vx_mps
      4 yaw_like_rate_radps
      5 is_yaw_reliable
    """
    base = np.stack([
        df["ax_mps2"].to_numpy(dtype=np.float32),
        df["throttle_pct"].to_numpy(dtype=np.float32),
        df["rpm"].to_numpy(dtype=np.float32),
        df["vx_mps"].to_numpy(dtype=np.float32),
        df["yaw_like_rate_radps"].to_numpy(dtype=np.float32),
        df["is_yaw_reliable"].to_numpy(dtype=np.float32),
    ], axis=-1)

    feats = []
    for h in range(history_steps):
        shifted = np.roll(base, shift=h, axis=0)
        if h > 0:
            shifted[:h, :] = shifted[h:h+1, :]
        feats.append(shifted)

    return np.concatenate(feats, axis=-1).astype(np.float32)


def build_can_target_arrays(
    df: pd.DataFrame,
    history_steps: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    can_feat = make_history_features(df, history_steps=history_steps)

    target_feat = np.stack([
        df["x_meas_m"].to_numpy(dtype=np.float32),            # 0
        df["y_meas_m"].to_numpy(dtype=np.float32),            # 1
        df["course_rad"].to_numpy(dtype=np.float32),          # 2
        df["vx_mps"].to_numpy(dtype=np.float32),              # 3
        df["yaw_like_rate_radps"].to_numpy(dtype=np.float32), # 4 yaw proxy
        df["dx_true_m"].to_numpy(dtype=np.float32),           # 5
        df["dy_true_m"].to_numpy(dtype=np.float32),           # 6
        df["dpsi_true_rad"].to_numpy(dtype=np.float32),       # 7
        df["dvx_true_mps"].to_numpy(dtype=np.float32),        # 8
    ], axis=-1)

    dt = df["dt_s"].to_numpy(dtype=np.float32)
    return can_feat, target_feat, dt


class CanSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int, history_steps: int = 3):
        self.seq_len = seq_len
        self.can_feat, self.target_feat, self.dt = build_can_target_arrays(
            df, history_steps=history_steps
        )

        n = len(df)
        if n <= seq_len:
            raise ValueError(f"Not enough rows ({n}) for seq_len={seq_len}")

        self.start_indices = np.arange(0, n - seq_len)

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int):
        s = self.start_indices[idx]
        e = s + self.seq_len
        can_seq = torch.from_numpy(self.can_feat[s:e])
        target_seq = torch.from_numpy(self.target_feat[s:e])
        dt_seq = torch.from_numpy(self.dt[s:e])
        return can_seq, target_seq, dt_seq


# =========================================================
# 7. Model
# =========================================================
class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsInformedVehicleModel(nn.Module):
    """
    State: [x, y, psi, vx, beta, r]
    Input: history-stacked CAN features

    Physics core:
      - dynamic bicycle model with explicit mu / Cf / Cr / Iz / mass
      - latent steering inferred from CAN history
      - small residual corrections on beta_dot / r_dot / dvx
    """

    def __init__(
        self,
        priors: VehicleSpecPriors,
        cfg: TrainConfig,
        mass_kg: float,
        drag_terms: Dict[str, float],
        input_dim: int = 18,  # 6 base channels * history_steps
    ):
        super().__init__()
        self.priors = priors
        self.cfg = cfg
        self.mass_fixed_init = float(mass_kg)
        self.drag_terms = drag_terms

        # Learnable physical parameters (bounded around prior/init)
        self.raw_mu = nn.Parameter(torch.tensor(
            inverse_sigmoid_from_bounded_value(0.95, *self.priors.mu_bounds),
            dtype=torch.float32
        ))
        self.raw_cf_scale = nn.Parameter(torch.tensor(
            inverse_sigmoid_from_bounded_value(1.0, *self.priors.cf_scale_bounds),
            dtype=torch.float32
        ))
        self.raw_cr_scale = nn.Parameter(torch.tensor(
            inverse_sigmoid_from_bounded_value(1.0, *self.priors.cr_scale_bounds),
            dtype=torch.float32
        ))
        self.raw_iz_scale = nn.Parameter(torch.tensor(
            inverse_sigmoid_from_bounded_value(1.0, *self.priors.iz_scale_bounds),
            dtype=torch.float32
        ))
        self.raw_mass_scale = nn.Parameter(torch.tensor(
            inverse_sigmoid_from_bounded_value(1.0, *self.priors.mass_scale_bounds),
            dtype=torch.float32
        ))

        # Small residual corrections
        self.residual = ResidualMLP(in_dim=input_dim + 6, hidden=cfg.hidden_dim, out_dim=3)

    def mu(self) -> torch.Tensor:
        return bounded_value(self.raw_mu, *self.priors.mu_bounds)

    def cf(self) -> torch.Tensor:
        scale = bounded_value(self.raw_cf_scale, *self.priors.cf_scale_bounds)
        return self.priors.cf_n_per_rad_init * scale

    def cr(self) -> torch.Tensor:
        scale = bounded_value(self.raw_cr_scale, *self.priors.cr_scale_bounds)
        return self.priors.cr_n_per_rad_init * scale

    def iz(self) -> torch.Tensor:
        scale = bounded_value(self.raw_iz_scale, *self.priors.iz_scale_bounds)
        return self.priors.iz_init * scale

    def mass_kg(self) -> torch.Tensor:
        scale = bounded_value(self.raw_mass_scale, *self.priors.mass_scale_bounds)
        return self.mass_fixed_init * scale

    def step_org(
        self,
        state: torch.Tensor,
        can_step: torch.Tensor,
        dt_step: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # state: [x, y, psi, vx, beta, r]
        x, y, psi, vx, beta, r = [state[..., i] for i in range(6)]

        # current-step channels are first 6 dims
        ax_meas = can_step[..., 0]
        throttle = can_step[..., 1]
        rpm = can_step[..., 2]
        vx_meas = can_step[..., 3]
        r_proxy = can_step[..., 4]
        yaw_rel = can_step[..., 5]

        dt = torch.clamp(dt_step.squeeze(-1), min=1e-2)
        vx_safe = torch.clamp(vx, min=1.0)

        model_in = torch.cat([state, can_step], dim=-1)

        #!!! 5m/s
        wheelbase = self.priors.wheelbase_m
        #delta_proxy = wheelbase * r_proxy / torch.clamp(vx_meas, min=5.0)
        #delta_eff = torch.clamp(delta_proxy, -self.cfg.max_steer_rad, self.cfg.max_steer_rad)

        vx_for_delta = torch.clamp(vx_meas, min=5.0)
        delta_proxy = wheelbase * r_proxy / vx_for_delta
        delta_eff = torch.clamp(delta_proxy, -self.cfg.max_steer_rad, self.cfg.max_steer_rad)

        #!!! さらに小さく制限
        delta_eff = torch.clamp(delta_proxy, -0.08, 0.08)

        # small residual corrections
        res = self.cfg.residual_scale * self.residual(model_in)
        beta_dot_res = res[..., 0]
        r_dot_res = res[..., 1]
        dvx_res = res[..., 2]

        m = self.mass_kg()
        mu = self.mu()
        cf = self.cf()
        cr = self.cr()
        iz = self.iz()
        g = 9.81

        lf = self.priors.lf_m
        lr = self.priors.lr_m

        # Dynamic bicycle model
        #alpha_f = beta + lf * r / vx_safe - delta_eff
        #alpha_r = beta - lr * r / vx_safe

        # r_nextを直接使う
        alpha_f = beta + lf * r_next / vx_safe - delta_eff
        alpha_r = beta - lr * r_next / vx_safe

        fzf = m * g * lr / (lf + lr)
        fzr = m * g * lf / (lf + lr)

        fyf_lin = -cf * alpha_f
        fyr_lin = -cr * alpha_r

        fyf = torch.clamp(fyf_lin, -mu * fzf, mu * fzf)
        fyr = torch.clamp(fyr_lin, -mu * fzr, mu * fzr)

        beta_dot_phys = (fyf + fyr) / (m * vx_safe) - r
        #r_dot_phys = (lf * fyf - lr * fyr) / iz

        #!!!
        r_dot_phys = torch.zeros_like(r_next)

        # Longitudinal:
        # start from measured ax, allow a tiny correction
        dvx_eff = ax_meas + dvx_res

        beta_dot_eff = beta_dot_phys + beta_dot_res
        #r_dot_eff = r_dot_phys + r_dot_res

        #!!!
        r_dot_eff = torch.zeros_like(r_next)

        # state deltas
        #dpsi = r * dt
        dvx = dvx_eff * dt
        dbeta = beta_dot_eff * dt
        #dr = r_dot_eff * dt
        #!!!
        dr = torch.zeros_like(r_next)

        ###！！！ r_next, dbetaに制限
        r_next = torch.clamp(r_proxy, -0.4, 0.4)
        dpsi = r_next * dt
        dbeta = torch.clamp(dbeta, -0.12, 0.12)

        dx = vx * torch.cos(psi + beta) * dt
        dy = vx * torch.sin(psi + beta) * dt

        next_state = torch.stack([
            x + dx,
            y + dy,
            psi + dpsi,
            vx + dvx,
            beta + dbeta,
            r_next,
        ], dim=-1)

        aux = {
            "dx_pred": dx,
            "dy_pred": dy,
            "dpsi_pred": dpsi,
            "dvx_pred": dvx,
            "dbeta_pred": dbeta,
            "dr_pred": dr,
            "r_pred": r + dr,

            "beta_dot_phys": beta_dot_phys,
            "r_dot_phys": r_dot_phys,
            "beta_dot_eff": beta_dot_eff,
            "r_dot_eff": r_dot_eff,
            "dvx_eff": dvx_eff,

            "delta_eff": delta_eff,
            "r_proxy": r_proxy,
            "yaw_rel": yaw_rel,

            "mu": mu,
            "cf": cf,
            "cr": cr,
            "iz": iz,
            "mass_kg": m,
        }
        return next_state, aux

    def step(
        self,
        state: torch.Tensor,
        can_step: torch.Tensor,
        dt_step: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # state: [x, y, psi, vx, beta, r]
        x, y, psi, vx, beta, r = [state[..., i] for i in range(6)]

        # current-step channels
        ax_meas = can_step[..., 0]
        throttle = can_step[..., 1]
        rpm = can_step[..., 2]
        vx_meas = can_step[..., 3]
        r_proxy = can_step[..., 4]
        yaw_rel = can_step[..., 5]

        dt = torch.clamp(dt_step.squeeze(-1), min=1e-2)
        vx_safe = torch.clamp(vx, min=1.0)

        # -----------------------------------------
        # use yaw-like proxy directly as yaw rate
        # -----------------------------------------
        r_next = torch.clamp(r_proxy, -0.4, 0.4)

        # steering proxy from curvature approximation
        wheelbase = self.priors.wheelbase_m
        vx_for_delta = torch.clamp(vx_meas, min=5.0)
        delta_proxy = wheelbase * r_proxy / vx_for_delta
        delta_eff = torch.clamp(delta_proxy, -0.08, 0.08)

        model_in = torch.cat([state, can_step], dim=-1)

        # residual only for beta_dot and dvx
        res = self.cfg.residual_scale * self.residual(model_in)
        beta_dot_res = res[..., 0]
        dvx_res = res[..., 2]

        m = self.mass_kg()
        mu = self.mu()
        cf = self.cf()
        cr = self.cr()
        iz = self.iz()
        g = 9.81

        lf = self.priors.lf_m
        lr = self.priors.lr_m

        # use r_next here, not r
        alpha_f = beta + lf * r_next / vx_safe - delta_eff
        alpha_r = beta - lr * r_next / vx_safe

        fzf = m * g * lr / (lf + lr)
        fzr = m * g * lf / (lf + lr)

        fyf_lin = -cf * alpha_f
        fyr_lin = -cr * alpha_r

        fyf = torch.clamp(fyf_lin, -mu * fzf, mu * fzf)
        fyr = torch.clamp(fyr_lin, -mu * fzr, mu * fzr)

        # beta dynamics only
        beta_dot_phys = (fyf + fyr) / (m * vx_safe) - r_next
        beta_dot_eff = beta_dot_phys + beta_dot_res

        dvx_eff = ax_meas + dvx_res

        dpsi = r_next * dt
        dvx = dvx_eff * dt
        dbeta = beta_dot_eff * dt

        beta_next = torch.clamp(beta + dbeta, -0.12, 0.12)

        dx = vx * torch.cos(psi + beta) * dt
        dy = vx * torch.sin(psi + beta) * dt

        next_state = torch.stack([
            x + dx,
            y + dy,
            psi + dpsi,
            vx + dvx,
            beta_next,
            r_next,
        ], dim=-1)

        aux = {
            "dx_pred": dx,
            "dy_pred": dy,
            "dpsi_pred": dpsi,
            "dvx_pred": dvx,
            "dbeta_pred": dbeta,
            "dr_pred": torch.zeros_like(r_next),
            "r_pred": r_next,

            "beta_dot_phys": beta_dot_phys,
            "r_dot_phys": torch.zeros_like(r_next),
            "beta_dot_eff": beta_dot_eff,
            "r_dot_eff": torch.zeros_like(r_next),
            "dvx_eff": dvx_eff,

            "delta_eff": delta_eff,
            "r_proxy": r_proxy,
            "yaw_rel": yaw_rel,

            "mu": mu,
            "cf": cf,
            "cr": cr,
            "iz": iz,
            "mass_kg": m,
        }
        return next_state, aux

    def rollout(
        self,
        can_seq: torch.Tensor,
        target_seq: torch.Tensor,
        dt_seq: torch.Tensor,
    ):
        batch, steps, _ = can_seq.shape
        device = can_seq.device

        x0 = target_seq[:, 0, 0]
        y0 = target_seq[:, 0, 1]
        psi0 = target_seq[:, 0, 2]
        vx0 = target_seq[:, 0, 3]
        r0 = target_seq[:, 0, 4]
        beta0 = torch.zeros_like(vx0)  # can be replaced later with better beta init

        state = torch.stack([x0, y0, psi0, vx0, beta0, r0], dim=-1).to(device)

        state_hist = []
        keys_bt = [
            "dx_pred", "dy_pred", "dpsi_pred", "dvx_pred", "dbeta_pred", "dr_pred",
            "r_pred", "beta_dot_phys", "r_dot_phys", "beta_dot_eff", "r_dot_eff",
            "dvx_eff", "delta_eff", "r_proxy", "yaw_rel"
        ]
        hist = {k: [] for k in keys_bt}
        scalars_last = {}

        for k in range(steps):
            dt = dt_seq[:, k].unsqueeze(-1)
            state, aux = self.step(state, can_seq[:, k, :], dt)
            state_hist.append(state)

            for kk in keys_bt:
                hist[kk].append(aux[kk])

            scalars_last["mu"] = aux["mu"]
            scalars_last["cf"] = aux["cf"]
            scalars_last["cr"] = aux["cr"]
            scalars_last["iz"] = aux["iz"]
            scalars_last["mass_kg"] = aux["mass_kg"]

        states = torch.stack(state_hist, dim=1)
        aux_seq = {k: torch.stack(v, dim=1) for k, v in hist.items()}
        aux_seq.update(scalars_last)
        return states, aux_seq


# =========================================================
# 8. Loss
# =========================================================
def compute_losses(
    model: PhysicsInformedVehicleModel,
    can_seq: torch.Tensor,
    target_seq: torch.Tensor,
    dt_seq: torch.Tensor,
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    states, aux = model.rollout(can_seq, target_seq, dt_seq)

    dx_pred = aux["dx_pred"]
    dy_pred = aux["dy_pred"]
    dpsi_pred = aux["dpsi_pred"]
    dvx_pred = aux["dvx_pred"]
    r_pred = aux["r_pred"]

    dx_true = target_seq[..., 5]
    dy_true = target_seq[..., 6]
    dpsi_true = target_seq[..., 7]
    dvx_true = target_seq[..., 8]
    r_proxy_true = target_seq[..., 4]

    yaw_rel = can_seq[..., 5]
    dt = torch.clamp(dt_seq, min=1e-2)

    sigma_dx = torch.clamp(dx_true.std(), min=1e-1)
    sigma_dy = torch.clamp(dy_true.std(), min=1e-1)
    sigma_dpsi = torch.clamp(dpsi_true.std(), min=1e-2)
    sigma_dvx = torch.clamp(dvx_true.std(), min=1e-2)
    sigma_r = torch.clamp(r_proxy_true.std(), min=5e-2)

    # data losses
    err_dx = (dx_pred - dx_true) / sigma_dx
    err_dy = (dy_pred - dy_true) / sigma_dy
    err_dvx = (dvx_pred - dvx_true) / sigma_dvx

    loss_xy = (
        1.0 * torch.nn.functional.huber_loss(err_dx, torch.zeros_like(err_dx), delta=2.0)
        + 2.0 * torch.nn.functional.huber_loss(err_dy, torch.zeros_like(err_dy), delta=2.0)
    )
    loss_v = torch.nn.functional.huber_loss(err_dvx, torch.zeros_like(err_dvx), delta=2.0)

    # yaw proxy bridge
    err_r = ((r_pred - r_proxy_true) * yaw_rel) / sigma_r
    #loss_r = torch.nn.functional.huber_loss(err_r, torch.zeros_like(err_r), delta=2.0)
    loss_r = torch.zeros((), device=dx_pred.device)

    # heading consistency
    err_psi = wrap_to_pi_torch((dpsi_pred - dpsi_true) * yaw_rel) / sigma_dpsi
    loss_psi = torch.nn.functional.huber_loss(err_psi, torch.zeros_like(err_psi), delta=2.0)

    # bicycle closeness: keep residual physics correction small
    beta_dot_phys = aux["beta_dot_phys"]
    r_dot_phys = aux["r_dot_phys"]
    beta_dot_eff = aux["beta_dot_eff"]
    r_dot_eff = aux["r_dot_eff"]

    sigma_beta_dot = torch.clamp(beta_dot_phys.std().detach(), min=5e-2)
    sigma_r_dot = torch.clamp(r_dot_phys.std().detach(), min=5e-2)

    err_bicycle_beta = (beta_dot_eff - beta_dot_phys) / sigma_beta_dot
    err_bicycle_r = (r_dot_eff - r_dot_phys) / sigma_r_dot
    loss_bicycle = (
        torch.nn.functional.huber_loss(err_bicycle_beta, torch.zeros_like(err_bicycle_beta), delta=2.0)
        + torch.nn.functional.huber_loss(err_bicycle_r, torch.zeros_like(err_bicycle_r), delta=2.0)
    )

    # small residuals / smooth latent steering
    delta_eff = aux["delta_eff"]
    dvx_eff = aux["dvx_eff"]

    delta_smooth = torch.diff(delta_eff, dim=1)
    loss_reg = (
        1e-2 * (delta_eff ** 2).mean()
        + 1e-2 * (delta_smooth ** 2).mean()
        + 1e-2 * ((dvx_eff - can_seq[..., 0]) ** 2).mean()
    )

    # parameter priors
    mu_rel = model.mu() / 0.95 - 1.0
    cf_rel = model.cf() / model.priors.cf_n_per_rad_init - 1.0
    cr_rel = model.cr() / model.priors.cr_n_per_rad_init - 1.0
    iz_rel = model.iz() / model.priors.iz_init - 1.0
    mass_rel = model.mass_kg() / model.mass_fixed_init - 1.0

    loss_param = (
        (mu_rel / cfg.mu_prior_sigma_rel) ** 2
        + (cf_rel / cfg.cf_prior_sigma_rel) ** 2
        + (cr_rel / cfg.cr_prior_sigma_rel) ** 2
        + (iz_rel / cfg.iz_prior_sigma_rel) ** 2
        + (mass_rel / cfg.mass_prior_sigma_rel) ** 2
    )

    total = (
        cfg.w_xy * loss_xy
        + cfg.w_v * loss_v
        + cfg.w_r * loss_r
        + cfg.w_psi * loss_psi
        + cfg.w_phys * loss_bicycle
        + cfg.w_param * loss_param
        + cfg.w_reg * loss_reg
    )

    logs = {
        "loss": float(total.detach().cpu().item()),
        "loss_xy": float(loss_xy.detach().cpu().item()),
        "loss_v": float(loss_v.detach().cpu().item()),
        "loss_r": float(loss_r.detach().cpu().item()),
        "loss_psi": float(loss_psi.detach().cpu().item()),
        "loss_bicycle": float(loss_bicycle.detach().cpu().item()),
        "loss_param": float(loss_param.detach().cpu().item()),
        "loss_reg": float(loss_reg.detach().cpu().item()),

        "mu": float(model.mu().detach().cpu().item()),
        "cf": float(model.cf().detach().cpu().item()),
        "cr": float(model.cr().detach().cpu().item()),
        "iz": float(model.iz().detach().cpu().item()),
        "mass_kg": float(model.mass_kg().detach().cpu().item()),
    }
    return total, logs


# =========================================================
# 9. Split / crop helpers
# =========================================================
def crop_df_by_time_range(
    df: pd.DataFrame,
    min_time: float | None = None,
    max_time: float | None = None,
) -> pd.DataFrame:
    out = df.sort_values("t_s").reset_index(drop=True).copy()
    if min_time is not None:
        out = out[out["t_s"] >= float(min_time)]
    if max_time is not None:
        out = out[out["t_s"] <= float(max_time)]
    out = out.reset_index(drop=True)

    if len(out) < 50:
        raise ValueError(
            f"Too few rows after time filtering: len={len(out)}, min_time={min_time}, max_time={max_time}"
        )
    return out


def make_time_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Dict[str, pd.DataFrame]:
    df = df.sort_values("t_s").reset_index(drop=True)
    n = len(df)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    if n_train < 100 or n_val < 50 or n_test < 50:
        raise ValueError(
            f"Not enough samples after split: n={n}, n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )

    return {
        "train": df.iloc[:n_train].reset_index(drop=True),
        "val": df.iloc[n_train:n_train + n_val].reset_index(drop=True),
        "test": df.iloc[n_train + n_val:].reset_index(drop=True),
    }


def save_split_summary(
    splits: Dict[str, pd.DataFrame],
    outdir: Path,
    min_time: float | None,
    max_time: float | None,
) -> None:
    rows = []
    for split_name, sdf in splits.items():
        rows.append({
            "split": split_name,
            "n_rows": len(sdf),
            "t_min": float(sdf["t_s"].min()),
            "t_max": float(sdf["t_s"].max()),
            "requested_min_time": min_time,
            "requested_max_time": max_time,
        })
    pd.DataFrame(rows).to_csv(outdir / "time_split_summary.csv", index=False)


# =========================================================
# 10. Evaluation / plots
# =========================================================
@torch.no_grad()
def evaluate_model_on_windows(
    model: nn.Module,
    df: pd.DataFrame,
    cfg: TrainConfig,
    seq_len: Optional[int] = None,
    stride: Optional[int] = None,
    history_steps: int = 3,
) -> Dict[str, float]:
    model.eval()

    can_feat, target_feat, dt = build_can_target_arrays(df, history_steps=history_steps)

    if seq_len is None:
        seq_len = cfg.seq_len
    if stride is None:
        stride = max(seq_len // cfg.eval_stride_div, 1)

    n = len(df)
    starts = np.arange(0, max(n - seq_len, 1), stride)

    dx_errs, dy_errs, dvx_errs, r_errs = [], [], [], []
    dx_true_all, dx_pred_all = [], []
    dy_true_all, dy_pred_all = [], []
    dvx_true_all, dvx_pred_all = [], []
    r_true_all, r_pred_all = [], []

    for s in starts:
        e = s + seq_len
        if e > n:
            break

        can_seq = torch.from_numpy(can_feat[s:e]).unsqueeze(0).to(cfg.device)
        target_seq = torch.from_numpy(target_feat[s:e]).unsqueeze(0).to(cfg.device)
        dt_seq = torch.from_numpy(dt[s:e]).unsqueeze(0).to(cfg.device)

        _, aux = model.rollout(can_seq, target_seq, dt_seq)

        dx_pred = aux["dx_pred"][0].detach().cpu().numpy()
        dy_pred = aux["dy_pred"][0].detach().cpu().numpy()
        dvx_pred = aux["dvx_pred"][0].detach().cpu().numpy()
        r_pred = aux["r_pred"][0].detach().cpu().numpy()

        dx_true = target_seq[0, :, 5].detach().cpu().numpy()
        dy_true = target_seq[0, :, 6].detach().cpu().numpy()
        dvx_true = target_seq[0, :, 8].detach().cpu().numpy()
        r_true = target_seq[0, :, 4].detach().cpu().numpy()

        dx_errs.append(float(np.sqrt(np.mean((dx_pred - dx_true) ** 2))))
        dy_errs.append(float(np.sqrt(np.mean((dy_pred - dy_true) ** 2))))
        dvx_errs.append(float(np.sqrt(np.mean((dvx_pred - dvx_true) ** 2))))
        r_errs.append(float(np.sqrt(np.mean((r_pred - r_true) ** 2))))

        dx_true_all.append(dx_true); dx_pred_all.append(dx_pred)
        dy_true_all.append(dy_true); dy_pred_all.append(dy_pred)
        dvx_true_all.append(dvx_true); dvx_pred_all.append(dvx_pred)
        r_true_all.append(r_true); r_pred_all.append(r_pred)

    def _cat(xs):
        return np.concatenate(xs) if len(xs) > 0 else np.array([])

    return {
        "rmse_dx": float(np.mean(dx_errs)) if dx_errs else np.nan,
        "rmse_dy": float(np.mean(dy_errs)) if dy_errs else np.nan,
        "rmse_dvx": float(np.mean(dvx_errs)) if dvx_errs else np.nan,
        "rmse_r_proxy": float(np.mean(r_errs)) if r_errs else np.nan,

        "dx_true_all": _cat(dx_true_all),
        "dx_pred_all": _cat(dx_pred_all),
        "dy_true_all": _cat(dy_true_all),
        "dy_pred_all": _cat(dy_pred_all),
        "dvx_true_all": _cat(dvx_true_all),
        "dvx_pred_all": _cat(dvx_pred_all),
        "r_true_all": _cat(r_true_all),
        "r_pred_all": _cat(r_pred_all),
    }


def _scatter_panel(ax, x, y, title, xlabel, ylabel):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) == 0:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        return

    ax.scatter(x, y, s=8, alpha=0.35)
    lo = np.nanmin([x.min(), y.min()])
    hi = np.nanmax([x.max(), y.max()])
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.0)

    corr = np.corrcoef(x, y)[0, 1] if (len(x) >= 2 and np.std(x) > 1e-12 and np.std(y) > 1e-12) else np.nan
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    mae = float(np.mean(np.abs(y - x)))

    ax.text(
        0.03, 0.97,
        f"n={len(x)}\nr={corr:.3f}\nRMSE={rmse:.3f}\nMAE={mae:.3f}",
        ha="left", va="top", transform=ax.transAxes
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_final_scatter_plots_windowed(preds: Dict[str, np.ndarray], split_name: str, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    _scatter_panel(axes[0], preds["dx_true_all"], preds["dx_pred_all"], f"{split_name}: dX_pred vs dX_true", "dX_true [m]", "dX_pred [m]")
    _scatter_panel(axes[1], preds["dy_true_all"], preds["dy_pred_all"], f"{split_name}: dY_pred vs dY_true", "dY_true [m]", "dY_pred [m]")
    _scatter_panel(axes[2], preds["dvx_true_all"], preds["dvx_pred_all"], f"{split_name}: dVx_pred vs dVx_true", "dVx_true [m/s]", "dVx_pred [m/s]")
    _scatter_panel(axes[3], preds["r_true_all"], preds["r_pred_all"], f"{split_name}: r_pred vs yaw_proxy", "r_proxy [rad/s]", "r_pred [rad/s]")

    fig.tight_layout()
    fig.savefig(outdir / f"{split_name.lower()}_scatter_final.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_metric_history(history_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.ravel()

    metric_names = [
        "loss",
        "rmse_dx",
        "rmse_dy",
        "rmse_dvx",
        "rmse_r_proxy",
    ]

    for ax, m in zip(axes, metric_names):
        if m == "loss":
            ax.plot(history_df["epoch"], history_df["train_loss"], label="train")
            ax.plot(history_df["epoch"], history_df["val_loss"], label="val")
            ax.plot(history_df["epoch"], history_df["test_loss"], label="test")
            ax.set_ylabel("loss")
        else:
            ax.plot(history_df["epoch"], history_df[f"train_{m}"], label="train")
            ax.plot(history_df["epoch"], history_df[f"val_{m}"], label="val")
            ax.plot(history_df["epoch"], history_df[f"test_{m}"], label="test")
            ax.set_ylabel(m)

        ax.set_title(m)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].axis("off")

    for ax in axes[:-1]:
        ax.set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(outdir / "epoch_split_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss_components(history_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "train_loss",
        "loss_xy",
        "loss_v",
        "loss_r",
        "loss_psi",
        "loss_bicycle",
        "loss_param",
        "loss_reg",
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=True)
    axes = axes.ravel()

    for ax, m in zip(axes, metrics):
        if m in history_df.columns:
            ax.plot(history_df["epoch"], history_df[m], linewidth=1.5)
            ax.set_title(m)
            ax.set_ylabel(m)
            ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(outdir / "training_loss_components.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_history(history_df: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = ["mu", "cf", "cr", "iz", "mass_kg"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, m in zip(axes, metrics):
        if m in history_df.columns:
            ax.plot(history_df["epoch"], history_df[m], linewidth=1.5)
            ax.set_ylabel(m)
            ax.set_title(m)
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(outdir / "parameter_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_preprocessed_can_dataframe(df: pd.DataFrame, outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    t = df["t_s"].to_numpy()

    plot_groups = [
        {
            "filename": "can_overview_motion.png",
            "title": "CAN Overview: Motion States",
            "columns": [
                ("vx_mps", "vx [m/s]"),
                ("gps_speed_mps", "gps speed [m/s]"),
                ("ax_mps2", "ax [m/s^2]"),
                ("yaw_like_rate_radps", "yaw-like [rad/s]"),
                ("r_xy_radps", "r_xy GPS [rad/s]"),
                ("raw_z_radps", "raw rot Z [rad/s]"),
                ("is_yaw_reliable", "yaw reliable"),
            ],
        },
        {
            "filename": "can_overview_powertrain.png",
            "title": "CAN Overview: Powertrain",
            "columns": [
                ("rpm", "rpm"),
                ("throttle_pct", "throttle [%]"),
            ],
        },
        {
            "filename": "can_overview_position.png",
            "title": "CAN Overview: Position / Heading",
            "columns": [
                ("x_meas_m", "x [m]"),
                ("y_meas_m", "y [m]"),
                ("course_rad", "course [rad]"),
                ("dx_true_m", "dx true [m]"),
                ("dy_true_m", "dy true [m]"),
                ("dvx_true_mps", "dvx true [m/s]"),
            ],
        },
        {
            "filename": "can_overview_gravity.png",
            "title": "CAN Overview: Gravity / Rotation",
            "columns": [
                ("accel_grav_x_mps2", "grav x [m/s^2]"),
                ("accel_grav_y_mps2", "grav y [m/s^2]"),
                ("accel_grav_z_mps2", "grav z [m/s^2]"),
                ("accel_grav_norm_mps2", "grav norm [m/s^2]"),
                ("rot_x_radps", "rot x [rad/s]"),
                ("rot_y_radps", "rot y [rad/s]"),
                ("raw_z_radps", "rot z [rad/s]"),
            ],
        },
    ]

    for group in plot_groups:
        cols = [(c, label) for c, label in group["columns"] if c in df.columns]
        if not cols:
            continue

        fig, axes = plt.subplots(len(cols), 1, figsize=(14, 2.6 * len(cols)), sharex=True)
        if len(cols) == 1:
            axes = [axes]

        for ax, (col, label) in zip(axes, cols):
            ax.plot(t, df[col].to_numpy(), linewidth=1.0)
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("time [s]")
        fig.suptitle(group["title"])
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, group["filename"]), dpi=150, bbox_inches="tight")
        plt.close(fig)


# =========================================================
# 11. Parameter summary
# =========================================================
def summarize_parameters(model: PhysicsInformedVehicleModel, mass_fit: Dict[str, float]) -> Dict[str, float]:
    return {
        "mass_fit_input_kg": float(mass_fit.get("mass_kg", np.nan)),
        "mass_kg": float(model.mass_kg().detach().cpu().item()),
        "mu": float(model.mu().detach().cpu().item()),
        "cf_n_per_rad": float(model.cf().detach().cpu().item()),
        "cr_n_per_rad": float(model.cr().detach().cpu().item()),
        "iz": float(model.iz().detach().cpu().item()),
    }


# =========================================================
# 12. Training
# =========================================================
def train_vehicle_model(
    df: pd.DataFrame,
    priors: VehicleSpecPriors,
    cfg: TrainConfig,
    mass_fit: Dict[str, float],
    outdir: Path,
    plot_every: int = 10,
    min_time: float | None = None,
    max_time: float | None = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    history_steps: int = 3,
) -> PhysicsInformedVehicleModel:
    outdir.mkdir(parents=True, exist_ok=True)

    # selected window
    df_fit = crop_df_by_time_range(df, min_time=min_time, max_time=max_time)
    pd.DataFrame([{
        "requested_min_time": min_time,
        "requested_max_time": max_time,
        "actual_min_time": float(df_fit["t_s"].min()),
        "actual_max_time": float(df_fit["t_s"].max()),
        "n_rows": len(df_fit),
    }]).to_csv(outdir / "fit_window_summary.csv", index=False)

    validate_can_training_signals(df_fit)
    plot_preprocessed_can_dataframe(df_fit, str(outdir / "preprocessed_overview"))

    splits = make_time_splits(df_fit, train_ratio=train_ratio, val_ratio=val_ratio)
    save_split_summary(splits, outdir, min_time=min_time, max_time=max_time)

    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    train_dataset = CanSequenceDataset(train_df, seq_len=cfg.seq_len, history_steps=history_steps)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    input_dim = 6 * history_steps

    model = PhysicsInformedVehicleModel(
        priors=priors,
        cfg=cfg,
        mass_kg=mass_fit["mass_kg"],
        drag_terms=mass_fit,
        input_dim=input_dim,
    ).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = np.inf
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(cfg.num_epochs):
        model.train()

        # -------------------------------------------------
        # Stage 1: fit Iz/Cf/Cr/mass + latent steering
        #          mu frozen, residual frozen
        # Stage 2: unfreeze mu
        # Stage 3: unfreeze residual as small correction
        # -------------------------------------------------
        if epoch < 20:
            set_requires_grad([model.raw_mu], False)
            set_requires_grad(model.residual.parameters(), False)
            cfg.w_phys = 0.03
            cfg.w_r = 3.0
            cfg.w_psi = 1.0
            cfg.w_xy = 5.0
        elif epoch < 50:
            set_requires_grad([model.raw_mu], True)
            set_requires_grad(model.residual.parameters(), False)
            cfg.w_phys = 0.03
            cfg.w_r = 3.0
            cfg.w_psi = 1.0
            cfg.w_xy = 5.0
        else:
            set_requires_grad([model.raw_mu], True)
            set_requires_grad(model.residual.parameters(), True)
            cfg.w_phys = 0.04
            cfg.w_r = 3.0
            cfg.w_psi = 1.0
            cfg.w_xy = 5.0

        #!!!
        cfg.w_psi = 0.0

        epoch_logs = []
        for can_seq, target_seq, dt_seq in train_loader:
            can_seq = can_seq.to(cfg.device)
            target_seq = target_seq.to(cfg.device)
            dt_seq = dt_seq.to(cfg.device)

            optimizer.zero_grad()
            loss, logs = compute_losses(model, can_seq, target_seq, dt_seq, cfg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_logs.append(logs)

        mean_train_logs = {
            k: float(np.mean([x[k] for x in epoch_logs]))
            for k in epoch_logs[0].keys()
        }

        train_preds = evaluate_model_on_windows(model, train_df, cfg, history_steps=history_steps)
        val_preds = evaluate_model_on_windows(model, val_df, cfg, history_steps=history_steps)
        test_preds = evaluate_model_on_windows(model, test_df, cfg, history_steps=history_steps)

        val_loss = val_preds["rmse_dx"] + val_preds["rmse_dy"] + val_preds["rmse_dvx"]
        test_loss = test_preds["rmse_dx"] + test_preds["rmse_dy"] + test_preds["rmse_dvx"]

        row = {
            "epoch": epoch,
            "train_loss": mean_train_logs["loss"],
            "val_loss": val_loss,
            "test_loss": test_loss,

            "train_rmse_dx": train_preds["rmse_dx"],
            "train_rmse_dy": train_preds["rmse_dy"],
            "train_rmse_dvx": train_preds["rmse_dvx"],
            "train_rmse_r_proxy": train_preds["rmse_r_proxy"],

            "val_rmse_dx": val_preds["rmse_dx"],
            "val_rmse_dy": val_preds["rmse_dy"],
            "val_rmse_dvx": val_preds["rmse_dvx"],
            "val_rmse_r_proxy": val_preds["rmse_r_proxy"],

            "test_rmse_dx": test_preds["rmse_dx"],
            "test_rmse_dy": test_preds["rmse_dy"],
            "test_rmse_dvx": test_preds["rmse_dvx"],
            "test_rmse_r_proxy": test_preds["rmse_r_proxy"],

            # train loss components
            "loss_xy": mean_train_logs["loss_xy"],
            "loss_v": mean_train_logs["loss_v"],
            "loss_r": mean_train_logs["loss_r"],
            "loss_psi": mean_train_logs["loss_psi"],
            "loss_bicycle": mean_train_logs["loss_bicycle"],
            "loss_param": mean_train_logs["loss_param"],
            "loss_reg": mean_train_logs["loss_reg"],

            # learned parameters
            "mu": mean_train_logs["mu"],
            "cf": mean_train_logs["cf"],
            "cr": mean_train_logs["cr"],
            "iz": mean_train_logs["iz"],
            "mass_kg": mean_train_logs["mass_kg"],
        }
        history.append(row)

        print(
            f"epoch={epoch:04d} "
            f"train_rmse=({row['train_rmse_dx']:.3f}, {row['train_rmse_dy']:.3f}, {row['train_rmse_dvx']:.3f}, r={row['train_rmse_r_proxy']:.3f}) "
            f"val_rmse=({row['val_rmse_dx']:.3f}, {row['val_rmse_dy']:.3f}, {row['val_rmse_dvx']:.3f}, r={row['val_rmse_r_proxy']:.3f}) "
            f"test_rmse=({row['test_rmse_dx']:.3f}, {row['test_rmse_dy']:.3f}, {row['test_rmse_dvx']:.3f}, r={row['test_rmse_r_proxy']:.3f}) "
            f"mu={row['mu']:.3f} cf={row['cf']:.1f} cr={row['cr']:.1f} iz={row['iz']:.1f} mass={row['mass_kg']:.1f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch % plot_every == 0) or (epoch == cfg.num_epochs - 1):
            hist_df = pd.DataFrame(history)
            hist_df.to_csv(outdir / "train_history.csv", index=False)
            plot_split_metric_history(hist_df, outdir)
            plot_training_loss_components(hist_df, outdir)
            plot_parameter_history(hist_df, outdir)

    if best_state is not None:
        model.load_state_dict(best_state)

    train_preds = evaluate_model_on_windows(model, train_df, cfg, history_steps=history_steps)
    val_preds = evaluate_model_on_windows(model, val_df, cfg, history_steps=history_steps)
    test_preds = evaluate_model_on_windows(model, test_df, cfg, history_steps=history_steps)

    plot_final_scatter_plots_windowed(train_preds, "Train", outdir)
    plot_final_scatter_plots_windowed(val_preds, "Val", outdir)
    plot_final_scatter_plots_windowed(test_preds, "Test", outdir)

    final_metrics = pd.DataFrame([
        {
            "split": "train",
            "rmse_dx": train_preds["rmse_dx"],
            "rmse_dy": train_preds["rmse_dy"],
            "rmse_dvx": train_preds["rmse_dvx"],
            "rmse_r_proxy": train_preds["rmse_r_proxy"],
        },
        {
            "split": "val",
            "rmse_dx": val_preds["rmse_dx"],
            "rmse_dy": val_preds["rmse_dy"],
            "rmse_dvx": val_preds["rmse_dvx"],
            "rmse_r_proxy": val_preds["rmse_r_proxy"],
        },
        {
            "split": "test",
            "rmse_dx": test_preds["rmse_dx"],
            "rmse_dy": test_preds["rmse_dy"],
            "rmse_dvx": test_preds["rmse_dvx"],
            "rmse_r_proxy": test_preds["rmse_r_proxy"],
        },
    ])
    final_metrics.to_csv(outdir / "final_split_metrics.csv", index=False)

    summary = summarize_parameters(model, mass_fit)
    pd.DataFrame([summary]).to_csv(outdir / "fit_summary.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "summary": summary}, outdir / "vehicle_model.pt")

    return model


# =========================================================
# 13. Example main
# =========================================================
if __name__ == "__main__":
    # Example:
    raw_df = pd.read_csv("./data/auto_iowa.csv")
    df = preprocess_can_dataframe(raw_df)
    priors = VehicleSpecPriors(mass_kg_init=1800.0, wheelbase_m=2.9, front_weight_fraction=0.55)
    cfg = TrainConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    mass_fit = {"mass_kg": 1800.0}
    outdir = Path("./out_physics_vehicle")
    model = train_vehicle_model(
        df=df,
        priors=priors,
        cfg=cfg,
        mass_fit=mass_fit,
        outdir=outdir,
        plot_every=10,
        min_time=500.0,
        max_time=3200.0,
        train_ratio=0.6,
        val_ratio=0.2,
        history_steps=3,
    )
    pass