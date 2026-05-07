import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. 基本設定
# =========================================================
EXPECTED_COLUMNS = {
    "time": "Time (sec)",
    "lat": " Latitude (deg)",
    "lon": " Longitude (deg)",
    "speed": " Vehicle speed (MPH)",
    "gps_speed": " GPS Speed (MPH)",
    "acceleration": " Acceleration (ft/s²)",
    "altitude": " Altitude (ft)",
    "accel_grav_x": " Accel (Grav) X (ft/s²)",
    "accel_grav_y": " Accel (Grav) Y (ft/s²)",
    "accel_grav_z": " Accel (Grav) Z (ft/s²)",
    "rot_x": " Rotation Rate X (deg/s)",
    "rot_y": " Rotation Rate Y (deg/s)",
    "rot_z": " Rotation Rate Z (deg/s)",
}

# 今回は time だけ除外。高さ・速度・加速度は含める
EXCLUDE_COLUMNS_DEFAULT = {
    "Time (sec)",
}

CATEGORY_RULES = [
    ("position_gps", [
        "latitude", "longitude", "bearing", "altitude", "horz accuracy", "trip distance"
    ]),
    ("speed_accel", [
        "speed", "accel", "acceleration", "hard brake", "hard accel", "grav"
    ]),
    ("powertrain", [
        "rpm", "torque", "load", "maf", "boost", "intake", "coolant",
        "throttle", "fuel rate", "fuel economy", "engine power", "o2 voltage",
        "voltage", "map"
    ]),
    ("attitude_rotation", [
        "rotation rate", "pitch", "roll", "yaw"
    ]),
    ("magnetometer", [
        "magnetometer"
    ]),
    ("trip_event", [
        "trip fuel", "trip duration", "idling", "seconds idling", "distance traveled while mil is activated"
    ]),
    ("attitude_rotation", [
    "rotation rate", "pitch", "roll", "yaw", "course"
    ]),
]


# =========================================================
# 2. 補助
# =========================================================
def _find_time_column(df: pd.DataFrame) -> str:
    candidates = [EXPECTED_COLUMNS["time"], " time", "time", "Time"]
    for c in candidates:
        if c in df.columns:
            return c
    stripped = {col.strip(): col for col in df.columns}
    for c in candidates:
        if c.strip() in stripped:
            return stripped[c.strip()]
    raise KeyError("time 列が見つかりません。")


def _clean_label(col: str) -> str:
    return re.sub(r"\s+", " ", col.strip())


def _to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _categorize_column(col: str) -> str:
    c = col.strip().lower()
    for cat, keywords in CATEGORY_RULES:
        if any(k in c for k in keywords):
            return cat
    return "other"


def _make_time_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[int, int, int]:
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def _chunk_list(xs: List[str], chunk_size: int) -> List[List[str]]:
    return [xs[i:i + chunk_size] for i in range(0, len(xs), chunk_size)]


def _find_column_by_expected_key(df: pd.DataFrame, expected_key: str) -> str | None:
    target = EXPECTED_COLUMNS[expected_key]
    if target in df.columns:
        return target

    stripped = {col.strip(): col for col in df.columns}
    if target.strip() in stripped:
        return stripped[target.strip()]

    return None


def _wrap_to_pi_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _latlon_to_local_xy_m(lat_deg: np.ndarray, lon_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple local tangent-plane approximation around the first valid point.
    """
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


def _compute_course_and_yawrate_from_latlon(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    t_s: np.ndarray,
    smooth_window: int = 9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    course = atan2(dy/dt, dx/dt)
    yaw rate = d(course)/dt
    """
    x_m, y_m = _latlon_to_local_xy_m(lat_deg, lon_deg)

    x_s = pd.Series(x_m).interpolate(limit_direction="both")
    y_s = pd.Series(y_m).interpolate(limit_direction="both")
    t_s = np.asarray(t_s, dtype=float)

    dxdt = np.gradient(x_s.to_numpy(), t_s)
    dydt = np.gradient(y_s.to_numpy(), t_s)

    if smooth_window and smooth_window > 1:
        dxdt = pd.Series(dxdt).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
        dydt = pd.Series(dydt).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()

    course = np.arctan2(dydt, dxdt)
    course_unwrap = np.unwrap(course)
    yawrate = np.gradient(course_unwrap, t_s)

    if smooth_window and smooth_window > 1:
        yawrate = pd.Series(yawrate).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()

    return course, yawrate


def _compute_yaw_like_rate_from_gravity_and_rotation(
    grav_x_ftps2: np.ndarray,
    grav_y_ftps2: np.ndarray,
    grav_z_ftps2: np.ndarray,
    rot_x_degps: np.ndarray,
    rot_y_degps: np.ndarray,
    rot_z_degps: np.ndarray,
    smooth_window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    yaw-like rate = omega dot u_down
    where u_down is the unit vector of gravity direction.
    Returns:
      yaw_like_degps, grav_norm_ftps2
    """
    gx = np.asarray(grav_x_ftps2, dtype=float)
    gy = np.asarray(grav_y_ftps2, dtype=float)
    gz = np.asarray(grav_z_ftps2, dtype=float)

    wx = np.asarray(rot_x_degps, dtype=float)
    wy = np.asarray(rot_y_degps, dtype=float)
    wz = np.asarray(rot_z_degps, dtype=float)

    gnorm = np.sqrt(gx**2 + gy**2 + gz**2)
    gnorm_safe = np.where(gnorm > 1e-6, gnorm, np.nan)

    ux = gx / gnorm_safe
    uy = gy / gnorm_safe
    uz = gz / gnorm_safe

    yaw_like = - (wx * ux + wy * uy + wz * uz)
#    yaw_like = - wz

    if smooth_window and smooth_window > 1:
        yaw_like = pd.Series(yaw_like).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()

    return yaw_like, gnorm

def plot_yaw_scatter_comparison(
    df: pd.DataFrame,
    outdir: Path,
    x_col: str = "d(course)/dt (deg/s)",
    y_col: str = "Yaw-like rate from grav+rot (deg/s)",
    negate_y: bool = False,
) -> str | None:
    if x_col not in df.columns or y_col not in df.columns:
        return None

    x = _to_numeric_series(df[x_col]).to_numpy(dtype=float)
    y = _to_numeric_series(df[y_col]).to_numpy(dtype=float)

    if negate_y:
        y = -y
        y_label = f"-{y_col}"
        title = f"-{y_col} vs {x_col}"
        stem = "timeseries_yaw_scatter_negated.png"
    else:
        y_label = y_col
        title = f"{y_col} vs {x_col}"
        stem = "timeseries_yaw_scatter.png"

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 3:
        return None

    speed_col = _find_column_by_expected_key(df, "speed")
    vx = _to_numeric_series(df[speed_col]).to_numpy() * 0.44704

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(vx) & (vx >= 5.0)
    x = x[m]
    y = y[m]

    corr = np.corrcoef(x, y)[0, 1] if (len(x) >= 2 and np.std(x) > 1e-12 and np.std(y) > 1e-12) else np.nan

    #corr = np.corrcoef(x, y)[0, 1] if (np.std(x) > 1e-12 and np.std(y) > 1e-12) else np.nan
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    mae = float(np.mean(np.abs(y - x)))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=8, alpha=0.35)

    lo = min(np.nanmin(x), np.nanmin(y))
    hi = max(np.nanmax(x), np.nanmax(y))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)

    ax.text(
        0.03, 0.97,
        f"n={len(x)}\nr={corr:.3f}\nRMSE={rmse:.3f}\nMAE={mae:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = outdir / stem
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame([{
        "x_col": x_col,
        "y_col": y_label,
        "n_used": len(x),
        "pearson_corr": corr,
        "rmse": rmse,
        "mae": mae,
        "negate_y": negate_y,
    }])
    csv_name = stem.replace(".png", ".csv")
    summary.to_csv(outdir / csv_name, index=False)

    return str(out_png)

# =========================================================
# 3. plot対象列の収集
# =========================================================
def collect_numeric_columns_for_timeseries(
    raw_df: pd.DataFrame,
    exclude_columns: set | None = None,
    min_non_nan_ratio: float = 0.3,
) -> Dict[str, List[str]]:
    if exclude_columns is None:
        exclude_columns = set(EXCLUDE_COLUMNS_DEFAULT)

    grouped: Dict[str, List[str]] = {}
    n = len(raw_df)

    for col in raw_df.columns:
        if col in exclude_columns:
            continue

        s = _to_numeric_series(raw_df[col])
        non_nan_ratio = float(s.notna().mean()) if n > 0 else 0.0
        if non_nan_ratio < min_non_nan_ratio:
            continue

        s_valid = s.dropna()
        if len(s_valid) < 2:
            continue
        if np.isclose(float(s_valid.std()), 0.0):
            continue

        cat = _categorize_column(col)
        grouped.setdefault(cat, []).append(col)

    for k in grouped:
        grouped[k] = sorted(grouped[k], key=lambda x: x.strip().lower())

    return grouped


# =========================================================
# 4. 時系列プロット
# =========================================================
def plot_timeseries_groups(
    raw_df: pd.DataFrame,
    outdir: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    max_series_per_figure: int = 5,
    exclude_columns: set | None = None,
    min_non_nan_ratio: float = 0.3,
    interpolate: bool = False,
) -> Dict[str, List[str]]:
    """
    数値列をカテゴリ別にまとめ、1図あたり最大5系列で時系列プロットする。
    train / val / test の区間は背景色で表示する。
    今回は 高さ・速度・加速度 も含める。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = raw_df.copy()

    # -----------------------------------------------------
    # Add derived columns:
    #  - Accel (Grav) norm
    #  - course from lat/lon
    #  - d(course)/dt
    #  - yaw-like rate from gravity vector and rotation vector
    # -----------------------------------------------------
    time_col = _find_time_column(df)
    df["t_s"] = _to_numeric_series(df[time_col])

    # sort first so derivatives use monotonic time
    df = df.sort_values("t_s").reset_index(drop=True)

    # Accel (Grav) norm
    grav_x_col = _find_column_by_expected_key(df, "accel_grav_x")
    grav_y_col = _find_column_by_expected_key(df, "accel_grav_y")
    grav_z_col = _find_column_by_expected_key(df, "accel_grav_z")

    if grav_x_col and grav_y_col and grav_z_col:
        gx = _to_numeric_series(df[grav_x_col]).to_numpy()
        gy = _to_numeric_series(df[grav_y_col]).to_numpy()
        gz = _to_numeric_series(df[grav_z_col]).to_numpy()

        grav_norm_ftps2 = np.sqrt(gx**2 + gy**2 + gz**2)
        df["Accel (Grav) Norm (ft/s²)"] = grav_norm_ftps2
        df["Accel (Grav) Norm (m/s²)"] = grav_norm_ftps2 * 0.3048

    # course / yaw rate from GPS lat-lon
    lat_col = _find_column_by_expected_key(df, "lat")
    lon_col = _find_column_by_expected_key(df, "lon")

    if lat_col and lon_col:
        lat = _to_numeric_series(df[lat_col]).to_numpy()
        lon = _to_numeric_series(df[lon_col]).to_numpy()
        t_arr = df["t_s"].to_numpy()

        course_rad, yawrate_radps = _compute_course_and_yawrate_from_latlon(lat, lon, t_arr, smooth_window=9)
        df["Course from GPS (rad)"] = course_rad
        df["Course from GPS (deg)"] = np.rad2deg(course_rad)
        df["d(course)/dt (rad/s)"] = yawrate_radps
        df["d(course)/dt (deg/s)"] = np.rad2deg(yawrate_radps)

    # yaw-like rate = rotation vector projected onto gravity direction
    rot_x_col = _find_column_by_expected_key(df, "rot_x")
    rot_y_col = _find_column_by_expected_key(df, "rot_y")
    rot_z_col = _find_column_by_expected_key(df, "rot_z")

    if grav_x_col and grav_y_col and grav_z_col and rot_x_col and rot_y_col and rot_z_col:
        gx = _to_numeric_series(df[grav_x_col]).to_numpy()
        gy = _to_numeric_series(df[grav_y_col]).to_numpy()
        gz = _to_numeric_series(df[grav_z_col]).to_numpy()

        wx = _to_numeric_series(df[rot_x_col]).to_numpy()
        wy = _to_numeric_series(df[rot_y_col]).to_numpy()
        wz = _to_numeric_series(df[rot_z_col]).to_numpy()

        yaw_like_degps, grav_norm_ftps2 = _compute_yaw_like_rate_from_gravity_and_rotation(
            gx, gy, gz, wx, wy, wz, smooth_window=5
        )
        df["Yaw-like rate from grav+rot (deg/s)"] = yaw_like_degps
        df["Yaw-like rate from grav+rot (rad/s)"] = np.deg2rad(yaw_like_degps)

    n_train, n_val, n_test = _make_time_splits(df, train_ratio=train_ratio, val_ratio=val_ratio)
    t = df["t_s"].to_numpy()

    grouped_cols = collect_numeric_columns_for_timeseries(
        df,
        exclude_columns=exclude_columns,
        min_non_nan_ratio=min_non_nan_ratio,
    )

    saved_files: Dict[str, List[str]] = {}

    split_t0 = float(t[0]) if len(t) > 0 else 0.0
    split_t1 = float(t[n_train - 1]) if n_train > 0 else split_t0
    split_t2 = float(t[n_train + n_val - 1]) if (n_train + n_val) > 0 else split_t1
    split_t3 = float(t[-1]) if len(t) > 0 else split_t2

    for category, cols in grouped_cols.items():
        print(category, cols)
        chunks = _chunk_list(cols, max_series_per_figure)
        saved_files[category] = []

        for i, chunk in enumerate(chunks, start=1):
            fig, axes = plt.subplots(len(chunk), 1, figsize=(14, 2.8 * len(chunk)), sharex=True)
            if len(chunk) == 1:
                axes = [axes]

            for ax, col in zip(axes, chunk):
                y = _to_numeric_series(df[col])
                if interpolate:
                    y = y.interpolate(limit_direction="both")

                ax.plot(t, y, linewidth=1.2)
                ax.set_ylabel(_clean_label(col), fontsize=9)
                ax.grid(True, alpha=0.3)

                # train / val / test 背景
                ax.axvspan(split_t0, split_t1, alpha=0.08, color="#1f77b4")
                ax.axvspan(split_t1, split_t2, alpha=0.08, color="#ff7f0e")
                ax.axvspan(split_t2, split_t3, alpha=0.08, color="#2ca02c")

            axes[0].set_title(
                f"{category} ({i}/{len(chunks)})  "
                f"[blue=train, orange=val, green=test]"
            )
            axes[-1].set_xlabel("time [s]")

            fig.tight_layout()
            out_png = outdir / f"timeseries_{category}_{i:02d}.png"
            fig.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close(fig)

            saved_files[category].append(str(out_png))

    # summary 保存
    summary_rows = []
    for category, cols in grouped_cols.items():
        for col in cols:
            summary_rows.append({
                "category": category,
                "column": col,
                "column_clean": _clean_label(col),
            })
    pd.DataFrame(summary_rows).to_csv(outdir / "timeseries_column_groups.csv", index=False)

    split_summary = pd.DataFrame([{
        "n_total": len(df),
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "t_start": split_t0,
        "t_train_end": split_t1,
        "t_val_end": split_t2,
        "t_end": split_t3,
    }])
    split_summary.to_csv(outdir / "timeseries_split_summary.csv", index=False)

    yaw_cmp = plot_yaw_comparison(df, outdir, train_ratio=train_ratio, val_ratio=val_ratio)
    if yaw_cmp is not None:
        saved_files.setdefault("attitude_rotation", []).append(yaw_cmp)

    yaw_scatter = plot_yaw_scatter_comparison(
        df,
        outdir,
        x_col="d(course)/dt (deg/s)",
        y_col="Yaw-like rate from grav+rot (deg/s)",
        negate_y=False,
    )
    if yaw_scatter is not None:
        saved_files.setdefault("attitude_rotation", []).append(yaw_scatter)

    return saved_files


def plot_yaw_comparison(
    df: pd.DataFrame,
    outdir: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> str | None:
    needed = [
        "Yaw-like rate from grav+rot (deg/s)",
        "d(course)/dt (deg/s)",
    ]
    if not all(c in df.columns for c in needed):
        return None

    n_train, n_val, n_test = _make_time_splits(df, train_ratio=train_ratio, val_ratio=val_ratio)
    t = df["t_s"].to_numpy()

    split_t0 = float(t[0]) if len(t) > 0 else 0.0
    split_t1 = float(t[n_train - 1]) if n_train > 0 else split_t0
    split_t2 = float(t[n_train + n_val - 1]) if (n_train + n_val) > 0 else split_t1
    split_t3 = float(t[-1]) if len(t) > 0 else split_t2

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, _to_numeric_series(df["Yaw-like rate from grav+rot (deg/s)"]), label="yaw-like from grav+rot", linewidth=1.2)
    ax.plot(t, _to_numeric_series(df["d(course)/dt (deg/s)"]), label="d(course)/dt from GPS", linewidth=1.2)

    ax.axvspan(split_t0, split_t1, alpha=0.08, color="#1f77b4")
    ax.axvspan(split_t1, split_t2, alpha=0.08, color="#ff7f0e")
    ax.axvspan(split_t2, split_t3, alpha=0.08, color="#2ca02c")

    ax.set_title("Yaw-like rate vs d(course)/dt  [blue=train, orange=val, green=test]")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("deg/s")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_png = outdir / "timeseries_yaw_comparison.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)

# =========================================================
# 5. 使用例
# =========================================================
if __name__ == "__main__":
    # 例:
    raw_df = pd.read_csv("./data/auto_iowa.csv")
    outputs = plot_timeseries_groups(
        raw_df,
        outdir="timeseries_groups",
        train_ratio=0.6,
        val_ratio=0.2,
        max_series_per_figure=5,
        interpolate=False,
    )
    print(outputs)
    pass