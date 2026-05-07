import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# PhysicsNeMo core model
from physicsnemo.models.mlp.fully_connected import FullyConnected


"""
Minimal CPU-friendly PhysicsNeMo PoC for skid-risk prediction and trajectory rollout.

This version separates:
1. Teacher physics model (deterministic dynamic bicycle step)
2. PhysicsNeMo surrogate model
3. Test functions to verify the teacher model behavior first

Recommended environment
-----------------------
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch nvidia-physicsnemo matplotlib
python physicsnemo_skid_poc.py --run-tests
python physicsnemo_skid_poc.py --train-and-compare
"""


torch.manual_seed(42)
DEVICE = torch.device("cpu")
DTYPE = torch.float32


@dataclass
class VehicleParams:
    mass: float = 1500.0
    iz: float = 2500.0
    lf: float = 1.2
    lr: float = 1.6
    cf: float = 80000.0
    cr: float = 80000.0
    g: float = 9.81
    dt: float = 0.1

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr


P = VehicleParams()


# -----------------------------------------------------------------------------
# 1. Teacher physics model
# -----------------------------------------------------------------------------

def teacher_one_step_from_features(x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Input columns:
      0  v_x
      1  beta
      2  r
      3  delta
      4  a_x
      5  mu
      6  bank
      7  grade
      8  lf
      9  lr
      10 cf
      11 cr
      12 mass
      13 iz

    Output columns:
      0 beta_next
      1 r_next
      2 ay_next
      3 slip_margin
      4 slip_risk
    """
    v_x = x[:, 0].clamp_min(1.0)
    beta = x[:, 1]
    r = x[:, 2]
    delta = x[:, 3]
    a_x = x[:, 4]
    mu = x[:, 5].clamp_min(0.05)
    bank = x[:, 6]
    grade = x[:, 7]
    lf = x[:, 8]
    lr = x[:, 9]
    cf = x[:, 10]
    cr = x[:, 11]
    mass = x[:, 12]
    iz = x[:, 13]

    alpha_f = delta - beta - (lf * r / v_x)
    alpha_r = -beta + (lr * r / v_x)

    fyf_lin = cf * alpha_f
    fyr_lin = cr * alpha_r

    fzf = mass * P.g * (lr / (lf + lr)) * torch.cos(grade)
    fzr = mass * P.g * (lf / (lf + lr)) * torch.cos(grade)

    fyf_max = mu * fzf
    fyr_max = mu * fzr
    fyf = fyf_lin.clamp(min=-fyf_max, max=fyf_max)
    fyr = fyr_lin.clamp(min=-fyr_max, max=fyr_max)

    ay = (fyf + fyr) / mass - P.g * torch.sin(bank)
    beta_dot = (fyf + fyr) / (mass * v_x) - r
    r_dot = (lf * fyf - lr * fyr) / iz

    beta_next = beta + P.dt * beta_dot
    r_next = r + P.dt * r_dot
    ay_next = ay

    ay_limit = mu * P.g * torch.cos(bank).clamp_min(0.5)
    slip_margin = 1.0 - torch.abs(ay_next) / ay_limit.clamp_min(1e-3)
    slip_risk = (slip_margin < 0.0).to(DTYPE)

    y = torch.stack([beta_next, r_next, ay_next, slip_margin, slip_risk], dim=1)
    aux = {
        "alpha_f": alpha_f,
        "alpha_r": alpha_r,
        "fyf": fyf,
        "fyr": fyr,
        "ay_limit": ay_limit,
        "v_x_next": (v_x + P.dt * a_x).clamp_min(0.5),
    }
    return y.to(DTYPE), aux


def teacher_rollout_step(
    v_x: float,
    beta: float,
    r: float,
    yaw: float,
    x_pos: float,
    y_pos: float,
    delta: float,
    a_x: float,
    mu: float,
    bank: float = 0.0,
    grade: float = 0.0,
    params: VehicleParams = P,
) -> Dict[str, float]:
    inp = torch.tensor(
        [[
            v_x,
            beta,
            r,
            delta,
            a_x,
            mu,
            bank,
            grade,
            params.lf,
            params.lr,
            params.cf,
            params.cr,
            params.mass,
            params.iz,
        ]],
        dtype=DTYPE,
    )
    out, _ = teacher_one_step_from_features(inp)
    beta_next = float(out[0, 0].item())
    r_next = float(out[0, 1].item())
    ay_next = float(out[0, 2].item())
    margin = float(out[0, 3].item())
    p_slip = float(out[0, 4].item())
    v_next = max(0.5, v_x + params.dt * a_x)
    yaw_next = yaw + params.dt * r_next
    heading_eff = yaw + beta
    x_next = x_pos + params.dt * v_x * math.cos(heading_eff)
    y_next = y_pos + params.dt * v_x * math.sin(heading_eff)
    return {
        "v_x": v_next,
        "beta": beta_next,
        "r": r_next,
        "yaw": yaw_next,
        "x": x_next,
        "y": y_next,
        "ay": ay_next,
        "margin": margin,
        "slip_prob": p_slip,
    }


# -----------------------------------------------------------------------------
# 2. Dataset generation for surrogate learning
# -----------------------------------------------------------------------------

def sample_inputs(n: int) -> torch.Tensor:
    v_x = torch.empty(n).uniform_(8.0, 35.0)
    beta = torch.empty(n).uniform_(-0.08, 0.08)
    r = torch.empty(n).uniform_(-0.6, 0.6)
    delta = torch.empty(n).uniform_(-0.12, 0.12)
    a_x = torch.empty(n).uniform_(-4.0, 2.0)
    mu = torch.empty(n).uniform_(0.15, 1.0)
    bank = torch.empty(n).uniform_(-0.08, 0.08)
    grade = torch.empty(n).uniform_(-0.06, 0.06)
    lf = torch.empty(n).uniform_(1.1, 1.4)
    lr = torch.empty(n).uniform_(1.4, 1.8)
    cf = torch.empty(n).uniform_(70000.0, 95000.0)
    cr = torch.empty(n).uniform_(70000.0, 95000.0)
    mass = torch.empty(n).uniform_(1300.0, 1900.0)
    iz = torch.empty(n).uniform_(2200.0, 3200.0)
    return torch.stack([v_x, beta, r, delta, a_x, mu, bank, grade, lf, lr, cf, cr, mass, iz], dim=1).to(DTYPE)


# -----------------------------------------------------------------------------
# 3. PhysicsNeMo surrogate
# -----------------------------------------------------------------------------
class SkidRiskNet(nn.Module):
    def __init__(self, in_features: int, hidden_layers: int = 3, layer_size: int = 128):
        super().__init__()
        self.backbone = FullyConnected(
            in_features=in_features,
            out_features=5,
            num_layers=hidden_layers,
            layer_size=layer_size,
            activation_fn="silu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def physics_consistency_loss(x: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    v_x = x[:, 0].clamp_min(1.0)
    beta = x[:, 1]
    r = x[:, 2]
    delta = x[:, 3]
    mu = x[:, 5].clamp_min(0.05)
    bank = x[:, 6]
    lf = x[:, 8]
    lr = x[:, 9]
    cf = x[:, 10]
    cr = x[:, 11]
    mass = x[:, 12]
    iz = x[:, 13]

    beta_next = pred[:, 0]
    r_next = pred[:, 1]
    ay_next = pred[:, 2]
    slip_margin = pred[:, 3]

    beta_dot_pred = (beta_next - beta) / P.dt
    r_dot_pred = (r_next - r) / P.dt

    alpha_f = delta - beta - (lf * r / v_x)
    alpha_r = -beta + (lr * r / v_x)
    fyf_lin = cf * alpha_f
    fyr_lin = cr * alpha_r

    fzf = mass * P.g * (lr / (lf + lr))
    fzr = mass * P.g * (lf / (lf + lr))
    fyf_max = mu * fzf
    fyr_max = mu * fzr
    fyf = fyf_lin.clamp(min=-fyf_max, max=fyf_max)
    fyr = fyr_lin.clamp(min=-fyr_max, max=fyr_max)

    beta_dot_phys = (fyf + fyr) / (mass * v_x) - r
    r_dot_phys = (lf * fyf - lr * fyr) / iz
    ay_phys = (fyf + fyr) / mass - P.g * torch.sin(bank)
    ay_limit = mu * P.g * torch.cos(bank).clamp_min(0.5)
    margin_phys = 1.0 - torch.abs(ay_phys) / ay_limit.clamp_min(1e-3)

    return (
        nn.functional.mse_loss(beta_dot_pred, beta_dot_phys)
        + nn.functional.mse_loss(r_dot_pred, r_dot_phys)
        + nn.functional.mse_loss(ay_next, ay_phys)
        + nn.functional.mse_loss(slip_margin, margin_phys)
    )


def make_dataloaders(n_train: int = 16000, n_val: int = 4000, batch_size: int = 256):
    x_train = sample_inputs(n_train)
    y_train, _ = teacher_one_step_from_features(x_train)
    x_val = sample_inputs(n_val)
    y_val, _ = teacher_one_step_from_features(x_val)
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False),
    )


def train_model(epochs: int = 20) -> SkidRiskNet:
    train_loader, val_loader = make_dataloaders()
    model = SkidRiskNet(in_features=14).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            data_loss = (
                nn.functional.mse_loss(pred[:, 0], yb[:, 0])
                + nn.functional.mse_loss(pred[:, 1], yb[:, 1])
                + nn.functional.mse_loss(pred[:, 2], yb[:, 2])
                + nn.functional.mse_loss(pred[:, 3], yb[:, 3])
            )
            cls_loss = bce(pred[:, 4], yb[:, 4])
            phys_loss = physics_consistency_loss(xb, pred)
            loss = data_loss + 0.5 * cls_loss + 0.2 * phys_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)

        model.eval()
        val_loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                data_loss = (
                    nn.functional.mse_loss(pred[:, 0], yb[:, 0])
                    + nn.functional.mse_loss(pred[:, 1], yb[:, 1])
                    + nn.functional.mse_loss(pred[:, 2], yb[:, 2])
                    + nn.functional.mse_loss(pred[:, 3], yb[:, 3])
                )
                cls_loss = bce(pred[:, 4], yb[:, 4])
                phys_loss = physics_consistency_loss(xb, pred)
                val_loss_sum += (data_loss + 0.5 * cls_loss + 0.2 * phys_loss).item() * xb.size(0)
                pred_cls = (torch.sigmoid(pred[:, 4]) > 0.5).to(DTYPE)
                correct += (pred_cls == yb[:, 4]).sum().item()
                total += xb.size(0)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss_sum / len(train_loader.dataset):.5f} | "
            f"val_loss={val_loss_sum / len(val_loader.dataset):.5f} | val_risk_acc={correct / total:.4f}"
        )
    return model


# -----------------------------------------------------------------------------
# 4. Road geometry and controller
# -----------------------------------------------------------------------------

def build_curvature_profile(
    speed_mps: float,
    dt: float,
    straight1_m: float = 100.0,
    turn_radius_m: float = 25.0,
    turn_angle_deg: float = 90.0,
    straight2_m: float = 100.0,
) -> torch.Tensor:
    turn_angle_rad = math.radians(turn_angle_deg)
    turn_arc_m = turn_radius_m * turn_angle_rad
    n1 = max(1, int(round(straight1_m / (speed_mps * dt))))
    n_turn = max(1, int(round(turn_arc_m / (speed_mps * dt))))
    n2 = max(1, int(round(straight2_m / (speed_mps * dt))))
    return torch.cat([
        torch.zeros(n1, dtype=DTYPE),
        torch.full((n_turn,), 1.0 / turn_radius_m, dtype=DTYPE),
        torch.zeros(n2, dtype=DTYPE),
    ])


def build_reference_path(
    speed_mps: float,
    dt: float,
    straight1_m: float = 100.0,
    turn_radius_m: float = 25.0,
    turn_angle_deg: float = 90.0,
    straight2_m: float = 100.0,
) -> Dict[str, torch.Tensor]:
    kappa_seq = build_curvature_profile(speed_mps, dt, straight1_m, turn_radius_m, turn_angle_deg, straight2_m)
    n = len(kappa_seq)
    ds = speed_mps * dt
    x_ref = torch.zeros(n + 1, dtype=DTYPE)
    y_ref = torch.zeros(n + 1, dtype=DTYPE)
    psi_ref = torch.zeros(n + 1, dtype=DTYPE)
    for t in range(n):
        psi_ref[t + 1] = psi_ref[t] + ds * kappa_seq[t]
        x_ref[t + 1] = x_ref[t] + ds * torch.cos(psi_ref[t])
        y_ref[t + 1] = y_ref[t] + ds * torch.sin(psi_ref[t])
    return {"x_ref": x_ref, "y_ref": y_ref, "psi_ref": psi_ref, "kappa_seq": kappa_seq}


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def steering_feedforward(kappa: float, wheelbase: float) -> float:
    return math.atan(wheelbase * kappa)


def steering_feedback(
    x: float,
    y: float,
    yaw: float,
    x_ref: float,
    y_ref: float,
    psi_ref: float,
    k_y: float = 0.01,
    k_psi: float = 0.40,
) -> Tuple[float, float, float]:
    dx = x - x_ref
    dy = y - y_ref
    e_y = -math.sin(psi_ref) * dx + math.cos(psi_ref) * dy
    e_psi = wrap_angle(yaw - psi_ref)
    delta_fb = -(k_y * e_y + k_psi * e_psi)
    return delta_fb, e_y, e_psi


# -----------------------------------------------------------------------------
# 5. Rollout functions
# -----------------------------------------------------------------------------

def rollout_teacher(
    v0: float,
    mu: float,
    steering_scale: float = 1.0,
    bank: float = 0.0,
    grade: float = 0.0,
    a_x_value: float = 0.0,
    k_y: float = 0.01,
    k_psi: float = 0.40,
) -> Dict[str, torch.Tensor]:
    road = build_reference_path(speed_mps=v0, dt=P.dt)
    x_ref, y_ref, psi_ref, kappa_seq = road["x_ref"], road["y_ref"], road["psi_ref"], road["kappa_seq"]
    n = len(kappa_seq)
    x = torch.zeros(n + 1, dtype=DTYPE)
    y = torch.zeros(n + 1, dtype=DTYPE)
    yaw = torch.zeros(n + 1, dtype=DTYPE)
    v_x = torch.zeros(n + 1, dtype=DTYPE)
    beta = torch.zeros(n + 1, dtype=DTYPE)
    r = torch.zeros(n + 1, dtype=DTYPE)
    ay = torch.zeros(n, dtype=DTYPE)
    margin = torch.zeros(n, dtype=DTYPE)
    slip_prob = torch.zeros(n, dtype=DTYPE)
    delta_ff_hist = torch.zeros(n, dtype=DTYPE)
    delta_fb_hist = torch.zeros(n, dtype=DTYPE)
    delta_used = torch.zeros(n, dtype=DTYPE)
    e_y_hist = torch.zeros(n, dtype=DTYPE)
    e_psi_hist = torch.zeros(n, dtype=DTYPE)
    v_x[0] = v0

    for t in range(n):
        delta_ff = steering_feedforward(float(kappa_seq[t].item()), P.wheelbase) * steering_scale
        delta_fb, e_y, e_psi = steering_feedback(
            x=float(x[t].item()),
            y=float(y[t].item()),
            yaw=float(yaw[t].item()),
            x_ref=float(x_ref[t].item()),
            y_ref=float(y_ref[t].item()),
            psi_ref=float(psi_ref[t].item()),
            k_y=k_y,
            k_psi=k_psi,
        )
        delta = max(min(delta_ff + delta_fb, 0.25), -0.25)
        nxt = teacher_rollout_step(
            v_x=float(v_x[t].item()),
            beta=float(beta[t].item()),
            r=float(r[t].item()),
            yaw=float(yaw[t].item()),
            x_pos=float(x[t].item()),
            y_pos=float(y[t].item()),
            delta=delta,
            a_x=a_x_value,
            mu=mu,
            bank=bank,
            grade=grade,
        )
        v_x[t + 1] = nxt["v_x"]
        beta[t + 1] = nxt["beta"]
        r[t + 1] = nxt["r"]
        yaw[t + 1] = nxt["yaw"]
        x[t + 1] = nxt["x"]
        y[t + 1] = nxt["y"]
        ay[t] = nxt["ay"]
        margin[t] = nxt["margin"]
        slip_prob[t] = nxt["slip_prob"]
        delta_ff_hist[t] = delta_ff
        delta_fb_hist[t] = delta_fb
        delta_used[t] = delta
        e_y_hist[t] = e_y
        e_psi_hist[t] = e_psi

    return {
        "x": x, "y": y, "yaw": yaw, "v_x": v_x, "beta": beta, "r": r,
        "ay": ay, "margin": margin, "slip_prob": slip_prob,
        "delta_ff": delta_ff_hist, "delta_fb": delta_fb_hist, "delta_used": delta_used,
        "e_y": e_y_hist, "e_psi": e_psi_hist,
        "x_ref": x_ref, "y_ref": y_ref, "psi_ref": psi_ref, "kappa_seq": kappa_seq,
    }


def rollout_surrogate(
    model: SkidRiskNet,
    v0: float,
    mu: float,
    steering_scale: float = 1.0,
    bank: float = 0.0,
    grade: float = 0.0,
    a_x_value: float = 0.0,
    k_y: float = 0.01,
    k_psi: float = 0.40,
) -> Dict[str, torch.Tensor]:
    road = build_reference_path(speed_mps=v0, dt=P.dt)
    x_ref, y_ref, psi_ref, kappa_seq = road["x_ref"], road["y_ref"], road["psi_ref"], road["kappa_seq"]
    n = len(kappa_seq)
    x = torch.zeros(n + 1, dtype=DTYPE)
    y = torch.zeros(n + 1, dtype=DTYPE)
    yaw = torch.zeros(n + 1, dtype=DTYPE)
    v_x = torch.zeros(n + 1, dtype=DTYPE)
    beta = torch.zeros(n + 1, dtype=DTYPE)
    r = torch.zeros(n + 1, dtype=DTYPE)
    ay = torch.zeros(n, dtype=DTYPE)
    margin = torch.zeros(n, dtype=DTYPE)
    slip_prob = torch.zeros(n, dtype=DTYPE)
    delta_ff_hist = torch.zeros(n, dtype=DTYPE)
    delta_fb_hist = torch.zeros(n, dtype=DTYPE)
    delta_used = torch.zeros(n, dtype=DTYPE)
    e_y_hist = torch.zeros(n, dtype=DTYPE)
    e_psi_hist = torch.zeros(n, dtype=DTYPE)
    v_x[0] = v0
    model.eval()

    for t in range(n):
        delta_ff = steering_feedforward(float(kappa_seq[t].item()), P.wheelbase) * steering_scale
        delta_fb, e_y, e_psi = steering_feedback(
            x=float(x[t].item()),
            y=float(y[t].item()),
            yaw=float(yaw[t].item()),
            x_ref=float(x_ref[t].item()),
            y_ref=float(y_ref[t].item()),
            psi_ref=float(psi_ref[t].item()),
            k_y=k_y,
            k_psi=k_psi,
        )
        delta = max(min(delta_ff + delta_fb, 0.25), -0.25)
        inp = torch.tensor([[v_x[t].item(), beta[t].item(), r[t].item(), delta, a_x_value, mu, bank, grade,
                             P.lf, P.lr, P.cf, P.cr, P.mass, P.iz]], dtype=DTYPE, device=DEVICE)
        with torch.no_grad():
            pred = model(inp)[0].cpu()
        beta[t + 1] = pred[0]
        r[t + 1] = pred[1]
        ay[t] = pred[2]
        margin[t] = pred[3]
        slip_prob[t] = torch.sigmoid(pred[4])
        v_x[t + 1] = max(0.5, v_x[t].item() + P.dt * a_x_value)
        yaw[t + 1] = yaw[t] + P.dt * r[t + 1]
        heading_eff = yaw[t] + beta[t]
        x[t + 1] = x[t] + P.dt * v_x[t] * torch.cos(heading_eff)
        y[t + 1] = y[t] + P.dt * v_x[t] * torch.sin(heading_eff)
        delta_ff_hist[t] = delta_ff
        delta_fb_hist[t] = delta_fb
        delta_used[t] = delta
        e_y_hist[t] = e_y
        e_psi_hist[t] = e_psi

    return {
        "x": x, "y": y, "yaw": yaw, "v_x": v_x, "beta": beta, "r": r,
        "ay": ay, "margin": margin, "slip_prob": slip_prob,
        "delta_ff": delta_ff_hist, "delta_fb": delta_fb_hist, "delta_used": delta_used,
        "e_y": e_y_hist, "e_psi": e_psi_hist,
        "x_ref": x_ref, "y_ref": y_ref, "psi_ref": psi_ref, "kappa_seq": kappa_seq,
    }


# -----------------------------------------------------------------------------
# 6. Tests
# -----------------------------------------------------------------------------

def test_straight_zero_steering() -> None:
    res = rollout_teacher(v0=10.0, mu=0.9, steering_scale=0.0, k_y=0.0, k_psi=0.0)
    max_abs_y = float(res["y"].abs().max().item())
    final_x = float(res["x"][-1].item())
    assert max_abs_y < 1e-3, f"Straight run should stay near y=0, got max |y|={max_abs_y:.6f}"
    assert final_x > 150.0, f"Straight run should move forward, got final x={final_x:.3f}"
    print("PASS: test_straight_zero_steering")


def test_turn_has_positive_lateral_motion() -> None:
    res = rollout_teacher(v0=10.0, mu=0.9)
    final_y = float(res["y"][-1].item())
    assert final_y > 80.0, f"Turn scenario should reach positive y after 90deg turn, got final y={final_y:.3f}"
    print("PASS: test_turn_has_positive_lateral_motion")


def test_low_mu_or_high_speed_increases_deviation() -> None:
    dry = rollout_teacher(v0=10.0, mu=0.9)
    icy_fast = rollout_teacher(v0=16.0, mu=0.2)
    dry_max_dev = float(dry["e_y"].abs().max().item())
    icy_max_dev = float(icy_fast["e_y"].abs().max().item())
    assert icy_max_dev > dry_max_dev, (
        f"Low-mu/high-speed case should deviate more. dry={dry_max_dev:.3f}, icy_fast={icy_max_dev:.3f}"
    )
    print("PASS: test_low_mu_or_high_speed_increases_deviation")


def test_feedback_steering_zero_on_initial_straight() -> None:
    road = build_reference_path(speed_mps=10.0, dt=P.dt)
    delta_ff = steering_feedforward(float(road["kappa_seq"][0].item()), P.wheelbase)
    delta_fb, e_y, e_psi = steering_feedback(0.0, 0.0, 0.0, float(road["x_ref"][0].item()), float(road["y_ref"][0].item()), float(road["psi_ref"][0].item()))
    assert abs(delta_ff) < 1e-9, f"Initial straight feedforward steering should be 0, got {delta_ff}"
    assert abs(delta_fb) < 1e-9, f"Initial feedback steering should be 0, got {delta_fb}"
    assert abs(e_y) < 1e-9 and abs(e_psi) < 1e-9
    print("PASS: test_feedback_steering_zero_on_initial_straight")


def run_tests() -> None:
    test_straight_zero_steering()
    test_turn_has_positive_lateral_motion()
    test_low_mu_or_high_speed_increases_deviation()
    test_feedback_steering_zero_on_initial_straight()
    print("All teacher-model tests passed.")


# -----------------------------------------------------------------------------
# 7. Visualization / comparison
# -----------------------------------------------------------------------------

def compare_teacher_scenarios() -> None:
    scenarios = [
        {"name": "dry_slow", "v0": 10.0, "mu": 0.90},
        {"name": "dry_nominal", "v0": 12.0, "mu": 0.90},
        {"name": "wet_nominal", "v0": 12.0, "mu": 0.50},
        {"name": "icy_nominal", "v0": 12.0, "mu": 0.20},
        {"name": "icy_fast", "v0": 16.0, "mu": 0.20},
    ]
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 7))
        ref_plotted = False
        for sc in scenarios:
            res = rollout_teacher(v0=sc["v0"], mu=sc["mu"])
            if not ref_plotted:
                plt.plot(res["x_ref"].numpy(), res["y_ref"].numpy(), "k--", linewidth=2.0, label="road centerline")
                ref_plotted = True
            plt.plot(res["x"].numpy(), res["y"].numpy(), linewidth=2.2, label=sc["name"])
            print(
                f"{sc['name']:<12} | max|e_y|={res['e_y'].abs().max().item():.2f} m | "
                f"max_P(slip)={res['slip_prob'].max().item():.3f}"
            )
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Teacher model: 100m straight -> 90deg turn -> 100m straight")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("teacher_trajectory_comparison.png", dpi=180)
        plt.close()
        print("Saved teacher_trajectory_comparison.png")
    except Exception as e:
        print(f"Warning: plotting failed: {e}")


def compare_surrogate_scenarios(model: SkidRiskNet) -> None:
    scenarios = [
        {"name": "dry_slow", "v0": 10.0, "mu": 0.90},
        {"name": "dry_nominal", "v0": 12.0, "mu": 0.90},
        {"name": "wet_nominal", "v0": 12.0, "mu": 0.50},
        {"name": "icy_nominal", "v0": 12.0, "mu": 0.20},
        {"name": "icy_fast", "v0": 16.0, "mu": 0.20},
    ]
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 7))
        ref_plotted = False
        for sc in scenarios:
            res = rollout_surrogate(model, v0=sc["v0"], mu=sc["mu"])
            if not ref_plotted:
                plt.plot(res["x_ref"].numpy(), res["y_ref"].numpy(), "k--", linewidth=2.0, label="road centerline")
                ref_plotted = True
            plt.plot(res["x"].numpy(), res["y"].numpy(), linewidth=2.2, label=sc["name"])
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Surrogate model rollout")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("surrogate_trajectory_comparison.png", dpi=180)
        plt.close()
        print("Saved surrogate_trajectory_comparison.png")
    except Exception as e:
        print(f"Warning: plotting failed: {e}")


# -----------------------------------------------------------------------------
# 8. Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if "--run-tests" in sys.argv:
        run_tests()
        compare_teacher_scenarios()
    elif "--train-and-compare" in sys.argv:
        run_tests()
        compare_teacher_scenarios()
        model = train_model(epochs=20)
        compare_surrogate_scenarios(model)
    else:
        print("Usage:")
        print("  python physicsnemo_skid_poc.py --run-tests")
        print("  python physicsnemo_skid_poc.py --train-and-compare")
