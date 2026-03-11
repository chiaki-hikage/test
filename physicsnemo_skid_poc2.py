import math
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# PhysicsNeMo core model
from physicsnemo.models.mlp.fully_connected import FullyConnected


"""
Improved CPU-friendly PhysicsNeMo PoC for skid-risk prediction and trajectory rollout.

Improvements over the minimal version
-------------------------------------
1. Predict beta_dot and r_dot instead of beta_next and r_next
2. Standardize inputs and regression targets
3. Proper train / val / test split
4. Save best model and normalization stats
5. Clear 1-step metrics and rollout comparison
6. Keep teacher model separate from surrogate model

Recommended environment
-----------------------
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch nvidia-physicsnemo matplotlib

Run
---
python physicsnemo_skid_poc_improved.py --run-tests
python physicsnemo_skid_poc_improved.py --train
python physicsnemo_skid_poc_improved.py --evaluate
"""


torch.manual_seed(42)
DEVICE = torch.device("cpu")
DTYPE = torch.float32
ARTIFACT_DIR = Path("artifacts_physicsnemo")
ARTIFACT_DIR.mkdir(exist_ok=True)


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

def teacher_targets_from_features(x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
      0 beta_dot
      1 r_dot
      2 ay
      3 slip_margin
      4 slip_risk (binary label)
    """
    v_x = x[:, 0].clamp_min(1.0)
    beta = x[:, 1]
    r = x[:, 2]
    delta = x[:, 3]
    a_x = x[:, 4]  # kept for consistency / future use
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

    ay_limit = mu * P.g * torch.cos(bank).clamp_min(0.5)
    slip_margin = 1.0 - torch.abs(ay) / ay_limit.clamp_min(1e-3)
    slip_risk = (slip_margin < 0.0).to(DTYPE)

    y = torch.stack([beta_dot, r_dot, ay, slip_margin, slip_risk], dim=1)
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
    out, _ = teacher_targets_from_features(inp)
    beta_dot = float(out[0, 0].item())
    r_dot = float(out[0, 1].item())
    ay = float(out[0, 2].item())
    margin = float(out[0, 3].item())
    p_slip = float(out[0, 4].item())

    beta_next = beta + params.dt * beta_dot
    r_next = r + params.dt * r_dot
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
        "ay": ay,
        "margin": margin,
        "slip_prob": p_slip,
    }


# -----------------------------------------------------------------------------
# 2. Dataset generation
# -----------------------------------------------------------------------------

def _sample_uniform_block(
    n: int,
    v_x_rng: Tuple[float, float],
    beta_rng: Tuple[float, float],
    r_rng: Tuple[float, float],
    delta_rng: Tuple[float, float],
    a_x_rng: Tuple[float, float],
    mu_rng: Tuple[float, float],
    bank_rng: Tuple[float, float],
    grade_rng: Tuple[float, float],
    lf_rng: Tuple[float, float],
    lr_rng: Tuple[float, float],
    cf_rng: Tuple[float, float],
    cr_rng: Tuple[float, float],
    mass_rng: Tuple[float, float],
    iz_rng: Tuple[float, float],
) -> torch.Tensor:
    def u(rng):
        return torch.empty(n).uniform_(rng[0], rng[1])

    return torch.stack(
        [
            u(v_x_rng),
            u(beta_rng),
            u(r_rng),
            u(delta_rng),
            u(a_x_rng),
            u(mu_rng),
            u(bank_rng),
            u(grade_rng),
            u(lf_rng),
            u(lr_rng),
            u(cf_rng),
            u(cr_rng),
            u(mass_rng),
            u(iz_rng),
        ],
        dim=1,
    ).to(DTYPE)


def sample_inputs_mixture(n: int) -> torch.Tensor:
    """
    Mixture sampling:
      60% nominal
      30% near-limit
      10% extreme
    """
    n_nom = int(0.60 * n)
    n_lim = int(0.30 * n)
    n_ext = n - n_nom - n_lim

    nominal = _sample_uniform_block(
        n_nom,
        v_x_rng=(8.0, 20.0),
        beta_rng=(-0.04, 0.04),
        r_rng=(-0.35, 0.35),
        delta_rng=(-0.08, 0.08),
        a_x_rng=(-2.0, 1.5),
        mu_rng=(0.50, 1.00),
        bank_rng=(-0.04, 0.04),
        grade_rng=(-0.03, 0.03),
        lf_rng=(1.1, 1.4),
        lr_rng=(1.4, 1.8),
        cf_rng=(70000.0, 95000.0),
        cr_rng=(70000.0, 95000.0),
        mass_rng=(1300.0, 1900.0),
        iz_rng=(2200.0, 3200.0),
    )

    near_limit = _sample_uniform_block(
        n_lim,
        v_x_rng=(12.0, 30.0),
        beta_rng=(-0.08, 0.08),
        r_rng=(-0.60, 0.60),
        delta_rng=(-0.12, 0.12),
        a_x_rng=(-4.0, 2.0),
        mu_rng=(0.15, 0.45),
        bank_rng=(-0.08, 0.08),
        grade_rng=(-0.06, 0.06),
        lf_rng=(1.1, 1.4),
        lr_rng=(1.4, 1.8),
        cf_rng=(70000.0, 95000.0),
        cr_rng=(70000.0, 95000.0),
        mass_rng=(1300.0, 1900.0),
        iz_rng=(2200.0, 3200.0),
    )

    extreme = _sample_uniform_block(
        n_ext,
        v_x_rng=(20.0, 35.0),
        beta_rng=(-0.12, 0.12),
        r_rng=(-0.90, 0.90),
        delta_rng=(-0.18, 0.18),
        a_x_rng=(-6.0, 3.0),
        mu_rng=(0.08, 0.25),
        bank_rng=(-0.10, 0.10),
        grade_rng=(-0.08, 0.08),
        lf_rng=(1.1, 1.4),
        lr_rng=(1.4, 1.8),
        cf_rng=(65000.0, 100000.0),
        cr_rng=(65000.0, 100000.0),
        mass_rng=(1200.0, 2100.0),
        iz_rng=(2000.0, 3400.0),
    )

    x = torch.cat([nominal, near_limit, extreme], dim=0)
    perm = torch.randperm(x.shape[0])
    return x[perm]


# -----------------------------------------------------------------------------
# 3. Standardization
# -----------------------------------------------------------------------------

@dataclass
class NormalizationStats:
    x_mean: torch.Tensor
    x_std: torch.Tensor
    y_reg_mean: torch.Tensor
    y_reg_std: torch.Tensor

    def to_dict(self) -> Dict[str, list]:
        return {
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_reg_mean": self.y_reg_mean.tolist(),
            "y_reg_std": self.y_reg_std.tolist(),
        }

    @staticmethod
    def from_dict(d: Dict[str, list]) -> "NormalizationStats":
        return NormalizationStats(
            x_mean=torch.tensor(d["x_mean"], dtype=DTYPE),
            x_std=torch.tensor(d["x_std"], dtype=DTYPE),
            y_reg_mean=torch.tensor(d["y_reg_mean"], dtype=DTYPE),
            y_reg_std=torch.tensor(d["y_reg_std"], dtype=DTYPE),
        )


def compute_normalization_stats(x_train: torch.Tensor, y_train: torch.Tensor) -> NormalizationStats:
    eps = 1e-6
    x_mean = x_train.mean(dim=0)
    x_std = x_train.std(dim=0).clamp_min(eps)

    # Only regression outputs are normalized: [beta_dot, r_dot, ay, slip_margin]
    y_reg = y_train[:, :4]
    y_reg_mean = y_reg.mean(dim=0)
    y_reg_std = y_reg.std(dim=0).clamp_min(eps)

    return NormalizationStats(
        x_mean=x_mean,
        x_std=x_std,
        y_reg_mean=y_reg_mean,
        y_reg_std=y_reg_std,
    )


def normalize_x(x: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return (x - stats.x_mean.to(x.device)) / stats.x_std.to(x.device)


def normalize_y_reg(y_reg: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return (y_reg - stats.y_reg_mean.to(y_reg.device)) / stats.y_reg_std.to(y_reg.device)


def denormalize_y_reg(y_reg_norm: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return y_reg_norm * stats.y_reg_std.to(y_reg_norm.device) + stats.y_reg_mean.to(y_reg_norm.device)


# -----------------------------------------------------------------------------
# 4. PhysicsNeMo surrogate
# -----------------------------------------------------------------------------

class SkidRiskNet(nn.Module):
    def __init__(self, in_features: int, hidden_layers: int = 4, layer_size: int = 128):
        super().__init__()
        self.backbone = FullyConnected(
            in_features=in_features,
            out_features=5,  # 4 regression + 1 classification logit
            num_layers=hidden_layers,
            layer_size=layer_size,
            activation_fn="silu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def physics_consistency_loss_denorm(x_raw: torch.Tensor, pred_reg_denorm: torch.Tensor) -> torch.Tensor:
    """
    Physics consistency in denormalized physical units.

    pred_reg_denorm columns:
      0 beta_dot
      1 r_dot
      2 ay
      3 slip_margin
    """
    v_x = x_raw[:, 0].clamp_min(1.0)
    beta = x_raw[:, 1]
    r = x_raw[:, 2]
    delta = x_raw[:, 3]
    mu = x_raw[:, 5].clamp_min(0.05)
    bank = x_raw[:, 6]
    grade = x_raw[:, 7]
    lf = x_raw[:, 8]
    lr = x_raw[:, 9]
    cf = x_raw[:, 10]
    cr = x_raw[:, 11]
    mass = x_raw[:, 12]
    iz = x_raw[:, 13]

    beta_dot_pred = pred_reg_denorm[:, 0]
    r_dot_pred = pred_reg_denorm[:, 1]
    ay_pred = pred_reg_denorm[:, 2]
    margin_pred = pred_reg_denorm[:, 3]

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

    beta_dot_phys = (fyf + fyr) / (mass * v_x) - r
    r_dot_phys = (lf * fyf - lr * fyr) / iz
    ay_phys = (fyf + fyr) / mass - P.g * torch.sin(bank)
    ay_limit = mu * P.g * torch.cos(bank).clamp_min(0.5)
    margin_phys = 1.0 - torch.abs(ay_phys) / ay_limit.clamp_min(1e-3)

    return (
        nn.functional.mse_loss(beta_dot_pred, beta_dot_phys)
        + nn.functional.mse_loss(r_dot_pred, r_dot_phys)
        + nn.functional.mse_loss(ay_pred, ay_phys)
        + nn.functional.mse_loss(margin_pred, margin_phys)
    )


# -----------------------------------------------------------------------------
# 5. Dataloaders
# -----------------------------------------------------------------------------

def make_splits(
    n_train: int = 24000,
    n_val: int = 6000,
    n_test: int = 6000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, NormalizationStats]:
    x_train = sample_inputs_mixture(n_train)
    y_train, _ = teacher_targets_from_features(x_train)

    x_val = sample_inputs_mixture(n_val)
    y_val, _ = teacher_targets_from_features(x_val)

    x_test = sample_inputs_mixture(n_test)
    y_test, _ = teacher_targets_from_features(x_test)

    stats = compute_normalization_stats(x_train, y_train)
    return x_train, y_train, x_val, y_val, x_test, y_test, stats


def make_dataloaders(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    stats: NormalizationStats,
    batch_size: int = 256,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    x_train_norm = normalize_x(x_train, stats)
    x_val_norm = normalize_x(x_val, stats)
    x_test_norm = normalize_x(x_test, stats)

    y_train_reg_norm = normalize_y_reg(y_train[:, :4], stats)
    y_val_reg_norm = normalize_y_reg(y_val[:, :4], stats)
    y_test_reg_norm = normalize_y_reg(y_test[:, :4], stats)

    y_train_all = torch.cat([y_train_reg_norm, y_train[:, 4:5]], dim=1)
    y_val_all = torch.cat([y_val_reg_norm, y_val[:, 4:5]], dim=1)
    y_test_all = torch.cat([y_test_reg_norm, y_test[:, 4:5]], dim=1)

    train_loader = DataLoader(
        TensorDataset(x_train_norm, y_train_all, x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_norm, y_val_all, x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(x_test_norm, y_test_all, x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# 6. Metrics / evaluation
# -----------------------------------------------------------------------------

def classification_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).to(DTYPE)

    tp = ((preds == 1) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    total = max(1, tp + tn + fp + fn)
    acc = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    stats: NormalizationStats,
    bce_weight: float = 0.2,
    phys_weight: float = 0.1,
) -> Dict[str, float]:
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_count = 0

    mae_beta_dot = 0.0
    mae_r_dot = 0.0
    mae_ay = 0.0
    mae_margin = 0.0

    logits_all = []
    cls_all = []

    with torch.no_grad():
        for x_norm, y_all_norm, x_raw, y_raw in loader:
            x_norm = x_norm.to(DEVICE)
            y_all_norm = y_all_norm.to(DEVICE)
            x_raw = x_raw.to(DEVICE)
            y_raw = y_raw.to(DEVICE)

            pred = model(x_norm)
            pred_reg_norm = pred[:, :4]
            pred_logit = pred[:, 4]

            target_reg_norm = y_all_norm[:, :4]
            target_cls = y_all_norm[:, 4]

            reg_loss = nn.functional.mse_loss(pred_reg_norm, target_reg_norm)
            cls_loss = bce(pred_logit, target_cls)

            pred_reg_denorm = denormalize_y_reg(pred_reg_norm, stats)
            phys_loss = physics_consistency_loss_denorm(x_raw, pred_reg_denorm)

            loss = reg_loss + bce_weight * cls_loss + phys_weight * phys_loss

            bs = x_norm.size(0)
            total_loss += loss.item() * bs
            total_count += bs

            mae_beta_dot += torch.abs(pred_reg_denorm[:, 0] - y_raw[:, 0]).sum().item()
            mae_r_dot += torch.abs(pred_reg_denorm[:, 1] - y_raw[:, 1]).sum().item()
            mae_ay += torch.abs(pred_reg_denorm[:, 2] - y_raw[:, 2]).sum().item()
            mae_margin += torch.abs(pred_reg_denorm[:, 3] - y_raw[:, 3]).sum().item()

            logits_all.append(pred_logit.cpu())
            cls_all.append(y_raw[:, 4].cpu())

    logits_all = torch.cat(logits_all)
    cls_all = torch.cat(cls_all)
    cls_metrics = classification_metrics_from_logits(logits_all, cls_all)

    return {
        "loss": total_loss / max(1, total_count),
        "mae_beta_dot": mae_beta_dot / max(1, total_count),
        "mae_r_dot": mae_r_dot / max(1, total_count),
        "mae_ay": mae_ay / max(1, total_count),
        "mae_margin": mae_margin / max(1, total_count),
        "acc": cls_metrics["acc"],
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1": cls_metrics["f1"],
    }


# -----------------------------------------------------------------------------
# 7. Training
# -----------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    stats: NormalizationStats,
    epoch: int,
    metrics: Dict[str, float],
    path: Path,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "stats": stats.to_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path) -> Tuple[SkidRiskNet, NormalizationStats, Dict[str, float], int]:
    ckpt = torch.load(path, map_location=DEVICE)
    model = SkidRiskNet(in_features=14).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    stats = NormalizationStats.from_dict(ckpt["stats"])
    metrics = ckpt.get("metrics", {})
    epoch = ckpt.get("epoch", -1)
    model.eval()
    return model, stats, metrics, epoch


def train_model(
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    bce_weight: float = 0.2,
    phys_weight: float = 0.1,
    early_stop_patience: int = 8,
) -> Tuple[SkidRiskNet, NormalizationStats]:
    x_train, y_train, x_val, y_val, x_test, y_test, stats = make_splits()
    train_loader, val_loader, test_loader = make_dataloaders(
        x_train, y_train, x_val, y_val, x_test, y_test, stats, batch_size=batch_size
    )

    model = SkidRiskNet(in_features=14).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_path = ARTIFACT_DIR / "best_model.pt"
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for x_norm, y_all_norm, x_raw, y_raw in train_loader:
            x_norm = x_norm.to(DEVICE)
            y_all_norm = y_all_norm.to(DEVICE)
            x_raw = x_raw.to(DEVICE)

            pred = model(x_norm)
            pred_reg_norm = pred[:, :4]
            pred_logit = pred[:, 4]

            target_reg_norm = y_all_norm[:, :4]
            target_cls = y_all_norm[:, 4]

            reg_loss = nn.functional.mse_loss(pred_reg_norm, target_reg_norm)
            cls_loss = bce(pred_logit, target_cls)

            pred_reg_denorm = denormalize_y_reg(pred_reg_norm, stats)
            phys_loss = physics_consistency_loss_denorm(x_raw, pred_reg_denorm)

            loss = reg_loss + bce_weight * cls_loss + phys_weight * phys_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = x_norm.size(0)
            train_loss_sum += loss.item() * bs
            train_count += bs

        train_loss = train_loss_sum / max(1, train_count)
        val_metrics = evaluate_loader(model, val_loader, stats, bce_weight=bce_weight, phys_weight=phys_weight)
        test_metrics = evaluate_loader(model, test_loader, stats, bce_weight=bce_weight, phys_weight=phys_weight)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae_beta_dot": val_metrics["mae_beta_dot"],
            "val_mae_r_dot": val_metrics["mae_r_dot"],
            "val_mae_ay": val_metrics["mae_ay"],
            "val_mae_margin": val_metrics["mae_margin"],
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "test_f1": test_metrics["f1"],
        }
        history.append(epoch_log)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_metrics['loss']:.5f} | "
            f"val_mae(beta_dot)={val_metrics['mae_beta_dot']:.5f} | "
            f"val_mae(r_dot)={val_metrics['mae_r_dot']:.5f} | "
            f"val_mae(ay)={val_metrics['mae_ay']:.5f} | "
            f"val_mae(margin)={val_metrics['mae_margin']:.5f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            no_improve = 0
            save_checkpoint(model, stats, epoch, val_metrics, best_path)
            print(f"  -> saved best checkpoint to {best_path}")
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    history_path = ARTIFACT_DIR / "train_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"Saved training history to {history_path}")

    best_model, best_stats, best_metrics, best_epoch = load_checkpoint(best_path)
    print(f"Loaded best checkpoint from epoch {best_epoch} with val_loss={best_metrics.get('loss', float('nan')):.5f}")

    return best_model, best_stats


# -----------------------------------------------------------------------------
# 8. Road geometry and controller
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
# 9. Rollout functions
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
    stats: NormalizationStats,
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

        inp_raw = torch.tensor(
            [[
                v_x[t].item(), beta[t].item(), r[t].item(), delta, a_x_value, mu, bank, grade,
                P.lf, P.lr, P.cf, P.cr, P.mass, P.iz
            ]],
            dtype=DTYPE,
            device=DEVICE,
        )
        inp_norm = normalize_x(inp_raw, stats)

        with torch.no_grad():
            pred = model(inp_norm)
            pred_reg_denorm = denormalize_y_reg(pred[:, :4], stats)[0]
            pred_logit = pred[0, 4]

        beta_dot = pred_reg_denorm[0]
        r_dot = pred_reg_denorm[1]
        ay[t] = pred_reg_denorm[2]
        margin[t] = pred_reg_denorm[3]
        slip_prob[t] = torch.sigmoid(pred_logit)

        beta[t + 1] = beta[t] + P.dt * beta_dot
        r[t + 1] = r[t] + P.dt * r_dot
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
# 10. Tests
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
    delta_fb, e_y, e_psi = steering_feedback(
        0.0,
        0.0,
        0.0,
        float(road["x_ref"][0].item()),
        float(road["y_ref"][0].item()),
        float(road["psi_ref"][0].item()),
    )
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
# 11. Visualization / comparison
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
        out = ARTIFACT_DIR / "teacher_trajectory_comparison.png"
        plt.savefig(out, dpi=180)
        plt.close()
        print(f"Saved {out}")
    except Exception as e:
        print(f"Warning: plotting failed: {e}")


def compare_surrogate_scenarios(model: SkidRiskNet, stats: NormalizationStats) -> None:
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
            res = rollout_surrogate(model, stats, v0=sc["v0"], mu=sc["mu"])
            if not ref_plotted:
                plt.plot(res["x_ref"].numpy(), res["y_ref"].numpy(), "k--", linewidth=2.0, label="road centerline")
                ref_plotted = True
            plt.plot(res["x"].numpy(), res["y"].numpy(), linewidth=2.2, label=sc["name"])
            print(
                f"{sc['name']:<12} | surrogate max|e_y|={res['e_y'].abs().max().item():.2f} m | "
                f"max_P(slip)={res['slip_prob'].max().item():.3f}"
            )
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("Surrogate model rollout")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        out = ARTIFACT_DIR / "surrogate_trajectory_comparison.png"
        plt.savefig(out, dpi=180)
        plt.close()
        print(f"Saved {out}")
    except Exception as e:
        print(f"Warning: plotting failed: {e}")


def compare_teacher_vs_surrogate(model: SkidRiskNet, stats: NormalizationStats) -> None:
    scenarios = [
        {"name": "dry_nominal", "v0": 12.0, "mu": 0.90},
        {"name": "wet_nominal", "v0": 12.0, "mu": 0.50},
        {"name": "icy_fast", "v0": 16.0, "mu": 0.20},
    ]
    try:
        import matplotlib.pyplot as plt
        for sc in scenarios:
            teacher = rollout_teacher(v0=sc["v0"], mu=sc["mu"])
            sur = rollout_surrogate(model, stats, v0=sc["v0"], mu=sc["mu"])

            final_pos_err = math.sqrt(
                (teacher["x"][-1].item() - sur["x"][-1].item()) ** 2
                + (teacher["y"][-1].item() - sur["y"][-1].item()) ** 2
            )
            max_ey_gap = abs(teacher["e_y"].abs().max().item() - sur["e_y"].abs().max().item())
            print(
                f"{sc['name']:<12} | final_pos_err={final_pos_err:.3f} m | "
                f"max|e_y| gap={max_ey_gap:.3f} m"
            )

            plt.figure(figsize=(8, 7))
            plt.plot(teacher["x_ref"].numpy(), teacher["y_ref"].numpy(), "k--", linewidth=2.0, label="road centerline")
            plt.plot(teacher["x"].numpy(), teacher["y"].numpy(), linewidth=2.2, label="teacher")
            plt.plot(sur["x"].numpy(), sur["y"].numpy(), linewidth=2.2, label="surrogate")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title(f"Teacher vs surrogate: {sc['name']}")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            out = ARTIFACT_DIR / f"teacher_vs_surrogate_{sc['name']}.png"
            plt.savefig(out, dpi=180)
            plt.close()
            print(f"Saved {out}")

    except Exception as e:
        print(f"Warning: plotting failed: {e}")


# -----------------------------------------------------------------------------
# 12. CLI
# -----------------------------------------------------------------------------

def evaluate_saved_model() -> None:
    ckpt_path = ARTIFACT_DIR / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model, stats, metrics, epoch = load_checkpoint(ckpt_path)
    print(f"Loaded model from epoch {epoch}")
    print("Stored validation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    compare_teacher_scenarios()
    compare_surrogate_scenarios(model, stats)
    compare_teacher_vs_surrogate(model, stats)


if __name__ == "__main__":
    import sys

    if "--run-tests" in sys.argv:
        run_tests()
        compare_teacher_scenarios()

    elif "--train" in sys.argv:
        run_tests()
        compare_teacher_scenarios()
        model, stats = train_model(
            epochs=40,
            batch_size=256,
            lr=1e-3,
            bce_weight=0.2,
            phys_weight=0.1,
            early_stop_patience=8,
        )
        compare_surrogate_scenarios(model, stats)
        compare_teacher_vs_surrogate(model, stats)

    elif "--evaluate" in sys.argv:
        evaluate_saved_model()

    else:
        print("Usage:")
        print("  python physicsnemo_skid_poc_improved.py --run-tests")
        print("  python physicsnemo_skid_poc_improved.py --train")
        print("  python physicsnemo_skid_poc_improved.py --evaluate")