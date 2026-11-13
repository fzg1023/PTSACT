import csv
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import DataLoader, Dataset

from .config import OUTPUT_ROOT, RESULTS_DIR
from .synthetic import FULL_RANGE_TEMPERATURE_REGIMES, PHANTOM_TYPES


DEFAULT_REGIME_LOSS_WEIGHTS = {
    "room_static": 4.0,
    "low_warm": 3.0,
    "mid_transition_1": 2.0,
    "mid_transition_2": 1.5,
    "high_running": 1.0,
    "legacy_high": 1.0,
}


SRC_ROOT = Path(__file__).resolve().parents[2] / "TempFieldReconstructionv2" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nn_prior import AttentionUNet  # noqa: E402


@dataclass(frozen=True)
class FactoryStage2Config:
    data_root: Union[Path, str] = OUTPUT_ROOT / "data_factory_geometry_tikhonov_12x12_pilot"
    output_dir: Union[Path, str] = OUTPUT_ROOT / "models" / "factory_stage2_tikhonov_12x12"
    report_path: Union[Path, str] = RESULTS_DIR / "factory_stage2_report.md"
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    base_ch: int = 16
    dropout_p: float = 0.1
    seed: int = 42
    device: Optional[str] = None
    max_train_per_type: Optional[int] = None
    max_val_per_type: Optional[int] = None
    phantom_types: Sequence[str] = PHANTOM_TYPES
    regime_loss_weights: Optional[Dict[str, float]] = None


class TemperatureResidualDataset(Dataset):
    def __init__(
        self,
        T_stage1: np.ndarray,
        T_true: np.ndarray,
        T_min: float,
        T_max: float,
        sample_weights: Optional[np.ndarray] = None,
    ):
        self.T_min = float(T_min)
        self.T_max = float(T_max)
        self.scale = self.T_max - self.T_min + 1e-14
        stage1 = T_stage1.astype(np.float32)
        true = T_true.astype(np.float32)
        x_norm = (stage1 - self.T_min) / self.scale
        residual_norm = (true - stage1) / self.scale
        self.X = torch.from_numpy(x_norm).float().unsqueeze(1)
        self.Y = torch.from_numpy(residual_norm).float().unsqueeze(1)
        if sample_weights is None:
            sample_weights = np.ones(stage1.shape[0], dtype=np.float32)
        self.W = torch.from_numpy(np.asarray(sample_weights, dtype=np.float32)).float()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx], self.W[idx]


def compute_regime_sample_weights(
    temperature_regimes: np.ndarray,
    regime_loss_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    weights = dict(DEFAULT_REGIME_LOSS_WEIGHTS)
    if regime_loss_weights:
        weights.update({str(key): float(value) for key, value in regime_loss_weights.items()})
    labels = np.asarray(temperature_regimes).astype(str)
    return np.asarray([weights.get(str(label), 1.0) for label in labels], dtype=np.float32)


def load_normalization(data_root: Union[Path, str]) -> Dict[str, float]:
    norm_path = Path(data_root) / "normalization.json"
    payload = json.loads(norm_path.read_text(encoding="utf-8"))
    return {"T_min": float(payload["T_min"]), "T_max": float(payload["T_max"])}


def load_split_arrays(
    data_root: Union[Path, str],
    split: str,
    phantom_types: Sequence[str] = PHANTOM_TYPES,
    limit_per_type: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    root = Path(data_root)
    tof_list = []
    stage1_list = []
    true_list = []
    type_labels = []
    regime_labels = []
    seed_list = []
    path_temperature_list = []
    for phantom_type in phantom_types:
        path = root / phantom_type / f"{split}.npz"
        data = np.load(path)
        n = data["T_stage1"].shape[0]
        if limit_per_type is not None:
            n = min(n, int(limit_per_type))
        tof_list.append(data["tof"][:n])
        stage1_list.append(data["T_stage1"][:n])
        true_list.append(data["T_true"][:n])
        if "phantom_type" in data:
            type_labels.extend(data["phantom_type"][:n].astype(str).tolist())
        else:
            type_labels.extend([phantom_type] * n)
        if "temperature_regime" in data:
            regime_labels.extend(data["temperature_regime"][:n].astype(str).tolist())
        else:
            regime_labels.extend(["legacy_high"] * n)
        if "seed" in data:
            seed_list.append(data["seed"][:n])
        if "path_temperatures" in data:
            path_temperature_list.append(data["path_temperatures"][:n])
    arrays = {
        "tof": np.concatenate(tof_list, axis=0),
        "T_stage1": np.concatenate(stage1_list, axis=0),
        "T_true": np.concatenate(true_list, axis=0),
        "phantom_type": np.asarray(type_labels),
        "temperature_regime": np.asarray(regime_labels),
    }
    if seed_list:
        arrays["seed"] = np.concatenate(seed_list, axis=0)
    if path_temperature_list:
        arrays["path_temperatures"] = np.concatenate(path_temperature_list, axis=0)
    return arrays


def predict_temperature_from_residual(
    residual_norm: np.ndarray,
    T_stage1: np.ndarray,
    T_min: float,
    T_max: float,
) -> np.ndarray:
    scale = float(T_max) - float(T_min) + 1e-14
    return T_stage1.astype(np.float32) + residual_norm.astype(np.float32) * scale


def compute_temperature_metrics(T_pred: np.ndarray, T_true: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(T_pred, dtype=np.float64).ravel()
    true = np.asarray(T_true, dtype=np.float64).ravel()
    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    max_ae = float(np.max(np.abs(err)))
    true_mean = float(np.mean(true))
    rmse_pct = float(rmse / (true_mean + 1e-14) * 100.0)
    are_pct = float(np.mean(np.abs(err) / (true_mean + 1e-14)) * 100.0)
    l2_rer_pct = float(np.linalg.norm(err) / (np.linalg.norm(true) + 1e-14) * 100.0)
    if rmse <= 0.0:
        psnr_db = 100.0
    else:
        psnr_db = float(20.0 * math.log10(float(np.max(true)) / (rmse + 1e-14)))
    return {
        "RMSE_C": rmse,
        "RMSE_percent": rmse_pct,
        "MAE_C": mae,
        "MaxAE_C": max_ae,
        "ARE_percent": are_pct,
        "L2_RER_percent": l2_rer_pct,
        "PSNR_dB": psnr_db,
    }


def zero_initialize_residual_head(model: torch.nn.Module) -> None:
    if not hasattr(model, "out_conv"):
        raise AttributeError("model must expose an out_conv layer for zero-residual initialization")
    torch.nn.init.zeros_(model.out_conv.weight)
    if model.out_conv.bias is not None:
        torch.nn.init.zeros_(model.out_conv.bias)


def summarize_improvement(
    baseline_average: Dict[str, float],
    stage2_average: Dict[str, float],
    min_rmse_gain_C: float = 0.0,
) -> Dict[str, object]:
    baseline_rmse = float(baseline_average["RMSE_C"])
    stage2_rmse = float(stage2_average["RMSE_C"])
    baseline_mae = float(baseline_average["MAE_C"])
    stage2_mae = float(stage2_average["MAE_C"])
    accepted = stage2_rmse < baseline_rmse - float(min_rmse_gain_C)
    return {
        "RMSE_C_percent": (baseline_rmse - stage2_rmse) / (baseline_rmse + 1e-14) * 100.0,
        "MAE_C_percent": (baseline_mae - stage2_mae) / (baseline_mae + 1e-14) * 100.0,
        "accepted": bool(accepted),
        "selected_method": "stage2_residual_unet" if accepted else "stage1_identity_residual",
    }


def _metrics_by_type(
    T_pred: np.ndarray,
    T_true: np.ndarray,
    labels: np.ndarray,
    phantom_types: Sequence[str],
) -> Dict[str, object]:
    per_type = {}
    for phantom_type in phantom_types:
        mask = labels == phantom_type
        if np.any(mask):
            per_type[phantom_type] = compute_temperature_metrics(T_pred[mask], T_true[mask])
    return {
        "per_type": per_type,
        "average": compute_temperature_metrics(T_pred, T_true),
        "n_samples": int(T_true.shape[0]),
    }


def _metrics_by_group(
    T_pred: np.ndarray,
    T_true: np.ndarray,
    labels: np.ndarray,
    label_order: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    per_group = {}
    for label in label_order:
        mask = labels == label
        if np.any(mask):
            per_group[str(label)] = compute_temperature_metrics(T_pred[mask], T_true[mask])
    for label in sorted(set(labels.astype(str)) - set(str(item) for item in label_order)):
        mask = labels == label
        if np.any(mask):
            per_group[str(label)] = compute_temperature_metrics(T_pred[mask], T_true[mask])
    return per_group


def evaluate_stage1_baseline(
    data_root: Union[Path, str],
    split: str = "val",
    phantom_types: Sequence[str] = PHANTOM_TYPES,
    limit_per_type: Optional[int] = None,
) -> Dict[str, object]:
    arrays = load_split_arrays(data_root, split, phantom_types, limit_per_type=limit_per_type)
    summary = _metrics_by_type(arrays["T_stage1"], arrays["T_true"], arrays["phantom_type"], phantom_types)
    summary["per_regime"] = _metrics_by_group(
        arrays["T_stage1"],
        arrays["T_true"],
        arrays["temperature_regime"],
        (*FULL_RANGE_TEMPERATURE_REGIMES, "legacy_high"),
    )
    return summary


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _predict_model(
    model: torch.nn.Module,
    T_stage1: np.ndarray,
    T_min: float,
    T_max: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    scale = float(T_max) - float(T_min) + 1e-14
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(T_stage1), batch_size):
            stage1_batch = T_stage1[start:start + batch_size].astype(np.float32)
            x = torch.from_numpy((stage1_batch - T_min) / scale).float().unsqueeze(1).to(device)
            residual_norm = model(x).cpu().numpy().squeeze(1)
            preds.append(predict_temperature_from_residual(residual_norm, stage1_batch, T_min, T_max))
    return np.concatenate(preds, axis=0)


def resize_stage1_field(stage1_field: np.ndarray, fine_grid: int = 32) -> np.ndarray:
    field = np.asarray(stage1_field, dtype=np.float32)
    if field.shape == (fine_grid, fine_grid):
        return field.copy()
    factor_y = float(fine_grid) / float(field.shape[0])
    factor_x = float(fine_grid) / float(field.shape[1])
    resized = zoom(field, zoom=(factor_y, factor_x), order=3)
    return np.asarray(resized, dtype=np.float32)


def apply_stage2_to_field(
    model: torch.nn.Module,
    stage1_field: np.ndarray,
    T_min: float,
    T_max: float,
    device: torch.device,
    fine_grid: int = 32,
) -> np.ndarray:
    stage1_fine = resize_stage1_field(stage1_field, fine_grid=fine_grid)
    scale = float(T_max) - float(T_min) + 1e-14
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy((stage1_fine - T_min) / scale).float().unsqueeze(0).unsqueeze(0).to(device)
        residual_norm = model(x).cpu().numpy().squeeze(0).squeeze(0)
    return predict_temperature_from_residual(residual_norm, stage1_fine, T_min, T_max)


def evaluate_stage2_model(
    model: torch.nn.Module,
    data_root: Union[Path, str],
    T_min: float,
    T_max: float,
    device: torch.device,
    split: str = "val",
    batch_size: int = 64,
    phantom_types: Sequence[str] = PHANTOM_TYPES,
    limit_per_type: Optional[int] = None,
) -> Dict[str, object]:
    arrays = load_split_arrays(data_root, split, phantom_types, limit_per_type=limit_per_type)
    pred = _predict_model(model, arrays["T_stage1"], T_min, T_max, device, batch_size)
    summary = _metrics_by_type(pred, arrays["T_true"], arrays["phantom_type"], phantom_types)
    summary["per_regime"] = _metrics_by_group(
        pred,
        arrays["T_true"],
        arrays["temperature_regime"],
        (*FULL_RANGE_TEMPERATURE_REGIMES, "legacy_high"),
    )
    return summary


def _write_training_log(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(summary: Dict[str, object], report_path: Path) -> None:
    baseline = summary["baseline"]["average"]
    stage2 = summary["stage2"]["average"]
    improvement = summary["improvement"]
    lines = [
        "# Factory Geometry Stage 2 Report",
        "",
        "## Configuration",
        "",
        f"- Data root: `{summary['config']['data_root']}`",
        f"- Epochs: `{summary['config']['epochs']}`",
        f"- Batch size: `{summary['config']['batch_size']}`",
        f"- Learning rate: `{summary['config']['learning_rate']}`",
        f"- Base channels: `{summary['config']['base_ch']}`",
        f"- Device: `{summary['device']}`",
        "",
        "## Validation Summary",
        "",
        f"- Accepted method: `{improvement['selected_method']}`",
        f"- Stage 2 accepted: `{improvement['accepted']}`",
        "",
        "| Method | RMSE C | RMSE % | MAE C | MaxAE C |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Stage 1 Tikhonov | {baseline['RMSE_C']:.4f} | "
            f"{baseline['RMSE_percent']:.4f} | {baseline['MAE_C']:.4f} | {baseline['MaxAE_C']:.4f} |"
        ),
        (
            f"| Stage 2 residual UNet | {stage2['RMSE_C']:.4f} | "
            f"{stage2['RMSE_percent']:.4f} | {stage2['MAE_C']:.4f} | {stage2['MaxAE_C']:.4f} |"
        ),
        "",
        "## Improvement",
        "",
        f"- RMSE C improvement: `{improvement['RMSE_C_percent']:.2f}%`",
        f"- MAE C improvement: `{improvement['MAE_C_percent']:.2f}%`",
        "",
        "## Per Phantom RMSE C",
        "",
        "| Phantom | Stage 1 | Stage 2 |",
        "|---|---:|---:|",
    ]
    for phantom_type, baseline_metrics in summary["baseline"]["per_type"].items():
        stage2_metrics = summary["stage2"]["per_type"][phantom_type]
        lines.append(
            f"| {phantom_type} | {baseline_metrics['RMSE_C']:.4f} | {stage2_metrics['RMSE_C']:.4f} |"
        )
    if summary["baseline"].get("per_regime"):
        lines.extend(
            [
                "",
                "## Per Temperature Regime RMSE C",
                "",
                "| Regime | Stage 1 | Stage 2 |",
                "|---|---:|---:|",
            ]
        )
        for regime, baseline_metrics in summary["baseline"]["per_regime"].items():
            stage2_metrics = summary["stage2"]["per_regime"][regime]
            lines.append(f"| {regime} | {baseline_metrics['RMSE_C']:.4f} | {stage2_metrics['RMSE_C']:.4f} |")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_stage2_model(config: FactoryStage2Config) -> Dict[str, object]:
    _set_seed(config.seed)
    data_root = Path(config.data_root)
    output_dir = Path(config.output_dir)
    report_path = Path(config.report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    norm = load_normalization(data_root)
    T_min = norm["T_min"]
    T_max = norm["T_max"]
    device = _resolve_device(config.device)

    train_arrays = load_split_arrays(
        data_root,
        "train",
        config.phantom_types,
        limit_per_type=config.max_train_per_type,
    )
    sample_weights = compute_regime_sample_weights(
        train_arrays["temperature_regime"],
        config.regime_loss_weights,
    )
    train_ds = TemperatureResidualDataset(
        train_arrays["T_stage1"],
        train_arrays["T_true"],
        T_min,
        T_max,
        sample_weights=sample_weights,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    baseline = evaluate_stage1_baseline(
        data_root,
        split="val",
        phantom_types=config.phantom_types,
        limit_per_type=config.max_val_per_type,
    )
    model = AttentionUNet(
        in_channels=1,
        base_ch=config.base_ch,
        dropout_p=config.dropout_p,
        use_attention=True,
    ).to(device)
    zero_initialize_residual_head(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    baseline_rmse = float(baseline["average"]["RMSE_C"])
    baseline_mae = float(baseline["average"]["MAE_C"])
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_rmse = baseline_rmse
    best_source = "stage1_identity_residual"
    history = [{
        "epoch": 0,
        "train_loss": 0.0,
        "val_rmse_C": baseline_rmse,
        "val_rmse_percent": float(baseline["average"]["RMSE_percent"]),
        "lr": float(config.learning_rate),
    }]
    start_time = time.time()

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for x, y, weights in train_loader:
            x = x.to(device)
            y = y.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            pred = model(x)
            sample_loss = torch.mean((pred - y) ** 2, dim=(1, 2, 3))
            loss = torch.sum(sample_loss * weights) / torch.sum(weights).clamp_min(1e-12)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * int(x.size(0))
        scheduler.step()
        epoch_loss /= max(len(train_ds), 1)

        val_metrics = evaluate_stage2_model(
            model,
            data_root,
            T_min,
            T_max,
            device,
            split="val",
            batch_size=config.batch_size,
            phantom_types=config.phantom_types,
            limit_per_type=config.max_val_per_type,
        )
        val_rmse = float(val_metrics["average"]["RMSE_C"])
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_source = f"epoch_{epoch + 1}"
        row = {
            "epoch": epoch + 1,
            "train_loss": float(epoch_loss),
            "val_rmse_C": val_rmse,
            "val_rmse_percent": float(val_metrics["average"]["RMSE_percent"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"[epoch {epoch + 1}/{config.epochs}] "
            f"loss={epoch_loss:.6f} val_RMSE_C={val_rmse:.4f} best={best_rmse:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model_path = output_dir / "stage2_best.pth"
    torch.save(model.state_dict(), model_path)

    stage2 = evaluate_stage2_model(
        model,
        data_root,
        T_min,
        T_max,
        device,
        split="val",
        batch_size=config.batch_size,
        phantom_types=config.phantom_types,
        limit_per_type=config.max_val_per_type,
    )
    stage2_rmse = float(stage2["average"]["RMSE_C"])
    stage2_mae = float(stage2["average"]["MAE_C"])
    improvement = summarize_improvement(baseline["average"], stage2["average"])

    log_path = output_dir / "training_log.csv"
    results_path = output_dir / "stage2_results.json"
    config_path = output_dir / "stage2_config.json"
    _write_training_log(log_path, history)

    config_payload = asdict(config)
    config_payload["data_root"] = str(config.data_root)
    config_payload["output_dir"] = str(config.output_dir)
    config_payload["report_path"] = str(config.report_path)
    summary = {
        "config": config_payload,
        "device": str(device),
        "n_train": int(len(train_ds)),
        "n_val": int(stage2["n_samples"]),
        "normalization": norm,
        "regime_loss_weights": dict(DEFAULT_REGIME_LOSS_WEIGHTS if config.regime_loss_weights is None else config.regime_loss_weights),
        "model_path": str(model_path),
        "training_log": str(log_path),
        "report_path": str(report_path),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "best_source": best_source,
        "baseline": baseline,
        "stage2": stage2,
        "improvement": improvement,
    }
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    norm_src = data_root / "normalization.json"
    if norm_src.exists():
        shutil.copy2(norm_src, output_dir / "normalization.json")
    _write_report(summary, report_path)
    print(json.dumps({
        "model_path": str(model_path),
        "results_path": str(results_path),
        "report_path": str(report_path),
        "stage1_RMSE_C": baseline_rmse,
        "stage2_RMSE_C": stage2_rmse,
        "RMSE_C_improvement_percent": improvement["RMSE_C_percent"],
        "accepted_method": improvement["selected_method"],
    }, indent=2, ensure_ascii=False))
    return summary
