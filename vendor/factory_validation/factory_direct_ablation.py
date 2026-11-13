import csv
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import OUTPUT_ROOT, PICTURE_DIR, RESULTS_DIR
from .stage2 import (
    AttentionUNet,
    _metrics_by_type,
    _resolve_device,
    _set_seed,
    _write_training_log,
    compute_temperature_metrics,
    evaluate_stage1_baseline,
    load_normalization,
    load_split_arrays,
)
from .synthetic import PHANTOM_TYPES


@dataclass(frozen=True)
class FactoryDirectConfig:
    data_root: Union[Path, str] = OUTPUT_ROOT / "data_factory_geometry_tikhonov_12x12"
    output_dir: Union[Path, str] = OUTPUT_ROOT / "models" / "factory_stage2_direct_ablation" / "seed42"
    report_path: Union[Path, str] = RESULTS_DIR / "factory_stage2_direct_seed42_report.md"
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    base_ch: int = 16
    dropout_p: float = 0.1
    seed: int = 42
    device: Optional[str] = None
    max_train_per_type: Optional[int] = None
    max_val_per_type: Optional[int] = None
    phantom_types: Sequence[str] = PHANTOM_TYPES


class TemperatureDirectDataset(Dataset):
    def __init__(self, T_stage1: np.ndarray, T_true: np.ndarray, T_min: float, T_max: float):
        self.T_min = float(T_min)
        self.T_max = float(T_max)
        self.scale = self.T_max - self.T_min + 1e-14
        stage1 = T_stage1.astype(np.float32)
        true = T_true.astype(np.float32)
        x_norm = (stage1 - self.T_min) / self.scale
        y_norm = (true - self.T_min) / self.scale
        self.X = torch.from_numpy(x_norm).float().unsqueeze(1)
        self.Y = torch.from_numpy(y_norm).float().unsqueeze(1)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def predict_temperature_direct(output_norm: np.ndarray, T_min: float, T_max: float) -> np.ndarray:
    scale = float(T_max) - float(T_min) + 1e-14
    return output_norm.astype(np.float32) * scale + float(T_min)


def _predict_direct_model(
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
            output_norm = model(x).cpu().numpy().squeeze(1)
            preds.append(predict_temperature_direct(output_norm, T_min, T_max))
    return np.concatenate(preds, axis=0)


def evaluate_direct_model(
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
    pred = _predict_direct_model(model, arrays["T_stage1"], T_min, T_max, device, batch_size)
    return _metrics_by_type(pred, arrays["T_true"], arrays["phantom_type"], phantom_types)


def _percent_gain(old: float, new: float) -> float:
    return (float(old) - float(new)) / (float(old) + 1e-14) * 100.0


def summarize_factory_direct_ablation(
    residual_summary: Dict[str, object],
    direct_summary: Dict[str, object],
) -> Dict[str, object]:
    stage1 = residual_summary["baseline"]["average"]
    residual = residual_summary["stage2"]["average"]
    direct = direct_summary["direct"]["average"]
    stage1_rmse = float(stage1["RMSE_C"])
    residual_rmse = float(residual["RMSE_C"])
    direct_rmse = float(direct["RMSE_C"])
    methods = {
        "stage1": stage1_rmse,
        "residual_stage2": residual_rmse,
        "direct_stage2": direct_rmse,
    }

    residual_per_type = residual_summary.get("stage2", {}).get("per_type", {})
    direct_per_type = direct_summary.get("direct", {}).get("per_type", {})
    stage1_per_type = residual_summary.get("baseline", {}).get("per_type", {})
    per_type = {}
    for phantom_type in sorted(set(stage1_per_type) | set(residual_per_type) | set(direct_per_type)):
        values = {}
        if phantom_type in stage1_per_type:
            values["stage1_rmse_C"] = float(stage1_per_type[phantom_type]["RMSE_C"])
        if phantom_type in residual_per_type:
            values["residual_rmse_C"] = float(residual_per_type[phantom_type]["RMSE_C"])
        if phantom_type in direct_per_type:
            values["direct_rmse_C"] = float(direct_per_type[phantom_type]["RMSE_C"])
        ranked = {
            "stage1": values.get("stage1_rmse_C", float("inf")),
            "residual_stage2": values.get("residual_rmse_C", float("inf")),
            "direct_stage2": values.get("direct_rmse_C", float("inf")),
        }
        values["best_method"] = min(ranked, key=ranked.get)
        if "direct_rmse_C" in values and "residual_rmse_C" in values:
            values["residual_vs_direct_rmse_improvement_percent"] = _percent_gain(
                values["direct_rmse_C"],
                values["residual_rmse_C"],
            )
        per_type[phantom_type] = values

    return {
        "stage1_rmse_C": stage1_rmse,
        "residual_rmse_C": residual_rmse,
        "direct_rmse_C": direct_rmse,
        "stage1_mae_C": float(stage1["MAE_C"]),
        "residual_mae_C": float(residual["MAE_C"]),
        "direct_mae_C": float(direct["MAE_C"]),
        "direct_vs_stage1_rmse_improvement_percent": _percent_gain(stage1_rmse, direct_rmse),
        "residual_vs_stage1_rmse_improvement_percent": _percent_gain(stage1_rmse, residual_rmse),
        "residual_vs_direct_rmse_improvement_percent": _percent_gain(direct_rmse, residual_rmse),
        "best_method": min(methods, key=methods.get),
        "per_type": per_type,
    }


def _write_direct_report(summary: Dict[str, object], report_path: Path) -> None:
    baseline = summary["baseline"]["average"]
    direct = summary["direct"]["average"]
    improvement = summary["improvement"]
    lines = [
        "# Factory Geometry Direct-Prediction Stage 2 Report",
        "",
        "## Configuration",
        "",
        f"- Data root: `{summary['config']['data_root']}`",
        f"- Epochs: `{summary['config']['epochs']}`",
        f"- Batch size: `{summary['config']['batch_size']}`",
        f"- Learning rate: `{summary['config']['learning_rate']}`",
        f"- Base channels: `{summary['config']['base_ch']}`",
        f"- Device: `{summary['device']}`",
        f"- Learning target: `{summary['learning_target']}`",
        "",
        "## Validation Summary",
        "",
        "| Method | RMSE C | RMSE % | MAE C | MaxAE C |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Stage 1 Tikhonov | {baseline['RMSE_C']:.4f} | "
            f"{baseline['RMSE_percent']:.4f} | {baseline['MAE_C']:.4f} | {baseline['MaxAE_C']:.4f} |"
        ),
        (
            f"| Stage 2 direct UNet | {direct['RMSE_C']:.4f} | "
            f"{direct['RMSE_percent']:.4f} | {direct['MAE_C']:.4f} | {direct['MaxAE_C']:.4f} |"
        ),
        "",
        "## Improvement",
        "",
        f"- Direct vs Stage 1 RMSE improvement: `{improvement['direct_vs_stage1_rmse_improvement_percent']:.2f}%`",
        f"- Direct vs Stage 1 MAE improvement: `{improvement['direct_vs_stage1_mae_improvement_percent']:.2f}%`",
        "",
        "## Per Phantom RMSE C",
        "",
        "| Phantom | Stage 1 | Direct Stage 2 |",
        "|---|---:|---:|",
    ]
    for phantom_type, baseline_metrics in summary["baseline"]["per_type"].items():
        direct_metrics = summary["direct"]["per_type"][phantom_type]
        lines.append(
            f"| {phantom_type} | {baseline_metrics['RMSE_C']:.4f} | {direct_metrics['RMSE_C']:.4f} |"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_direct_model(config: FactoryDirectConfig) -> Dict[str, object]:
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
    train_ds = TemperatureDirectDataset(train_arrays["T_stage1"], train_arrays["T_true"], T_min, T_max)
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

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))

    initial_metrics = evaluate_direct_model(
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
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_rmse = float(initial_metrics["average"]["RMSE_C"])
    best_source = "initial_direct_network"
    history = [{
        "epoch": 0,
        "train_loss": 0.0,
        "val_rmse_C": best_rmse,
        "val_rmse_percent": float(initial_metrics["average"]["RMSE_percent"]),
        "lr": float(config.learning_rate),
    }]
    start_time = time.time()

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * int(x.size(0))
        scheduler.step()
        epoch_loss /= max(len(train_ds), 1)

        val_metrics = evaluate_direct_model(
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
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(epoch_loss),
            "val_rmse_C": val_rmse,
            "val_rmse_percent": float(val_metrics["average"]["RMSE_percent"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        print(
            f"[direct epoch {epoch + 1}/{config.epochs}] "
            f"loss={epoch_loss:.6f} val_RMSE_C={val_rmse:.4f} best={best_rmse:.4f}"
        )

    model.load_state_dict(best_state)
    model_path = output_dir / "direct_best.pth"
    torch.save(model.state_dict(), model_path)

    direct = evaluate_direct_model(
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
    improvement = {
        "direct_vs_stage1_rmse_improvement_percent": _percent_gain(
            float(baseline["average"]["RMSE_C"]),
            float(direct["average"]["RMSE_C"]),
        ),
        "direct_vs_stage1_mae_improvement_percent": _percent_gain(
            float(baseline["average"]["MAE_C"]),
            float(direct["average"]["MAE_C"]),
        ),
    }

    log_path = output_dir / "direct_training_log.csv"
    results_path = output_dir / "direct_results.json"
    config_path = output_dir / "direct_config.json"
    _write_training_log(log_path, history)

    config_payload = asdict(config)
    config_payload["data_root"] = str(config.data_root)
    config_payload["output_dir"] = str(config.output_dir)
    config_payload["report_path"] = str(config.report_path)
    summary = {
        "config": config_payload,
        "device": str(device),
        "n_train": int(len(train_ds)),
        "n_val": int(direct["n_samples"]),
        "normalization": norm,
        "model_path": str(model_path),
        "training_log": str(log_path),
        "report_path": str(report_path),
        "elapsed_seconds": round(time.time() - start_time, 2),
        "best_source": best_source,
        "learning_target": "direct_normalized_temperature",
        "baseline": baseline,
        "direct": direct,
        "improvement": improvement,
    }
    results_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    norm_src = data_root / "normalization.json"
    if norm_src.exists():
        shutil.copy2(norm_src, output_dir / "normalization.json")
    _write_direct_report(summary, report_path)
    print(json.dumps({
        "model_path": str(model_path),
        "results_path": str(results_path),
        "report_path": str(report_path),
        "stage1_RMSE_C": float(baseline["average"]["RMSE_C"]),
        "direct_RMSE_C": float(direct["average"]["RMSE_C"]),
        "direct_vs_stage1_RMSE_improvement_percent": improvement["direct_vs_stage1_rmse_improvement_percent"],
    }, indent=2, ensure_ascii=False))
    return summary


def _read_json(path: Union[Path, str]) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def _aggregate_comparisons(comparisons: Sequence[Dict[str, object]], seeds: Sequence[int]) -> Dict[str, object]:
    if not comparisons:
        raise ValueError("no direct ablation comparisons available")
    stage1 = _mean_std([float(item["stage1_rmse_C"]) for item in comparisons])
    residual = _mean_std([float(item["residual_rmse_C"]) for item in comparisons])
    direct = _mean_std([float(item["direct_rmse_C"]) for item in comparisons])
    direct_gain = _mean_std([float(item["direct_vs_stage1_rmse_improvement_percent"]) for item in comparisons])
    residual_gain = _mean_std([float(item["residual_vs_stage1_rmse_improvement_percent"]) for item in comparisons])
    residual_vs_direct = _mean_std(
        [float(item["residual_vs_direct_rmse_improvement_percent"]) for item in comparisons]
    )
    phantom_types = sorted(set().union(*[set(item["per_type"].keys()) for item in comparisons]))
    per_type = {}
    for phantom_type in phantom_types:
        type_rows = [item["per_type"][phantom_type] for item in comparisons if phantom_type in item["per_type"]]
        per_type[phantom_type] = {
            "stage1_rmse_C": _mean_std([float(row["stage1_rmse_C"]) for row in type_rows if "stage1_rmse_C" in row]),
            "residual_rmse_C": _mean_std(
                [float(row["residual_rmse_C"]) for row in type_rows if "residual_rmse_C" in row]
            ),
            "direct_rmse_C": _mean_std([float(row["direct_rmse_C"]) for row in type_rows if "direct_rmse_C" in row]),
        }
        per_type[phantom_type]["residual_vs_direct_rmse_improvement_percent"] = _mean_std(
            [
                float(row["residual_vs_direct_rmse_improvement_percent"])
                for row in type_rows
                if "residual_vs_direct_rmse_improvement_percent" in row
            ]
        )
    methods = {
        "stage1": stage1["mean"],
        "residual_stage2": residual["mean"],
        "direct_stage2": direct["mean"],
    }
    return {
        "n_seeds": int(len(comparisons)),
        "seeds": [int(seed) for seed in seeds],
        "stage1_rmse_C_mean": stage1["mean"],
        "stage1_rmse_C_std": stage1["std"],
        "residual_rmse_C_mean": residual["mean"],
        "residual_rmse_C_std": residual["std"],
        "direct_rmse_C_mean": direct["mean"],
        "direct_rmse_C_std": direct["std"],
        "direct_vs_stage1_rmse_improvement_percent_mean": direct_gain["mean"],
        "direct_vs_stage1_rmse_improvement_percent_std": direct_gain["std"],
        "residual_vs_stage1_rmse_improvement_percent_mean": residual_gain["mean"],
        "residual_vs_stage1_rmse_improvement_percent_std": residual_gain["std"],
        "residual_vs_direct_rmse_improvement_percent_mean": residual_vs_direct["mean"],
        "residual_vs_direct_rmse_improvement_percent_std": residual_vs_direct["std"],
        "best_method": min(methods, key=methods.get),
        "per_type": per_type,
    }


def _write_comparison_rows(path: Path, seed_rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "seed",
        "stage1_rmse_C",
        "residual_rmse_C",
        "direct_rmse_C",
        "direct_vs_stage1_rmse_improvement_percent",
        "residual_vs_stage1_rmse_improvement_percent",
        "residual_vs_direct_rmse_improvement_percent",
        "best_method",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(seed_rows)


def _write_ablation_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Factory Geometry Residual vs Direct Ablation",
        "",
        "## Summary",
        "",
        f"- Seeds: `{summary['seeds']}`",
        f"- Best method by mean RMSE: `{summary['best_method']}`",
        f"- Stage 1 RMSE: `{summary['stage1_rmse_C_mean']:.4f} +/- {summary['stage1_rmse_C_std']:.4f} C`",
        f"- Residual Stage 2 RMSE: `{summary['residual_rmse_C_mean']:.4f} +/- {summary['residual_rmse_C_std']:.4f} C`",
        f"- Direct Stage 2 RMSE: `{summary['direct_rmse_C_mean']:.4f} +/- {summary['direct_rmse_C_std']:.4f} C`",
        f"- Residual vs Direct RMSE improvement: `{summary['residual_vs_direct_rmse_improvement_percent_mean']:.2f} +/- {summary['residual_vs_direct_rmse_improvement_percent_std']:.2f}%`",
        "",
        "## Interpretation",
        "",
        "- This ablation uses the same factory-geometry synthetic dataset and the same UNet family.",
        "- Residual Stage 2 predicts a correction to Stage 1; Direct Stage 2 predicts the full normalized temperature field.",
        "- The comparison is valid only in the factory-geometry synthetic domain where `T_true` is available.",
        "",
        "## Per Phantom",
        "",
        "| Phantom | Stage 1 RMSE C | Residual RMSE C | Direct RMSE C | Residual vs Direct % |",
        "|---|---:|---:|---:|---:|",
    ]
    for phantom_type, stats in summary["per_type"].items():
        lines.append(
            f"| {phantom_type} | "
            f"{stats['stage1_rmse_C']['mean']:.4f} +/- {stats['stage1_rmse_C']['std']:.4f} | "
            f"{stats['residual_rmse_C']['mean']:.4f} +/- {stats['residual_rmse_C']['std']:.4f} | "
            f"{stats['direct_rmse_C']['mean']:.4f} +/- {stats['direct_rmse_C']['std']:.4f} | "
            f"{stats['residual_vs_direct_rmse_improvement_percent']['mean']:.2f} +/- "
            f"{stats['residual_vs_direct_rmse_improvement_percent']['std']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ablation_plot(path_base: Path, summary: Dict[str, object]) -> Dict[str, str]:
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["Stage 1", "Residual\nStage 2", "Direct\nStage 2"]
    means = [
        summary["stage1_rmse_C_mean"],
        summary["residual_rmse_C_mean"],
        summary["direct_rmse_C_mean"],
    ]
    stds = [
        summary["stage1_rmse_C_std"],
        summary["residual_rmse_C_std"],
        summary["direct_rmse_C_std"],
    ]
    colors = ["#777777", "#2f6f9f", "#c76f2f"]
    fig, ax = plt.subplots(figsize=(4.8, 3.2), dpi=300)
    ax.bar(labels, means, yerr=stds, capsize=3, color=colors, edgecolor="#222222", linewidth=0.6)
    ax.set_ylabel("RMSE (C)")
    ax.set_title("Factory-geometry synthetic ablation")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    outputs = {}
    for ext in ("png", "svg", "pdf"):
        out = path_base.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight")
        outputs[ext] = str(out)
    plt.close(fig)
    return outputs


def run_factory_direct_ablation(
    data_root: Path,
    residual_summary_path: Path = RESULTS_DIR / "factory_stage2_multiseed_summary.json",
    seeds: Iterable[int] = (42, 123, 456),
    epochs: int = 40,
    batch_size: int = 64,
    base_ch: int = 16,
    device: Optional[str] = None,
    output_root: Path = OUTPUT_ROOT / "models" / "factory_stage2_direct_ablation",
    report_dir: Path = RESULTS_DIR,
    picture_dir: Path = PICTURE_DIR,
) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    picture_dir.mkdir(parents=True, exist_ok=True)
    seed_list = [int(seed) for seed in seeds]

    residual_payload = _read_json(residual_summary_path)
    residual_seed_summaries = residual_payload.get("seeds", [])
    residual_by_seed = {
        int(summary["config"]["seed"]): summary
        for summary in residual_seed_summaries
    }
    missing = [seed for seed in seed_list if seed not in residual_by_seed]
    if missing:
        raise ValueError(f"residual multiseed summary is missing seeds: {missing}")

    direct_summaries = []
    comparisons = []
    seed_rows = []
    for idx, seed in enumerate(seed_list, start=1):
        print(f"[{idx}/{len(seed_list)}] training factory direct Stage 2 seed={seed}")
        config = FactoryDirectConfig(
            data_root=data_root,
            output_dir=output_root / f"seed{seed}",
            report_path=report_dir / f"factory_stage2_direct_seed{seed}_report.md",
            epochs=epochs,
            batch_size=batch_size,
            base_ch=base_ch,
            seed=seed,
            device=device,
            phantom_types=PHANTOM_TYPES,
        )
        direct_summary = train_direct_model(config)
        direct_summaries.append(direct_summary)
        comparison = summarize_factory_direct_ablation(residual_by_seed[seed], direct_summary)
        comparison["seed"] = seed
        comparisons.append(comparison)
        seed_rows.append({
            "seed": seed,
            "stage1_rmse_C": comparison["stage1_rmse_C"],
            "residual_rmse_C": comparison["residual_rmse_C"],
            "direct_rmse_C": comparison["direct_rmse_C"],
            "direct_vs_stage1_rmse_improvement_percent": comparison["direct_vs_stage1_rmse_improvement_percent"],
            "residual_vs_stage1_rmse_improvement_percent": comparison["residual_vs_stage1_rmse_improvement_percent"],
            "residual_vs_direct_rmse_improvement_percent": comparison["residual_vs_direct_rmse_improvement_percent"],
            "best_method": comparison["best_method"],
        })

    summary = _aggregate_comparisons(comparisons, seed_list)
    summary_path = report_dir / "factory_stage2_direct_ablation_summary.json"
    rows_path = report_dir / "factory_stage2_direct_ablation_runs.csv"
    report_path = report_dir / "factory_stage2_direct_ablation_report.md"
    figure_paths = _write_ablation_plot(
        picture_dir / "factory_stage2_direct_ablation_rmse",
        summary,
    )
    payload = {
        "summary": summary,
        "seed_rows": seed_rows,
        "comparisons": comparisons,
        "direct_summaries": direct_summaries,
        "residual_summary_path": str(residual_summary_path),
        "data_root": str(data_root),
        "figure_paths": figure_paths,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_comparison_rows(rows_path, seed_rows)
    _write_ablation_report(report_path, summary)
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "rows_path": str(rows_path),
        "report_path": str(report_path),
        "figure_paths": figure_paths,
        "output_root": str(output_root),
    }
