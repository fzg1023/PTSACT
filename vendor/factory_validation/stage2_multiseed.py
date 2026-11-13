import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .config import OUTPUT_ROOT, RESULTS_DIR
from .stage2 import FactoryStage2Config, train_stage2_model
from .synthetic import PHANTOM_TYPES


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}


def summarize_multiseed_results(seed_summaries: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not seed_summaries:
        raise ValueError("no seed summaries available")
    seeds = [int(summary["config"]["seed"]) for summary in seed_summaries]
    stage2_rmse = [float(summary["stage2"]["average"]["RMSE_C"]) for summary in seed_summaries]
    baseline_rmse = [float(summary["baseline"]["average"]["RMSE_C"]) for summary in seed_summaries]
    improvements = [float(summary["improvement"]["RMSE_C_percent"]) for summary in seed_summaries]
    accepted = [bool(summary["improvement"]["accepted"]) for summary in seed_summaries]

    phantom_types = sorted(
        set().union(*[set(summary["stage2"]["per_type"].keys()) for summary in seed_summaries])
    )
    per_type = {}
    for phantom_type in phantom_types:
        base_values = [
            float(summary["baseline"]["per_type"][phantom_type]["RMSE_C"])
            for summary in seed_summaries
            if phantom_type in summary["baseline"]["per_type"]
        ]
        stage2_values = [
            float(summary["stage2"]["per_type"][phantom_type]["RMSE_C"])
            for summary in seed_summaries
            if phantom_type in summary["stage2"]["per_type"]
        ]
        base_stats = _mean_std(base_values)
        stage2_stats = _mean_std(stage2_values)
        per_type[phantom_type] = {
            "baseline_rmse_C_mean": base_stats["mean"],
            "baseline_rmse_C_std": base_stats["std"],
            "stage2_rmse_C_mean": stage2_stats["mean"],
            "stage2_rmse_C_std": stage2_stats["std"],
        }

    regime_names = sorted(
        set().union(*[set(summary["stage2"].get("per_regime", {}).keys()) for summary in seed_summaries])
    )
    per_regime = {}
    for regime in regime_names:
        base_values = [
            float(summary["baseline"]["per_regime"][regime]["RMSE_C"])
            for summary in seed_summaries
            if regime in summary["baseline"].get("per_regime", {})
        ]
        stage2_values = [
            float(summary["stage2"]["per_regime"][regime]["RMSE_C"])
            for summary in seed_summaries
            if regime in summary["stage2"].get("per_regime", {})
        ]
        if not base_values or not stage2_values:
            continue
        base_stats = _mean_std(base_values)
        stage2_stats = _mean_std(stage2_values)
        per_regime[regime] = {
            "baseline_rmse_C_mean": base_stats["mean"],
            "baseline_rmse_C_std": base_stats["std"],
            "stage2_rmse_C_mean": stage2_stats["mean"],
            "stage2_rmse_C_std": stage2_stats["std"],
            "rmse_improvement_percent_mean": float(
                (base_stats["mean"] - stage2_stats["mean"]) / (base_stats["mean"] + 1e-14) * 100.0
            ),
        }

    stage2_stats = _mean_std(stage2_rmse)
    baseline_stats = _mean_std(baseline_rmse)
    improvement_stats = _mean_std(improvements)
    return {
        "n_seeds": int(len(seed_summaries)),
        "seeds": seeds,
        "all_accepted": all(accepted),
        "baseline_rmse_C_mean": baseline_stats["mean"],
        "baseline_rmse_C_std": baseline_stats["std"],
        "stage2_rmse_C_mean": stage2_stats["mean"],
        "stage2_rmse_C_std": stage2_stats["std"],
        "rmse_improvement_percent_mean": improvement_stats["mean"],
        "rmse_improvement_percent_std": improvement_stats["std"],
        "per_type": per_type,
        "per_regime": per_regime,
    }


def _write_seed_rows(path: Path, seed_summaries: Sequence[Dict[str, object]]) -> None:
    rows = []
    for summary in seed_summaries:
        rows.append(
            {
                "seed": int(summary["config"]["seed"]),
                "baseline_rmse_C": float(summary["baseline"]["average"]["RMSE_C"]),
                "stage2_rmse_C": float(summary["stage2"]["average"]["RMSE_C"]),
                "rmse_improvement_percent": float(summary["improvement"]["RMSE_C_percent"]),
                "accepted": bool(summary["improvement"]["accepted"]),
                "model_path": summary["model_path"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, summary: Dict[str, object]) -> None:
    lines = [
        "# Factory Stage 2 Multiseed Report",
        "",
        "## Summary",
        "",
        f"- Seeds: `{summary['seeds']}`",
        f"- All accepted: `{summary['all_accepted']}`",
        f"- Stage 1 RMSE: `{summary['baseline_rmse_C_mean']:.4f} +/- {summary['baseline_rmse_C_std']:.4f} C`",
        f"- Stage 2 RMSE: `{summary['stage2_rmse_C_mean']:.4f} +/- {summary['stage2_rmse_C_std']:.4f} C`",
        f"- RMSE improvement: `{summary['rmse_improvement_percent_mean']:.2f} +/- {summary['rmse_improvement_percent_std']:.2f}%`",
        "",
        "## Per Phantom",
        "",
        "| Phantom | Stage 1 RMSE C | Stage 2 RMSE C |",
        "|---|---:|---:|",
    ]
    for phantom_type, stats in summary["per_type"].items():
        lines.append(
            f"| {phantom_type} | {stats['baseline_rmse_C_mean']:.4f} +/- {stats['baseline_rmse_C_std']:.4f} | "
            f"{stats['stage2_rmse_C_mean']:.4f} +/- {stats['stage2_rmse_C_std']:.4f} |"
        )
    if summary.get("per_regime"):
        lines.extend(
            [
                "",
                "## Per Temperature Regime",
                "",
                "| Regime | Stage 1 RMSE C | Stage 2 RMSE C | Improvement |",
                "|---|---:|---:|---:|",
            ]
        )
        for regime, stats in summary["per_regime"].items():
            lines.append(
                f"| {regime} | {stats['baseline_rmse_C_mean']:.4f} +/- {stats['baseline_rmse_C_std']:.4f} | "
                f"{stats['stage2_rmse_C_mean']:.4f} +/- {stats['stage2_rmse_C_std']:.4f} | "
                f"{stats['rmse_improvement_percent_mean']:.2f}% |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage2_multiseed(
    data_root: Path,
    seeds: Iterable[int] = (42, 123, 456),
    epochs: int = 40,
    batch_size: int = 64,
    base_ch: int = 16,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    dropout_p: float = 0.1,
    device: str = None,
    output_root: Path = OUTPUT_ROOT / "models" / "factory_stage2_multiseed",
    report_dir: Path = RESULTS_DIR,
    output_stem: str = "factory_stage2_multiseed",
) -> Dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    seed_summaries: List[Dict[str, object]] = []
    seed_list = [int(seed) for seed in seeds]
    for idx, seed in enumerate(seed_list, start=1):
        print(f"[{idx}/{len(seed_list)}] training factory Stage 2 seed={seed}")
        config = FactoryStage2Config(
            data_root=data_root,
            output_dir=output_root / f"seed{seed}",
            report_path=report_dir / f"{output_stem}_seed{seed}_report.md",
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            base_ch=base_ch,
            dropout_p=dropout_p,
            seed=seed,
            device=device,
            phantom_types=PHANTOM_TYPES,
        )
        seed_summaries.append(train_stage2_model(config))

    summary = summarize_multiseed_results(seed_summaries)
    summary_path = report_dir / f"{output_stem}_summary.json"
    rows_path = report_dir / f"{output_stem}_runs.csv"
    report_path = report_dir / f"{output_stem}_report.md"
    summary_path.write_text(
        json.dumps({"summary": summary, "seeds": seed_summaries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_seed_rows(rows_path, seed_summaries)
    _write_report(report_path, summary)
    return {
        "summary": summary,
        "seed_summaries": seed_summaries,
        "summary_path": str(summary_path),
        "rows_path": str(rows_path),
        "report_path": str(report_path),
        "output_root": str(output_root),
    }
