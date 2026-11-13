from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .paper_palette import PAPER_THERMAL_CMAP  # noqa: E402


def plot_timeseries(metrics, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, metrics["path_mae_c"], color="#1f77b4", linewidth=1.5)
    axes[0].set_ylabel("Path MAE (C)")
    axes[1].plot(x, metrics["max_temperature_diff_c"], color="#d62728", linewidth=1.5)
    axes[1].set_ylabel("Max diff (C)")
    axes[2].plot(x, metrics["smoothness_c_per_px"], color="#2ca02c", linewidth=1.5)
    axes[2].set_ylabel("Smoothness")
    axes[2].set_xlabel("Sample index")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_reconstruction_examples(examples, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not examples:
        return
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False, constrained_layout=True)
    values = [example["temperature"] for example in examples]
    vmin = min(float(np.min(value)) for value in values)
    vmax = max(float(np.max(value)) for value in values)
    image = None
    for ax, example in zip(axes.ravel(), examples):
        image = ax.imshow(example["temperature"], origin="lower", cmap=PAPER_THERMAL_CMAP, vmin=vmin, vmax=vmax)
        ax.set_title(f"row {example['row_id']}")
        ax.set_xticks([])
        ax.set_yticks([])
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="Temperature (C)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
