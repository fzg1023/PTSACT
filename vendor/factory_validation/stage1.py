from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[2] / "TempFieldReconstructionv2" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reconstruction import TikhonovReconstructor  # noqa: E402
from ray_tracer import slowness_to_temperature  # noqa: E402

from .conversions import tof_to_path_temperature


@dataclass(frozen=True)
class Stage1Reconstruction:
    slowness: np.ndarray
    temperature: np.ndarray
    alpha: float
    grid_shape: Tuple[int, int]


def reconstruct_tikhonov_temperature(
    A,
    tof_s,
    grid_shape: Tuple[int, int],
    alpha: float,
    reg_type: str = "gradient",
) -> Stage1Reconstruction:
    tikhonov = TikhonovReconstructor(
        A,
        np.asarray(tof_s, dtype=np.float64),
        alpha=float(alpha),
        reg_type=reg_type,
        grid_shape=grid_shape,
    )
    slowness = tikhonov.solve()
    temperature = slowness_to_temperature(slowness).reshape(grid_shape)
    return Stage1Reconstruction(
        slowness=slowness,
        temperature=temperature,
        alpha=float(alpha),
        grid_shape=grid_shape,
    )


def predicted_path_temperatures(A, slowness, effective_lengths_m):
    predicted_tof = A.dot(np.asarray(slowness, dtype=np.float64))
    return tof_to_path_temperature(predicted_tof, np.asarray(effective_lengths_m, dtype=np.float64))


def compute_path_temperature_metrics(observed, predicted) -> Dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    diff = predicted - observed
    return {
        "path_mae_c": float(np.mean(np.abs(diff))),
        "path_rmse_c": float(np.sqrt(np.mean(diff ** 2))),
        "path_bias_c": float(np.mean(diff)),
        "path_max_abs_c": float(np.max(np.abs(diff))),
    }


def _partition_edges(size: int, n_parts: int) -> np.ndarray:
    base = size // n_parts
    edges = [0]
    for _ in range(n_parts - 1):
        edges.append(edges[-1] + base)
    edges.append(size)
    return np.asarray(edges, dtype=int)


def proxy_area_means(field, n_rows: int = 4, n_cols: int = 6) -> np.ndarray:
    field = np.asarray(field, dtype=np.float64)
    row_edges = _partition_edges(field.shape[0], n_rows)
    col_edges = _partition_edges(field.shape[1], n_cols)
    means = []
    for row in range(n_rows):
        for col in range(n_cols):
            tile = field[row_edges[row]:row_edges[row + 1], col_edges[col]:col_edges[col + 1]]
            means.append(float(np.mean(tile)))
    return np.asarray(means, dtype=np.float64)


def field_smoothness(field) -> float:
    field = np.asarray(field, dtype=np.float64)
    gy, gx = np.gradient(field)
    return float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
