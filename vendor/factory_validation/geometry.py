from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class GeometryFitResult:
    positions: np.ndarray
    residuals_m: np.ndarray
    rmse_m: float
    max_abs_error_m: float


def _initial_positions(n_sensors: int, radius: float) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n_sensors, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def fit_sensor_geometry(
    pairs_0based: Iterable[Tuple[int, int]],
    lengths_m: Iterable[float],
    n_sensors: int = 8,
) -> GeometryFitResult:
    pairs = list(pairs_0based)
    lengths = np.asarray(list(lengths_m), dtype=np.float64)
    if len(pairs) != len(lengths):
        raise ValueError("pairs_0based and lengths_m must have the same length")
    if n_sensors < 3:
        raise ValueError("at least three sensors are required")

    radius = max(float(np.median(lengths)) / 2.0, 1.0)
    initial = _initial_positions(n_sensors, radius)
    initial[0] = [0.0, 0.0]
    initial[1, 1] = 0.0

    def pack(positions: np.ndarray) -> np.ndarray:
        return np.concatenate([[positions[1, 0]], positions[2:].ravel()])

    def unpack(vector: np.ndarray) -> np.ndarray:
        positions = np.zeros((n_sensors, 2), dtype=np.float64)
        positions[1] = [vector[0], 0.0]
        positions[2:] = vector[1:].reshape(n_sensors - 2, 2)
        return positions

    def residual(vector: np.ndarray) -> np.ndarray:
        positions = unpack(vector)
        predicted = np.array(
            [np.linalg.norm(positions[i] - positions[j]) for i, j in pairs],
            dtype=np.float64,
        )
        return predicted - lengths

    result = least_squares(
        residual,
        pack(initial),
        max_nfev=20000,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    positions = unpack(result.x)
    residuals = residual(result.x)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    max_abs = float(np.max(np.abs(residuals)))
    return GeometryFitResult(
        positions=positions,
        residuals_m=residuals,
        rmse_m=rmse,
        max_abs_error_m=max_abs,
    )


def normalize_positions_to_bbox(positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    mins = positions.min(axis=0)
    return positions - mins

