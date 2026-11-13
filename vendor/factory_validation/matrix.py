from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Tuple

import numpy as np


SRC_ROOT = Path(__file__).resolve().parents[2] / "TempFieldReconstructionv2" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ray_tracer import GridConfig, SiddonRayTracer  # noqa: E402


@dataclass(frozen=True)
class MatrixBuildResult:
    A: object
    grid: GridConfig
    positions: np.ndarray
    pairs_0based: list[Tuple[int, int]]


def build_system_matrix(
    positions: np.ndarray,
    pairs_0based: Iterable[Tuple[int, int]],
    grid_size: int,
) -> MatrixBuildResult:
    positions = np.asarray(positions, dtype=np.float64)
    pairs = list(pairs_0based)
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    lx, ly = maxs - mins
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("sensor positions must span a non-zero 2D bounding box")

    grid = GridConfig(
        nx=int(grid_size),
        ny=int(grid_size),
        lx=float(lx),
        ly=float(ly),
        x0=float(mins[0]),
        y0=float(mins[1]),
    )
    tracer = SiddonRayTracer(grid)
    A = tracer.build_system_matrix(positions, pairs)
    return MatrixBuildResult(A=A, grid=grid, positions=positions, pairs_0based=pairs)
