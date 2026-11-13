from functools import lru_cache

import numpy as np
from scipy.ndimage import zoom

from .common import register_vendor_paths


@lru_cache(maxsize=8)
def projection(grid_size=16):
    register_vendor_paths()
    from ray_tracer import TransducerConfig, GridConfig, SiddonRayTracer
    trans = TransducerConfig(n_transducers=16, geometry="square", side_length=20.0, center=(10.0, 10.0))
    grid = GridConfig(nx=grid_size, ny=grid_size, lx=20.0, ly=20.0)
    positions = trans.get_positions()
    pairs = np.asarray(trans.get_ray_pairs())
    matrix = SiddonRayTracer(grid).build_system_matrix(positions, pairs)
    return matrix, grid, positions, pairs


def reconstruct_tof(tof, grid_size=16, method="tikhonov"):
    register_vendor_paths()
    from reconstruction import TikhonovReconstructor, RBFReconstructor
    from ray_tracer import slowness_to_temperature
    matrix, grid, _, _ = projection(grid_size)
    tof = np.asarray(tof, dtype=np.float64)
    if tof.shape != (120,) or not np.isfinite(tof).all() or np.any(tof <= 0):
        raise ValueError("Expected one finite, strictly positive 120-path TOF vector")
    if method == "tikhonov":
        model = TikhonovReconstructor(matrix, tof, alpha=0.01, reg_type="gradient",
                                      grid_shape=(grid_size, grid_size))
    elif method == "rbf":
        xx, yy = grid.pixel_centers()
        centers = np.column_stack([xx.ravel(), yy.ravel()])
        model = RBFReconstructor(matrix, tof, grid_shape=(grid_size, grid_size), pixel_centers=centers,
                                 n_centers=50, kernel="multiquadric", alpha=1e-3, center_strategy="grid")
    else:
        raise ValueError("Unknown Stage 1 solver")
    field = slowness_to_temperature(model.solve()).reshape(grid_size, grid_size)
    return (zoom(field, 32.0 / grid_size, order=3) if grid_size != 32 else field).astype(np.float32)


def equivalent_temperature(tof, lengths):
    return 273.15 * ((np.asarray(lengths) / np.asarray(tof) / 331.3) ** 2 - 1.0)


def equivalent_tof(temperature, lengths):
    return np.asarray(lengths) / (331.3 * np.sqrt(1.0 + np.asarray(temperature) / 273.15))


def solve_tv(tof, settings):
    register_vendor_paths()
    from tv_solver import TVSolver
    from ray_tracer import slowness_to_temperature
    grid_size = settings["selected_grid"]
    matrix, _, _, _ = projection(grid_size)
    scale = 20.0 * np.sqrt(120.0)
    solver = TVSolver(matrix / scale, (grid_size, grid_size), **settings["solver"])
    result = solver.solve(np.asarray(tof) / (scale / 343.0), beta=settings["selected_lambda"] / grid_size)
    field = slowness_to_temperature(result["solution"] / 343.0).reshape(grid_size, grid_size)
    return field.astype(np.float32), {k: v for k, v in result.items() if k != "solution"}
