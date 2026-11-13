import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence, Union

import numpy as np
from scipy.ndimage import zoom

from .config import OUTPUT_ROOT, PATH_SETS, chord_lengths, effective_lengths, path_pairs_0based
from .conversions import tof_to_path_temperature
from .geometry import fit_sensor_geometry
from .matrix import build_system_matrix
from .stage1 import reconstruct_tikhonov_temperature


SRC_ROOT = Path(__file__).resolve().parents[2] / "TempFieldReconstructionv2" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ray_tracer import temperature_to_slowness  # noqa: E402


PHANTOM_TYPES = ("single_peak", "dual_peak", "triple_peak")
FULL_RANGE_TEMPERATURE_REGIMES = (
    "room_static",
    "low_warm",
    "mid_transition_1",
    "mid_transition_2",
    "high_running",
)

_FULL_RANGE_REGIME_SPECS = {
    "room_static": {
        "background": (18.0, 25.0),
        "clip": (15.0, 35.0),
        "gradient_fraction": 0.020,
        "single_scale": ((0.16, 0.30), (0.08, 0.18)),
        "multi_scale": (0.12, 0.28),
    },
    "low_warm": {
        "background": (45.0, 90.0),
        "clip": (35.0, 150.0),
        "gradient_fraction": 0.035,
        "single_scale": ((0.18, 0.34), (0.09, 0.20)),
        "multi_scale": (0.13, 0.29),
    },
    "mid_transition_1": {
        "background": (170.0, 240.0),
        "clip": (150.0, 350.0),
        "gradient_fraction": 0.045,
        "single_scale": ((0.20, 0.36), (0.10, 0.22)),
        "multi_scale": (0.14, 0.30),
    },
    "mid_transition_2": {
        "background": (380.0, 500.0),
        "clip": (350.0, 650.0),
        "gradient_fraction": 0.050,
        "single_scale": ((0.20, 0.36), (0.10, 0.22)),
        "multi_scale": (0.14, 0.30),
    },
    "high_running": {
        "background": (650.0, 820.0),
        "clip": (650.0, 1200.0),
        "gradient_fraction": 0.050,
        "single_scale": ((0.18, 0.34), (0.10, 0.22)),
        "multi_scale": (0.13, 0.30),
    },
}


@dataclass(frozen=True)
class FactorySyntheticConfig:
    output_root: Union[Path, str] = OUTPUT_ROOT / "data_factory_geometry_tikhonov_12x12"
    path_set_name: str = "sensor_config"
    coarse_grid: int = 12
    fine_grid: int = 32
    n_total_per_type: int = 300
    n_train_per_type: int = 240
    n_total_per_regime_type: Optional[int] = None
    n_train_per_regime_type: Optional[int] = None
    seed: int = 20260607
    noise_std_s: float = 3e-5
    alpha: float = 1e-4
    phantom_types: Sequence[str] = PHANTOM_TYPES
    temperature_regimes: Sequence[str] = ()

    @property
    def n_val_per_type(self) -> int:
        return self.n_total_per_type - self.n_train_per_type

    @property
    def n_total_per_combo(self) -> int:
        if self.temperature_regimes and self.n_total_per_regime_type is not None:
            return int(self.n_total_per_regime_type)
        return int(self.n_total_per_type)

    @property
    def n_train_per_combo(self) -> int:
        if self.temperature_regimes and self.n_train_per_regime_type is not None:
            return int(self.n_train_per_regime_type)
        return int(self.n_train_per_type)

    @property
    def n_val_per_combo(self) -> int:
        return self.n_total_per_combo - self.n_train_per_combo


def _field_grid(nx: int, ny: int, lx: float, ly: float):
    x = (np.arange(nx) + 0.5) * lx / nx
    y = (np.arange(ny) + 0.5) * ly / ny
    return np.meshgrid(x, y)


def _add_gaussian(field, X, Y, cx: float, cy: float, amp: float, sigma: float) -> None:
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    field += amp * np.exp(-r2 / (2.0 * sigma ** 2))


def _make_legacy_high_temperature_field(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    phantom_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    X, Y = _field_grid(nx, ny, lx, ly)
    background = rng.uniform(620.0, 820.0)
    field = np.full((ny, nx), background, dtype=np.float64)
    scale = min(lx, ly)

    if phantom_type == "single_peak":
        cx = rng.uniform(0.25 * lx, 0.75 * lx)
        cy = rng.uniform(0.25 * ly, 0.75 * ly)
        _add_gaussian(field, X, Y, cx, cy, rng.uniform(180.0, 320.0), rng.uniform(0.15, 0.26) * scale)
        _add_gaussian(field, X, Y, cx, cy, rng.uniform(100.0, 220.0), rng.uniform(0.07, 0.14) * scale)
    elif phantom_type == "dual_peak":
        for _ in range(2):
            _add_gaussian(
                field,
                X,
                Y,
                rng.uniform(0.18 * lx, 0.82 * lx),
                rng.uniform(0.18 * ly, 0.82 * ly),
                rng.uniform(130.0, 280.0),
                rng.uniform(0.08, 0.18) * scale,
            )
    else:
        for _ in range(3):
            _add_gaussian(
                field,
                X,
                Y,
                rng.uniform(0.15 * lx, 0.85 * lx),
                rng.uniform(0.15 * ly, 0.85 * ly),
                rng.uniform(90.0, 220.0),
                rng.uniform(0.06, 0.15) * scale,
            )

    gradient_x = rng.uniform(-25.0, 25.0) * (X / max(lx, 1e-12) - 0.5)
    gradient_y = rng.uniform(-25.0, 25.0) * (Y / max(ly, 1e-12) - 0.5)
    field += gradient_x + gradient_y
    return np.clip(field, 500.0, 1200.0)


def _make_fullrange_temperature_field(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    phantom_type: str,
    temperature_regime: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if temperature_regime not in _FULL_RANGE_REGIME_SPECS:
        raise ValueError(f"unknown temperature_regime={temperature_regime!r}")
    spec = _FULL_RANGE_REGIME_SPECS[temperature_regime]
    X, Y = _field_grid(nx, ny, lx, ly)
    clip_min, clip_max = spec["clip"]
    span = float(clip_max - clip_min)
    field = np.full((ny, nx), rng.uniform(*spec["background"]), dtype=np.float64)
    scale = min(lx, ly)

    if phantom_type == "single_peak":
        cx = rng.uniform(0.25 * lx, 0.75 * lx)
        cy = rng.uniform(0.25 * ly, 0.75 * ly)
        broad_scale, narrow_scale = spec["single_scale"]
        _add_gaussian(
            field,
            X,
            Y,
            cx,
            cy,
            rng.uniform(*broad_scale) * span,
            rng.uniform(0.15, 0.26) * scale,
        )
        _add_gaussian(
            field,
            X,
            Y,
            cx,
            cy,
            rng.uniform(*narrow_scale) * span,
            rng.uniform(0.07, 0.14) * scale,
        )
    else:
        n_peaks = 2 if phantom_type == "dual_peak" else 3
        for _ in range(n_peaks):
            _add_gaussian(
                field,
                X,
                Y,
                rng.uniform(0.15 * lx, 0.85 * lx),
                rng.uniform(0.15 * ly, 0.85 * ly),
                rng.uniform(*spec["multi_scale"]) * span,
                rng.uniform(0.06, 0.18) * scale,
            )

    gradient_span = float(spec["gradient_fraction"]) * span
    field += rng.uniform(-gradient_span, gradient_span) * (X / max(lx, 1e-12) - 0.5)
    field += rng.uniform(-gradient_span, gradient_span) * (Y / max(ly, 1e-12) - 0.5)
    return np.clip(field, clip_min, clip_max)


def make_factory_temperature_field(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    phantom_type: str,
    rng: np.random.Generator,
    temperature_regime: Optional[str] = None,
) -> np.ndarray:
    if phantom_type not in PHANTOM_TYPES:
        raise ValueError(f"unknown phantom_type={phantom_type!r}")
    if temperature_regime is None:
        return _make_legacy_high_temperature_field(nx, ny, lx, ly, phantom_type, rng)
    return _make_fullrange_temperature_field(nx, ny, lx, ly, phantom_type, temperature_regime, rng)


def _resize_to_fine(field: np.ndarray, fine_grid: int) -> np.ndarray:
    factor_y = fine_grid / field.shape[0]
    factor_x = fine_grid / field.shape[1]
    resized = zoom(field, zoom=(factor_y, factor_x), order=3)
    if resized.shape[0] != fine_grid or resized.shape[1] != fine_grid:
        fixed = np.zeros((fine_grid, fine_grid), dtype=resized.dtype)
        rows = min(fine_grid, resized.shape[0])
        cols = min(fine_grid, resized.shape[1])
        fixed[:rows, :cols] = resized[:rows, :cols]
        if rows < fine_grid:
            fixed[rows:] = fixed[rows - 1]
        if cols < fine_grid:
            fixed[:, cols:] = fixed[:, cols - 1:cols]
        resized = fixed
    return resized


def _build_factory_matrices(path_set_name: str, fine_grid: int, coarse_grid: int):
    paths = PATH_SETS[path_set_name]
    pairs = path_pairs_0based(paths)
    chord = np.asarray(chord_lengths(paths), dtype=np.float64)
    geometry = fit_sensor_geometry(pairs, chord, n_sensors=8)
    fine = build_system_matrix(geometry.positions, pairs, fine_grid)
    coarse = build_system_matrix(geometry.positions, pairs, coarse_grid)
    return geometry, fine, coarse


def generate_one_sample(
    phantom_type: str,
    coarse_grid: int,
    seed: int,
    noise_std_s: float = 3e-5,
    path_set_name: str = "sensor_config",
    fine_grid: int = 32,
    alpha: float = 1e-4,
    matrices: Optional[Dict[str, object]] = None,
    temperature_regime: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    if matrices is None:
        _, fine_matrix, coarse_matrix = _build_factory_matrices(path_set_name, fine_grid, coarse_grid)
    else:
        fine_matrix = matrices["fine"]
        coarse_matrix = matrices["coarse"]

    T_true = make_factory_temperature_field(
        nx=fine_grid,
        ny=fine_grid,
        lx=fine_matrix.grid.lx,
        ly=fine_matrix.grid.ly,
        phantom_type=phantom_type,
        rng=rng,
        temperature_regime=temperature_regime,
    )
    slowness = temperature_to_slowness(T_true.ravel())
    tof = fine_matrix.A.dot(slowness)
    if noise_std_s > 0.0:
        tof = tof + rng.normal(0.0, noise_std_s, size=tof.shape)
        tof = np.maximum(tof, 1e-8)
    path_temperatures = tof_to_path_temperature(
        tof,
        np.asarray(effective_lengths(PATH_SETS[path_set_name]), dtype=np.float64),
    )

    stage1 = reconstruct_tikhonov_temperature(
        coarse_matrix.A,
        tof,
        grid_shape=(coarse_grid, coarse_grid),
        alpha=alpha,
    )
    T_stage1 = _resize_to_fine(stage1.temperature, fine_grid)
    return {
        "tof": tof.astype(np.float64),
        "T_stage1": T_stage1.astype(np.float32),
        "T_true": T_true.astype(np.float32),
        "path_temperatures": path_temperatures.astype(np.float64),
        "phantom_type": phantom_type,
        "temperature_regime": temperature_regime or "legacy_high",
        "seed": int(seed),
    }


def _save_split(output_dir: Path, split_name: str, samples: Sequence[Dict[str, np.ndarray]], indices: np.ndarray) -> None:
    selected = [samples[int(idx)] for idx in indices]
    payload = {
        "tof": np.stack([sample["tof"] for sample in selected]),
        "T_stage1": np.stack([sample["T_stage1"] for sample in selected]),
        "T_true": np.stack([sample["T_true"] for sample in selected]),
        "path_temperatures": np.stack([sample["path_temperatures"] for sample in selected]),
        "phantom_type": np.asarray([sample["phantom_type"] for sample in selected]),
        "temperature_regime": np.asarray([sample["temperature_regime"] for sample in selected]),
        "seed": np.asarray([sample["seed"] for sample in selected], dtype=np.int64),
    }
    np.savez_compressed(output_dir / f"{split_name}.npz", **payload)


def _split_samples_stratified(
    samples: Sequence[Dict[str, np.ndarray]],
    n_train: int,
    seed: int,
) -> Dict[str, Sequence[Dict[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    return {
        "train": [samples[int(idx)] for idx in train_idx],
        "val": [samples[int(idx)] for idx in val_idx],
    }


def generate_factory_dataset(config: FactorySyntheticConfig) -> Dict[str, object]:
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    geometry, fine_matrix, coarse_matrix = _build_factory_matrices(
        config.path_set_name,
        config.fine_grid,
        config.coarse_grid,
    )
    matrices = {"fine": fine_matrix, "coarse": coarse_matrix}
    all_true_values = []
    total_samples = 0

    temperature_regimes = tuple(config.temperature_regimes)
    if temperature_regimes:
        unknown = sorted(set(temperature_regimes) - set(FULL_RANGE_TEMPERATURE_REGIMES))
        if unknown:
            raise ValueError(f"unknown full-range temperature regimes: {unknown}")

    for type_idx, phantom_type in enumerate(config.phantom_types):
        output_dir = output_root / phantom_type
        output_dir.mkdir(parents=True, exist_ok=True)
        train_samples = []
        val_samples = []
        regimes_for_generation = temperature_regimes or (None,)
        for regime_idx, regime in enumerate(regimes_for_generation):
            samples = []
            regime_label = regime or "legacy_high"
            n_total = config.n_total_per_combo
            n_train = config.n_train_per_combo
            for i in range(n_total):
                global_index = ((type_idx * len(regimes_for_generation) + regime_idx) * n_total) + i
                if i == 0 or (i + 1) % 10 == 0 or i + 1 == n_total:
                    print(f"[{phantom_type}/{regime_label}] generating {i + 1}/{n_total}")
                sample = generate_one_sample(
                    phantom_type=phantom_type,
                    coarse_grid=config.coarse_grid,
                    seed=config.seed + global_index,
                    noise_std_s=config.noise_std_s,
                    path_set_name=config.path_set_name,
                    fine_grid=config.fine_grid,
                    alpha=config.alpha,
                    matrices=matrices,
                    temperature_regime=regime,
                )
                samples.append(sample)
                all_true_values.append(sample["T_true"].ravel())
                total_samples += 1
            split = _split_samples_stratified(
                samples,
                n_train=n_train,
                seed=config.seed + type_idx * 1009 + regime_idx,
            )
            train_samples.extend(split["train"])
            val_samples.extend(split["val"])

        _save_split(output_dir, "train", train_samples, np.arange(len(train_samples)))
        _save_split(output_dir, "val", val_samples, np.arange(len(val_samples)))
        meta = {
            "phantom_type": phantom_type,
            "path_set_name": config.path_set_name,
            "coarse_grid": config.coarse_grid,
            "fine_grid": config.fine_grid,
            "temperature_regimes": list(temperature_regimes) if temperature_regimes else ["legacy_high"],
            "n_total_per_combo": config.n_total_per_combo,
            "n_train_per_combo": config.n_train_per_combo,
            "n_val_per_combo": config.n_val_per_combo,
            "n_total": len(train_samples) + len(val_samples),
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "noise_std_s": config.noise_std_s,
            "alpha": config.alpha,
            "geometry_rmse_m": geometry.rmse_m,
        }
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    all_true = np.concatenate(all_true_values)
    normalization = {
        "T_min": float(np.min(all_true)),
        "T_max": float(np.max(all_true)),
        "path_set_name": config.path_set_name,
        "coarse_grid": config.coarse_grid,
        "fine_grid": config.fine_grid,
        "temperature_regimes": list(temperature_regimes) if temperature_regimes else ["legacy_high"],
    }
    (output_root / "normalization.json").write_text(
        json.dumps(normalization, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    config_payload = asdict(config)
    config_payload["output_root"] = str(config.output_root)
    (output_root / "dataset_config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "output_root": str(output_root),
        "total_samples": total_samples,
        "geometry_rmse_m": float(geometry.rmse_m),
        "normalization": normalization,
    }
