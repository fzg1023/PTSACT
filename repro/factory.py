"""Bounded inference for the factory-geometry synthetic validation assets.

Importing this module requires neither vendor sources nor PyTorch. Scientific
generation, inference, and pixel metrics are delegated to packaged originals.
"""

from contextlib import contextmanager
import importlib
from importlib.machinery import ModuleSpec
import json
from numbers import Integral
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import numpy as np


_ASSETS = Path("assets/NewExperiment/factory_validation")
_SEEDS = (42, 123, 456)
_DATASETS = (
    (
        "high_data",
        "data_factory_geometry_tikhonov_12x12",
        (("residual", "factory_stage2_multiseed"), ("direct", "factory_stage2_direct_ablation")),
    ),
    (
        "fullrange",
        "data_factory_geometry_tikhonov_12x12_fullrange_v1",
        (("residual", "factory_stage2_tikhonov_12x12_fullrange_v1_multiseed"),),
    ),
)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _package_path(package_root, relative):
    root = Path(package_root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path must remain inside the package: {relative}")
    return path


def _output_path(package_root, output_dir):
    root = Path(package_root).resolve()
    directory = Path(output_dir).resolve()
    outputs = root / "outputs"
    if directory == outputs or not directory.is_relative_to(outputs):
        raise ValueError("output_dir must be strictly beneath package_root/outputs/")
    target = directory / "factory_summary.json"
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"Summary output already exists: {target}")
    return target


def _read_json(root, relative):
    return json.loads(_package_path(root, relative).read_text(encoding="utf-8"))


def _normalization_values(payload):
    minimum, maximum = float(payload["T_min"]), float(payload["T_max"])
    if not np.isfinite([minimum, maximum]).all() or maximum <= minimum:
        raise ValueError("Invalid normalization temperature range")
    return minimum, maximum


def _aggregate_seed_metrics(runs):
    """Average per-seed metrics, never fields, family means, or squared RMSEs."""
    if not runs:
        raise ValueError("No seed results available")
    seeds = [int(run["seed"]) for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seed results")
    if len({run["n_samples"] for run in runs}) != 1:
        raise ValueError("Seed results must use identical sample counts")

    def aggregate_metrics(metrics):
        keys = set(metrics[0])
        if any(set(item) != keys for item in metrics):
            raise ValueError("Seed metric keys must match")
        result = {}
        for key in sorted(keys):
            values = np.asarray([item[key] for item in metrics], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Seed metrics must be finite")
            result[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
        return result

    summary = {
        "n_seeds": len(runs),
        "seeds": seeds,
        "n_samples": int(runs[0]["n_samples"]),
        "average": aggregate_metrics([run["average"] for run in runs]),
    }
    for grouping in ("per_type", "per_regime"):
        groups = [run.get(grouping, {}) for run in runs]
        labels = set(groups[0])
        if any(set(group) != labels for group in groups):
            raise ValueError("Seed group labels must match")
        summary[grouping] = {
            label: aggregate_metrics([group[label] for group in groups]) for label in sorted(labels)
        }
    return summary


def _example_kwargs(config, family, seed, temperature_regime):
    if family not in config["phantom_types"]:
        raise ValueError(f"Unknown family: {family!r}")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    regimes = config.get("temperature_regimes", [])
    if (temperature_regime is None and regimes) or (
        temperature_regime is not None and temperature_regime not in regimes
    ):
        raise ValueError(f"Temperature regime does not match dataset config: {temperature_regime!r}")
    settings = ("path_set_name", "coarse_grid", "fine_grid", "noise_std_s", "alpha")
    for key in settings:
        if key not in config:
            raise ValueError(f"Missing matched scientific setting: {key}")
    return {
        "phantom_type": family,
        "seed": int(seed),
        "temperature_regime": temperature_regime,
        **{key: config[key] for key in settings},
    }


@contextmanager
def _vendor_modules(package_root, evaluation=True):
    """Load only this release's originals; undo legacy import aliases afterward."""
    prefix = "_factory_release_" + uuid4().hex
    vendor = _package_path(package_root, "vendor")
    physics_names = ("ray_tracer", "reconstruction", "nn_prior") if evaluation else (
        "ray_tracer", "reconstruction"
    )
    missing = object()
    old_aliases = {name: sys.modules.get(name, missing) for name in physics_names}
    old_path = sys.path[:]
    old_bytecode = sys.dont_write_bytecode

    def namespace(name, directory):
        module = ModuleType(name)
        module.__package__ = name
        module.__path__ = [str(directory)]
        module.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        module.__spec__.submodule_search_locations = module.__path__
        sys.modules[name] = module

    try:
        sys.dont_write_bytecode = True
        namespace(prefix, vendor)
        namespace(prefix + ".physics", vendor / "physics")
        namespace(prefix + ".factory_validation", vendor / "factory_validation")
        for name in physics_names:
            source = _package_path(package_root, Path("vendor/physics") / f"{name}.py")
            if not source.is_file():
                raise FileNotFoundError(f"Missing packaged vendor source: {source}")
            sys.modules[name] = importlib.import_module(prefix + ".physics." + name)
        factory_prefix = prefix + ".factory_validation."
        synthetic = importlib.import_module(factory_prefix + "synthetic")
        if evaluation:
            yield SimpleNamespace(
                synthetic=synthetic,
                stage2=importlib.import_module(factory_prefix + "stage2"),
                direct=importlib.import_module(factory_prefix + "factory_direct_ablation"),
            )
        else:
            yield SimpleNamespace(synthetic=synthetic)
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
        for name, previous in old_aliases.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]


def _batch_progress(model, label, n_samples, batch_size):
    batches = (n_samples + batch_size - 1) // batch_size
    completed = 0
    last_update = time.monotonic()

    def report(_model, _inputs, _output):
        nonlocal completed, last_update
        completed += 1
        now = time.monotonic()
        if completed == 1 or completed == batches or now - last_update >= 25:
            print(f"[factory {label}] batch {completed}/{batches}", flush=True)
            last_update = now

    return model.register_forward_hook(report)


def evaluate_factory(package_root, output_dir, device="cpu", batch_size=64, limit_per_family=None):
    """Evaluate saved synthetic validation arrays; write only factory_summary.json.

    Every reported prediction is from an individual saved seed, not an ensemble.
    A positive limit takes the first N saved validation samples of each family;
    such a smoke run cannot reproduce full-validation reference numbers.
    """
    _positive_integer(batch_size, "batch_size")
    if limit_per_family is not None:
        _positive_integer(limit_per_family, "limit_per_family")
    root = Path(package_root).resolve()
    target = _output_path(root, output_dir)
    import torch

    resolved_device = torch.device(device)
    summary = {
        "domain": "factory_geometry_synthetic",
        "split": "val",
        "independent_test": False,
        "validation_used_for_checkpoint_selection": True,
        "prediction_policy": "raw_residual_or_direct; no_DC; no_display_processing; no_ensemble",
        "device": str(resolved_device),
        "batch_size": int(batch_size),
        "limit_per_family": None if limit_per_family is None else int(limit_per_family),
        "aggregation": {
            "within_seed": "original stage2._metrics_by_type: flatten all validation pixels",
            "across_seeds": "arithmetic metric mean; population standard deviation (ddof=0)",
            "percent_denominator": "mean true temperature, not per-pixel true temperatures",
        },
        "datasets": {},
    }
    completed = 0
    total = sum(1 + len(methods) * len(_SEEDS) for _, _, methods in _DATASETS)
    with _vendor_modules(root) as vendor:
        for dataset_name, dataset_dir, methods in _DATASETS:
            relative_data = _ASSETS / dataset_dir
            data_root = _package_path(root, relative_data)
            data_config = _read_json(root, relative_data / "dataset_config.json")
            families = tuple(data_config["phantom_types"])
            if not families or set(families) != set(vendor.synthetic.PHANTOM_TYPES):
                raise ValueError(f"Unexpected phantom families in {dataset_name}")
            for family in families:
                path = _package_path(root, relative_data / family / "val.npz")
                if not path.is_file():
                    raise FileNotFoundError(f"Missing packaged validation data: {path}")
            data_norm = _normalization_values(_read_json(root, relative_data / "normalization.json"))
            completed += 1
            print(f"[factory {completed}/{total}] {dataset_name}: saved Stage 1 baseline", flush=True)
            baseline = vendor.stage2.evaluate_stage1_baseline(
                data_root, split="val", phantom_types=families, limit_per_type=limit_per_family
            )
            dataset_summary = {
                "data_root": relative_data.as_posix(),
                "baseline": baseline,
            }
            for role, model_dir in methods:
                runs = []
                for seed in _SEEDS:
                    completed += 1
                    label = f"{dataset_name} {role} seed{seed}"
                    print(f"[factory {completed}/{total}] {label}", flush=True)
                    model_relative = _ASSETS / "models" / model_dir / f"seed{seed}"
                    stem = "stage2" if role == "residual" else "direct"
                    config = _read_json(root, model_relative / f"{stem}_config.json")
                    if int(config["seed"]) != seed or tuple(config["phantom_types"]) != families:
                        raise ValueError(f"Checkpoint config does not match validation run: {label}")
                    norm = _normalization_values(_read_json(root, model_relative / "normalization.json"))
                    if norm != data_norm:
                        raise ValueError(f"Checkpoint normalization does not match dataset: {label}")
                    model = vendor.stage2.AttentionUNet(
                        in_channels=1,
                        base_ch=config["base_ch"],
                        dropout_p=config["dropout_p"],
                        use_attention=True,
                    ).to(resolved_device)
                    checkpoint_relative = model_relative / f"{stem}_best.pth"
                    state = torch.load(
                        _package_path(root, checkpoint_relative),
                        map_location=resolved_device,
                        weights_only=True,
                    )
                    model.load_state_dict(state, strict=True)
                    del state
                    evaluator = vendor.stage2.evaluate_stage2_model if role == "residual" else (
                        vendor.direct.evaluate_direct_model
                    )
                    progress = _batch_progress(model, label, baseline["n_samples"], int(batch_size))
                    try:
                        metrics = evaluator(
                            model, data_root, norm[0], norm[1], resolved_device,
                            split="val", batch_size=int(batch_size), phantom_types=families,
                            limit_per_type=limit_per_family,
                        )
                    finally:
                        progress.remove()
                    runs.append({
                        "seed": seed,
                        "checkpoint": checkpoint_relative.as_posix(),
                        "normalization": {"T_min": norm[0], "T_max": norm[1]},
                        **metrics,
                    })
                    del model
                dataset_summary[role] = {"runs": runs, "aggregate": _aggregate_seed_metrics(runs)}
            summary["datasets"][dataset_name] = dataset_summary

    # Recheck after inference; exclusive creation also closes the file-existence race.
    target = _output_path(root, output_dir)
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _output_path(root, output_dir)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(f"[factory complete] validation summary: {target}", flush=True)
    return summary


def generate_factory_example(package_root, family="single_peak", seed=20260607, temperature_regime=None):
    """Return one synthetic sample using the matched packaged dataset config.

    This does not write files, train, evaluate, or replay a historical dataset.
    None selects the legacy high-temperature data; a named regime selects the
    fullrange dataset and must occur in its original configuration.
    """
    root = Path(package_root).resolve()
    dataset_dir = _DATASETS[0 if temperature_regime is None else 1][1]
    config = _read_json(root, _ASSETS / dataset_dir / "dataset_config.json")
    kwargs = _example_kwargs(config, family, seed, temperature_regime)
    print(f"[factory example 1/1] {family}/{temperature_regime or 'legacy_high'} seed={seed}", flush=True)
    with _vendor_modules(root, evaluation=False) as vendor:
        return vendor.synthetic.generate_one_sample(**kwargs)
