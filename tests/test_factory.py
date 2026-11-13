"""Factory adapter tests: no vendor modules, checkpoints, or datasets required."""

import builtins
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ADAPTER_PATH = Path(__file__).resolve().parents[1] / "repro" / "factory.py"


def adapter():
    assert ADAPTER_PATH.is_file(), "The bounded factory adapter has not been implemented"
    spec = importlib.util.spec_from_file_location("factory_adapter_under_test", ADAPTER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def example_config():
    return {
        "path_set_name": "matched_path_set",
        "coarse_grid": 7,
        "fine_grid": 24,
        "noise_std_s": 0.000073,
        "alpha": 0.0023,
        "phantom_types": ["single_peak", "dual_peak", "triple_peak"],
    }


def seed_results():
    return [
        {
            "seed": seed,
            "average": {"RMSE_C": rmse, "MAE_C": mae},
            "per_type": {
                "single_peak": {"RMSE_C": rmse + 1.0, "MAE_C": mae},
                "dual_peak": {"RMSE_C": rmse + 5.0, "MAE_C": mae},
            },
            "per_regime": {"room_static": {"RMSE_C": rmse / 2.0, "MAE_C": mae}},
            "n_samples": 13,
        }
        for seed, rmse, mae in [(42, 1.0, 0.5), (123, 2.0, 1.0), (456, 9.0, 4.5)]
    ]


def test_seed_average_uses_existing_pooled_metrics_not_family_mean_or_ensemble():
    summary = adapter()._aggregate_seed_metrics(seed_results())
    assert summary["seeds"] == [42, 123, 456]
    assert summary["n_seeds"] == 3
    assert summary["n_samples"] == 13
    assert summary["average"]["RMSE_C"]["mean"] == 4.0
    assert summary["average"]["MAE_C"]["mean"] == 2.0
    assert summary["per_type"]["single_peak"]["RMSE_C"]["mean"] == 5.0
    assert summary["per_type"]["dual_peak"]["RMSE_C"]["mean"] == 9.0
    assert summary["per_regime"]["room_static"]["RMSE_C"]["mean"] == 2.0


def test_seed_standard_deviation_is_population_standard_deviation():
    stats = adapter()._aggregate_seed_metrics(seed_results())["average"]["RMSE_C"]
    assert stats["std"] == pytest.approx(np.std([1.0, 2.0, 9.0], ddof=0))
    assert stats["std"] != pytest.approx(np.std([1.0, 2.0, 9.0], ddof=1))


def test_seed_aggregation_does_not_mutate_inputs():
    runs = seed_results()
    before = repr(runs)
    adapter()._aggregate_seed_metrics(runs)
    assert repr(runs) == before


def test_empty_seed_results_are_rejected():
    with pytest.raises(ValueError, match="seed"):
        adapter()._aggregate_seed_metrics([])


def test_incomparable_seed_sample_counts_are_rejected():
    runs = seed_results()
    runs[1]["n_samples"] = 12
    with pytest.raises(ValueError, match="sample"):
        adapter()._aggregate_seed_metrics(runs)


def test_duplicate_seed_results_are_rejected():
    runs = seed_results()
    runs[1]["seed"] = 42
    with pytest.raises(ValueError, match="seed"):
        adapter()._aggregate_seed_metrics(runs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_seed_metrics_are_rejected(value):
    runs = seed_results()
    runs[0]["average"]["RMSE_C"] = value
    with pytest.raises(ValueError, match="finite"):
        adapter()._aggregate_seed_metrics(runs)


def test_legacy_direct_summary_need_not_have_regime_metrics():
    runs = seed_results()
    for run in runs:
        del run["per_regime"]
    assert adapter()._aggregate_seed_metrics(runs)["per_regime"] == {}


def test_seed_aggregation_rejects_missing_group_instead_of_averaging_partial_runs():
    runs = seed_results()
    del runs[1]["per_type"]["dual_peak"]
    with pytest.raises(ValueError, match="group"):
        adapter()._aggregate_seed_metrics(runs)


def test_seed_aggregation_rejects_mismatched_metric_keys():
    runs = seed_results()
    del runs[1]["average"]["MAE_C"]
    with pytest.raises(ValueError, match="metric"):
        adapter()._aggregate_seed_metrics(runs)


def test_normalization_preserves_supplied_temperature_range():
    assert adapter()._normalization_values({"T_min": 13.25, "T_max": 1189.75}) == (13.25, 1189.75)


@pytest.mark.parametrize("minimum,maximum", [(1, 1), (2, 1), (float("nan"), 3), (1, float("inf"))])
def test_invalid_normalization_is_rejected(minimum, maximum):
    with pytest.raises(ValueError, match="normalization"):
        adapter()._normalization_values({"T_min": minimum, "T_max": maximum})


def test_example_forwards_matched_scientific_settings_without_defaults():
    config = example_config()
    before = repr(config)
    kwargs = adapter()._example_kwargs(config, "dual_peak", 1234, None)
    assert kwargs == {
        "phantom_type": "dual_peak",
        "seed": 1234,
        "temperature_regime": None,
        "path_set_name": "matched_path_set",
        "coarse_grid": 7,
        "fine_grid": 24,
        "noise_std_s": 0.000073,
        "alpha": 0.0023,
    }
    assert repr(config) == before


def test_example_uses_fullrange_regime_from_matched_config():
    config = example_config()
    config["temperature_regimes"] = ["room_static", "high_running"]
    kwargs = adapter()._example_kwargs(config, "single_peak", 20260607, "room_static")
    assert kwargs["temperature_regime"] == "room_static"


@pytest.mark.parametrize("regime", ["unknown", "", "legacy_high"])
def test_example_rejects_regime_not_supported_by_selected_config(regime):
    with pytest.raises(ValueError, match="regime"):
        adapter()._example_kwargs(example_config(), "single_peak", 1, regime)


def test_fullrange_config_cannot_silently_generate_legacy_high_example():
    config = example_config()
    config["temperature_regimes"] = ["room_static"]
    with pytest.raises(ValueError, match="regime"):
        adapter()._example_kwargs(config, "single_peak", 1, None)


def test_example_rejects_unknown_family():
    with pytest.raises(ValueError, match="family"):
        adapter()._example_kwargs(example_config(), "other", 1, None)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_example_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        adapter()._example_kwargs(example_config(), "single_peak", seed, None)


def test_missing_scientific_setting_is_not_replaced_with_a_constant():
    config = example_config()
    del config["alpha"]
    with pytest.raises(ValueError, match="alpha"):
        adapter()._example_kwargs(config, "single_peak", 1, None)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_evaluation_rejects_invalid_batch_size_before_loading_vendor(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        adapter().evaluate_factory("missing-package", "unused-output", batch_size=batch_size)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_evaluation_rejects_invalid_limit_before_loading_vendor(limit):
    with pytest.raises(ValueError, match="limit_per_family"):
        adapter().evaluate_factory("missing-package", "unused-output", limit_per_family=limit)


def test_package_path_rejects_traversal_without_needing_assets():
    root = ADAPTER_PATH.parents[1]
    with pytest.raises(ValueError, match="package"):
        adapter()._package_path(root, "../outside.json")


def test_package_path_keeps_repo_relative_asset_path():
    root = ADAPTER_PATH.parents[1]
    relative = "assets/NewExperiment/factory_validation/models/seed42/stage2_best.pth"
    assert adapter()._package_path(root, relative) == (root / relative).resolve()


@pytest.mark.parametrize("directory", ["assets", "vendor", "repro", "tests"])
def test_output_cannot_target_package_inputs(directory):
    root = ADAPTER_PATH.parents[1]
    with pytest.raises(ValueError, match="output"):
        adapter()._output_path(root, root / directory / "results")


def test_output_path_is_confined_to_requested_directory():
    root = ADAPTER_PATH.parents[1]
    output = root / "outputs" / "factory"
    assert adapter()._output_path(root, output) == output.resolve() / "factory_summary.json"


@pytest.mark.parametrize("relative", [".", "outputs", "scratch/run", "configs/run", "outputs_other/run"])
def test_output_must_be_strictly_beneath_package_outputs(tmp_path, relative):
    root = tmp_path / "release"
    with pytest.raises(ValueError, match="outputs"):
        adapter()._output_path(root, root / relative)


def test_output_rejects_outside_package(tmp_path):
    with pytest.raises(ValueError, match="outputs"):
        adapter()._output_path(tmp_path / "release", tmp_path / "outside")


def reject_loading(monkeypatch, module):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "torch", "Output preflight must precede PyTorch loading"
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("Output preflight must precede vendor/asset loading")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(module, "_vendor_modules", forbidden)
    monkeypatch.setattr(module, "_read_json", forbidden)


@pytest.mark.parametrize("relative", [".", "outputs", "configs/run", "../outside"])
def test_invalid_output_is_rejected_before_loading(tmp_path, monkeypatch, relative):
    module = adapter()
    root = tmp_path / "release"
    reject_loading(monkeypatch, module)
    with pytest.raises(ValueError, match="outputs"):
        module.evaluate_factory(root, root / relative)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_existing_summary_is_rejected_before_loading(tmp_path, monkeypatch, kind):
    module = adapter()
    root = tmp_path / "release"
    output = root / "outputs" / "existing"
    output.mkdir(parents=True)
    target = output / "factory_summary.json"
    if kind == "file":
        target.write_text("existing result", encoding="utf-8")
    else:
        target.mkdir()
    reject_loading(monkeypatch, module)
    with pytest.raises(FileExistsError):
        module.evaluate_factory(root, output)
    if kind == "file":
        assert target.read_text(encoding="utf-8") == "existing result"


def create_symlink(link, target, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"OS does not permit test symlinks: {error}")


@pytest.mark.parametrize("dangling", [False, True])
def test_existing_summary_symlink_is_rejected_even_if_dangling(tmp_path, dangling):
    root = tmp_path / "release"
    output = root / "outputs" / "linked"
    output.mkdir(parents=True)
    destination = output / "other.json"
    if not dangling:
        destination.write_text("untouched", encoding="utf-8")
    target = output / "factory_summary.json"
    create_symlink(target, destination)
    with pytest.raises(FileExistsError):
        adapter()._output_path(root, output)
    assert target.is_symlink()
    if not dangling:
        assert destination.read_text(encoding="utf-8") == "untouched"


@pytest.mark.parametrize("link_outputs_root", [False, True])
def test_output_directory_symlink_cannot_escape_outputs(tmp_path, link_outputs_root):
    root = tmp_path / "release"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    if link_outputs_root:
        link = root / "outputs"
        create_symlink(link, outside, directory=True)
        output = link / "run"
    else:
        outputs = root / "outputs"
        outputs.mkdir()
        output = outputs / "linked"
        create_symlink(output, outside, directory=True)
    with pytest.raises(ValueError, match="outputs"):
        adapter()._output_path(root, output)
    assert list(outside.iterdir()) == []


def stub_inference(monkeypatch, module):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda value: value))
    monkeypatch.setattr(module, "_DATASETS", ())
    monkeypatch.setattr(module, "_vendor_modules", lambda root: nullcontext())


def test_summary_is_written_exclusively_and_round_trips(tmp_path, monkeypatch):
    module = adapter()
    stub_inference(monkeypatch, module)
    root = tmp_path / "release"
    output = root / "outputs" / "new"
    target = output / "factory_summary.json"
    original_open = Path.open
    modes = []

    def tracked_open(path, mode="r", *args, **kwargs):
        if path == target:
            modes.append(mode)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    result = module.evaluate_factory(root, output)
    assert modes == ["x"]
    assert json.loads(target.read_text(encoding="utf-8")) == result
    assert list(output.iterdir()) == [target]


def test_exclusive_write_preserves_file_created_after_preflight(tmp_path, monkeypatch):
    module = adapter()
    stub_inference(monkeypatch, module)
    root = tmp_path / "release"
    output = root / "outputs" / "raced"
    target = output / "factory_summary.json"
    original_open = Path.open

    def competing_open(path, mode="r", *args, **kwargs):
        if path == target and mode in ("w", "x"):
            with original_open(path, "x", encoding="utf-8") as stream:
                stream.write("concurrent result")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", competing_open)
    with pytest.raises(FileExistsError):
        module.evaluate_factory(root, output)
    assert target.read_text(encoding="utf-8") == "concurrent result"


def test_factory_documentation_is_portable_and_describes_saved_layout():
    document = (ADAPTER_PATH.parents[1] / "FACTORY.md").read_text(encoding="utf-8")
    assert re.search(r"[A-Za-z]:[/\\]", document) is None
    assert "python -B -c" in document
    assert "python -B -m pytest" in document
    assert "teacher coordinates" not in document
    assert "retains saved factory synthetic" in document
    assert "not a new layout" in document
    assert "fixture-free" not in document
