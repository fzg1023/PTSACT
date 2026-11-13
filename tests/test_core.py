import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_public_entrypoint_exists():
    assert (ROOT / "repro" / "core.py").is_file(), "Portable core implementation is not available yet"


def test_fixed_normalization_and_inverse():
    from repro.core import normalize_initial, restore_prediction
    norm = {"T_min": 10.0, "T_max": 30.0}
    field = np.full((2, 32, 32), 20.0, dtype=np.float32)
    x = normalize_initial(field, norm)
    assert x.shape == (2, 1, 32, 32)
    np.testing.assert_allclose(x, 0.5)
    np.testing.assert_allclose(restore_prediction(np.full_like(field, 0.1), field, norm), 22.0)
    np.testing.assert_allclose(restore_prediction(np.full_like(field, 0.1), field, norm, "direct"), 12.0)


def test_truth_not_in_prediction_interface():
    from repro.core import predict_field
    assert "truth" not in inspect.signature(predict_field).parameters
    assert "T_true" not in inspect.signature(predict_field).parameters


def test_invalid_scale_and_shape_are_rejected():
    from repro.core import normalize_initial
    with pytest.raises(ValueError):
        normalize_initial(np.zeros((1, 32, 32)), {"T_min": 5.0, "T_max": 5.0})
    with pytest.raises(ValueError):
        normalize_initial(np.zeros((3, 3)), {"T_min": 0.0, "T_max": 10.0})


def test_metrics_use_fixed_reference_and_family_mean():
    from repro.metrics import field_metrics
    truth = np.full((2, 32, 32), 100.0)
    prediction = truth + 2.0
    result = field_metrics(prediction, truth, 10.0)
    assert result["RMSE_C"] == 2.0
    assert result["RMSE_percent"] == 2.0
    assert result["MAE_C"] == 2.0
    assert result["MaxAE_C"] == 2.0
    assert result["PSNR_dB"] == pytest.approx(20 * np.log10(5.0))


def test_output_must_be_new_and_inside_package(tmp_path):
    from repro.common import create_output
    package = tmp_path / "package"
    package.mkdir()
    out = create_output(package, "outputs/first")
    assert out.is_dir()
    with pytest.raises(FileExistsError):
        create_output(package, "outputs/first")
    with pytest.raises(ValueError):
        create_output(package, "../escape")
    with pytest.raises(ValueError):
        create_output(package, "assets/NewExperiment")


def test_roundtrip_tof_and_temperature():
    from repro.physics import equivalent_temperature, equivalent_tof
    lengths = np.array([5.0, 12.0, 20.0])
    times = np.array([0.01, 0.02, 0.03])
    np.testing.assert_allclose(equivalent_tof(equivalent_temperature(times, lengths), lengths), times, atol=1e-15)


def test_dual_input_is_distributed_as_weights_only():
    import torch
    for seed in (42, 123, 456):
        path = ROOT / "assets/NewExperiment/revision_experiments/r3_01_hybrid_baseline/runs" / (
            "dual_seed{}_v1/model_state.pth".format(seed))
        assert path.is_file(), "Portable tensor-only checkpoint has not been exported"
        state = torch.load(path, map_location="cpu", weights_only=True)
        assert isinstance(state, dict) and state
        assert all(isinstance(value, torch.Tensor) for value in state.values())


@pytest.mark.parametrize("times", [np.zeros(120), np.full(120, -0.01), np.r_[0.0, np.ones(119)]])
def test_nonpositive_tof_is_rejected(times):
    from repro.physics import reconstruct_tof
    with pytest.raises(ValueError, match="positive"):
        reconstruct_tof(times)


@pytest.mark.parametrize("suite,limit,valid", [("factory", 500, True), ("factory", 1000, True),
                                               ("main", 500, False), ("all", 500, False),
                                               ("baselines", 0, False), ("ablations", 401, False)])
def test_subset_limit_matches_the_selected_dataset(suite, limit, valid):
    import run
    args = run.parser().parse_args(["evaluate", "--suite", suite, "--limit-per-family", str(limit)])
    if valid:
        run.validate_options(args)
    else:
        with pytest.raises(ValueError):
            run.validate_options(args)
