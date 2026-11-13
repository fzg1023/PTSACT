import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.mark.parametrize("label,size,method", [("tikhonov12", 12, "tikhonov"),
                                               ("tikhonov20", 20, "tikhonov"),
                                               ("rbf16", 16, "rbf")])
@pytest.mark.parametrize("family", ["single_peak", "dual_peak", "triple_peak"])
def test_alternative_stage1_solver_matches_supplied_fields(label, size, method, family):
    from repro.common import asset
    from repro.physics import reconstruct_tof
    base = "NewExperiment/revision_experiments/"
    with np.load(asset(base + "r2_02_tv_baseline/runs/independent_test_20261030_v1/" + family + ".npz"), allow_pickle=False) as data:
        tof = data["tof"][0]
    with np.load(asset(base + "r2_04_independent_ablation/runs/independent_ablation_20261101_v1/stage1_inputs/" + label + ".npz"), allow_pickle=False) as archive:
        expected = archive[family][0]
    actual = reconstruct_tof(tof, size, method)
    np.testing.assert_allclose(actual, expected, atol=0.01, rtol=0)


def test_main_prediction_needs_no_target():
    from repro.common import asset, load_json
    from repro.core import load_refiner, predict_field
    initial = np.full((1, 32, 32), 1000.0, dtype=np.float32)
    norm = load_json(ROOT / "configs/main.json")["normalization"]
    model = load_refiner(asset("NewExperiment/models_tikhonov_16x16/seed42.pth"))
    actual = predict_field(model, initial, norm)
    assert actual.shape == initial.shape and np.isfinite(actual).all()
