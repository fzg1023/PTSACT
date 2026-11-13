import numpy as np

from .common import FAMILIES, asset, load_json, register_vendor_paths, write_json
from .physics import projection, reconstruct_tof, equivalent_temperature, equivalent_tof

TV = "NewExperiment/revision_experiments/r2_02_tv_baseline/"


def generate_independent(root, out, count_per_family=3):
    """Recreate the fixed test seed/noise stream, writing only into a new output."""
    if not 1 <= count_per_family <= 400:
        raise ValueError("Count must be between 1 and 400")
    register_vendor_paths(root)
    import phantom_generators
    from ray_tracer import temperature_to_slowness
    plan = load_json(asset(TV + "runs/independent_test_20261030_v1/manifest.json", root))["generation"]
    noise = np.random.default_rng(plan["noise_seed"]).normal(0.0, plan["noise_std_s"], size=(3, 400, 120))
    matrix, _, positions, pairs = projection(32)
    lengths = np.asarray(matrix.sum(axis=1)).ravel()
    checks = {}
    for family_index, family in enumerate(FAMILIES):
        fields, tofs, initial = [], [], []
        function = getattr(phantom_generators, family + "_isotropic")
        for index, seed in enumerate(plan["phantom_seeds"][family][:count_per_family]):
            truth = function(32, 32, seed=seed).reshape(32, 32)
            tof = np.maximum(matrix.dot(temperature_to_slowness(truth.ravel())) + noise[family_index, index], 1e-12)
            fields.append(truth.astype(np.float32))
            tofs.append(tof)
            initial.append(reconstruct_tof(tof))
        generated = dict(T_true=np.asarray(fields), tof=np.asarray(tofs), T_stage1=np.asarray(initial))
        np.savez_compressed(out / (family + ".npz"), **generated)
        with np.load(asset(TV + "runs/independent_test_20261030_v1/" + family + ".npz", root), allow_pickle=False) as saved:
            checks[family] = {key + "_max_abs_difference": float(np.max(np.abs(
                generated[key].astype(np.float64) - saved[key][:count_per_family].astype(np.float64))))
                for key in generated}
        path_t = equivalent_temperature(generated["tof"], lengths)
        checks[family]["tof_temperature_roundtrip_max_abs_s"] = float(np.max(np.abs(equivalent_tof(path_t, lengths) - generated["tof"])))
        print("[generate {}] {} fields; differences {}".format(family, count_per_family, checks[family]), flush=True)
    np.savez_compressed(out / "geometry.npz", positions=positions, pairs_0based=pairs,
                        A_fine=matrix.toarray(), A_coarse=projection(16)[0].toarray())
    report = dict(scope="same frozen independent-test generator/noise stream; existing assets untouched",
                  count_per_family=count_per_family, checks=checks)
    write_json(out / "generation_check.json", report)
    return report


def verify_normalization(root, out):
    norm = load_json(asset("NewExperiment/data_tikhonov_16x16/normalization.json", root))
    rows = {}
    for split in ("train", "val"):
        values = []
        for family in FAMILIES:
            with np.load(asset("NewExperiment/data_tikhonov_16x16/" + family + "/" + split + ".npz", root), allow_pickle=False) as data:
                truth = data["T_true"]
                values.append(dict(family=family, count=len(truth), minimum=float(truth.min()), maximum=float(truth.max())))
        rows[split] = dict(count=sum(v["count"] for v in values), minimum=min(v["minimum"] for v in values),
                           maximum=max(v["maximum"] for v in values), families=values)
    result = dict(stored_constants=norm, data=rows,
                  training_only_exactly_matches_stored=rows["train"]["minimum"] == norm["T_min"] and rows["train"]["maximum"] == norm["T_max"],
                  validation_does_not_expand_limits=rows["val"]["minimum"] >= norm["T_min"] and rows["val"]["maximum"] <= norm["T_max"],
                  test_truth_opened_for_this_check=False,
                  note="Historical preparation scanned train and validation; recomputing from train alone is numerically identical here.")
    write_json(out / "normalization_check.json", result)
    if not result["training_only_exactly_matches_stored"]:
        raise ValueError("Stored normalization differs from training-only extrema")
    return result
