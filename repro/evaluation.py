from pathlib import Path

import numpy as np

from .common import FAMILIES, SEEDS, asset, load_json, register_vendor_paths, write_json
from .core import load_refiner, predict_field
from .metrics import field_metrics, macro_rmse, sample_metrics

TV = "NewExperiment/revision_experiments/r2_02_tv_baseline/"
AB = "NewExperiment/revision_experiments/r2_04_independent_ablation/"
HY = "NewExperiment/revision_experiments/r3_01_hybrid_baseline/"


def load_test(root, limit=None):
    result = {}
    for family in FAMILIES:
        with np.load(asset(TV + "runs/independent_test_20261030_v1/" + family + ".npz", root), allow_pickle=False) as data:
            result[family] = {key: data[key][:limit].copy() for key in ("T_stage1", "tof", "T_true")}
    return result


def evaluate_main(root, out, device="cpu", batch_size=64, limit=None):
    register_vendor_paths(root)
    from comparison_statistics import paired_bootstrap
    data = load_test(root, limit)
    norm = load_json(Path(root) / "configs/main.json")["normalization"]
    span = norm["T_max"] - norm["T_min"]
    initial = {f: field_metrics(data[f]["T_stage1"], data[f]["T_true"], span) for f in FAMILIES}
    per_seed, errors = {}, {}
    for number, seed in enumerate(SEEDS, 1):
        model = load_refiner(asset("NewExperiment/models_tikhonov_16x16/seed{}.pth".format(seed), root), device)
        prediction = {f: predict_field(model, data[f]["T_stage1"], norm, device, batch_size) for f in FAMILIES}
        per_seed[str(seed)] = {f: field_metrics(prediction[f], data[f]["T_true"], span) for f in FAMILIES}
        errors[seed] = {f: sample_metrics(prediction[f], data[f]["T_true"])["mse_C2"] for f in FAMILIES}
        np.savez_compressed(out / ("main_predictions_seed{}.npz".format(seed)), **prediction)
        print("[main model {}/3] seed {} RMSE={:.7f}%".format(number, seed, macro_rmse(per_seed[str(seed)])), flush=True)
    scores = [macro_rmse(per_seed[str(s)]) for s in SEEDS]
    cases = {f: dict(baseline_mse=sample_metrics(data[f]["T_stage1"], data[f]["T_true"])["mse_C2"],
                     model_mse=np.stack([errors[s][f] for s in SEEDS]),
                     truth_means=data[f]["T_true"].mean(axis=(1, 2), dtype=np.float64)) for f in FAMILIES}
    result = dict(scope="fresh three-model inference on supplied independent synthetic test",
                  limit_per_family=limit, samples=sum(len(v["T_true"]) for v in data.values()),
                  stage1_families=initial, stage1_macro_RMSE_percent=macro_rmse(initial),
                  stage2_per_seed=per_seed, stage2_mean_RMSE_percent=float(np.mean(scores)),
                  stage2_population_std_percentage_points=float(np.std(scores)),
                  comparison=paired_bootstrap(cases, 2000, 20261030),
                  fixed_reference_range_C=span, inference_uses_test_truth=False)
    write_json(out / "main_summary.json", result)
    return result


def evaluate_baselines(root, out, device="cpu", batch_size=64, limit=None):
    import torch
    register_vendor_paths(root)
    from direct_networks import E2ENet, E2ENetV1, E2ENetV1_Multi, E2ENetV3
    from image_baseline import EndToEndUNetA
    from dual_network import DualInputResidual
    data = load_test(root, limit)
    norm = load_json(asset("NewExperiment/data_e2e_baseline/normalization.json", root))
    lo, span = norm["T_min"], norm["T_max"] - norm["T_min"] + 1e-14
    classes = dict(baseline=E2ENet, v1=E2ENetV1, v1_multi=E2ENetV1_Multi, v3=E2ENetV3, e2e_a=EndToEndUNetA)
    folders = dict(baseline="models", v1="models_capacity/v1", v1_multi="models_multiscale/v1_multi",
                   v3="models_capacity/v3", e2e_a="models_e2e_a")
    results = {}
    for version in list(classes) + ["dual"]:
        seed_rows = {}
        for seed in SEEDS:
            if version == "dual":
                folder = HY + "runs/dual_seed{}_v1/".format(seed)
                config = load_json(asset(folder + "config.json", root))
                model = DualInputResidual().to(device)
                model.load_state_dict(torch.load(asset(folder + "model_state.pth", root), map_location=device, weights_only=True))
            else:
                model = classes[version](tof_dim=120).to(device)
                model.load_state_dict(torch.load(asset("NewExperiment/e2e_baseline/" + folders[version]
                    + "/seed{}.pth".format(seed), root), map_location=device, weights_only=True))
            model.eval()
            metrics = {}
            with torch.no_grad():
                for family, d in data.items():
                    chunks = []
                    for first in range(0, len(d["tof"]), batch_size):
                        if version == "dual":
                            field = d["T_stage1"][first:first + batch_size].astype(np.float32)
                            mean, std = np.asarray(config["tof_mean"]), np.asarray(config["tof_std"])
                            tof = ((d["tof"][first:first + batch_size] - mean) / std).astype(np.float32)
                            image = ((field - config["temperature_min_C"]) / config["temperature_scale_C"])[:, None]
                            raw = model(torch.from_numpy(image).to(device), torch.from_numpy(tof).to(device)).squeeze(1).cpu().numpy()
                            chunks.append(raw * config["temperature_scale_C"] + field)
                        else:
                            x = ((d["tof"][first:first + batch_size] - norm["tof_mean"]) / norm["tof_std"]).astype(np.float32)
                            raw = model(torch.from_numpy(x).to(device)).squeeze(1).cpu().numpy()
                            chunks.append(raw * span + lo)
                    metrics[family] = field_metrics(np.concatenate(chunks), d["T_true"], span)
            seed_rows[str(seed)] = dict(families=metrics, macro_RMSE_percent=macro_rmse(metrics),
                                       parameters=sum(p.numel() for p in model.parameters()))
            print("[baseline {} seed {}] {:.7f}%".format(version, seed, macro_rmse(metrics)), flush=True)
        scores = [r["macro_RMSE_percent"] for r in seed_rows.values()]
        results[version] = dict(per_seed=seed_rows, mean_RMSE_percent=float(np.mean(scores)),
                                population_std_percentage_points=float(np.std(scores)))
    tv = {}
    for family, d in data.items():
        with np.load(asset(TV + "runs/independent_evaluation_20261030_v1/tv_" + family + ".npz", root), allow_pickle=False) as archive:
            tv[family] = field_metrics(archive["prediction"][:limit], d["T_true"], span)
    results["tv"] = dict(families=tv, macro_RMSE_percent=macro_rmse(tv),
                         evaluation_source="metrics recomputed from supplied converged TV solutions; not fresh TV solves")
    result = dict(scope="fresh direct/dual network inference; TV reference-solution metrics", results=results,
                  samples=sum(len(v["T_true"]) for v in data.values()), limit_per_family=limit)
    write_json(out / "baseline_summary.json", result)
    return result


def evaluate_ablations(root, out, device="cpu", batch_size=64, limit=None):
    data = load_test(root, limit)
    configs = load_json(Path(root) / "configs/ablations.json")
    stages = {"tikhonov16": {f: d["T_stage1"] for f, d in data.items()}}
    for label in ("tikhonov12", "tikhonov20", "rbf16"):
        with np.load(asset(AB + "runs/independent_ablation_20261101_v1/stage1_inputs/" + label + ".npz", root), allow_pickle=False) as archive:
            stages[label] = {f: archive[f][:limit].copy() for f in FAMILIES}
    results = {}
    for number, (key, config) in enumerate(configs.items(), 1):
        group, name = key.split("/")
        label = config["stage1"]
        norm_path = "NewExperiment/models/normalization.json" if label == "rbf16" else (
            "NewExperiment/data_tikhonov_{0}x{0}/normalization.json".format(label.replace("tikhonov", "")))
        norm = load_json(asset(norm_path, root))
        model = load_refiner(asset("NewExperiment/" + config["checkpoint"], root), device,
                             model_type=name if group == "attention" else "attention")
        scores = {}
        for family in FAMILIES:
            pred = predict_field(model, stages[label][family], norm, device, batch_size, config["mode"])
            scores[family] = field_metrics(pred, data[family]["T_true"], norm["T_max"] - norm["T_min"])
        results[key] = dict(families=scores, macro_RMSE_percent=macro_rmse(scores))
        print("[ablation {}/{}] {} {:.7f}%".format(number, len(configs), key, macro_rmse(scores)), flush=True)
    rbf = []
    norm = load_json(asset("NewExperiment/models/normalization.json", root))
    for seed in SEEDS:
        model = load_refiner(asset("NewExperiment/ablation/models_rbf/baseline_seed{}.pth".format(seed), root), device)
        scores = {f: field_metrics(predict_field(model, stages["rbf16"][f], norm, device, batch_size),
                                   data[f]["T_true"], norm["T_max"] - norm["T_min"]) for f in FAMILIES}
        rbf.append(macro_rmse(scores))
    result = dict(scope="fresh inference; supplied Stage 1 inputs used for alternative discretizations",
                  variants=results, rbf_seed_macro_RMSE_percent=rbf,
                  rbf_mean_RMSE_percent=float(np.mean(rbf)), rbf_std_percentage_points=float(np.std(rbf)),
                  alternative_stage1_source="saved deterministic Stage 1 fields; see generation check for solver verification")
    write_json(out / "ablation_summary.json", result)
    return result
