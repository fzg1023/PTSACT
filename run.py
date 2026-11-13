"""Command-line entry point; all outputs stay in a new outputs/ directory."""

import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def parser():
    cli = argparse.ArgumentParser(description="Synthetic acoustic-CT reproducibility package")
    commands = cli.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--suite", choices=("main", "baselines", "ablations", "factory", "all"), default="main")
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--limit-per-family", type=int, default=None, help="Smoke subset only; omit for paper scores")
    evaluate.add_argument("--output", default="outputs/evaluation")
    generate = commands.add_parser("generate")
    generate.add_argument("--count-per-family", type=int, default=3)
    generate.add_argument("--output", default="outputs/generated")
    norm = commands.add_parser("normalization")
    norm.add_argument("--output", default="outputs/normalization")
    predict = commands.add_parser("predict")
    predict.add_argument("--input", required=True, help="NPZ with tof and/or T_stage1; T_true not required or read")
    predict.add_argument("--seed", type=int, choices=(42, 123, 456), default=42)
    predict.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    predict.add_argument("--output", default="outputs/prediction")
    tv = commands.add_parser("tv-solve")
    tv.add_argument("--family", choices=("single_peak", "dual_peak", "triple_peak"), default="single_peak")
    tv.add_argument("--index", type=int, default=0)
    tv.add_argument("--output", default="outputs/tv_solve")
    commands.add_parser("verify")
    return cli


def validate_options(args):
    limit = getattr(args, "limit_per_family", None)
    upper = 1000 if getattr(args, "suite", None) == "factory" else 400
    if limit is not None and not 1 <= limit <= upper:
        raise ValueError("Selected-suite subset must contain 1 to {} fields per family".format(upper))
    if getattr(args, "batch_size", 64) < 1:
        raise ValueError("Batch size must be positive")


def main():
    args = parser().parse_args()
    from repro.common import asset, create_output, load_json, write_json
    if args.command == "verify":
        from repro.integrity import verify_manifest
        return verify_manifest(ROOT)
    validate_options(args)
    output = create_output(ROOT, args.output)
    if args.command == "generate":
        from repro.generation import generate_independent
        generate_independent(ROOT, output, args.count_per_family)
    elif args.command == "normalization":
        from repro.generation import verify_normalization
        result = verify_normalization(ROOT, output)
        print("Training-only constants exactly match:", result["training_only_exactly_matches_stored"], flush=True)
    elif args.command == "tv-solve":
        import numpy as np
        from repro.physics import solve_tv
        directory = "NewExperiment/revision_experiments/r2_02_tv_baseline/"
        settings = load_json(asset(directory + "runs/tuning_confirm_20261030_v1/frozen_config.json"))
        with np.load(asset(directory + "runs/independent_test_20261030_v1/" + args.family + ".npz"), allow_pickle=False) as data:
            times = data["tof"][args.index]
        field, state = solve_tv(times, settings)
        np.savez_compressed(output / "tv_prediction.npz", prediction=field)
        write_json(output / "solver_status.json", state)
        print("TV converged:", state["converged"], "iterations:", state["iterations"], flush=True)
        if not state["converged"]:
            raise RuntimeError("TV solution did not meet the frozen stopping tolerances")
    else:
        import torch
        torch.set_num_threads(2)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Match the original synthetic evaluators' convolution-precision default.
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        import platform
        import numpy as np
        import scipy
        write_json(output / "run_environment.json", dict(python=platform.python_version(),
                   numpy=np.__version__, scipy=scipy.__version__, torch=str(torch.__version__),
                   device=args.device, cudnn_deterministic=True, cudnn_benchmark=False,
                   cudnn_allow_tf32=True, cuda_matmul_allow_tf32=False))
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; choose --device cpu")
        if args.command == "predict":
            import numpy as np
            from repro.core import load_refiner, predict_field
            from repro.physics import reconstruct_tof
            with np.load(args.input, allow_pickle=False) as data:
                if "T_stage1" in data:
                    initial = data["T_stage1"].copy()
                elif "tof" in data:
                    times = np.atleast_2d(data["tof"])
                    initial = np.stack([reconstruct_tof(tof) for tof in times])
                else:
                    raise ValueError("Input requires tof or T_stage1")
            norm = load_json(ROOT / "configs/main.json")["normalization"]
            model = load_refiner(asset("NewExperiment/models_tikhonov_16x16/seed{}.pth".format(args.seed)), args.device)
            predicted = predict_field(model, initial, norm, args.device)
            np.savez_compressed(output / "prediction.npz", T_stage1=initial, T_stage2=predicted)
            write_json(output / "prediction_info.json", dict(seed=args.seed, input_truth_read=False,
                       input_mode="Stage1 if supplied, otherwise reconstructed from TOF", samples=len(predicted)))
        else:
            from repro.evaluation import evaluate_main, evaluate_baselines, evaluate_ablations
            suites = ("main", "baselines", "ablations", "factory") if args.suite == "all" else (args.suite,)
            for suite in suites:
                print("[suite]", suite, flush=True)
                if suite == "factory":
                    from repro.factory import evaluate_factory
                    result = evaluate_factory(ROOT, output, device=args.device, batch_size=args.batch_size,
                                              limit_per_family=args.limit_per_family)
                else:
                    function = dict(main=evaluate_main, baselines=evaluate_baselines, ablations=evaluate_ablations)[suite]
                    result = function(ROOT, output, args.device, args.batch_size, args.limit_per_family)
                print("[suite complete]", suite, flush=True)
    print("Output:", output.relative_to(ROOT).as_posix(), flush=True)


if __name__ == "__main__":
    main()
