"""Paired stratified bootstrap; model seeds remain a separate averaging axis."""

import numpy as np


def paired_bootstrap(cases, replicates=2000, seed=20261030):
    if not cases or replicates < 1:
        raise ValueError('Empty comparison or invalid replicate count')
    checked = {}
    seed_count = None
    for family, values in cases.items():
        b, m, t = [np.asarray(values[k], dtype=float) for k in ('baseline_mse', 'model_mse', 'truth_means')]
        if b.ndim != 1 or m.ndim != 2 or t.shape != b.shape or m.shape[1] != len(b) or len(b) == 0:
            raise ValueError('Invalid paired sample shapes')
        if not all(np.isfinite(x).all() for x in (b, m, t)) or np.any(b < 0) or np.any(m < 0) or np.any(t <= 0):
            raise ValueError('Invalid paired values')
        if seed_count is not None and m.shape[0] != seed_count:
            raise ValueError('Model seed count differs across families')
        seed_count = m.shape[0]
        if seed_count == 0:
            raise ValueError('Empty model seeds')
        checked[family] = b, m, t

    def scores(selections):
        baseline, model = [], []
        for family, (b, m, t) in checked.items():
            ids = selections[family]
            denominator = t[ids].mean()
            baseline.append(np.sqrt(b[ids].mean()) / denominator * 100.)
            model.append(np.sqrt(m[:, ids].mean(axis=1)) / denominator * 100.)
        return float(np.mean(baseline)), np.mean(model, axis=0)

    base, model_seeds = scores({f: np.arange(len(v[0])) for f, v in checked.items()})
    if base <= 0:
        raise ValueError('Relative improvement undefined for zero baseline')
    rng = np.random.default_rng(seed)
    differences, reductions = [], []
    for _ in range(replicates):
        selections = {f: rng.integers(0, len(v[0]), len(v[0])) for f, v in checked.items()}
        b, m = scores(selections)
        if b <= 0:
            raise ValueError('Zero bootstrap baseline; relative interval undefined')
        differences.append(b - float(m.mean()))
        reductions.append((b - float(m.mean())) / b * 100.)
    return dict(baseline_score=base, model_mean_score=float(model_seeds.mean()),
                model_seed_scores=model_seeds.tolist(), model_seed_std_ddof0=float(model_seeds.std()),
                difference_percentage_points=base - float(model_seeds.mean()),
                relative_reduction_percent=(base - float(model_seeds.mean())) / base * 100.,
                difference_ci95_percentage_points=np.percentile(differences, [2.5, 97.5]).tolist(),
                relative_reduction_ci95_percent=np.percentile(reductions, [2.5, 97.5]).tolist(),
                sample_count=sum(len(v[0]) for v in checked.values()), replicates=replicates, seed=seed,
                interval_scope='paired within-family sample resampling, not training uncertainty')
