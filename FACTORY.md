# Factory-layout synthetic validation

## Data and models

This evaluation uses synthetic temperature fields with known two-dimensional
truth and the factory sensor layout. It does not read industrial CSV records.
The supplied validation splits were also used for checkpoint selection; they
are not independent test sets.

| Dataset | Validation fields | Models evaluated |
|---|---:|---|
| High-temperature synthetic | 180 | Residual Stage 2 and direct-field ablation |
| Full-range synthetic | 3,000 | Residual Stage 2 |

Each model role has checkpoints for seeds 42, 123, and 456, giving nine
checkpoints in total. Models are evaluated separately, without ensembling.
Their own configurations and normalization constants are used.

- Stage 1 uses the saved `T_stage1` fields. Inversion is not repeated during
  this evaluation.
- Residual Stage 2 predicts a normalized correction that is rescaled and added
  to Stage 1. Evaluation uses the raw prediction, without DC or display processing.
- The high-temperature direct-field ablation predicts the full normalized field
  from Stage 1. It is not a direct TOF-to-field network.
- Full-range models are separate from the high-temperature models.

Packaged inputs retain these relative paths:

```text
assets/NewExperiment/factory_validation/
  data_factory_geometry_tikhonov_12x12/
  data_factory_geometry_tikhonov_12x12_fullrange_v1/
  models/factory_stage2_multiseed/seed{42,123,456}/
  models/factory_stage2_direct_ablation/seed{42,123,456}/
  models/factory_stage2_tikhonov_12x12_fullrange_v1_multiseed/seed{42,123,456}/
```

The datasets contain configuration and normalization JSON files and one
`val.npz` file for each phantom family. Training arrays are not opened by the
evaluation command. Model weights are loaded from the package with strict
state matching and `weights_only=True`.

## Run the evaluation

From the package directory:

```text
python run.py evaluate --suite factory --output outputs/factory
```

For GPU inference, add `--device cuda`. For a short execution check, add
`--limit-per-family 2`. This limit selects the first two saved fields of each
phantom family, not two per temperature regime. Omit it for the full evaluation.
Every output directory must be a new subdirectory of `outputs/`.

The result is written to `factory_summary.json`. It contains `datasets.high_data`
and `datasets.fullrange`, each with Stage 1 metrics and the corresponding model
results. Individual seed results appear under `runs`; their mean and population
standard deviation appear under `aggregate`.

The Python interface is:

```python
from pathlib import Path
from repro.factory import evaluate_factory

root = Path.cwd()
summary = evaluate_factory(root, root / "outputs" / "factory_api", device="cpu")
```

The function accepts `device`, `batch_size` (default 64), and
`limit_per_family` (default `None`). Call this interface serially because its
compatibility imports use process-global Python module state. The command-line
interface handles package imports and output-directory creation.

A short check of the Python interface can also be run directly:

```text
python -B -c "from pathlib import Path; from repro.factory import evaluate_factory; r=Path.cwd(); evaluate_factory(r, r/'outputs'/'factory_api_check', limit_per_family=2)"
```

## Metrics

The factory evaluator retains the original factory metric aggregation. Overall
RMSE is computed after pooling all validation pixels, rather than averaging
sample RMSEs or phantom-family RMSEs. Per-family and per-regime values use the
same calculation within the selected group; the direct-field evaluator does
not report per-regime metrics.

`RMSE_percent` and `ARE_percent` use the group's mean true temperature as the
denominator. `L2_RER_percent`, `MaxAE_C`, and `PSNR_dB` retain the factory
definitions. Factory PSNR uses maximum true temperature, whereas the revised
main-benchmark PSNR uses a fixed reference range. Use RMSE in degrees Celsius
for the principal factory synthetic comparison.

Seed aggregates are arithmetic means of the separately evaluated metrics.
Standard deviations use `numpy.std` with `ddof=0`. Predictions are not pooled
across seeds.

Reference values for the complete saved validation splits are:

| Dataset | Stage 1 RMSE (degrees C) | Residual three-seed mean | Direct-field three-seed mean |
|---|---:|---:|---:|
| High-temperature synthetic | 21.8086836 | 9.9548932 | 18.2722 |
| Full-range synthetic | 8.20316 | 4.19347 | Not evaluated |

These values describe synthetic validation, not measured full-field accuracy
or independent-test generalization. Device-dependent numerical differences are
documented in `VALIDATION.json`.

## Generate an example

```python
from repro.factory import generate_factory_example

sample = generate_factory_example(root, family="single_peak", seed=20260607)
room_sample = generate_factory_example(
    root, family="dual_peak", seed=20260607, temperature_regime="room_static"
)
```

Without `temperature_regime`, the high-temperature configuration is used.
Example generation retains saved factory synthetic geometry; it is not a new layout.
A named regime selects the full-range configuration. Grid sizes, path set,
noise level, and Tikhonov regularization are taken from the matching saved
configuration and passed to the supplied generator.

The returned sample contains `tof`, `T_stage1`, `T_true`, `path_temperatures`,
`phantom_type`, `temperature_regime`, and `seed`. This function returns an
in-memory example; it does not replace the saved validation data or write files.

## Tests and recorded checks

```text
python -B -m pytest -q -p no:cacheprovider tests/test_factory.py
```

The tests cover metric aggregation, normalization matching, generator arguments,
input limits, and output handling. Real symlink checks may be skipped when the
operating system does not permit their creation.

The recorded integration check evaluated all nine checkpoints on CPU and CUDA
using both complete validation splits. It also checked example generation and
execution from an independent directory without access to the original project.
The numerical validation record is in `VALIDATION.json`.
