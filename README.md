# Acoustic-CT reconstruction: data and evaluation

Materials accompanying manuscript MEAS-D-26-12759. This package contains the
synthetic data generator, Siddon projection and reconstruction code, trained
models, normalization settings, and evaluation commands. Materials are provided
by arrangement with the corresponding author. Access and redistribution should
be agreed with the authors; the package does not carry an open-source license.

For Chinese instructions, see `README_ZH.md`. Factory-layout synthetic data and
their model settings are described in `FACTORY.md`.

## Contents

| Material | Location |
|---|---|
| Command-line interface | `run.py` |
| Generation, reconstruction, inference, and metrics | `repro/` |
| Physics and network definitions | `vendor/` |
| Model and evaluation settings | `configs/` |
| Synthetic data, weights, and reference solutions | `assets/` |
| Tests | `tests/` |
| Numerical validation record | `VALIDATION.json` |
| File origins and checksums | `SOURCE_MAP.json`, `MANIFEST_SHA256.csv` |

The main benchmark contains 4,800 training fields, 1,200 validation fields, and
1,200 independently generated test fields, with weights for three training seeds.
Comparison materials include five direct networks, a dual-input reference,
17 ablation configurations, three RBF-based models, and TV reference solutions.
Separate high-temperature and full-range factory-layout synthetic datasets and
models are also included.

Industrial measurements, measured-path derivative tables, interface images,
manuscript files, and reviewer correspondence are excluded. This release supports
synthetic-data generation and evaluation with trained models. It does not include
a complete workflow for retraining every configuration or reproducing all
manuscript figures, real-record displays, timing tests, and spectral analyses.

## Environment

The numerical checks used Python 3.9.2 and the versions in `requirements.txt`.
Install them in a compatible Python environment:

```text
python -m pip install -r requirements.txt
```

The default device is CPU. GPU evaluation requires a compatible PyTorch/CUDA
installation and the option `--device cuda`.

## Quick start

Open a terminal in this package directory and run:

```text
python run.py verify
python run.py normalization --output outputs/normalization_check
python run.py generate --count-per-family 3 --output outputs/generator_check
python run.py evaluate --suite main --output outputs/main_test
```

Each output path must be a new subdirectory of `outputs/`. Existing directories
are not overwritten; choose a different name when repeating a command.

The full main evaluation uses 1,200 test fields and all three model seeds. The
reference mean normalized RMSE is 1.9153% for Stage 1 and 0.5497% for the two-stage
method. The population standard deviation across the three two-stage seed means
is approximately 0.0007 percentage points. Families receive equal weight; these
statistics are not computed from an ensemble of averaged temperature fields.

To regenerate all test fields, including the fixed noise stream:

```text
python run.py generate --count-per-family 400 --output outputs/generated_test
```

The command also exports coarse/fine projection matrices, sensor coordinates,
and ordered ray pairs in `geometry.npz`. Packaged data and weights remain unchanged.

## Prediction with new inputs

Supply an NPZ file containing either `T_stage1` of shape `(N,32,32)` or `tof` of
shape `(N,120)`, in seconds. If both arrays are present, `T_stage1` is used.
The TOF input must correspond to the supplied main-benchmark sensor layout and
path order. `T_true` is not required or read during prediction.

```text
python run.py predict --input my_measurements.npz --seed 42 --output outputs/prediction
```

Input normalization and residual rescaling use fixed, saved constants. Those
constants equal the global extrema of the training data. The historical data
preparation also scanned the validation fields; excluding them gives the same
values. The `normalization` command checks this provenance without using test
truth. Evaluation reads ground truth separately to calculate errors.

## Other evaluations

```text
python run.py evaluate --suite baselines --output outputs/baselines
python run.py evaluate --suite ablations --output outputs/ablations
python run.py evaluate --suite factory --output outputs/factory
python run.py evaluate --suite all --device cuda --output outputs/all_evaluations
```

`--limit-per-family 3` selects a small execution check. Omit this option for the
full reported evaluations. Direct networks, the dual-input reference, and Stage 2
ablation models are inferred from their weights. Alternative-grid and RBF Stage 1
fields are supplied as deterministic reference inputs; their reconstruction
functions are in `repro/physics.py`.

The baseline suite calculates TV metrics from the 1,200 supplied converged
solutions. To run the TV optimizer for an individual test field:

```text
python run.py tv-solve --family single_peak --index 0 --output outputs/tv_example
```

The optimizer uses the supplied regularization and stopping settings. A full
optimization of all 1,200 TV cases is separate from the default evaluation.
The dual-input comparator is the study-specific reference described in the
manuscript. Its tensor-only weights reproduce the original saved model states;
model settings are stored separately in JSON.

## Metrics and numerical precision

- Main normalized RMSE divides pooled pixelwise RMSE by each family's mean true
  temperature in degrees Celsius; the reported mean gives each family equal weight.
- MAE and MaxAE are in degrees Celsius; ARE and L2-RER are percentages.
- Main-benchmark and controlled-comparison PSNR uses the fixed reference range,
  approximately 1251.371 degrees Celsius. Historical reference JSON may contain
  older PSNR values; the evaluation commands calculate the revised metric.
- Factory-layout synthetic evaluation retains its original metric definitions,
  including maximum-true-temperature PSNR. Its principal comparison is RMSE in
  degrees Celsius. See `FACTORY.md` for aggregation details.
- Synthetic full-field results use Stage 1 and raw residual Stage 2, without DC
  or display priors. Factory synthetic validation is distinct from real-record
  replay and is not an independent test set.

Small differences can arise across devices and numerical libraries. Under the
original TF32 settings, changing GPU inference batches from 64 fields to three
fields produced a largest observed pixelwise difference of about 0.056 degrees C.
The prediction check with and without ground truth used identical inputs and
batch shapes. Full numerical check results and their scope are in `VALIDATION.json`.

## File integrity and access

`python run.py verify` checks the distributed files against
`MANIFEST_SHA256.csv`. The ZIP is accompanied by a separate SHA-256 checksum.
`SOURCE_MAP.json` records relative file origins and any packaging adaptations;
data and model tensors retain their original numerical values.

Requests for these materials or for separately authorized industrial data should
be directed to the corresponding author identified in the manuscript.
