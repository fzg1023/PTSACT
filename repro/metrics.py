import numpy as np


def sample_metrics(prediction, truth):
    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Expected matching, finite (sample, height, width) fields")
    error = x - y
    return {"mse_C2": np.mean(error ** 2, axis=(1, 2)),
            "mae_C": np.mean(np.abs(error), axis=(1, 2)),
            "maxae_C": np.max(np.abs(error), axis=(1, 2))}


def field_metrics(prediction, truth, reference_range):
    values = sample_metrics(prediction, truth)
    truth = np.asarray(truth, dtype=np.float64)
    error = np.asarray(prediction, dtype=np.float64) - truth
    mean = float(truth.mean())
    if mean <= 0 or reference_range <= 0:
        raise ValueError("Expected positive reference mean and fixed range")
    rmse = float(np.sqrt(values["mse_C2"].mean()))
    mae = float(values["mae_C"].mean())
    return {"RMSE_C": rmse, "RMSE_percent": 100.0 * rmse / mean, "MAE_C": mae,
            "MaxAE_C": float(values["maxae_C"].max()), "ARE_percent": 100.0 * mae / mean,
            "L2_RER_percent": float(100.0 * np.linalg.norm(error) / np.linalg.norm(truth)),
            "PSNR_dB": None if rmse == 0 else float(20 * np.log10(reference_range / rmse)),
            "reference_mean_C": mean, "fixed_reference_range_C": float(reference_range)}


def macro_rmse(families):
    return float(np.mean([value["RMSE_percent"] for value in families.values()]))
