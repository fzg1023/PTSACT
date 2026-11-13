import numpy as np

from .common import register_vendor_paths


def fixed_scale(normalization):
    lo = float(normalization["T_min"])
    span = float(normalization["T_max"]) - lo + 1e-14
    if not np.isfinite(lo) or not np.isfinite(span) or span <= 1e-13:
        raise ValueError("Invalid fixed normalization limits")
    return lo, span


def normalize_initial(initial, normalization):
    field = np.asarray(initial, dtype=np.float32)
    if field.ndim == 2 and field.shape == (32, 32):
        field = field[None]
    if field.ndim != 3 or field.shape[1:] != (32, 32) or not np.isfinite(field).all():
        raise ValueError("Expected finite 32 by 32 Stage 1 fields")
    lo, span = fixed_scale(normalization)
    return ((field - lo) / span)[:, None]


def restore_prediction(raw, initial, normalization, mode="residual"):
    lo, span = fixed_scale(normalization)
    if mode == "residual":
        return raw * span + np.asarray(initial, dtype=np.float32)
    if mode == "direct":
        return raw * span + lo
    raise ValueError("Unknown learning target")


def load_refiner(checkpoint, device="cpu", base_ch=24, model_type="attention"):
    register_vendor_paths()
    import torch
    from nn_prior import AttentionUNet, PlainUNet
    if model_type == "plain":
        model = PlainUNet(in_channels=1, base_ch=base_ch, dropout_p=0.1)
    else:
        model = AttentionUNet(in_channels=1, base_ch=base_ch, dropout_p=0.1,
                              use_attention=model_type != "base")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def predict_field(model, initial, normalization, device="cpu", batch_size=64, mode="residual"):
    """Inference takes Stage 1 fields and fixed constants, never target truth."""
    import torch
    field = np.asarray(initial, dtype=np.float32)
    if field.ndim == 2:
        field = field[None]
    normalize_initial(field[:1], normalization)
    if batch_size < 1 or len(field) < 1:
        raise ValueError("Invalid batch size or empty input")
    outputs = []
    with torch.no_grad():
        for first in range(0, len(field), batch_size):
            batch = field[first:first + batch_size]
            x = torch.from_numpy(normalize_initial(batch, normalization)).to(device)
            raw = model(x).squeeze(1).cpu().numpy()
            outputs.append(restore_prediction(raw, batch, normalization, mode))
    result = np.concatenate(outputs)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite prediction")
    return result
