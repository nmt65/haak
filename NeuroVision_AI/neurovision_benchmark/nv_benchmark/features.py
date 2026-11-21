"""Fixed-channel CSI representation. Real sensor measurements, never synthetic labels."""
import numpy as np

FEATURE_NAMES = [
    "mean", "std", "q05", "q25", "median", "q75", "q95", "rms",
    "min", "max", "mean_absolute_delta", "std_delta"
]


def csi_to_features(raw: np.ndarray, steps: int = 128) -> np.ndarray:
    """Return float32 [12, T] from WiMANS CSI [time, ...] or MM-Fi adaptation.

    Assumption: first dimension is time. Convert complex -> amplitude. CSI
    format, time axis and sample rate MUST be verified for new hardware.
    """
    a = np.asarray(raw)
    if a.ndim < 2 or a.shape[0] < 2:
        raise ValueError(f"Expected time-first CSI with >=2 samples, got {a.shape}")
    a = np.abs(a).reshape(a.shape[0], -1).astype(np.float32)
    if a.shape[1] == 0 or not np.any(np.isfinite(a)):
        raise ValueError("CSI has no finite values")
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    delta = np.diff(a, axis=0, prepend=a[:1])
    features = np.stack([
        a.mean(axis=1), a.std(axis=1),
        np.quantile(a, .05, axis=1), np.quantile(a, .25, axis=1),
        np.median(a, axis=1), np.quantile(a, .75, axis=1),
        np.quantile(a, .95, axis=1), np.sqrt((a*a).mean(axis=1)),
        a.min(axis=1), a.max(axis=1),
        np.abs(delta).mean(axis=1), delta.std(axis=1)
    ], axis=0)
    source_t = np.linspace(0, 1, features.shape[1])
    target_t = np.linspace(0, 1, steps)
    out = np.stack([np.interp(target_t, source_t, f) for f in features])
    return np.nan_to_num(out).astype(np.float32)


def standardize_fit(x_train: np.ndarray):
    """Training only: (N, C, T) -> per-channel mu/sigma."""
    mu = x_train.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    sd = x_train.std(axis=(0, 2), keepdims=True).astype(np.float32)
    return mu, np.maximum(sd, 1e-5)
