"""NaN-aware Gaussian smoothing (normalised convolution)."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def nan_gaussian(field: np.ndarray, sigma: float, mask: np.ndarray | None = None, min_weight: float = 0.05) -> np.ndarray:
    """Gaussian smoothing that ignores NaN and out-of-mask points (normalised convolution).

    ``field`` is ``(h, w)`` or ``(C, h, w)``. Points outside ``mask`` (or NaN) stay NaN; points
    whose neighbourhood holds less than ``min_weight`` of the kernel mass become NaN.
    """
    if sigma <= 0:
        return np.array(field, dtype=np.float32, copy=True)
    f = field[None] if field.ndim == 2 else field
    valid = np.isfinite(f).all(axis=0)
    if mask is not None:
        valid &= mask
    wv = valid.astype(np.float64)
    wsum = ndimage.gaussian_filter(wv, sigma, mode="constant")
    out = np.empty(f.shape, dtype=np.float32)
    for c in range(f.shape[0]):
        num = ndimage.gaussian_filter(np.where(valid, f[c], 0.0).astype(np.float64), sigma, mode="constant")
        with np.errstate(invalid="ignore", divide="ignore"):
            r = num / wsum
        r[~valid | (wsum < min_weight)] = np.nan
        out[c] = r
    return out[0] if field.ndim == 2 else out


def temporal_gaussian(stack: np.ndarray, sigma_frames: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing along axis 0 of a ``(T, ...)`` stack."""
    if sigma_frames <= 0:
        return np.array(stack, dtype=np.float32, copy=True)
    valid = np.isfinite(stack)
    num = ndimage.gaussian_filter1d(np.where(valid, stack, 0.0).astype(np.float64), sigma_frames, axis=0, mode="nearest")
    den = ndimage.gaussian_filter1d(valid.astype(np.float64), sigma_frames, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~valid] = np.nan
    return out.astype(np.float32)
