"""NaN-aware bilinear interpolation for displacement field sampling.

Standard cv2.remap fills NaN with 0 before interpolation, which contaminates
boundary pixels.  This module provides a pure-NumPy alternative that excludes
NaN and out-of-bounds neighbours, renormalising the remaining weights so that
valid data is never mixed with fabricated zeros.
"""

import numpy as np


def bilinear_sample_nan_aware(
    field: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Sample *field* at sub-pixel locations (*x*, *y*) with NaN awareness.

    Parameters
    ----------
    field : ndarray, shape (H, W)
        2-D float array that may contain NaN values.
    x : ndarray, shape (N,)
        Column (horizontal) coordinates.
    y : ndarray, shape (N,)
        Row (vertical) coordinates.

    Returns
    -------
    ndarray, shape (N,)
        Interpolated values.  If all four neighbours of a query point are
        invalid (NaN or out-of-bounds), the result is NaN.  Otherwise only
        valid neighbours contribute, with weights renormalised to sum to 1.
    """
    H, W = field.shape
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Floor coordinates and fractional parts
    x0 = np.floor(x).astype(np.intp)
    y0 = np.floor(y).astype(np.intp)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = x - x0
    fy = y - y0

    # Bilinear weights
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx * fy

    # Validity masks: in-bounds check for each corner
    valid00 = (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H)
    valid10 = (x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H)
    valid01 = (x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H)
    valid11 = (x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H)

    # Clamp indices for safe array access (clamped values will be masked out)
    cx0 = np.clip(x0, 0, W - 1)
    cx1 = np.clip(x1, 0, W - 1)
    cy0 = np.clip(y0, 0, H - 1)
    cy1 = np.clip(y1, 0, H - 1)

    # Fetch values at the four corners
    v00 = field[cy0, cx0]
    v10 = field[cy0, cx1]
    v01 = field[cy1, cx0]
    v11 = field[cy1, cx1]

    # Also exclude NaN values in the field itself
    valid00 = valid00 & ~np.isnan(v00)
    valid10 = valid10 & ~np.isnan(v10)
    valid01 = valid01 & ~np.isnan(v01)
    valid11 = valid11 & ~np.isnan(v11)

    # Zero out weights and values for invalid corners
    w00 = np.where(valid00, w00, 0.0)
    w10 = np.where(valid10, w10, 0.0)
    w01 = np.where(valid01, w01, 0.0)
    w11 = np.where(valid11, w11, 0.0)

    v00 = np.where(valid00, v00, 0.0)
    v10 = np.where(valid10, v10, 0.0)
    v01 = np.where(valid01, v01, 0.0)
    v11 = np.where(valid11, v11, 0.0)

    # Weighted sum and weight sum
    weighted = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11
    w_sum = w00 + w10 + w01 + w11

    # Renormalise; return NaN where no valid neighbours
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(w_sum > 0.0, weighted / w_sum, np.nan)
    return result
