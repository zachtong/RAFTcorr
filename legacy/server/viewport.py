"""Viewport-aware downsampling for render endpoints."""

from typing import Tuple

import cv2
import numpy as np


def downsample_for_viewport(
    full_data: np.ndarray,
    bg_img: np.ndarray,
    vw: int,
    vh: int,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Downsample data and background to fit the browser viewport.

    Parameters
    ----------
    full_data : (H, W) scalar field (may contain NaN)
    bg_img : (H, W, 3) or (H, W) background image
    vw, vh : viewport width and height in CSS pixels

    Returns
    -------
    (data_out, bg_out, out_h, out_w) — downsampled arrays and their dimensions.
    If viewport is 0 or larger than image, returns inputs unchanged.
    """
    h, w = full_data.shape[:2]

    if vw <= 0 or vh <= 0:
        return full_data, bg_img, h, w

    scale = min(vw / w, vh / h, 1.0)
    if scale >= 1.0:
        return full_data, bg_img, h, w

    out_w = max(1, int(w * scale))
    out_h = max(1, int(h * scale))

    # Downsample scalar data (NaN-safe)
    mask = np.isfinite(full_data)
    filled = np.nan_to_num(full_data, nan=0.0).astype(np.float32)
    data_small = cv2.resize(filled, (out_w, out_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(
        mask.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA
    )
    data_out = data_small.astype(np.float64)
    data_out[mask_small < 0.5] = np.nan

    # Downsample background
    bg_out = cv2.resize(bg_img, (out_w, out_h), interpolation=cv2.INTER_AREA)

    return data_out, bg_out, out_h, out_w
