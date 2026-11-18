"""pyALDIC's mask / image reading helpers, implemented with RAFTcorr's readers (no OpenCV)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from raftcorr.io.images import read_raw, to_grey
from raftcorr.roi.masks import load_mask


def _read_unchanged(path: str | Path) -> NDArray:
    """Image as stored (bit depth and channels preserved)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"image not found: {p}")
    return read_raw(p)


def _to_grayscale(img: NDArray) -> NDArray:
    """Single channel, dtype preserved where possible."""
    if img.ndim == 2:
        return img
    g = to_grey(img)
    return g.astype(img.dtype) if np.issubdtype(img.dtype, np.integer) else g


def _to_uint8(img: NDArray) -> NDArray[np.uint8]:
    """Any bit depth to uint8 [0, 255]."""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.uint16:
        return (a / 256).astype(np.uint8)
    a = a.astype(np.float64)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def read_mask_as_bool(path: str | Path, target_shape: tuple[int, int] | None = None) -> NDArray[np.bool_]:
    """A mask image (white = inside) as a boolean array, resized (nearest) to ``target_shape``."""
    return load_mask(path, target_shape)
