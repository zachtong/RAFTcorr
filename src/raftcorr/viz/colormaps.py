"""Colormaps as 256-entry RGBA lookup tables (from matplotlib), shared by the GUI and exports."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

#: Offered in the GUI, in this order. Sequential first, then diverging (zero-centred), then grey.
SEQUENTIAL: tuple[str, ...] = ("turbo", "viridis", "plasma", "inferno", "cividis", "jet")
DIVERGING: tuple[str, ...] = ("RdBu_r", "coolwarm", "seismic", "PuOr_r")
COLORMAPS: tuple[str, ...] = SEQUENTIAL + DIVERGING + ("gray",)

DEFAULT_COLORMAP = "turbo"


def is_diverging(name: str) -> bool:
    return name in DIVERGING


@lru_cache(maxsize=32)
def lut(name: str, n: int = 256) -> np.ndarray:
    """``(n, 4)`` uint8 RGBA table, read-only."""
    import matplotlib

    try:
        cmap = matplotlib.colormaps[name]
    except KeyError:
        raise KeyError(f"unknown colormap {name!r}; available: {', '.join(COLORMAPS)}") from None
    table = (cmap(np.linspace(0.0, 1.0, n)) * 255.0 + 0.5).astype(np.uint8)
    table.setflags(write=False)
    return table


def discrete_lut(name: str, levels: int) -> np.ndarray:
    """A table with ``levels`` flat colour bands (contour-like display)."""
    base = lut(name)
    if levels <= 1 or levels >= base.shape[0]:
        return base
    idx = np.minimum((np.arange(256) * levels // 256), levels - 1)
    centres = ((idx + 0.5) * 256 / levels).astype(int).clip(0, 255)
    return base[centres]
