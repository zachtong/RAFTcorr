"""Turning fields into pixels: colour mapping, ranges, compositing (Qt-free).

The GUI converts the RGBA arrays produced here into QImages; the image and animation exporters
write them directly. One renderer means the exported pictures are the pictures on screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .colormaps import is_diverging, lut


@dataclass(frozen=True)
class ColorRange:
    vmin: float
    vmax: float

    @property
    def span(self) -> float:
        return self.vmax - self.vmin


def auto_range(values: np.ndarray, colormap: str = "turbo", percentile: float = 2.0) -> ColorRange:
    """Range of the finite values, trimmed by ``percentile`` % at each end (2-98 %, pyALDIC's rule).

    Diverging colormaps get a range symmetric about zero, so white means zero. The one auto rule
    of the application: the main window, the strain window and the exports all call it.
    """
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return ColorRange(-1.0, 1.0) if is_diverging(colormap) else ColorRange(0.0, 1.0)
    if percentile > 0:
        lo, hi = np.percentile(v, [percentile, 100.0 - percentile])
    else:
        lo, hi = float(v.min()), float(v.max())
    lo, hi = float(lo), float(hi)
    if is_diverging(colormap):
        m = max(abs(lo), abs(hi)) or 1e-12
        return ColorRange(-m, m)
    if hi - lo < 1e-12:
        pad = max(abs(lo) * 1e-3, 1e-9)
        lo, hi = lo - pad, hi + pad
    return ColorRange(lo, hi)


def colorize(values: np.ndarray, rng: ColorRange, colormap: str = "turbo", alpha: float = 1.0,
             table: np.ndarray | None = None) -> np.ndarray:
    """``(h, w, 4)`` uint8 RGBA; NaN pixels are fully transparent."""
    v = np.asarray(values, dtype=np.float32)
    t = lut(colormap) if table is None else table
    n = t.shape[0]
    span = rng.span if rng.span != 0 else 1.0
    finite = np.isfinite(v)
    idx = np.clip(((np.where(finite, v, rng.vmin) - rng.vmin) / span * (n - 1) + 0.5), 0, n - 1).astype(np.intp)
    rgba = t[idx]
    rgba[..., 3] = np.where(finite, np.uint8(round(255 * float(np.clip(alpha, 0.0, 1.0)))), 0)
    return rgba


def grey_rgba(image: np.ndarray, contrast: tuple[float, float] | None = None) -> np.ndarray:
    """Grey image (0-255 float) as opaque RGBA, optionally contrast-stretched to ``(lo, hi)``."""
    g = np.asarray(image, dtype=np.float32)
    if contrast is not None:
        lo, hi = contrast
        g = (g - lo) * (255.0 / max(hi - lo, 1e-6))
    g8 = np.clip(g + 0.5, 0, 255).astype(np.uint8)
    out = np.empty(g8.shape + (4,), dtype=np.uint8)
    out[..., 0] = out[..., 1] = out[..., 2] = g8
    out[..., 3] = 255
    return out


def composite(background: np.ndarray, overlay: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """Alpha-blend an RGBA overlay onto an RGBA background at ``(x0, y0)`` (clipped). New array."""
    out = background.copy()
    H, W = out.shape[:2]
    h, w = overlay.shape[:2]
    bx0, by0 = max(0, x0), max(0, y0)
    bx1, by1 = min(W, x0 + w), min(H, y0 + h)
    if bx1 <= bx0 or by1 <= by0:
        return out
    ov = overlay[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0].astype(np.float32)
    bg = out[by0:by1, bx0:bx1].astype(np.float32)
    a = ov[..., 3:4] / 255.0
    bg[..., :3] = ov[..., :3] * a + bg[..., :3] * (1.0 - a)
    bg[..., 3:4] = np.maximum(bg[..., 3:4], ov[..., 3:4])
    out[by0:by1, bx0:bx1] = (bg + 0.5).astype(np.uint8)
    return out


def mask_rgba(mask: np.ndarray, color: tuple[int, int, int] = (74, 108, 247), alpha: int = 90) -> np.ndarray:
    """Translucent fill of a boolean mask (ROI display)."""
    out = np.zeros(mask.shape + (4,), dtype=np.uint8)
    out[mask, 0], out[mask, 1], out[mask, 2] = color
    out[mask, 3] = alpha
    return out


def nice_ticks(vmin: float, vmax: float, n: int = 5) -> list[float]:
    """Round tick values inside ``[vmin, vmax]``."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return [vmin]
    raw = (vmax - vmin) / max(1, n - 1)
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = np.ceil(vmin / step) * step
    ticks = np.arange(start, vmax + step * 1e-6, step)
    return [float(0.0 if abs(t) < step * 1e-9 else t) for t in ticks]


def format_tick(value: float, span: float) -> str:
    if span == 0 or not np.isfinite(span):
        return f"{value:g}"
    digits = max(0, int(-np.floor(np.log10(span / 5))) + 1) if span < 10 else 0
    if abs(value) >= 1e4 or (0 < abs(value) < 1e-3 and span < 1e-2):
        return f"{value:.2e}"
    return f"{value:.{min(digits, 6)}f}"
