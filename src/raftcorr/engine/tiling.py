"""Work areas and tiles for one image pair.

A point of the ROI can only be matched if its displaced position lies inside the window fed to
the network, so every window extends past the region it reports by a *margin* that covers the
expected displacement (default: the model's nominal lookup radius ``32 s`` px). The deformed
window is the same rectangle of the deformed frame and is never padded beyond the frame.

* One window when ``ROI bounding box + margin`` fits the memory budget.
* Otherwise the output region is cut into *cores*; each tile's window is its core grown by the
  margin (the *halo*). A tile writes only its core. Neighbouring cores overlap by ``blend_px``
  and are feathered with linear weights, so no seam is visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MIN_CORE_PX = 48


@dataclass(frozen=True)
class Window:
    """Half-open rectangle ``[x0, x1) x [y0, y1)`` in image pixels."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y1), slice(self.x0, self.x1)

    def grow(self, px: int, shape: tuple[int, int]) -> "Window":
        h, w = shape
        return Window(max(0, self.x0 - px), max(0, self.y0 - px), min(w, self.x1 + px), min(h, self.y1 + px))

    def at_least(self, side: int, shape: tuple[int, int]) -> "Window":
        """Grow (about the centre, inside the image) until both sides are >= ``side``."""
        h, w = shape
        x0, x1 = _widen(self.x0, self.x1, side, w)
        y0, y1 = _widen(self.y0, self.y1, side, h)
        return Window(x0, y0, x1, y1)

    def to_tuple(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


def _widen(a: int, b: int, side: int, limit: int) -> tuple[int, int]:
    if b - a >= side or limit <= side:
        return (0, limit) if limit <= side else (a, b)
    extra = side - (b - a)
    a2 = max(0, a - extra // 2)
    b2 = min(limit, a2 + side)
    a2 = max(0, b2 - side)
    return a2, b2


@dataclass(frozen=True)
class Tile:
    core: Window    # pixels this tile writes
    window: Window  # pixels fed to the network


@dataclass(frozen=True)
class TilePlan:
    output: Window
    tiles: tuple[Tile, ...]
    margin_px: int
    blend_px: int
    reduced_margin: bool = False

    @property
    def single(self) -> bool:
        return len(self.tiles) == 1

    @property
    def max_window_pixels(self) -> int:
        return max(t.window.area for t in self.tiles)

    def describe(self) -> str:
        if self.single:
            w = self.tiles[0].window
            return f"one window of {w.width} x {w.height} px (margin {self.margin_px} px)"
        w = max((t.window for t in self.tiles), key=lambda t: t.area)
        note = " (margin reduced to fit memory)" if self.reduced_margin else ""
        return f"{len(self.tiles)} tiles, windows up to {w.width} x {w.height} px, margin {self.margin_px} px{note}"


def plan_tiles(
    image_shape: tuple[int, int],
    roi_bbox: tuple[int, int, int, int],
    margin_px: int,
    max_pixels: int,
    min_side: int,
    blend_px: int = 32,
    out_pad: int = 8,
) -> TilePlan:
    """Windows and cores covering ``roi_bbox`` (``x0, y0, width, height``) for one pair."""
    h, w = image_shape
    if min(h, w) < min_side:
        raise ValueError(f"image {h} x {w} px is smaller than the model's {min_side} px minimum")
    bx, by, bw, bh = roi_bbox
    output = Window(bx, by, bx + bw, by + bh).grow(out_pad, image_shape)
    margin = max(0, int(margin_px))
    single = output.grow(margin, image_shape).at_least(min_side, image_shape)
    if single.area <= max_pixels:
        return TilePlan(output, (Tile(output, single),), margin, blend_px)

    side = int(math.isqrt(max(0, int(max_pixels))))
    if side < min_side:
        raise MemoryError(
            f"the memory budget allows windows of about {side} x {side} px, below the model's {min_side} px minimum"
        )
    blend = max(0, int(blend_px))
    reduced = False
    core = side - 2 * margin
    min_core = max(MIN_CORE_PX, 2 * blend + 16)
    if core < min_core:
        core = min(min_core, side)
        margin = max(0, (side - core) // 2)
        reduced = True
    xs = _starts(output.x0, output.width, core, blend)
    ys = _starts(output.y0, output.height, core, blend)
    tiles = []
    for y0 in ys:
        for x0 in xs:
            c = Window(x0, y0, min(x0 + core, output.x1), min(y0 + core, output.y1))
            win = c.grow(margin, image_shape).at_least(min_side, image_shape)
            tiles.append(Tile(c, win))
    return TilePlan(output, tuple(tiles), margin, blend, reduced)


def _starts(origin: int, length: int, core: int, blend: int) -> list[int]:
    if length <= core:
        return [origin]
    step_max = max(1, core - blend)
    n = 1 + math.ceil((length - core) / step_max)
    step = (length - core) / (n - 1)
    return [origin + int(round(i * step)) for i in range(n)]


def blend_weights(tile: Tile, output: Window, blend_px: int) -> np.ndarray:
    """Feathering weights over a tile's core: linear ramps where the core borders another tile."""
    c = tile.core

    def ramp(n: int, start_inner: bool, end_inner: bool) -> np.ndarray:
        wts = np.ones(n, dtype=np.float32)
        b = min(blend_px, n // 2)
        if b > 0:
            r = (np.arange(b, dtype=np.float32) + 0.5) / b
            if start_inner:
                wts[:b] = np.minimum(wts[:b], r)
            if end_inner:
                wts[-b:] = np.minimum(wts[-b:], r[::-1])
        return wts

    wx = ramp(c.width, c.x0 > output.x0, c.x1 < output.x1)
    wy = ramp(c.height, c.y0 > output.y0, c.y1 < output.y1)
    return wy[:, None] * wx[None, :]
