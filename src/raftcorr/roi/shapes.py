"""ROI shapes, their rasterisation, and boolean composition (add = OR, cut = AND NOT).

Coordinates are image pixels with the origin at the centre of the top-left pixel: pixel
``[row, col]`` covers ``x in [col - 0.5, col + 0.5)``. A pixel belongs to a shape when its
centre does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

MODES: tuple[str, ...] = ("add", "cut")


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def normalized(self) -> "Rect":
        return Rect(min(self.x0, self.x1), min(self.y0, self.y1), max(self.x0, self.x1), max(self.y0, self.y1))

    def to_dict(self) -> dict:
        return {"type": "rect", "x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float

    def to_dict(self) -> dict:
        return {"type": "circle", "cx": self.cx, "cy": self.cy, "r": self.r}


@dataclass(frozen=True)
class Polygon:
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        pts = tuple((float(x), float(y)) for x, y in self.points)
        object.__setattr__(self, "points", pts)

    def to_dict(self) -> dict:
        return {"type": "polygon", "points": [list(p) for p in self.points]}


Shape = Union[Rect, Circle, Polygon]


def shape_from_dict(d: dict) -> Shape:
    t = d.get("type")
    if t == "rect":
        return Rect(float(d["x0"]), float(d["y0"]), float(d["x1"]), float(d["y1"]))
    if t == "circle":
        return Circle(float(d["cx"]), float(d["cy"]), float(d["r"]))
    if t in ("polygon", "poly"):
        return Polygon(tuple((float(x), float(y)) for x, y in d["points"]))
    raise ValueError(f"unknown shape type {t!r}")


def rasterize(shape: Shape, height: int, width: int) -> np.ndarray:
    """Boolean mask of the pixels whose centres lie inside ``shape``."""
    if isinstance(shape, Rect):
        r = shape.normalized()
        mask = np.zeros((height, width), dtype=bool)
        c0, c1 = int(np.ceil(r.x0)), int(np.floor(r.x1))
        r0, r1 = int(np.ceil(r.y0)), int(np.floor(r.y1))
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, width - 1), min(r1, height - 1)
        if c1 >= c0 and r1 >= r0:
            mask[r0 : r1 + 1, c0 : c1 + 1] = True
        return mask
    if isinstance(shape, Circle):
        if shape.r <= 0:
            return np.zeros((height, width), dtype=bool)
        y0, y1 = max(0, int(np.floor(shape.cy - shape.r))), min(height, int(np.ceil(shape.cy + shape.r)) + 1)
        x0, x1 = max(0, int(np.floor(shape.cx - shape.r))), min(width, int(np.ceil(shape.cx + shape.r)) + 1)
        mask = np.zeros((height, width), dtype=bool)
        if y1 > y0 and x1 > x0:
            yy, xx = np.mgrid[y0:y1, x0:x1]
            mask[y0:y1, x0:x1] = (xx - shape.cx) ** 2 + (yy - shape.cy) ** 2 <= shape.r**2
        return mask
    if isinstance(shape, Polygon):
        if len(shape.points) < 3:
            return np.zeros((height, width), dtype=bool)
        return _polygon_mask(shape.points, height, width)
    raise TypeError(f"not a shape: {type(shape).__name__}")


def _polygon_mask(points: tuple[tuple[float, float], ...], height: int, width: int) -> np.ndarray:
    """Even-odd fill of pixel centres, vectorised per row (no cv2 / Pillow rounding)."""
    pts = np.asarray(points, dtype=np.float64)
    xs, ys = pts[:, 0], pts[:, 1]
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
    mask = np.zeros((height, width), dtype=bool)
    r0 = max(0, int(np.ceil(ys.min())))
    r1 = min(height - 1, int(np.floor(ys.max())))
    if r1 < r0:
        return mask
    rows = np.arange(r0, r1 + 1, dtype=np.float64)[:, None]  # (R, 1)
    # edges crossing the row centre y: (y1 <= y < y2) or (y2 <= y < y1)
    crosses = ((ys[None] <= rows) & (y2[None] > rows)) | ((y2[None] <= rows) & (ys[None] > rows))
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = xs[None] + (rows - ys[None]) * (x2 - xs)[None] / (y2 - ys)[None]
    xi = np.where(crosses, xi, np.nan)
    for k, row in enumerate(range(r0, r1 + 1)):
        xr = np.sort(xi[k][np.isfinite(xi[k])])
        for a, b in zip(xr[0::2], xr[1::2]):
            c0, c1 = max(0, int(np.ceil(a))), min(width - 1, int(np.floor(b)))
            if c1 >= c0:
                mask[row, c0 : c1 + 1] = True
    return mask


def apply_shape(mask: np.ndarray, shape: Shape, mode: str) -> np.ndarray:
    """New mask with ``shape`` added (OR) or cut (AND NOT). ``mask`` is not modified."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    region = rasterize(shape, *mask.shape)
    return (mask | region) if mode == "add" else (mask & ~region)


def brush_stroke(height: int, width: int, points: list[tuple[float, float]], radius: float) -> np.ndarray:
    """Mask of a round brush of ``radius`` px dragged through ``points``."""
    mask = np.zeros((height, width), dtype=bool)
    if not points or radius <= 0:
        return mask
    pts = np.asarray(points, dtype=np.float64)
    # densify so consecutive stamps overlap
    dense = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / max(radius / 2.0, 0.5))))
        t = np.linspace(0, 1, n + 1)[1:, None]
        dense.extend(a + t * (b - a))
    r2 = radius * radius
    for cx, cy in dense:
        y0, y1 = max(0, int(np.floor(cy - radius))), min(height, int(np.ceil(cy + radius)) + 1)
        x0, x1 = max(0, int(np.floor(cx - radius))), min(width, int(np.ceil(cx + radius)) + 1)
        if y1 <= y0 or x1 <= x0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        mask[y0:y1, x0:x1] |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r2
    return mask
