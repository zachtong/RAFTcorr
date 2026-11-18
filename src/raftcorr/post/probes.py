"""Probes ("virtual sensors"): points, lines (extensometers) and areas, with their time series.

Probe geometry is in frame-0 image pixels (material coordinates), so a probe follows the material
and survives a change of units. Values are read from the reference-configuration fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engine.mapping import sample_bilinear
from ..roi.shapes import Circle, Polygon, Rect, rasterize

KINDS: tuple[str, ...] = ("point", "line", "rect", "circle", "polygon")
AREA_KINDS: tuple[str, ...] = ("rect", "circle", "polygon")
METRICS: tuple[str, ...] = ("mean", "min", "max", "std", "median")

#: Colour-blind-friendly cycle (Okabe-Ito, without yellow which vanishes on light images).
COLORS: tuple[str, ...] = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#882255")

LINE_SAMPLES = 100


@dataclass(frozen=True)
class Probe:
    id: int
    kind: str
    points: tuple[tuple[float, float], ...]
    color: str = COLORS[0]
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"probe kind must be one of {KINDS}, got {self.kind!r}")
        pts = tuple((float(x), float(y)) for x, y in self.points)
        need = {"point": 1, "line": 2, "rect": 2, "circle": 2}.get(self.kind)
        if need is not None and len(pts) != need:
            raise ValueError(f"a {self.kind} probe needs {need} points, got {len(pts)}")
        if self.kind == "polygon" and len(pts) < 3:
            raise ValueError("a polygon probe needs at least 3 points")
        object.__setattr__(self, "points", pts)
        if not self.label:
            prefix = {"point": "P", "line": "L"}.get(self.kind, "A")
            object.__setattr__(self, "label", f"{prefix}{self.id}")

    @property
    def is_area(self) -> bool:
        return self.kind in AREA_KINDS

    def shape(self):
        if self.kind == "rect":
            (x0, y0), (x1, y1) = self.points
            return Rect(x0, y0, x1, y1)
        if self.kind == "circle":
            (cx, cy), (ex, ey) = self.points
            return Circle(cx, cy, float(np.hypot(ex - cx, ey - cy)))
        if self.kind == "polygon":
            return Polygon(self.points)
        raise ValueError(f"a {self.kind} probe has no area")

    def describe(self) -> str:
        if self.kind == "point":
            x, y = self.points[0]
            return f"({x:.1f}, {y:.1f})"
        if self.kind == "line":
            (x0, y0), (x1, y1) = self.points
            return f"({x0:.0f}, {y0:.0f}) - ({x1:.0f}, {y1:.0f}), L0 = {np.hypot(x1 - x0, y1 - y0):.1f} px"
        if self.kind == "circle":
            (cx, cy), (ex, ey) = self.points
            return f"centre ({cx:.0f}, {cy:.0f}), r = {np.hypot(ex - cx, ey - cy):.1f} px"
        if self.kind == "rect":
            (x0, y0), (x1, y1) = self.points
            return f"{abs(x1 - x0):.0f} x {abs(y1 - y0):.0f} px at ({min(x0, x1):.0f}, {min(y0, y1):.0f})"
        return f"{len(self.points)} vertices"

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "points": [list(p) for p in self.points],
                "color": self.color, "label": self.label}

    @classmethod
    def from_dict(cls, d: dict) -> "Probe":
        kind = "polygon" if d["kind"] == "poly" else d["kind"]
        return cls(int(d["id"]), kind, tuple(tuple(p) for p in d["points"]), d.get("color", COLORS[0]), d.get("label", ""))


def color_for(index: int) -> str:
    return COLORS[index % len(COLORS)]


def _reduce(vals: np.ndarray, metric: str) -> float:
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return float("nan")
    return float({"mean": np.mean, "min": np.min, "max": np.max, "std": np.std, "median": np.median}[metric](v))


def outline(probe: Probe, n: int = 48) -> np.ndarray:
    """``(N, 2)`` points describing the probe for drawing: the point, the sampled line, or the
    closed boundary of an area (edges subdivided so that the outline can be warped)."""
    if probe.kind == "point":
        return np.asarray(probe.points, dtype=np.float64)
    if probe.kind == "line":
        return line_points(probe, n)
    if probe.kind == "circle":
        (cx, cy), (ex, ey) = probe.points
        r = float(np.hypot(ex - cx, ey - cy))
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)
    if probe.kind == "rect":
        (x0, y0), (x1, y1) = probe.points
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    else:
        corners = list(probe.points)
    pts = []
    k = max(2, n // len(corners))
    for a, b in zip(corners, corners[1:] + corners[:1]):
        t = np.linspace(0, 1, k, endpoint=False)[:, None]
        pts.append(np.asarray(a) + t * (np.asarray(b) - np.asarray(a)))
    return np.concatenate(pts)


def line_points(probe: Probe, n: int = LINE_SAMPLES) -> np.ndarray:
    (x0, y0), (x1, y1) = probe.points
    t = np.linspace(0.0, 1.0, n)
    return np.stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0)], axis=1)


def area_mask(probe: Probe, x0: float, y0: float, step: int, shape: tuple[int, int]) -> np.ndarray:
    """The probe's area on a field grid placed at ``(x0, y0)`` with spacing ``step`` px."""
    sh = probe.shape()
    if isinstance(sh, Rect):
        sh = Rect((sh.x0 - x0) / step, (sh.y0 - y0) / step, (sh.x1 - x0) / step, (sh.y1 - y0) / step)
    elif isinstance(sh, Circle):
        sh = Circle((sh.cx - x0) / step, (sh.cy - y0) / step, sh.r / step)
    else:
        sh = Polygon(tuple(((x - x0) / step, (y - y0) / step) for x, y in sh.points))
    return rasterize(sh, *shape)


def probe_value(probe: Probe, values: np.ndarray, x0: float, y0: float, step: int = 1, metric: str = "mean") -> float:
    """One number for one frame: the field ``values`` placed at ``(x0, y0)`` with spacing ``step``."""
    if probe.kind == "point":
        x, y = probe.points[0]
        return float(sample_bilinear(values, np.array([(x - x0) / step]), np.array([(y - y0) / step]))[0])
    if probe.kind == "line":
        pts = line_points(probe)
        vals = sample_bilinear(values, (pts[:, 0] - x0) / step, (pts[:, 1] - y0) / step)
        return _reduce(vals, metric)
    m = area_mask(probe, x0, y0, step, values.shape)
    return _reduce(values[m], metric)


def probe_series(probe: Probe, provider, key: str, metric: str = "mean") -> np.ndarray:
    """``(T,)`` values of field ``key`` at the probe for every frame (NaN where not computed)."""
    n = provider.series.n_frames
    out = np.full(n, np.nan)
    for k in range(n):
        if provider.has_frame(key, k):
            vals, pl = provider.values(key, k)
            out[k] = probe_value(probe, vals, pl.x0, pl.y0, pl.step, metric)
    return out


def extensometer(probe: Probe, series) -> tuple[np.ndarray, np.ndarray]:
    """Deformed length and engineering strain ``(L - L0) / L0`` of a line probe over frames."""
    if probe.kind != "line":
        raise ValueError("an extensometer needs a line probe")
    (xa, ya), (xb, yb) = probe.points
    L0 = float(np.hypot(xb - xa, yb - ya))
    g = series.grid
    n = series.n_frames
    length = np.full(n, np.nan)
    for k in range(n):
        if not series.done[k]:
            continue
        uv = sample_bilinear(series.field(k), np.array([xa, xb]), np.array([ya, yb]), g.x0, g.y0)
        pa = (xa + uv[0, 0], ya + uv[1, 0])
        pb = (xb + uv[0, 1], yb + uv[1, 1])
        length[k] = np.hypot(pb[0] - pa[0], pb[1] - pa[1])
    strain = (length - L0) / L0 if L0 > 0 else np.full(n, np.nan)
    return length, strain
