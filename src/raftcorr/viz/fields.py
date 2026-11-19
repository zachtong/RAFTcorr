"""The catalogue of displayable fields and their extraction in either configuration.

``reference`` configuration: values at the frame-0 position of each material point (the grid).
``deformed`` configuration: the same values drawn where the points are in the current frame,
through the inverse map (cached per frame and shared by every component).

Velocity has exactly one definition everywhere (display, probes, exports): the central difference
of the Lagrangian displacement over neighbouring computed frames, one-sided at the ends, times the
frame rate (per frame when no rate is set).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from ..engine.mapping import inverse_map, sample_bilinear
from ..post.strain import COMPONENT_LABELS, COMPONENTS
from ..post.strain import fill_trimmed as fill_trimmed_band
from ..results import DisplacementSeries, StrainSeries


def fill_untracked(values: np.ndarray, inside: np.ndarray, keep_empty: np.ndarray | None = None) -> np.ndarray:
    """NaN points of ``inside`` filled from the nearest valid value ("Hide untracked regions" off).

    The one rule of the main window, the strain window and exported pictures. Region by region,
    as :func:`raftcorr.post.strain.fill_trimmed` fills the trimmed band: each point takes the value
    of the nearest finite point of its own connected part of ``inside``, so the fill never takes
    values from across a crack or a hole cut out of the region of interest. Points of
    ``keep_empty`` (the edge-trimmed strain band) stay as they are: only "Fill trimmed edges"
    fills them. Returns ``values`` itself when there is nothing to fill, else a new array.
    """
    from scipy import ndimage

    finite = np.isfinite(values)
    bad = inside & ~finite
    if keep_empty is not None and keep_empty.shape == bad.shape:
        bad &= ~keep_empty
    if not bad.any() or not finite.any():
        return values
    out = values.copy()
    labels, _n = ndimage.label(inside)
    for lab, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        region = labels[sl] == lab
        need = bad[sl] & region
        src = finite[sl] & region
        if not need.any() or not src.any():
            continue
        _d, (iy, ix) = ndimage.distance_transform_edt(~src, return_indices=True)
        block = out[sl]                                     # a view: writes go to ``out``
        block[need] = block[iy[need], ix[need]]
    return out


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    kind: str  # displacement | velocity | strain | angle

    @property
    def is_strain(self) -> bool:
        return self.kind in ("strain", "angle")


DISPLACEMENT_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("u", "u (horizontal displacement)", "displacement"),
    FieldSpec("v", "v (vertical displacement)", "displacement"),
    FieldSpec("magnitude", "|u| (displacement magnitude)", "displacement"),
)
VELOCITY_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("vel_u", "du/dt (horizontal velocity)", "velocity"),
    FieldSpec("vel_v", "dv/dt (vertical velocity)", "velocity"),
    FieldSpec("speed", "|du/dt| (speed)", "velocity"),
)
QUALITY_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("residual", "intensity residual (tracking quality)", "quality"),
)
STRAIN_FIELDS: tuple[FieldSpec, ...] = tuple(
    FieldSpec(k, COMPONENT_LABELS[k], "angle" if k in ("rotation", "principal_angle") else "strain") for k in COMPONENTS
)
ALL_FIELDS: dict[str, FieldSpec] = {f.key: f for f in DISPLACEMENT_FIELDS + VELOCITY_FIELDS + QUALITY_FIELDS + STRAIN_FIELDS}


@dataclass(frozen=True)
class Units:
    """Display units: ``pixel_size`` length units per px, ``frame_rate`` frames per second."""

    pixel_size: float = 1.0
    length_unit: str = "px"
    frame_rate: float | None = None

    @property
    def physical(self) -> bool:
        return self.length_unit != "px"

    @property
    def scale(self) -> float:
        """Length units per px actually applied (1 when the unit is px; the calibration is kept)."""
        return 1.0 if self.length_unit == "px" else self.pixel_size

    def factor(self, spec: FieldSpec) -> float:
        if spec.kind == "displacement":
            return self.scale
        if spec.kind == "velocity":
            return self.scale * (self.frame_rate or 1.0)
        return 1.0

    def unit(self, spec: FieldSpec) -> str:
        if spec.kind == "displacement":
            return self.length_unit
        if spec.kind == "velocity":
            return f"{self.length_unit}/s" if self.frame_rate else f"{self.length_unit}/frame"
        if spec.kind == "angle":
            return "deg"
        if spec.kind == "quality":
            return "grey levels"
        return "-"


@dataclass(frozen=True)
class Placement:
    """Where an array goes in the image: pixel ``(x0, y0)``, one array cell = ``step`` px."""

    x0: float
    y0: float
    step: int = 1


class FieldProvider:
    """Values of any field for any frame, in either configuration, with small LRU caches."""

    def __init__(self, series: DisplacementSeries, strain: StrainSeries | None = None, cache_size: int = 16) -> None:
        self.series = series
        self.strain = strain
        self._cache_size = cache_size
        self._inv: OrderedDict[int, tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]] = OrderedDict()
        self._def: OrderedDict[tuple[str, int, bool], np.ndarray] = OrderedDict()

    def with_strain(self, strain: StrainSeries | None) -> "FieldProvider":
        out = FieldProvider(self.series, strain, self._cache_size)
        out._inv = self._inv  # the inverse maps do not depend on strain
        return out

    def available(self) -> list[FieldSpec]:
        out = list(DISPLACEMENT_FIELDS) + list(VELOCITY_FIELDS)
        if self.series.residual is not None:
            out += list(QUALITY_FIELDS)
        if self.strain is not None:
            out += [f for f in STRAIN_FIELDS if f.key in self.strain.components]
        return out

    def has_frame(self, key: str, frame: int) -> bool:
        if ALL_FIELDS[key].is_strain:
            return self.strain is not None and bool(self.strain.done[frame])
        return bool(self.series.done[frame])

    # -- reference configuration ------------------------------------------------------------------
    def values(self, key: str, frame: int, fill_trimmed: bool = False) -> tuple[np.ndarray, Placement]:
        """Values of a field in the reference configuration.

        *fill_trimmed* fills the edge-trimmed band of a strain field from its reliable points
        (pyALDIC's "Fill trimmed edges"), for pictures only; it changes nothing else.
        """
        s = self.series
        g = s.grid
        if key == "u":
            return s.u[frame], Placement(g.x0, g.y0)
        if key == "v":
            return s.v[frame], Placement(g.x0, g.y0)
        if key == "magnitude":
            return np.hypot(s.u[frame], s.v[frame]), Placement(g.x0, g.y0)
        if key == "residual":
            if s.residual is None:
                raise KeyError("this result has no residual")
            return s.residual[frame], Placement(g.x0, g.y0)
        if key in ("vel_u", "vel_v", "speed"):
            du, dv = self._velocity(frame)
            val = du if key == "vel_u" else dv if key == "vel_v" else np.hypot(du, dv)
            return val, Placement(g.x0, g.y0)
        if self.strain is None or key not in self.strain.components:
            raise KeyError(f"field {key!r} is not available")
        vals = self.strain.components[key][frame]
        trimmed = self.strain.trimmed_in(frame) if fill_trimmed else None
        if trimmed is not None:
            vals = fill_trimmed_band(vals, trimmed)
        return vals, Placement(g.x0, g.y0, self.strain.step)

    def trimmed_band(self, key: str, frame: int) -> np.ndarray | None:
        """Edge-trimmed points of strain field *key* on *frame*, on the field's own grid.

        None for a displacement field or when nothing is trimmed; :func:`fill_untracked` leaves
        these points to "Fill trimmed edges".
        """
        if not ALL_FIELDS[key].is_strain or self.strain is None:
            return None
        return self.strain.trimmed_in(frame)

    def _velocity(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.series
        n = s.n_frames
        prev = next((k for k in range(frame - 1, -1, -1) if s.done[k]), None)
        nxt = next((k for k in range(frame + 1, n) if s.done[k]), None)
        a, b = (prev, nxt) if prev is not None and nxt is not None else (
            (frame, nxt) if nxt is not None else (prev, frame) if prev is not None else (None, None))
        if a is None or b is None or a == b:
            nan = np.full(s.grid.shape, np.nan, dtype=np.float32)
            return nan, nan
        dt = float(b - a)
        return (s.u[b] - s.u[a]) / dt, (s.v[b] - s.v[a]) / dt

    def series_values(self, key: str) -> np.ndarray:
        """``(T, h, w)`` values of a field for every frame (NaN for frames not computed)."""
        frames = [self.values(key, k)[0] if self.has_frame(key, k) else None for k in range(self.series.n_frames)]
        shape = next((f.shape for f in frames if f is not None), self.series.grid.shape)
        return np.stack([f if f is not None else np.full(shape, np.nan, np.float32) for f in frames])

    # -- deformed configuration -------------------------------------------------------------------
    def inverse(self, frame: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        hit = self._inv.get(frame)
        if hit is not None:
            self._inv.move_to_end(frame)
            return hit
        s = self.series
        inv = inverse_map(s.field(frame), s.grid, s.mask)
        self._inv[frame] = inv
        while len(self._inv) > self._cache_size:
            self._inv.popitem(last=False)
        return inv

    def deformed(self, key: str, frame: int, fill_trimmed: bool = False) -> tuple[np.ndarray, Placement]:
        """Values over the deformed frame's window (NaN where no material point lands)."""
        ck = (key, frame, bool(fill_trimmed))
        X, Y, (x0, y0, _x1, _y1) = self.inverse(frame)
        hit = self._def.get(ck)
        if hit is None:
            vals, pl = self.values(key, frame, fill_trimmed=fill_trimmed)
            if X.size == 0:
                hit = np.full((0, 0), np.nan, dtype=np.float32)
            else:
                xs = (X - pl.x0) / pl.step
                ys = (Y - pl.y0) / pl.step
                hit = sample_bilinear(vals, xs, ys).astype(np.float32)
            self._def[ck] = hit
            while len(self._def) > self._cache_size * 4:
                self._def.popitem(last=False)
        else:
            self._def.move_to_end(ck)
        return hit, Placement(x0, y0)

    def deformed_mask(self, frame: int) -> tuple[np.ndarray, Placement]:
        X, _Y, (x0, y0, _x1, _y1) = self.inverse(frame)
        return np.isfinite(X), Placement(x0, y0)

    def clear(self) -> None:
        self._inv.clear()
        self._def.clear()
