"""Read probes across a finished run of dense fields.

RAFTcorr's results are dense: every pixel of the frame-0 ROI's bounding box
(``DisplacementSeries.grid``) carries a displacement, and strain sits on the
same box subsampled by ``StrainSeries.step``. Field values come from
``raftcorr.viz.fields.FieldProvider`` in the reference configuration, so a
probe reads exactly what the field views draw.

A probe's samples are placed once per field grid and reused for every frame.
A point or a line is read by NaN-aware bilinear interpolation
(``raftcorr.engine.mapping.sample_bilinear``); a region reads every grid cell
inside it, which weights its statistics by area since the cells are evenly
spaced. Reading a frame touches only the small window of the field the
samples need, so a long sequence stored as memory maps is never loaded whole.

What makes a sample invalid, and why
------------------------------------
The nearest displacement pixel of a sample decides (pixel ``[row, col]``
covers ``x in [col - 0.5, col + 0.5)``, as for ROI shapes).

``OUTSIDE``
    Never material: off the result grid, or on a pixel outside the frame-0
    ROI (a hole or a notch present from the start). Excluded from the valid
    fraction -- nothing was lost there.
``UNTRACKED``
    Material at frame 0 without a displacement on this frame: excluded by the
    frame's own ROI (a crack or a hole cut out of the deformed image), out of
    the field of view, or dropped by the run. The run writes NaN there, so the
    loss is found in frame-0 coordinates without any extra mask.
``UNRELIABLE``
    Tracked, but without a value: a strain field is NaN where its virtual
    strain gauge held too few valid points or its fit was ill-conditioned.
    Displacement is never trimmed.

A frame's reduced value uses the valid samples. Its ``FrameStatus`` says why
the frame is not plain OK, independently of whether a value survived: a line
across a lost region keeps a value from the material either side and is
marked ``UNTRACKED``; it is not blanked.

Units
-----
Everything is computed in pixels. Only quantities that are lengths --
displacement fields, elongation, crack opening -- are multiplied by
``pixel_size`` on the way out. Gauge strain is a ratio of two pixel lengths and
never sees the pixel size; strains and angles are never scaled.

Frames
------
Frame ``k`` is image ``k`` of the sequence, 0 the reference (zero
displacement), as in ``raftcorr.results``. Values are in image axes: ``v``
positive down.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from raftcorr.analysis.geometry_sampling import area_cells, area_centroid
from raftcorr.analysis.probes import (
    GAUGE_QUANTITIES,
    SPATIAL_STATISTICS,
    AreaGeom,
    Geometry,
    LineGeom,
    PointGeom,
    Probe,
    allowed_reductions,
)
from raftcorr.analysis.series import FrameStatus, TimeSeries
from raftcorr.engine.mapping import sample_bilinear
from raftcorr.gui.support.fields import (
    field_unit,
    is_strain_field,
    provider_key,
    validate_field,
)
from raftcorr.results import DisplacementSeries, StrainSeries
from raftcorr.viz.fields import FieldProvider

#: Samples per grid cell along a line. Finer than the grid invents resolution
#: the data does not have.
SAMPLES_PER_STEP = 1.0
_MIN_LINE_SAMPLES = 2
_MAX_LINE_SAMPLES = 4096

_LENGTH_GAUGES = frozenset({"elongation", "cod", "cod_sliding", "cod_magnitude"})

DEFAULT_MIN_VALID_FRACTION = 0.5

_EMPTY_WINDOW = (0, 0, 0, 0)


class SampleStatus(enum.IntEnum):
    """Verdict on one sample on one frame. Stored as ``uint8`` arrays."""

    VALID = 0
    OUTSIDE = 1
    UNTRACKED = 2
    UNRELIABLE = 3


# ---------------------------------------------------------------------------
# Plans and results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldGrid:
    """Where one field's array sits: cell ``(i, j)`` at image pixel
    ``(x0 + j * step, y0 + i * step)``, ``height`` x ``width`` cells."""

    x0: float
    y0: float
    step: int
    height: int
    width: int


@dataclass(frozen=True, eq=False)
class SamplePlan:
    """Interpolated samples on one field grid -- a point, a line, or a region
    too small to hold a cell -- fixed for the whole run.

    ``window`` (``r0, r1, c0, c1``, half-open) is the part of the field grid
    the samples need; they sit at fractional window coordinates ``fx``,
    ``fy``. ``di``, ``dj`` locate each sample's nearest pixel inside
    ``disp_window`` of the displacement grid, which says whether the sample is
    material and whether it is tracked.
    """

    xy: NDArray[np.float64]
    window: tuple[int, int, int, int]
    fx: NDArray[np.float64]
    fy: NDArray[np.float64]
    disp_window: tuple[int, int, int, int]
    di: NDArray[np.int64]
    dj: NDArray[np.int64]
    material: NDArray[np.bool_]
    distance: NDArray[np.float64] | None = None

    @property
    def size(self) -> int:
        return int(self.xy.shape[0])

    def gather(self, window_values: NDArray) -> NDArray[np.float64]:
        """The samples' values from the field over ``window``."""
        return np.asarray(sample_bilinear(window_values, self.fx, self.fy), dtype=np.float64)

    def pick(self, disp_values: NDArray) -> NDArray:
        """Each sample's nearest pixel from an array over ``disp_window``."""
        return disp_values[self.di, self.dj]


@dataclass(frozen=True, eq=False)
class CellPlan:
    """A region: every cell of one field grid inside it, ``cells`` over
    ``window`` -- a byte per cell, however large the region.

    ``disp_offset`` (rows, cols) and the grid's step place cell ``(i, j)`` of
    the window on displacement pixel ``(i * step, j * step)`` of
    ``disp_window``.
    """

    grid: FieldGrid
    window: tuple[int, int, int, int]
    cells: NDArray[np.bool_]
    disp_window: tuple[int, int, int, int]
    material: NDArray[np.bool_]
    distance: None = None

    @property
    def size(self) -> int:
        return int(self.material.size)

    @property
    def xy(self) -> NDArray[np.float64]:
        """Image positions of the cells, computed on demand."""
        rows, cols = np.nonzero(self.cells)
        g, (r0, _r1, c0, _c1) = self.grid, self.window
        return np.column_stack([g.x0 + (cols + c0) * float(g.step), g.y0 + (rows + r0) * float(g.step)])

    def gather(self, window_values: NDArray) -> NDArray[np.float64]:
        return np.asarray(window_values[self.cells], dtype=np.float64)

    def pick(self, disp_values: NDArray) -> NDArray:
        s = self.grid.step
        return disp_values[::s, ::s][self.cells]


Plan = SamplePlan | CellPlan


@dataclass(frozen=True)
class Samples:
    """One frame of one plan: values in pixel units, NaN unless VALID."""

    values: NDArray[np.float64]
    status: NDArray[np.uint8]

    @property
    def valid(self) -> NDArray[np.bool_]:
        return self.status == SampleStatus.VALID

    @property
    def material(self) -> NDArray[np.bool_]:
        return self.status != SampleStatus.OUTSIDE

    @property
    def valid_fraction(self) -> float:
        n = int(self.material.sum())
        return float(self.valid.sum()) / n if n else 0.0

    @property
    def untracked(self) -> bool:
        return bool((self.status == SampleStatus.UNTRACKED).any())


@dataclass(frozen=True)
class Profile:
    """A field along a line on one frame, against reference arc length."""

    distance: NDArray[np.float64]
    xy: NDArray[np.float64]
    values: NDArray[np.float64]
    status: NDArray[np.uint8]
    unit: str
    distance_unit: str


@dataclass(frozen=True)
class Kymograph:
    """A field along a line on every frame: ``values[frame, sample]``.

    ``xy`` holds each sample's reference position in image pixels, in the
    order of ``distance``.
    """

    frames: NDArray[np.int64]
    distance: NDArray[np.float64]
    xy: NDArray[np.float64]
    values: NDArray[np.float64]
    status: NDArray[np.uint8]
    unit: str
    distance_unit: str


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AnalysisEngine:
    """Samples one run. Build once per result; every read reuses its plans.

    Parameters
    ----------
    series:
        The displacement of every frame relative to frame 0.
    strain:
        The strain computed from it, if any.
    """

    def __init__(self, series: DisplacementSeries, strain: StrainSeries | None = None) -> None:
        self._series = series
        self._strain = strain
        self._provider = FieldProvider(series, strain)
        g = series.grid
        self._disp_grid = FieldGrid(float(g.x0), float(g.y0), 1, g.height, g.width)
        self._strain_grid: FieldGrid | None = None
        if strain is not None and strain.components:
            hs, ws = next(iter(strain.components.values())).shape[1:]
            self._strain_grid = FieldGrid(float(strain.grid.x0), float(strain.grid.y0),
                                          int(strain.step), int(hs), int(ws))
        self._plans: dict[tuple, Plan] = {}

    # -- run facts ----------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Frames a probe can report on, the reference frame included."""
        return self._series.n_frames

    def has_strain(self) -> bool:
        return self._strain_grid is not None

    def grid_for(self, field: str) -> FieldGrid | None:
        """The grid *field* is stored on; None for strain not computed."""
        return self._strain_grid if is_strain_field(field) else self._disp_grid

    # -- plans ----------------------------------------------------------------

    def plan(self, geometry: Geometry, field: str = "disp_u") -> Plan:
        """Where *geometry* reads *field*; raises ValueError without that field."""
        grid = self.grid_for(field)
        if grid is None:
            raise ValueError(f"{field!r} has not been computed for this run.")
        return self._plan_on(geometry, grid)

    def _plan_on(self, geometry: Geometry, grid: FieldGrid) -> Plan:
        key = (geometry, grid)
        cached = self._plans.get(key)
        if cached is None:
            cached = self._build_plan(geometry, grid)
            self._plans[key] = cached
        return cached

    def _build_plan(self, geometry: Geometry, grid: FieldGrid) -> Plan:
        if isinstance(geometry, PointGeom):
            return self._interp_plan(np.array([[geometry.x, geometry.y]], dtype=np.float64), grid)
        if isinstance(geometry, LineGeom):
            n = self.line_sample_count(geometry, grid)
            t = np.linspace(0.0, 1.0, n)
            xy = np.column_stack([
                geometry.x0 + t * (geometry.x1 - geometry.x0),
                geometry.y0 + t * (geometry.y1 - geometry.y0),
            ])
            return replace(self._interp_plan(xy, grid), distance=t * geometry.length())
        if isinstance(geometry, AreaGeom):
            found = area_cells(geometry, grid.x0, grid.y0, grid.step, (grid.height, grid.width))
            if found is None:
                # Smaller than a cell, or off the grid: read at its centre.
                cx, cy = area_centroid(geometry)
                return self._interp_plan(np.array([[cx, cy]], dtype=np.float64), grid)
            return self._cell_plan(*found, grid)
        raise TypeError(f"Unsupported geometry {type(geometry).__name__}.")

    @staticmethod
    def line_sample_count(geometry: LineGeom, grid: FieldGrid) -> int:
        n = int(round(geometry.length() / grid.step * SAMPLES_PER_STEP)) + 1
        return int(np.clip(n, _MIN_LINE_SAMPLES, _MAX_LINE_SAMPLES))

    def _interp_plan(self, xy: NDArray[np.float64], grid: FieldGrid) -> SamplePlan:
        fx = (xy[:, 0] - grid.x0) / grid.step
        fy = (xy[:, 1] - grid.y0) / grid.step
        r0 = max(int(np.floor(fy.min())), 0)
        r1 = min(int(np.floor(fy.max())) + 2, grid.height)
        c0 = max(int(np.floor(fx.min())), 0)
        c1 = min(int(np.floor(fx.max())) + 2, grid.width)
        window = (r0, r1, c0, c1) if r1 > r0 and c1 > c0 else _EMPTY_WINDOW
        disp_window, di, dj, material = self._lookup(xy)
        return SamplePlan(xy=xy, window=window, fx=fx - window[2], fy=fy - window[0],
                          disp_window=disp_window, di=di, dj=dj, material=material)

    def _cell_plan(self, window: tuple[int, int, int, int], cells: NDArray[np.bool_],
                   grid: FieldGrid) -> CellPlan:
        d = self._disp_grid
        # Cell (i, j) sits on displacement pixel (oy + i * s, ox + j * s).
        s = grid.step
        oy = int(round(grid.y0 - d.y0))
        ox = int(round(grid.x0 - d.x0))
        r0, r1, c0, c1 = window
        disp_window = (oy + r0 * s, oy + (r1 - 1) * s + 1, ox + c0 * s, ox + (c1 - 1) * s + 1)
        dr0, dr1, dc0, dc1 = disp_window
        plan = CellPlan(grid=grid, window=window, cells=cells, disp_window=disp_window,
                        material=np.zeros(0, dtype=bool))
        roi = np.asarray(self._series.mask[dr0:dr1, dc0:dc1], dtype=bool)
        return replace(plan, material=plan.pick(roi))

    def _lookup(self, xy: NDArray[np.float64]):
        """Nearest displacement pixel of each sample: window, indices, material."""
        g = self._disp_grid
        dj = np.floor(xy[:, 0] - g.x0 + 0.5).astype(np.int64)
        di = np.floor(xy[:, 1] - g.y0 + 0.5).astype(np.int64)
        on = (di >= 0) & (di < g.height) & (dj >= 0) & (dj < g.width)
        if not on.any():
            zeros = np.zeros(len(xy), dtype=np.int64)
            return _EMPTY_WINDOW, zeros, zeros, np.zeros(len(xy), dtype=bool)
        r0, r1 = int(di[on].min()), int(di[on].max()) + 1
        c0, c1 = int(dj[on].min()), int(dj[on].max()) + 1
        di_w = np.where(on, di - r0, 0)
        dj_w = np.where(on, dj - c0, 0)
        roi = np.asarray(self._series.mask[r0:r1, c0:c1], dtype=bool)
        material = on & roi[di_w, dj_w]
        return (r0, r1, c0, c1), di_w, dj_w, material

    # -- reading one frame --------------------------------------------------

    def _has(self, field: str, frame: int) -> bool:
        if not 0 <= frame < self.n_frames:
            return False
        key = provider_key(field)
        if is_strain_field(field):
            return (self._strain is not None and key in self._strain.components
                    and self._provider.has_frame(key, frame))
        return self._provider.has_frame(key, frame)

    def _tracked(self, plan: Plan, frame: int) -> NDArray[np.bool_]:
        """Samples whose nearest pixel has a displacement on *frame*."""
        r0, r1, c0, c1 = plan.disp_window
        if r1 <= r0 or c1 <= c0 or not self._has("disp_u", frame):
            return np.zeros(plan.size, dtype=bool)
        s = self._series
        u = np.asarray(s.u[frame][r0:r1, c0:c1])
        v = np.asarray(s.v[frame][r0:r1, c0:c1])
        alive = np.asarray(s.mask[r0:r1, c0:c1], dtype=bool) & np.isfinite(u) & np.isfinite(v)
        return plan.material & plan.pick(alive)

    def _window(self, key: str, frame: int, window: tuple[int, int, int, int]) -> NDArray:
        """*key* on *frame* over *window* of its grid, in the stored precision.

        Only the window is read, so a memory-mapped series stays on disk.
        """
        r0, r1, c0, c1 = window
        if key == "magnitude":
            # FieldProvider's definition, hypot(u, v) per pixel, on the window only.
            return np.hypot(self._window("u", frame, window), self._window("v", frame, window))
        arr, _placement = self._provider.values(key, frame)
        return np.asarray(arr[r0:r1, c0:c1])

    def _values(self, plan: Plan, field: str, frame: int) -> NDArray[np.float64]:
        r0, r1, c0, c1 = plan.window
        if r1 <= r0 or c1 <= c0:
            return np.full(plan.size, np.nan)
        return plan.gather(self._window(provider_key(field), frame, plan.window))

    def sample(self, plan: Plan, field: str, frame: int) -> Samples:
        """Read *plan* on *frame*; every sample OUTSIDE when the field is absent."""
        got = self._read(plan, field, frame)
        if got is None:
            return Samples(np.full(plan.size, np.nan),
                           np.full(plan.size, SampleStatus.OUTSIDE, dtype=np.uint8))
        return got

    def _read(self, plan: Plan, field: str, frame: int) -> Samples | None:
        if not self._has(field, frame):
            return None
        vals = self._values(plan, field, frame)
        tracked = self._tracked(plan, frame)
        finite = np.isfinite(vals)
        status = np.full(plan.size, SampleStatus.OUTSIDE, dtype=np.uint8)
        status[plan.material] = SampleStatus.UNTRACKED
        status[tracked] = SampleStatus.UNRELIABLE
        ok = tracked & finite
        status[ok] = SampleStatus.VALID
        return Samples(np.where(ok, vals, np.nan), status)

    def node_field(self, field: str, frame: int) -> tuple[NDArray[np.float64], NDArray[np.bool_]] | None:
        """The whole field on its grid in pixel units, with its validity mask
        (ROI, tracked, finite); None when *field* does not exist for *frame*."""
        validate_field(field)
        grid = self.grid_for(field)
        if grid is None or not self._has(field, frame):
            return None
        window = (0, grid.height, 0, grid.width)
        vals = self._window(provider_key(field), frame, window)
        s = self._series
        alive = s.mask & np.isfinite(s.u[frame]) & np.isfinite(s.v[frame])
        alive = alive[::grid.step, ::grid.step][:grid.height, :grid.width]
        return vals, alive & np.isfinite(vals)

    # -- capabilities -----------------------------------------------------

    @staticmethod
    def supports(probe: Probe, statistic: str) -> bool:
        """Whether *probe* produces *statistic* (and so joins that chart)."""
        return statistic in allowed_reductions(probe.kind)

    # -- series -------------------------------------------------------------

    def series(
        self,
        probe: Probe,
        field: str | None,
        statistic: str,
        *,
        pixel_size: float = 1.0,
        length_unit: str = "px",
        min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
        frames: Sequence[int] | None = None,
    ) -> TimeSeries:
        """One curve: *probe* reduced to one number per frame.

        ``field`` is ignored for gauge quantities.
        """
        if not self.supports(probe, statistic):
            raise ValueError(
                f"{statistic!r} does not apply to a {probe.kind} probe."
            )
        idxs = list(frames) if frames is not None else list(range(self.n_frames))
        if statistic in GAUGE_QUANTITIES:
            return self._gauge_series(probe, statistic, idxs, pixel_size, length_unit)
        if field is None:
            raise ValueError(f"{statistic!r} needs a field.")
        validate_field(field)
        scale = 1.0 if is_strain_field(field) else float(pixel_size)
        unit = "" if statistic == "valid_fraction" else field_unit(field, length_unit)

        values = np.full(len(idxs), np.nan, dtype=np.float64)
        fractions = np.zeros(len(idxs), dtype=np.float64)
        status: list[FrameStatus] = []
        grid = self.grid_for(field)
        plan = None if grid is None else self._plan_on(probe.geometry, grid)
        for i, frame in enumerate(idxs):
            s = None if plan is None else self._read(plan, field, frame)
            if s is None:
                status.append(FrameStatus.NOT_COMPUTED)
                continue
            fractions[i] = s.valid_fraction
            lost = s.untracked
            if not s.material.any():
                status.append(FrameStatus.NO_DATA)
                continue
            if statistic == "valid_fraction":
                # A measure of the data, not a measurement made from it: the
                # threshold must not suppress it, and an empty frame is 0.
                values[i] = s.valid_fraction
                status.append(FrameStatus.UNTRACKED if lost else FrameStatus.OK)
                continue
            if not s.valid.any():
                if lost:
                    status.append(FrameStatus.UNTRACKED)
                elif (s.status == SampleStatus.UNRELIABLE).any():
                    status.append(FrameStatus.UNRELIABLE)
                else:
                    status.append(FrameStatus.NO_DATA)
                continue
            if probe.kind != "point" and s.valid_fraction < min_valid_fraction:
                status.append(FrameStatus.BELOW_THRESHOLD)
                continue
            values[i] = _reduce(s.values[s.valid], statistic) * scale
            status.append(FrameStatus.UNTRACKED if lost else FrameStatus.OK)

        return TimeSeries(
            frames=np.asarray(idxs, dtype=np.int64),
            values=values,
            valid_fraction=fractions,
            status=status,
            unit=unit,
        )

    def _gauge_series(
        self,
        probe: Probe,
        quantity: str,
        idxs: Sequence[int],
        pixel_size: float,
        length_unit: str,
    ) -> TimeSeries:
        geom: LineGeom = probe.geometry  # type: ignore[assignment]
        ends = self._interp_plan(
            np.array([[geom.x0, geom.y0], [geom.x1, geom.y1]], dtype=np.float64), self._disp_grid)
        path = self._plan_on(geom, self._disp_grid)
        l0 = geom.length()
        ex, ey = geom.direction()
        unit = length_unit if quantity in _LENGTH_GAUGES else ""
        scale = float(pixel_size) if quantity in _LENGTH_GAUGES else 1.0

        values = np.full(len(idxs), np.nan, dtype=np.float64)
        fractions = np.zeros(len(idxs), dtype=np.float64)
        status: list[FrameStatus] = []
        for i, frame in enumerate(idxs):
            su = self._read(ends, "disp_u", frame)
            sv = self._read(ends, "disp_v", frame)
            if su is None or sv is None:
                status.append(FrameStatus.NOT_COMPUTED)
                continue
            if not (su.valid.all() and sv.valid.all()):
                status.append(FrameStatus.ENDPOINT_LOST)
                continue
            crossing = self.sample(path, "disp_u", frame).untracked
            (du0, du1), (dv0, dv1) = su.values, sv.values
            fractions[i] = 1.0
            jx, jy = du1 - du0, dv1 - dv0
            if quantity in ("strain", "true_strain", "elongation"):
                length = math.hypot(
                    (geom.x1 + du1) - (geom.x0 + du0),
                    (geom.y1 + dv1) - (geom.y0 + dv0),
                )
                if quantity == "strain":
                    val = (length - l0) / l0
                elif quantity == "true_strain":
                    val = math.log(length / l0) if length > 0 else math.nan
                else:
                    val = length - l0
            elif quantity == "cod":
                val = jx * ex + jy * ey
            elif quantity == "cod_sliding":
                val = jx * (-ey) + jy * ex
            else:
                val = math.hypot(jx, jy)
            values[i] = val * scale
            status.append(FrameStatus.UNTRACKED if crossing else FrameStatus.OK)

        return TimeSeries(
            frames=np.asarray(idxs, dtype=np.int64),
            values=values,
            valid_fraction=fractions,
            status=status,
            unit=unit,
        )

    # -- line views -----------------------------------------------------------

    def profile(
        self,
        probe: Probe,
        field: str,
        frame: int,
        *,
        pixel_size: float = 1.0,
        length_unit: str = "px",
    ) -> Profile:
        k = self.kymograph(probe, field, frames=[frame], pixel_size=pixel_size,
                           length_unit=length_unit)
        return Profile(
            distance=k.distance, xy=k.xy, values=k.values[0], status=k.status[0],
            unit=k.unit, distance_unit=k.distance_unit,
        )

    def kymograph(
        self,
        probe: Probe,
        field: str,
        *,
        frames: Sequence[int] | None = None,
        pixel_size: float = 1.0,
        length_unit: str = "px",
    ) -> Kymograph:
        if probe.kind != "line":
            raise ValueError("Profiles and kymographs need a line probe.")
        validate_field(field)
        grid = self.grid_for(field) or self._disp_grid
        plan = self._plan_on(probe.geometry, grid)
        idxs = list(frames) if frames is not None else list(range(self.n_frames))
        scale = 1.0 if is_strain_field(field) else float(pixel_size)
        values = np.full((len(idxs), plan.size), np.nan, dtype=np.float64)
        status = np.full((len(idxs), plan.size), SampleStatus.OUTSIDE, dtype=np.uint8)
        for i, frame in enumerate(idxs):
            s = self.sample(plan, field, frame)
            values[i] = s.values * scale
            status[i] = s.status
        return Kymograph(
            frames=np.asarray(idxs, dtype=np.int64),
            distance=plan.distance * float(pixel_size),
            xy=plan.xy,
            values=values,
            status=status,
            unit=field_unit(field, length_unit),
            distance_unit=length_unit,
        )


def _reduce(good: NDArray[np.float64], statistic: str) -> float:
    if not good.size:
        return math.nan
    if statistic == "value":
        return float(good[0])
    if statistic == "mean":
        return float(np.mean(good))
    if statistic == "median":
        return float(np.median(good))
    if statistic == "std":
        return float(np.std(good))
    if statistic == "max":
        return float(np.max(good))
    if statistic == "min":
        return float(np.min(good))
    raise ValueError(f"Unknown statistic {statistic!r}.")


__all__ = [
    "AnalysisEngine", "CellPlan", "DEFAULT_MIN_VALID_FRACTION", "FieldGrid", "GAUGE_QUANTITIES",
    "Kymograph", "Profile", "SPATIAL_STATISTICS", "SamplePlan", "SampleStatus",
    "Samples",
]
