"""Probe geometry and identity.

A probe is a shape placed on the reference image plus the metadata needed to
show it and tell it apart. It carries no results: reading a probe -- which
samples are valid, and the series they give -- is
``raftcorr.analysis.engine``; editing one on screen is
``raftcorr.analysis.geometry_edit``.

Coordinates are frame-0 (reference) image pixels, x = column, y = row, with
the centre of the top-left pixel at (0, 0): pixel ``[row, col]`` covers
``x in [col - 0.5, col + 0.5)``. That is the system of
``raftcorr.engine.mapping.Grid`` and of ``raftcorr.roi.shapes``, so no probe
ever needs a coordinate transform to be read. Geometry stays in pixels even
when the project displays physical units; the conversion happens at read-out,
so changing the unit setting cannot invalidate a probe someone already saved.

Probes are frozen. Editing goes through ``dataclasses.replace`` and
``ProbeSet.replace``.

On disk (``probe_to_dict``) a probe uses the layout of
``raftcorr.post.probes.Probe.to_dict`` -- ``kind`` one of point, line, rect,
circle, polygon and its defining ``points`` -- plus ``visible``, so a project
file is read by both RAFTcorr interfaces.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Literal, Sequence

ProbeKind = Literal["point", "line", "area"]
AreaShape = Literal["rect", "circle", "polygon"]

#: Statistics over a probe's samples on one frame, in display order.
SPATIAL_STATISTICS: tuple[str, ...] = (
    "mean", "median", "max", "min", "std", "valid_fraction",
)
#: Quantities from a line's two endpoints, in display order. The field does
#: not apply: engineering and true (log) strain, elongation, and the crack
#: opening's components along and across the gauge and its magnitude.
GAUGE_QUANTITIES: tuple[str, ...] = (
    "strain", "true_strain", "elongation",
    "cod", "cod_sliding", "cod_magnitude",
)
#: What a single point stands in for: a one-sample region. Not ``std`` or
#: ``valid_fraction``, which say nothing about one sample.
POINT_STATISTICS: tuple[str, ...] = ("value", "mean", "median", "max", "min")

_COLOUR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")

#: Assigned in order to new probes. Chosen to stay distinguishable on the dark
#: theme and, as far as eight colours can, under common colour-vision
#: deficiencies.
DEFAULT_COLOURS: tuple[str, ...] = (
    "#ef4444", "#22c55e", "#3b82f6", "#eab308",
    "#a855f7", "#06b6d4", "#f97316", "#ec4899",
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PointGeom:
    """A single location."""

    x: float
    y: float


@dataclass(frozen=True)
class LineGeom:
    """A straight gauge between two endpoints. Not a polyline."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.length() <= 0.0:
            raise ValueError(
                "Line probe has zero length; its two endpoints coincide."
            )

    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    def direction(self) -> tuple[float, float]:
        """Unit vector from the first endpoint to the second."""
        n = self.length()
        return ((self.x1 - self.x0) / n, (self.y1 - self.y0) / n)


@dataclass(frozen=True)
class AreaGeom:
    """A closed region: an axis-aligned rectangle, a circle or a polygon.

    ``shape`` is spelled ``"polygon"`` everywhere; ``"poly"`` is accepted
    only when reading a file.
    """

    shape: AreaShape
    data: tuple[float, ...] | tuple[tuple[float, float], ...]

    @staticmethod
    def rect(x0: float, y0: float, x1: float, y1: float) -> "AreaGeom":
        """Normalised so a rectangle dragged in any direction is the same one."""
        lo_x, hi_x = sorted((float(x0), float(x1)))
        lo_y, hi_y = sorted((float(y0), float(y1)))
        if hi_x <= lo_x or hi_y <= lo_y:
            raise ValueError("Rectangle probe has zero width or height.")
        return AreaGeom(shape="rect", data=(lo_x, lo_y, hi_x, hi_y))

    @staticmethod
    def circle(cx: float, cy: float, radius: float) -> "AreaGeom":
        if radius <= 0.0:
            raise ValueError("Circle probe radius must be positive.")
        return AreaGeom(shape="circle", data=(float(cx), float(cy), float(radius)))

    @staticmethod
    def polygon(vertices: Sequence[tuple[float, float]]) -> "AreaGeom":
        if len(vertices) < 3:
            raise ValueError("Polygon probe needs at least three vertices.")
        pts = tuple((float(x), float(y)) for x, y in vertices)
        n = len(pts)
        twice_area = sum(pts[i][0] * pts[(i + 1) % n][1]
                         - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))
        if abs(twice_area) < 1e-9:
            raise ValueError(
                "Polygon probe encloses no area; its vertices lie on one line."
            )
        return AreaGeom(shape="polygon", data=pts)


Geometry = PointGeom | LineGeom | AreaGeom

_GEOM_FOR_KIND: dict[str, type] = {
    "point": PointGeom,
    "line": LineGeom,
    "area": AreaGeom,
}


def allowed_reductions(kind: ProbeKind) -> frozenset[str]:
    """Reductions that mean something for *kind*.

    Asking for one that does not is an error rather than something to ignore
    silently.
    """
    if kind == "point":
        # A point is a one-sample region, so it can join a chart of region or
        # line means -- the same quantity on the same axis.
        return frozenset(POINT_STATISTICS)
    if kind == "line":
        return frozenset(SPATIAL_STATISTICS) | frozenset(GAUGE_QUANTITIES)
    if kind == "area":
        return frozenset(SPATIAL_STATISTICS)
    raise ValueError(f"Unknown probe kind {kind!r}.")


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Probe:
    """A placed probe: geometry plus the metadata that identifies it."""

    id: int
    kind: ProbeKind
    geometry: Geometry
    label: str
    color: str
    visible: bool = True

    def __post_init__(self) -> None:
        expected = _GEOM_FOR_KIND.get(self.kind)
        if expected is None:
            raise ValueError(f"Unknown probe kind {self.kind!r}.")
        if not isinstance(self.geometry, expected):
            raise ValueError(
                f"Probe kind {self.kind!r} needs {expected.__name__}, "
                f"got {type(self.geometry).__name__}."
            )
        if not _COLOUR_RE.match(self.color):
            raise ValueError(
                f"Probe color must be '#RRGGBB', got {self.color!r}."
            )


def _file_kind_and_points(geometry: Geometry) -> tuple[str, list[list[float]]]:
    """The on-disk kind and defining points of *geometry*."""
    if isinstance(geometry, PointGeom):
        return "point", [[float(geometry.x), float(geometry.y)]]
    if isinstance(geometry, LineGeom):
        return "line", [[float(geometry.x0), float(geometry.y0)],
                        [float(geometry.x1), float(geometry.y1)]]
    if geometry.shape == "rect":
        x0, y0, x1, y1 = geometry.data  # type: ignore[misc]
        return "rect", [[float(x0), float(y0)], [float(x1), float(y1)]]
    if geometry.shape == "circle":
        cx, cy, r = geometry.data  # type: ignore[misc]
        # Centre and a point on the edge, as the other interface stores it.
        return "circle", [[float(cx), float(cy)], [float(cx + r), float(cy)]]
    return "polygon", [[float(x), float(y)] for x, y in geometry.data]  # type: ignore[misc]


def _geometry_from_file(kind: str, points: Any) -> tuple[ProbeKind, Geometry]:
    if not isinstance(points, (list, tuple)) or not points:
        raise ValueError(f"A {kind} probe needs a list of points, got {points!r}.")
    try:
        pts = [(float(p[0]), float(p[1])) for p in points]
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Unreadable probe points {points!r}: {exc}") from None
    need = {"point": 1, "line": 2, "rect": 2, "circle": 2}.get(kind)
    if need is not None and len(pts) != need:
        raise ValueError(f"A {kind} probe needs {need} points, got {len(pts)}.")
    if kind == "point":
        return "point", PointGeom(*pts[0])
    if kind == "line":
        (x0, y0), (x1, y1) = pts
        return "line", LineGeom(x0, y0, x1, y1)
    if kind == "rect":
        (x0, y0), (x1, y1) = pts
        return "area", AreaGeom.rect(x0, y0, x1, y1)
    if kind == "circle":
        (cx, cy), (ex, ey) = pts
        return "area", AreaGeom.circle(cx, cy, math.hypot(ex - cx, ey - cy))
    if kind in ("polygon", "poly"):
        return "area", AreaGeom.polygon(pts)
    raise ValueError(
        f"Unknown probe kind {kind!r}; expected point, line, rect, circle or polygon."
    )


def probe_to_dict(probe: Probe) -> dict[str, Any]:
    """JSON-safe form, in the layout both RAFTcorr interfaces read."""
    kind, points = _file_kind_and_points(probe.geometry)
    return {
        "id": int(probe.id),
        "kind": kind,
        "points": points,
        "label": probe.label,
        "color": probe.color,
        "visible": bool(probe.visible),
    }


def probe_from_dict(d: dict[str, Any]) -> Probe:
    """Inverse of ``probe_to_dict``; ValueError for anything malformed."""
    if not isinstance(d, dict):
        raise ValueError(f"A probe must be an object, got {d!r}.")
    try:
        kind, geometry = _geometry_from_file(str(d["kind"]), d["points"])
        return Probe(
            id=int(d["id"]),
            kind=kind,
            geometry=geometry,
            label=str(d["label"]),
            color=str(d["color"]),
            visible=bool(d.get("visible", True)),
        )
    except KeyError as exc:
        raise ValueError(f"A probe entry lacks {exc}: {d!r}.") from None
    except TypeError as exc:
        raise ValueError(f"Unreadable probe entry {d!r}: {exc}") from None


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

@dataclass
class ProbeSet:
    """Ordered probes with unique ids.

    Ids are unique across every kind and are never reused, so a probe can be
    deleted, referenced or coloured by id alone. Default names (P1, P2, ...)
    count the probes added since the set was last empty: deleting every probe
    starts them again at P1 while the ids keep counting.
    """

    _probes: list[Probe] = field(default_factory=list)
    _next_id: int = 1
    _next_number: int = 1

    def __iter__(self) -> Iterator[Probe]:
        return iter(self._probes)

    def __len__(self) -> int:
        return len(self._probes)

    def get(self, probe_id: int) -> Probe:
        for p in self._probes:
            if p.id == probe_id:
                return p
        raise KeyError(f"No probe with id {probe_id}.")

    def of_kind(self, kind: ProbeKind) -> list[Probe]:
        return [p for p in self._probes if p.kind == kind]

    def add(
        self,
        kind: ProbeKind,
        geometry: Geometry,
        *,
        label: str | None = None,
        color: str | None = None,
    ) -> Probe:
        probe_id = self._next_id
        number = self._next_number
        probe = Probe(
            id=probe_id,
            kind=kind,
            geometry=geometry,
            label=label if label else f"P{number}",
            color=color or self._free_colour(probe_id),
        )
        # Counters move only once the probe is known to be valid.
        self._next_id += 1
        self._next_number += 1
        self._probes.append(probe)
        return probe

    def _free_colour(self, probe_id: int) -> str:
        """First palette colour no current probe uses; cycle once all are.

        Following the id counter instead made colours repeat on one chart as
        soon as probes had been deleted.
        """
        used = {p.color.lower() for p in self._probes}
        for colour in DEFAULT_COLOURS:
            if colour.lower() not in used:
                return colour
        return DEFAULT_COLOURS[(probe_id - 1) % len(DEFAULT_COLOURS)]

    def replace(self, probe: Probe) -> None:
        """Swap the probe with the same id for *probe*, keeping its position."""
        for i, existing in enumerate(self._probes):
            if existing.id == probe.id:
                self._probes[i] = probe
                return
        raise KeyError(f"No probe with id {probe.id}.")

    def remove(self, probe_id: int) -> None:
        for i, existing in enumerate(self._probes):
            if existing.id == probe_id:
                del self._probes[i]
                if not self._probes:
                    self._next_number = 1
                return
        raise KeyError(f"No probe with id {probe_id}.")

    def clear(self) -> None:
        self._probes.clear()
        self._next_number = 1

    # -- persistence ------------------------------------------------------

    def to_list(self) -> list[dict[str, Any]]:
        return [probe_to_dict(p) for p in self._probes]

    def counters(self) -> dict[str, int]:
        """The id and name counters a file keeps beside the probes.

        The id counter matters: rebuilt from the surviving ids alone, a
        deleted probe's id comes back after a reload.
        """
        return {"next_id": int(self._next_id), "next_number": int(self._next_number)}

    def to_payload(self) -> dict[str, Any]:
        """The probes and both counters in one JSON-safe object."""
        return {"items": self.to_list(), **self.counters()}

    #: What ``raftcorr.gui.project_bridge`` stores in a project file.
    to_dicts = to_list

    def load_dicts(
        self,
        items: Sequence[dict[str, Any]],
        next_id: int | None = None,
        next_number: int | None = None,
    ) -> None:
        """Replace the probes in place with *items* (the set is shared state).

        Everything is parsed before anything changes: a malformed entry
        raises ValueError and leaves the set as it was.
        """
        loaded = ProbeSet.from_list(items, next_id, next_number)
        self._probes = loaded._probes
        self._next_id = loaded._next_id
        self._next_number = loaded._next_number

    @staticmethod
    def from_list(
        items: Sequence[dict[str, Any]],
        next_id: int | None = None,
        next_number: int | None = None,
    ) -> "ProbeSet":
        if not isinstance(items, (list, tuple)):
            raise ValueError("Probe items must be a list.")
        probes = [probe_from_dict(d) for d in items]
        ids = [p.id for p in probes]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Probe ids must be unique; duplicate in {ids}.")
        floor = max(ids, default=0) + 1
        start = max(int(next_id), floor) if next_id is not None else floor
        # Without a name counter the next name follows the id counter; an
        # empty set starts at P1.
        if not probes:
            number = 1
        elif next_number is not None:
            number = max(int(next_number), 1)
        else:
            number = start
        return ProbeSet(_probes=probes, _next_id=start, _next_number=number)

    @staticmethod
    def from_payload(payload: Any) -> "ProbeSet":
        """Accept ``to_payload`` output, or a bare list of probes."""
        if payload is None:
            return ProbeSet()
        if isinstance(payload, dict):
            return ProbeSet.from_list(
                payload.get("items", []), payload.get("next_id"),
                payload.get("next_number"),
            )
        return ProbeSet.from_list(payload)


__all__ = [
    "GAUGE_QUANTITIES", "POINT_STATISTICS", "SPATIAL_STATISTICS",
    "AreaGeom", "AreaShape", "DEFAULT_COLOURS", "Geometry", "LineGeom",
    "PointGeom", "Probe", "ProbeKind", "ProbeSet", "allowed_reductions",
    "probe_from_dict", "probe_to_dict", "replace",
]
