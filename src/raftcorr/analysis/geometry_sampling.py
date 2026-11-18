"""Where a probe's samples go on a dense field grid.

Pure geometry, no Qt: the cells of a field grid a region covers, the even-odd
test for polygons, and a region's bounds and centroid.

A field grid places cell ``(i, j)`` at image pixel ``(x0 + j * step,
y0 + i * step)`` -- ``raftcorr.viz.fields.Placement``. A region reads every
cell whose position lies inside it (the rule ``raftcorr.roi.shapes`` uses for
pixels), so a region's statistics are area-weighted by construction: the
cells are evenly spaced.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from raftcorr.analysis.probes import AreaGeom


def in_area(geom: AreaGeom, pts: NDArray[np.float64]) -> NDArray[np.bool_]:
    x, y = pts[:, 0], pts[:, 1]
    if geom.shape == "rect":
        x0, y0, x1, y1 = geom.data  # type: ignore[misc]
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if geom.shape == "circle":
        cx, cy, r = geom.data  # type: ignore[misc]
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    if geom.shape == "polygon":
        return points_in_polygon(x, y, geom.data)  # type: ignore[arg-type]
    raise ValueError(f"Unknown area shape {geom.shape!r}.")


def area_bounds(geom: AreaGeom) -> tuple[float, float, float, float]:
    if geom.shape == "rect":
        return tuple(geom.data)  # type: ignore[return-value]
    if geom.shape == "circle":
        cx, cy, r = geom.data  # type: ignore[misc]
        return (cx - r, cy - r, cx + r, cy + r)
    pts = np.asarray(geom.data, dtype=np.float64)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def area_centroid(geom: AreaGeom) -> tuple[float, float]:
    """Centre of a rectangle or circle; vertex mean of a polygon."""
    if geom.shape == "circle":
        return geom.data[0], geom.data[1]  # type: ignore[return-value]
    x0, y0, x1, y1 = area_bounds(geom)
    if geom.shape == "rect":
        return (x0 + x1) / 2, (y0 + y1) / 2
    pts = np.asarray(geom.data, dtype=np.float64)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


#: Rows of cells tested at once: bounds the temporary coordinate arrays of a
#: region covering a whole large image.
_BLOCK_ROWS = 256


def area_cells(geom: AreaGeom, x0: float, y0: float, step: int, shape: tuple[int, int],
               ) -> tuple[tuple[int, int, int, int], NDArray[np.bool_]] | None:
    """The cells of a field grid inside *geom*, as ``(window, mask)``.

    The grid is placed at ``(x0, y0)`` with spacing ``step`` px and has
    ``shape`` cells. ``window`` (``r0, r1, c0, c1``, half-open) is the
    smallest block of the grid holding every cell inside; ``mask`` marks them
    over it -- one byte per cell, whatever the region's size. None when no
    cell lies inside (a region smaller than a cell, or off the grid).
    """
    h, w = shape
    bx0, by0, bx1, by1 = area_bounds(geom)
    c0 = max(int(math.ceil((bx0 - x0) / step)), 0)
    c1 = min(int(math.floor((bx1 - x0) / step)), w - 1)
    r0 = max(int(math.ceil((by0 - y0) / step)), 0)
    r1 = min(int(math.floor((by1 - y0) / step)), h - 1)
    if c1 < c0 or r1 < r0:
        return None
    n_rows, n_cols = r1 - r0 + 1, c1 - c0 + 1
    if geom.shape == "rect":
        # Every cell of the clipped index range lies inside by construction.
        return (r0, r1 + 1, c0, c1 + 1), np.ones((n_rows, n_cols), dtype=bool)
    xs = x0 + np.arange(c0, c1 + 1, dtype=np.float64) * step
    mask = np.empty((n_rows, n_cols), dtype=bool)
    for start in range(0, n_rows, _BLOCK_ROWS):
        rows = np.arange(r0 + start, min(r0 + start + _BLOCK_ROWS, r1 + 1))
        gx, gy = np.meshgrid(xs, y0 + rows.astype(np.float64) * step)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        mask[start:start + len(rows)] = in_area(geom, pts).reshape(len(rows), n_cols)
    if not mask.any():
        return None
    keep_r = np.flatnonzero(mask.any(axis=1))
    keep_c = np.flatnonzero(mask.any(axis=0))
    mask = mask[keep_r[0]:keep_r[-1] + 1, keep_c[0]:keep_c[-1] + 1]
    window = (r0 + int(keep_r[0]), r0 + int(keep_r[-1]) + 1,
              c0 + int(keep_c[0]), c0 + int(keep_c[-1]) + 1)
    return window, mask


def points_in_polygon(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    vertices: Sequence[tuple[float, float]],
) -> NDArray[np.bool_]:
    """Even-odd ray casting, vectorised over the query points."""
    verts = np.asarray(vertices, dtype=np.float64)
    inside = np.zeros(x.shape, dtype=bool)
    j = len(verts) - 1
    for i in range(len(verts)):
        xi, yi = verts[i]
        xj, yj = verts[j]
        straddles = (yi > y) != (yj > y)
        denom = np.where(yj != yi, yj - yi, 1.0)
        crossing_x = (xj - xi) * (y - yi) / denom + xi
        inside ^= straddles & (x < crossing_x)
        j = i
    return inside


__all__ = [
    "area_bounds", "area_cells", "area_centroid", "in_area", "points_in_polygon",
]
