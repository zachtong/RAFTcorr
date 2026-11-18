"""Tabular exports: probe time series as CSV, and summary statistics of one frame.

CSV layout: a header row ``frame`` (0-based image index), ``time_s`` (``frame / frame_rate``,
only when a frame rate is given), then the value columns in the order given. Values are written
with the shortest representation that round-trips their own dtype; NaN is an empty field. The
file is UTF-8 without a byte-order mark and is written atomically.
"""

from __future__ import annotations

import csv
import math
import operator
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..results import DisplacementSeries
from .scientific import atomic_output

__all__ = ["STATISTICS", "probe_table_csv", "series_statistics"]

STATISTICS: tuple[str, ...] = ("min", "max", "mean", "std", "median")
_RESERVED = ("frame", "time_s")


def probe_table_csv(
    path: str | Path,
    frames: Sequence[int],
    columns: Mapping[str, np.ndarray],
    frame_rate: float | None = None,
) -> Path:
    """Write one row per entry of ``frames`` with ``columns[name][i]`` as the value columns.

    Raises ValueError for a column whose length differs from ``frames``, a column named like a
    reserved column (``frame``, ``time_s``), a negative frame or a non-positive frame rate.
    """
    path = Path(path)
    index = _frame_indices(frames)
    if frame_rate is not None and not (math.isfinite(frame_rate) and frame_rate > 0):
        raise ValueError(f"frame_rate must be None or a positive finite number, got {frame_rate!r}")
    cols = {str(name): _column(str(name), values, len(index)) for name, values in columns.items()}
    header = ["frame"] + (["time_s"] if frame_rate is not None else []) + list(cols)
    with atomic_output(path) as tmp:
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for i, k in enumerate(index):
                time = [repr(k / frame_rate)] if frame_rate is not None else []
                writer.writerow([str(k)] + time + [_cell(c[i]) for c in cols.values()])
    return path


def series_statistics(series: DisplacementSeries, frame: int) -> dict[str, dict[str, float]]:
    """``{"U" | "V" | "Magnitude": {min, max, mean, std, median, count}}`` over the finite ROI
    points of ``frame`` (px; ``std`` is the population standard deviation). A frame that was not
    computed, or has no finite point, gives NaN statistics and ``count`` 0."""
    if not 0 <= frame < series.n_frames:
        raise IndexError(f"frame {frame} out of range [0, {series.n_frames - 1}]")
    valid = series.frame_mask(frame) if bool(series.done[frame]) else np.zeros(series.grid.shape, dtype=bool)
    u = np.asarray(series.u[frame], dtype=np.float64)[valid]
    v = np.asarray(series.v[frame], dtype=np.float64)[valid]
    return {name: _statistics(x) for name, x in (("U", u), ("V", v), ("Magnitude", np.hypot(u, v)))}


def _frame_indices(frames: Sequence[int]) -> list[int]:
    out = []
    for f in frames:
        try:
            k = operator.index(f)
        except TypeError as exc:
            raise TypeError(f"frame indices must be integers, got {f!r}") from exc
        if k < 0:
            raise ValueError(f"frame indices must be >= 0, got {k}")
        out.append(int(k))
    return out


def _column(name: str, values: Any, n: int) -> np.ndarray:
    if not name or name in _RESERVED:
        raise ValueError(f"invalid column name {name!r} (empty or reserved: {', '.join(_RESERVED)})")
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"column {name!r} has shape {arr.shape}, expected ({n},)")
    return arr


def _cell(value: Any) -> str:
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return ""
    return str(value)


def _statistics(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {**{k: math.nan for k in STATISTICS}, "count": 0}
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "count": int(x.size),
    }
