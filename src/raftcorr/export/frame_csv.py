"""Per-frame CSV tables of a displacement series (and its strain), pyALDIC's ``export_csv``.

pyALDIC wrote one CSV per frame with a row per mesh node. RAFTcorr fields are dense, so a row is a
grid point inside the frame-0 ROI. One file per computed frame, ``frame_<n>.csv`` with ``n`` the
4-digit frame number: the 0-based image index by default (frame 0 is the reference), 1-based with
``first_number=1`` as the GUI numbers frames.

Columns: ``x [px]``, ``y [px]`` (image pixel coordinates of the point in frame 0), then the chosen
variables with their unit, in the order of :func:`csv_variables`: ``U [px]``, ``V [px]``,
``Magnitude [px]``, ``Residual [grey levels]``, then the strain components (``exx [-]``, ...,
``rotation [deg]``). Displacements are always in pixels (+u right, +v down); the calibration lives
in the parameters file written next to the tables.

Strain on a coarser grid (step > 1) does not share the displacement rows; its columns then go into a
second file per frame, ``frame_<k>_strain.csv``, with the coordinates of the strain samples.

Values are written with 9 significant digits (a float32 round-trips); NaN (not tracked) is an empty
field. UTF-8 without a byte-order mark; each file is written atomically.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import numpy as np

from ..callbacks import check_stop, coerce
from ..post.strain import COMPONENT_UNITS, COMPONENTS
from ..results import DisplacementSeries, StrainSeries
from .scientific import atomic_output

__all__ = ["DISPLACEMENT_COLUMNS", "csv_variables", "export_frame_csv", "frame_csv_paths"]

#: Per-frame displacement variables a table can hold (``Residual`` only when the series has one).
DISPLACEMENT_COLUMNS: tuple[str, ...] = ("U", "V", "Magnitude", "Residual")
_UNITS: dict[str, str] = {"U": "px", "V": "px", "Magnitude": "px", "Residual": "grey levels"}
_ROWS_PER_BLOCK = 200_000
_FORMAT = "%.9g"


def csv_variables(strain: StrainSeries | None = None, series: DisplacementSeries | None = None) -> tuple[str, ...]:
    """Every variable a table can hold, in column order."""
    has_residual = series is not None and series.residual is not None
    disp = tuple(v for v in DISPLACEMENT_COLUMNS if v != "Residual" or has_residual)
    if strain is None:
        return disp
    return disp + tuple(c for c in COMPONENTS if c in strain.components)


def frame_csv_paths(folder: str | Path, series: DisplacementSeries, strain: StrainSeries | None = None,
                    variables: Iterable[str] | None = None, frames: Iterable[int] | None = None,
                    first_number: int = 0) -> list[Path]:
    """The files :func:`export_frame_csv` would write, in writing order."""
    disp, comps = _split(variables, strain, series)
    out = Path(folder)
    paths: list[Path] = []
    for k in _frames(series, frames):
        if disp or (comps and _merged(strain)):
            paths.append(out / f"frame_{k + first_number:04d}.csv")
        if comps and not _merged(strain):
            paths.append(out / f"frame_{k + first_number:04d}_strain.csv")
    return paths


def export_frame_csv(folder: str | Path, series: DisplacementSeries, strain: StrainSeries | None = None,
                     variables: Iterable[str] | None = None, frames: Iterable[int] | None = None,
                     overwrite: bool = False, callbacks: object = None, first_number: int = 0) -> list[Path]:
    """Write one table per computed frame of ``frames`` (default: every computed frame).

    Raises ValueError for an unknown variable (or a strain variable without ``strain``),
    FileExistsError when a table exists and ``overwrite`` is False. Progress is reported once per
    frame; ``should_stop()`` cancels (:class:`~raftcorr.callbacks.StopRequested`); tables already
    written stay, the one being written is not left behind half-done.
    """
    cb = coerce(callbacks)
    disp, comps = _split(variables, strain, series)
    if not disp and not comps:
        raise ValueError("no variables selected")
    todo = _frames(series, frames)
    paths = frame_csv_paths(folder, series, strain, list(disp) + list(comps), todo, first_number)
    if not overwrite:
        existing = [p for p in paths if p.exists()]
        if existing:
            raise FileExistsError(f"{existing[0]} already exists (pass overwrite=True to replace it)")
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    merged = bool(comps) and _merged(strain)
    rows = np.nonzero(series.mask)
    xy = (series.grid.x0 + rows[1].astype(np.float64), series.grid.y0 + rows[0].astype(np.float64))
    if comps and not merged:
        s_rows = np.nonzero(series.mask[:: strain.step, :: strain.step])
        s_xy = (series.grid.x0 + s_rows[1] * float(strain.step), series.grid.y0 + s_rows[0] * float(strain.step))
    written: list[Path] = []
    for i, k in enumerate(todo):
        check_stop(cb)
        columns = _displacement_columns(series, k, disp, rows)
        if merged:
            columns.update(_strain_columns(strain, k, comps, rows))
        if columns:
            written.append(_write_table(out / f"frame_{k + first_number:04d}.csv", xy, columns))
        if comps and not merged:
            written.append(_write_table(out / f"frame_{k + first_number:04d}_strain.csv", s_xy,
                                        _strain_columns(strain, k, comps, s_rows)))
        cb.progress((i + 1) / len(todo), f"csv: frame {i + 1}/{len(todo)}")
    return written


# -- helpers ------------------------------------------------------------------------------------


def _split(variables: Iterable[str] | None, strain: StrainSeries | None,
           series: DisplacementSeries | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    valid = csv_variables(strain, series)
    if variables is None:
        requested = set(valid)
    else:
        requested = {variables} if isinstance(variables, str) else set(variables)
        unknown = sorted(requested - set(valid))
        if unknown:
            hint = "" if strain is not None else " (strain variables need a strain series)"
            raise ValueError(f"unknown variable(s) {unknown}; valid names: {', '.join(valid)}{hint}")
    disp = tuple(v for v in valid if v in requested and v in DISPLACEMENT_COLUMNS)
    comps = tuple(v for v in valid if v in requested and v not in DISPLACEMENT_COLUMNS)
    return disp, comps


def _frames(series: DisplacementSeries, frames: Iterable[int] | None) -> list[int]:
    if frames is None:
        return series.completed_frames
    out = []
    for k in frames:
        k = int(k)
        if not 0 <= k < series.n_frames:
            raise IndexError(f"frame {k} out of range [0, {series.n_frames - 1}]")
        if bool(series.done[k]):
            out.append(k)
    return out


def _merged(strain: StrainSeries | None) -> bool:
    """Strain columns share the displacement rows when the strain grid is the displacement grid."""
    return strain is not None and int(strain.step) == 1


def _displacement_columns(series: DisplacementSeries, k: int, names: tuple[str, ...],
                          rows: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if not names:
        return out
    u = np.asarray(series.u[k], dtype=np.float32)[rows]
    v = np.asarray(series.v[k], dtype=np.float32)[rows]
    values = {"U": u, "V": v, "Magnitude": np.hypot(u, v)}
    if "Residual" in names:
        values["Residual"] = np.asarray(series.residual[k], dtype=np.float32)[rows]
    for name in names:
        out[f"{name} [{_UNITS[name]}]"] = values[name]
    return out


def _strain_columns(strain: StrainSeries, k: int, names: tuple[str, ...],
                    rows: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in names:
        unit = COMPONENT_UNITS.get(name, "-")
        if bool(strain.done[k]):
            out[f"{name} [{unit}]"] = np.asarray(strain.components[name][k], dtype=np.float32)[rows]
        else:
            out[f"{name} [{unit}]"] = np.full(rows[0].shape, np.nan, dtype=np.float32)
    return out


def _write_table(path: Path, xy: tuple[np.ndarray, np.ndarray], columns: dict[str, np.ndarray]) -> Path:
    header = ["x [px]", "y [px]", *columns]
    data = [xy[0], xy[1], *columns.values()]
    n = int(xy[0].shape[0])
    with atomic_output(path) as tmp:
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            fh.write(",".join(header) + "\n")
            for start in range(0, n, _ROWS_PER_BLOCK):
                block = np.column_stack([np.asarray(c[start:start + _ROWS_PER_BLOCK], dtype=np.float64) for c in data])
                buf = io.StringIO()
                np.savetxt(buf, block, fmt=_FORMAT, delimiter=",")
                # NaN is the only value whose text contains "nan"; it becomes an empty field.
                fh.write(buf.getvalue().replace("nan", ""))
    return path
