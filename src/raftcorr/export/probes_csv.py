"""Probe time series and line profiles to CSV.

Two tables. ``export_probe_csv`` writes one row per frame and a column per
probe curve; ``export_line_csv`` writes a field along one line, a row per
sample and a column per frame (every profile, and the kymograph).

Both files describe themselves in a ``#`` comment header: each probe's
geometry, what every column holds and in what unit, the axis convention, how
the frame numbers map onto the images and onto the scientific export, the run
parameters the iDICs Good Practices Guide asks a report to state, and the
RAFTcorr version. Quality columns travel with the data by default: a value
alone cannot distinguish a mean over two hundred valid points from a mean over
three. A cell without a valid measurement is left empty, never 0.

Axes. Values are written as RAFTcorr stores them, in image axes: x right,
y down, so ``v`` is positive down and the strain components are those of
``raftcorr.post.strain`` -- the same convention as the scientific export.
Crack sliding is positive when the second endpoint moves to the right of the
gauge direction as seen on screen.

The files are written as UTF-8 with a byte-order mark: without it, Excel
reads them in the system code page and CJK probe labels arrive garbled.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from raftcorr._version import __version__
from raftcorr.analysis.engine import Kymograph
from raftcorr.analysis.probes import (
    GAUGE_QUANTITIES,
    AreaGeom,
    LineGeom,
    PointGeom,
    Probe,
)
from raftcorr.analysis.series import TimeSeries
from raftcorr.gui.support.fields import is_strain_field
from raftcorr.results import DisplacementSeries, StrainSeries

#: Written for a frame that holds no value, so a reader is never asked to guess
#: whether an empty cell means zero.
_EMPTY = ""

_COORDINATES = ("Probe coordinates are reference-frame (frame 1) image pixels, x = column, "
                "y = row, with the centre of the top-left pixel at (0, 0).")
_AXES = ("Values are in image axes (x right, y down), as every RAFTcorr result: v is "
         "positive down; strain components as computed by RAFTcorr. Crack sliding is "
         "positive when the second endpoint moves to the right of the gauge direction "
         "as seen on screen.")


@dataclass(frozen=True)
class ProbeSeries:
    """One curve, with enough context to name and describe its columns.

    ``field`` is None for a gauge quantity, which does not read a field.
    """

    probe: Probe
    field: str | None
    reduction: str
    series: TimeSeries


def run_parameters(series: DisplacementSeries | None,
                   strain: StrainSeries | None = None) -> dict[str, str]:
    """The run and strain settings a report should state, as recorded with
    the results (``series.info``, ``strain.settings``)."""
    out = {"software": f"RAFTcorr {__version__}"}
    if series is None:
        return out
    info = series.info
    model = info.model_name or "unknown"
    if info.model_sha256:
        model += f" (sha256 {info.model_sha256[:12]})"
    out["model"] = model
    out["iterations"] = str(info.iters)
    if info.corr_mode:
        out["correlation"] = info.corr_mode
    tracking = (info.config or {}).get("tracking") or {}
    mode = tracking.get("mode")
    if mode == "every_n":
        out["tracking"] = f"incremental, new reference every {tracking.get('every_n')} frames"
    elif mode == "custom":
        frames = ", ".join(str(int(k) + 1) for k in tracking.get("key_frames", []))
        out["tracking"] = f"incremental, reference frames {frames}"
    elif mode:
        out["tracking"] = str(mode)
    sigma = (info.config or {}).get("smoothing_sigma")
    if sigma:
        out["displacement smoothing"] = f"Gaussian, sigma {float(sigma):g} px"
    if strain is not None:
        st = strain.settings
        fit = "plane" if st.poly_order == 1 else "quadratic"
        out["strain"] = (f"{st.measure.replace('_', '-')}, virtual strain gauge {st.gauge_px} px "
                         f"({st.weighting} weighting, {fit} fit), step {st.step} px")
        if st.smoothing_sigma:
            out["strain pre-smoothing"] = f"Gaussian, sigma {st.smoothing_sigma:g} px"
    return out


def _clean(text: str) -> str:
    """One line, no control characters: a newline in a label broke the header."""
    return " ".join(str(text).split())


def _geometry_text(probe: Probe) -> str:
    g = probe.geometry
    if isinstance(g, PointGeom):
        return f"point at ({g.x:.3f}, {g.y:.3f}) px"
    if isinstance(g, LineGeom):
        return (
            f"line ({g.x0:.3f}, {g.y0:.3f}) -> ({g.x1:.3f}, {g.y1:.3f}) px, "
            f"length {g.length():.3f} px"
        )
    if isinstance(g, AreaGeom):
        if g.shape == "rect":
            x0, y0, x1, y1 = g.data  # type: ignore[misc]
            return f"rect ({x0:.3f}, {y0:.3f}) -> ({x1:.3f}, {y1:.3f}) px"
        if g.shape == "circle":
            cx, cy, r = g.data  # type: ignore[misc]
            return f"circle center ({cx:.3f}, {cy:.3f}) px, radius {r:.3f} px"
        pts = ", ".join(f"({x:.3f}, {y:.3f})" for x, y in g.data)  # type: ignore[misc]
        return f"polygon [{pts}] px"
    return "unknown geometry"


def _stem(entry: ProbeSeries) -> str:
    label = _clean(entry.probe.label)
    if entry.reduction in GAUGE_QUANTITIES or entry.field is None:
        # A gauge reads no field; naming one ("P1_strain_eyy_strain") would lie.
        return f"{label}_{entry.reduction}"
    return f"{label}_{entry.field}_{entry.reduction}"


def _column_stems(entries: Sequence[ProbeSeries]) -> list[str]:
    """Column prefixes, disambiguated when two probes share a label."""
    stems = [_stem(e) for e in entries]
    seen: dict[str, int] = {}
    for stem in stems:
        seen[stem] = seen.get(stem, 0) + 1
    return [
        f"{stem}_id{entry.probe.id}" if seen[stem] > 1 else stem
        for entry, stem in zip(entries, stems)
    ]


def _parameter_lines(parameters: Mapping[str, str] | None) -> list[str]:
    if not parameters:
        return []
    lines = ["", "Run parameters (iDICs Good Practices Guide, ch. 6):"]
    lines += [f"    {_clean(key)}: {_clean(value)}" for key, value in parameters.items()]
    return lines


def _header_lines(
    entries: Sequence[ProbeSeries],
    stems: Sequence[str],
    frame_rate: float | None,
    parameters: Mapping[str, str] | None,
    notes: Sequence[str] = (),
    with_images: bool = False,
) -> list[str]:
    lines = [
        f"RAFTcorr {__version__} probe export",
        _COORDINATES,
        "frame is 1-based as in the application: frame 1 is the reference image; "
        "frame N is image N of the sequence, index N-1 of the scientific export.",
        _AXES,
        "An empty cell means no valid measurement; the matching _flag column says why.",
    ]
    if with_images:
        lines.append("image is the file name of the frame.")
    if frame_rate:
        lines.append(f"time_s = (frame - 1) / {frame_rate:g}")
    lines.extend(_clean(note) if not note.startswith(" ") else note.rstrip()
                 for note in notes)
    lines += _parameter_lines(parameters)
    lines.append("")
    for entry, stem in zip(entries, stems):
        lines.append(
            f"{stem}: {entry.probe.kind} probe id {entry.probe.id} "
            f"'{_clean(entry.probe.label)}', {_geometry_text(entry.probe)}"
        )
        unit = entry.series.unit or "dimensionless"
        what = (f"gauge {entry.reduction}" if entry.field is None
                else f"field {entry.field}, reduction {entry.reduction}")
        lines.append(f"    {what}, unit {unit}")
    return lines


def _cell(value: float) -> str:
    return _EMPTY if not np.isfinite(value) else f"{value:.9g}"


def export_probe_csv(
    path: str | Path,
    entries: Sequence[ProbeSeries],
    *,
    frame_rate: float | None = None,
    include_quality: bool = True,
    parameters: Mapping[str, str] | None = None,
    extra_columns: Mapping[str, np.ndarray] | None = None,
    notes: Sequence[str] = (),
    frame_names: Sequence[str] | None = None,
) -> Path:
    """Write *entries* as one table and return the path written.

    Parameters
    ----------
    frame_rate:
        Frames per second. When given, a ``time_s`` column is added.
    include_quality:
        Add ``_valid_fraction`` and ``_flag`` columns beside every value.
    parameters:
        Run parameters for the header, e.g. ``run_parameters(series, strain)``.
    extra_columns:
        Per-frame values written after ``frame`` (and ``time_s``), indexed by
        frame -- a machine's load and stress. Empty where a frame has none.
    notes:
        Header lines describing the extra columns.
    frame_names:
        The image file of every frame (``DisplacementSeries.frame_names``);
        when given, an ``image`` column follows ``frame``.
    """
    out = Path(path)
    if not entries:
        raise ValueError("Nothing to export: no probe series were given.")

    lengths = {len(e.series.frames) for e in entries}
    if len(lengths) > 1:
        raise ValueError(
            f"All series must cover the same frames, got lengths {sorted(lengths)}."
        )

    stems = _column_stems(entries)
    frames = entries[0].series.frames
    names = list(frame_names) if frame_names is not None else None

    extra = dict(extra_columns or {})
    header = ["frame"]
    if names is not None:
        header.append("image")
    if frame_rate:
        header.append("time_s")
    header.extend(extra)
    for stem in stems:
        header.append(stem)
        if include_quality:
            header.append(f"{stem}_valid_fraction")
            header.append(f"{stem}_flag")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        for line in _header_lines(entries, stems, frame_rate, parameters, notes,
                                  with_images=names is not None):
            fh.write(f"# {line}\n" if line else "#\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, frame in enumerate(frames):
            k = int(frame)
            row: list[object] = [k + 1]      # 1-based in the file
            if names is not None:
                row.append(names[k] if 0 <= k < len(names) else _EMPTY)
            if frame_rate:
                row.append(f"{k / frame_rate:.6g}")
            for values in extra.values():
                row.append(_cell(values[k]) if k < len(values) else _EMPTY)
            for entry in entries:
                row.append(_cell(entry.series.values[i]))
                if include_quality:
                    row.append(f"{entry.series.valid_fraction[i]:.4g}")
                    row.append(entry.series.status[i].value)
            writer.writerow(row)
    return out


def _line_header_lines(
    probe: Probe,
    kymograph: Kymograph,
    field: str,
    frame_rate: float | None,
    parameters: Mapping[str, str] | None,
) -> list[str]:
    unit = kymograph.unit or "dimensionless"
    measured = f"field {field}, unit {unit}"
    if is_strain_field(field):
        measured += ", as computed by RAFTcorr (image axes)"
    elif field == "disp_v":
        measured += ", image axes (positive down)"
    lines = [
        f"RAFTcorr {__version__} line export",
        f"{_clean(probe.label)}: line probe id {probe.id}, {_geometry_text(probe)}",
        f"    {measured}",
        _COORDINATES,
        f"distance_{kymograph.distance_unit} is reference arc length from the "
        "first endpoint; x_px, y_px locate each sample.",
        "frame is 1-based: frame_1 is the reference image; frame_N is image N of "
        "the sequence, index N-1 of the scientific export.",
        "An empty cell means no valid measurement at that sample and frame: "
        "off the measured area, untracked on that frame, or no strain value there.",
    ]
    if frame_rate:
        lines.append(f"frame_N is at time (N - 1) / {frame_rate:g} s")
    lines += _parameter_lines(parameters)
    return lines


def export_line_csv(
    path: str | Path,
    probe: Probe,
    kymograph: Kymograph,
    *,
    field: str,
    frame_rate: float | None = None,
    parameters: Mapping[str, str] | None = None,
) -> Path:
    """Write *kymograph* -- *field* along *probe* -- and return the path."""
    out = Path(path)
    values = np.asarray(kymograph.values, dtype=np.float64)
    positions = np.asarray(kymograph.xy, dtype=np.float64)
    n_samples = values.shape[1]
    if positions.shape != (n_samples, 2) or len(kymograph.distance) != n_samples:
        raise ValueError(
            f"Sample positions {positions.shape} do not match the "
            f"{n_samples} samples along the line."
        )

    header = [f"distance_{kymograph.distance_unit}", "x_px", "y_px"]
    header += [f"frame_{int(f) + 1}" for f in kymograph.frames]

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        for line in _line_header_lines(probe, kymograph, field, frame_rate, parameters):
            fh.write(f"# {line}\n" if line else "#\n")
        writer = csv.writer(fh)
        writer.writerow(header)
        for s in range(n_samples):
            row: list[object] = [
                f"{kymograph.distance[s]:.6g}",
                f"{positions[s, 0]:.3f}",
                f"{positions[s, 1]:.3f}",
            ]
            row += [_cell(v) for v in values[:, s]]
            writer.writerow(row)
    return out


__all__ = ["ProbeSeries", "export_line_csv", "export_probe_csv", "run_parameters"]
