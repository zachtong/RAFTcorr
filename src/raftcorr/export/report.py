"""Self-contained HTML report of a run (pyALDIC's ``export_report``, RAFTcorr results).

Sections, each optional:

1. header: frames (computed / total), ROI size, grid, software version, creation time;
2. run parameters: one row per entry of :func:`raftcorr.export.params.run_parameters` (model,
   checkpoint hash, iterations, tracking, tiling, strain settings, calibration ...);
3. per-field statistics: min / max / mean / std of every computed frame, over the finite ROI
   points (displacements scaled to the length unit of ``units``);
4. sample pictures of each field (PNG embedded as base64), every ``sample_every``-th computed frame
   plus the last one, composed exactly like the image export (:mod:`raftcorr.export.figures`)
   with one colour range per field so that colours compare across the samples.

No external CSS, fonts or scripts: the file opens in any browser, offline. Frame numbers in the
report are 1-based like the GUI's frame slider (frame 1 is the reference image, index 0 of the
data files).
"""

from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .._version import __version__
from ..callbacks import check_stop, coerce
from ..viz.fields import ALL_FIELDS, FieldProvider, Units
from ..viz.render import ColorRange
from .figures import FrameStyle, _crop_box, color_range, compose, figure_rgba, render_figure
from .params import run_parameters
from .scientific import PhysicalUnits, atomic_output

__all__ = ["REPORT_STATISTICS", "export_html_report", "frame_statistics", "sampled_frames"]

REPORT_STATISTICS: tuple[str, ...] = ("min", "max", "mean", "std")
_THUMB_DPI = 60
_THUMB_MAX_SIDE = 480

_CSS = """
body { font-family: Arial, sans-serif; background: #f5f5f5; color: #222; margin: 0; padding: 16px; }
h1   { color: #1a3a5c; border-bottom: 2px solid #1a3a5c; padding-bottom: 8px; }
h2   { color: #2c5f8a; margin-top: 32px; }
h3   { color: #2c5f8a; margin-top: 24px; margin-bottom: 8px; }
table { border-collapse: collapse; margin-bottom: 24px; font-size: 13px; }
th   { background: #1a3a5c; color: #fff; padding: 6px 12px; text-align: left; }
td   { padding: 5px 12px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) td { background: #eef2f7; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
img  { max-width: 320px; border: 1px solid #ccc; margin: 4px; vertical-align: top; }
.field-row { margin-bottom: 12px; }
.meta { font-size: 13px; color: #555; margin-bottom: 8px; }
.note { font-size: 12px; color: #666; }
"""


def frame_statistics(provider: FieldProvider, key: str, frame: int, units: Units = Units()) -> dict[str, float] | None:
    """Min / max / mean / std (population) of ``key`` in ``frame`` over its finite ROI points,
    in display units; None when the frame was not computed or has no finite point."""
    if not provider.has_frame(key, frame):
        return None
    spec = ALL_FIELDS[key]
    values = np.asarray(provider.values(key, frame)[0], dtype=np.float64)
    if not spec.is_strain:
        values = values[provider.series.mask]
    finite = values[np.isfinite(values)] * units.factor(spec)
    if finite.size == 0:
        return None
    return {"min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean()),
            "std": float(finite.std())}


def sampled_frames(frames: Sequence[int], every: int) -> list[int]:
    """Every ``every``-th entry of ``frames``, plus the last one."""
    frames = list(frames)
    out = frames[:: max(1, int(every))]
    if frames and out[-1] != frames[-1]:
        out.append(frames[-1])
    return out


def export_html_report(
    path: str | Path,
    provider: FieldProvider,
    images: Any = None,
    fields: Sequence[str] = ("u", "v", "magnitude"),
    *,
    units: Units = Units(),
    styles: Mapping[str, FrameStyle] | None = None,
    include_parameters: bool = True,
    include_statistics: bool = True,
    include_images: bool = True,
    sample_every: int = 5,
    title: str = "RAFTcorr report",
    overwrite: bool = False,
    callbacks: object = None,
) -> Path:
    """Write the report of ``provider``'s results to ``path``.

    ``fields`` are :data:`raftcorr.viz.fields.ALL_FIELDS` keys; fields that are not available
    (strain without a strain series, residual of a run without one) are skipped. ``styles[key]``
    sets the colour map / range / opacity / configuration / background of a field's pictures
    (default: :class:`FrameStyle` of the field); without ``images`` the pictures have no background.
    Raises FileExistsError when ``path`` exists and ``overwrite`` is False.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (pass overwrite=True to replace it)")
    cb = coerce(callbacks)
    available = {spec.key for spec in provider.available()}
    keys = [k for k in fields if k in available]
    styles = dict(styles or {})
    series = provider.series
    computed = {k: [f for f in range(series.n_frames) if provider.has_frame(k, f)] for k in keys}
    samples = {k: sampled_frames(computed[k], sample_every) if include_images else [] for k in keys}
    total = max(1, sum(len(computed[k]) * include_statistics + len(samples[k]) for k in keys))
    done = 0

    def step(message: str) -> None:
        nonlocal done
        done += 1
        cb.progress(done / total, message)

    sections = [_header(provider, title)]
    if include_parameters:
        sections.append("<h2>Run parameters</h2>")
        sections.append(_parameters_table(provider, units))
    if keys:
        sections.append("<h2>Fields</h2>")
        sections.append('<p class="note">Frame numbers as in the frame slider: frame 1 is the reference image. '
                        "Statistics over the finite points of the region of interest.</p>")
    for key in keys:
        spec = ALL_FIELDS[key]
        sections.append(f"<h3>{html.escape(spec.label)} [{html.escape(units.unit(spec))}]</h3>")
        if include_statistics:
            rows = []
            for k in computed[key]:
                check_stop(cb)
                rows.append((k, frame_statistics(provider, key, k, units)))
                step(f"report: statistics of {key}")
            sections.append(_statistics_table(rows))
        if samples[key]:
            style = _thumbnail_style(styles.get(key), key, images)
            rng = color_range(provider, style, units, samples[key])
            pictures = []
            for k in samples[key]:
                check_stop(cb)
                pictures.append(_picture(provider, images, k, style, units, rng, spec))
                step(f"report: picture of {key}")
            sections.append("<div class='field-row'>" + "".join(pictures) + "</div>")
    page = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"UTF-8\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        + "\n".join(sections)
        + "\n</body>\n</html>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output(path) as tmp:
        tmp.write_text(page, encoding="utf-8")
    cb.progress(1.0, f"report: saved {path.name}")
    return path


# -- sections -----------------------------------------------------------------------------------


def _header(provider: FieldProvider, title: str) -> str:
    s = provider.series
    g = s.grid
    strain = provider.strain
    items = [
        ("Frames computed", f"{len(s.completed_frames)} / {s.n_frames}"),
        ("ROI points", f"{int(np.count_nonzero(s.mask))}"),
        ("Grid", f"{g.width} x {g.height} px at ({g.x0}, {g.y0}) in a {g.image_width} x {g.image_height} px image"),
        ("Model", s.info.model_name),
        ("Strain", "none" if strain is None else f"{len(np.flatnonzero(strain.done))} frames, step {strain.step} px"),
        ("Software", f"RAFTcorr {__version__}"),
        ("Created", _dt.datetime.now().isoformat(timespec="seconds")),
    ]
    meta = " &nbsp;|&nbsp; ".join(f"<b>{html.escape(k)}:</b> {html.escape(v)}" for k, v in items)
    return f"<h1>{html.escape(title)}</h1>\n<div class=\"meta\">{meta}</div>"


def _parameters_table(provider: FieldProvider, units: Units) -> str:
    phys = PhysicalUnits(units.pixel_size, units.length_unit, units.frame_rate) if units.physical else PhysicalUnits()
    params = run_parameters(provider.series, provider.strain, phys)
    params.pop("frame_names", None)  # one per frame: the header gives the count
    params.pop("created", None)
    rows = ["<table>", "<tr><th>Parameter</th><th>Value</th></tr>"]
    for name, value in _flatten(params):
        rows.append(f"<tr><td>{html.escape(name)}</td><td>{html.escape(value)}</td></tr>")
    rows.append("</table>")
    return "\n".join(rows)


def _flatten(d: Mapping[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            out.extend(_flatten(value, name + "."))
        elif isinstance(value, (list, tuple)) and len(value) > 12:
            out.append((name, "[" + ", ".join(_text(v) for v in value[:6]) + f", ...] ({len(value)} entries)"))
        else:
            out.append((name, _text(value)))
    return out


def _text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_text(v) for v in value) + "]"
    return str(value)


def _statistics_table(rows: list[tuple[int, dict[str, float] | None]]) -> str:
    out = ["<table>", "<tr><th>Frame</th>" + "".join(f"<th>{s.capitalize()}</th>" for s in REPORT_STATISTICS)
           + "</tr>"]
    for k, stats in rows:
        if stats is None:
            out.append(f"<tr><td>{k + 1}</td><td colspan='{len(REPORT_STATISTICS)}'>N/A</td></tr>")
            continue
        cells = "".join(f"<td class='num'>{_number(stats[s])}</td>" for s in REPORT_STATISTICS)
        out.append(f"<tr><td>{k + 1}</td>{cells}</tr>")
    out.append("</table>")
    return "\n".join(out)


def _number(x: float) -> str:
    return "N/A" if not math.isfinite(x) else f"{x + 0.0:.4g}"   # + 0.0: no "-0"


# -- pictures -----------------------------------------------------------------------------------


def _thumbnail_style(style: FrameStyle | None, key: str, images: Any) -> FrameStyle:
    base = style if style is not None else FrameStyle(field=key)
    background = "none" if images is None else base.background
    # The thumbnails are embedded without alpha: a transparent hidden background is white there.
    hidden = "white" if base.hidden_background == "transparent" else base.hidden_background
    return replace(base, field=key, background=background, dpi=_THUMB_DPI, title=True, hidden_background=hidden)


def _picture(provider: FieldProvider, images: Any, k: int, style: FrameStyle, units: Units, rng: ColorRange,
             spec: Any) -> str:
    from PIL import Image

    rgba = compose(provider, images, k, style, units, rng)
    if style.crop_to_roi:
        x0, y0, x1, y1 = _crop_box(provider)
        rgba = rgba[y0:y1, x0:x1]
    h, w = rgba.shape[:2]
    if max(h, w) > _THUMB_MAX_SIDE:
        stride = int(math.ceil(max(h, w) / _THUMB_MAX_SIDE))
        rgba = rgba[::stride, ::stride]
    label = f"{style.field} ({units.unit(spec)})"
    fig = render_figure(rgba, style, rng, label, f"Frame {k + 1}")
    buf = io.BytesIO()
    Image.fromarray(figure_rgba(fig)[..., :3]).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" title="Frame {k + 1}" alt="Frame {k + 1}">'
