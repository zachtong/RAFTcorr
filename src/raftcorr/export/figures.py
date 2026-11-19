"""Pictures of results: single frames (PNG, TIFF, JPEG, PDF, SVG) and animations (GIF, MP4).

Frames are composed with the same code as the GUI canvas (:mod:`raftcorr.viz`), then framed with
matplotlib for the colour bar, units and title (geometry: :mod:`raftcorr.export.figure_layout`).
One colour range is used for every frame of an animation (``range_mode="sequence"``) unless asked
otherwise, so colours mean the same in every frame.

Like the windows, a picture can hide its background image (``background="none"``) and show white,
black or nothing (``hidden_background="transparent"``, kept by PNG and TIFF; the other formats get
white), and can fill untracked points inside the region of interest with the nearest value
(``hide_untracked=False``, reference configuration).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..callbacks import Callbacks, check_stop, coerce
from ..post.strain import COMPONENT_LABELS
from ..viz.colormaps import discrete_lut, lut
from ..viz.fields import ALL_FIELDS, FieldProvider, Units, fill_untracked
from ..viz.render import ColorRange, auto_range, colorize, composite, grey_rgba
from .figure_layout import (
    COLORBAR_POSITIONS,
    FONT_FAMILIES,
    MARGIN_COLORS,
    PICTURE_IN,
    TICK_RATIO,
    TITLE_PT,
    Box,
    ColorbarStyle,
    FigureLayout,
    effective_dpi,
    figure_layout,
)

__all__ = [
    "ALPHA_FORMATS", "ANIMATION_FORMATS", "COLORBAR_POSITIONS", "FONT_FAMILIES", "HIDDEN_BACKGROUNDS",
    "DISPLAY_NAMES", "IMAGE_FORMATS", "ColorbarStyle", "FrameStyle", "colorbar_label", "color_range", "compose",
    "export_animation", "export_frames", "figure_rgba", "render_figure", "style_for_format",
]

#: The name of each field on a colour bar: the name the windows show (U, V, Magnitude, and the strain
#: symbols of :data:`raftcorr.post.strain.COMPONENT_LABELS`), in English (the export has no Qt to
#: translate the words). The keys are :data:`raftcorr.viz.fields.ALL_FIELDS` keys.
DISPLAY_NAMES: dict[str, str] = {
    "u": "U", "v": "V", "magnitude": "Magnitude",
    "vel_u": "U velocity", "vel_v": "V velocity", "speed": "Velocity",
    "residual": "Residual",
    **COMPONENT_LABELS,
}

IMAGE_FORMATS = (".png", ".tif", ".tiff", ".pdf", ".svg", ".jpg")
ANIMATION_FORMATS = (".gif", ".mp4")
#: Formats that keep a transparent hidden background; the others get white.
ALPHA_FORMATS = (".png", ".tif", ".tiff")
#: What replaces the background image when it is hidden.
HIDDEN_BACKGROUNDS = ("white", "black", "transparent")
#: Animation frames are drawn at least this sharp (colour bar text stays legible on small images).
ANIMATION_MIN_DPI = 100.0


@dataclass(frozen=True)
class FrameStyle:
    field: str = "u"
    configuration: str = "reference"   # reference | deformed
    background: str = "auto"           # auto | frame | reference | none
    colormap: str = "turbo"
    levels: int = 0
    opacity: float = 0.75
    vmin: float | None = None          # None = automatic
    vmax: float | None = None
    range_mode: str = "sequence"       # sequence | frame (when vmin/vmax are None)
    colorbar: bool = True
    title: bool = True
    crop_to_roi: bool = False
    dpi: float = 150
    colorbar_style: ColorbarStyle = ColorbarStyle()   # frozen, so one shared default is safe
    margin: float = 0.0                # around the content, fraction of its long edge
    margin_color: str = "white"        # white | black
    hidden_background: str = "white"   # white | black | transparent (background "none")
    hide_untracked: bool = True        # False: untracked points in the ROI get the nearest value
    fill_trimmed_edges: bool = False   # fill the edge-trimmed strain band (pictures only)

    def __post_init__(self) -> None:
        if self.hidden_background not in HIDDEN_BACKGROUNDS:
            raise ValueError(f"hidden_background must be one of {HIDDEN_BACKGROUNDS}, got {self.hidden_background!r}")
        if self.margin_color not in MARGIN_COLORS:
            raise ValueError(f"margin_color must be one of {MARGIN_COLORS}, got {self.margin_color!r}")
        if not 0.0 <= float(self.margin) <= 1.0:
            raise ValueError(f"margin must be in [0, 1], got {self.margin}")

    @property
    def transparent(self) -> bool:
        """True when the picture has no background image and nothing in its place."""
        return self.background == "none" and self.hidden_background == "transparent"


def style_for_format(style: FrameStyle, ext: str) -> FrameStyle:
    """``style`` as a file of extension ``ext`` can hold it: transparency becomes white without alpha."""
    if style.hidden_background == "transparent" and ext.lower() not in ALPHA_FORMATS:
        return replace(style, hidden_background="white")
    return style


def colorbar_label(field: str, units: Units) -> str:
    """Colour-bar label of ``field``: its display name and unit, e.g. ``"U (px)"``, ``"εxx (-)"``."""
    return f"{DISPLAY_NAMES.get(field, field)} ({units.unit(ALL_FIELDS[field])})"


# -- composing a frame --------------------------------------------------------------------------


def _blank(h: int, w: int, color: str) -> np.ndarray:
    """RGBA of the hidden background: white, black or fully transparent."""
    out = np.zeros((h, w, 4), np.uint8)
    if color != "transparent":
        out[..., :3] = 255 if color == "white" else 0
        out[..., 3] = 255
    return out


def _place(canvas: np.ndarray, overlay: np.ndarray, x0: int, y0: int) -> np.ndarray:
    """``overlay`` pasted onto a transparent ``canvas`` at ``(x0, y0)`` (clipped), where it has a value.

    Over nothing the field keeps its colour and its opacity becomes its alpha; blending toward the
    transparent black of an empty canvas would only darken it. New array.
    """
    out = canvas.copy()
    H, W = out.shape[:2]
    h, w = overlay.shape[:2]
    bx0, by0, bx1, by1 = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if bx1 > bx0 and by1 > by0:
        ov = overlay[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0]
        region = out[by0:by1, bx0:bx1]
        out[by0:by1, bx0:bx1] = np.where(ov[..., 3:] > 0, ov, region)
    return out


def _reference_values(provider: FieldProvider, style: FrameStyle, frame: int) -> tuple[np.ndarray, Any]:
    spec = ALL_FIELDS[style.field]
    vals, pl = provider.values(style.field, frame, fill_trimmed=style.fill_trimmed_edges)
    mask = provider.series.mask
    if not spec.is_strain:
        vals = np.where(mask, vals, np.nan)
    if not style.hide_untracked:
        inside = mask[:: pl.step, :: pl.step] if pl.step > 1 else mask
        if inside.shape == vals.shape:
            vals = fill_untracked(np.asarray(vals, dtype=np.float32), inside,
                                  provider.trimmed_band(style.field, frame))
    return vals, pl


def compose(provider: FieldProvider, images: Any, frame: int, style: FrameStyle, units: Units,
            rng: ColorRange) -> np.ndarray:
    """RGBA image of one frame: background with the coloured field on top."""
    spec = ALL_FIELDS[style.field]
    if style.background == "reference":
        k_bg = 0
    elif style.background == "frame":
        k_bg = frame
    else:
        k_bg = 0 if style.configuration == "reference" else frame
    if style.background == "none":
        h, w = provider.series.grid.image_shape
        bg = _blank(h, w, style.hidden_background)
    else:
        bg = grey_rgba(np.asarray(images[k_bg]))
    if not provider.has_frame(style.field, frame):
        return bg
    fill = style.fill_trimmed_edges
    if style.configuration == "deformed":
        vals, pl = provider.deformed(style.field, frame, fill_trimmed=fill)
    else:
        vals, pl = _reference_values(provider, style, frame)
    vals = vals * units.factor(spec)
    table = discrete_lut(style.colormap, style.levels) if style.levels else lut(style.colormap)
    ov = colorize(vals, rng, style.colormap, alpha=style.opacity, table=table)
    if pl.step > 1:
        ov = np.repeat(np.repeat(ov, pl.step, axis=0), pl.step, axis=1)
    x0, y0 = int(round(pl.x0)), int(round(pl.y0))
    return _place(bg, ov, x0, y0) if style.transparent else composite(bg, ov, x0, y0)


def color_range(provider: FieldProvider, style: FrameStyle, units: Units, frames: Sequence[int]) -> ColorRange:
    if style.vmin is not None and style.vmax is not None:
        return ColorRange(float(style.vmin), float(style.vmax))
    spec = ALL_FIELDS[style.field]
    f = units.factor(spec)
    samples = [np.asarray(provider.values(style.field, k)[0])[::2, ::2].ravel() * f
               for k in frames if provider.has_frame(style.field, k)]
    return auto_range(np.concatenate(samples) if samples else np.array([]), style.colormap)


def _crop_box(provider: FieldProvider, pad: int = 20) -> tuple[int, int, int, int]:
    g = provider.series.grid
    H, W = g.image_shape
    return max(0, g.x0 - pad), max(0, g.y0 - pad), min(W, g.x0 + g.width + pad), min(H, g.y0 + g.height + pad)


# -- framing it ---------------------------------------------------------------------------------


def render_figure(rgba: np.ndarray, style: FrameStyle, rng: ColorRange, label: str, title: str,
                  max_side: int = 0):
    """A matplotlib figure of the composed frame with colour bar and title.

    The figure's DPI is ``style.dpi``, lowered so its long edge is at most ``max_side`` pixels
    (0: no cap). A bare ``Figure`` draws through the Agg / PDF / SVG canvases directly, so no pyplot
    backend is selected: this runs in worker threads beside the Qt application.
    """
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    cbs = style.colorbar_style
    lay = figure_layout(rgba.shape[:2], cbs if style.colorbar else None, style.title, style.margin)
    dpi = effective_dpi(style.dpi, lay, max_side)
    fig = Figure(figsize=(lay.width, lay.height), dpi=dpi)
    # Transparent: only the colour bar strip stays opaque, so its text remains readable.
    fig.patch.set_facecolor((0.0, 0.0, 0.0, 0.0) if style.transparent else style.margin_color)
    backdrop = lay.strip if style.transparent else lay.content
    if backdrop is not None:
        fig.add_artist(Rectangle((backdrop.x / lay.width, backdrop.y / lay.height), backdrop.w / lay.width,
                                 backdrop.h / lay.height, transform=fig.transFigure, facecolor=cbs.background,
                                 edgecolor="none", zorder=-1))
    ax = fig.add_axes(lay.picture.fraction(lay.width, lay.height))
    ax.imshow(_thin(rgba, PICTURE_IN * dpi), interpolation="nearest", aspect="auto")
    ax.set_axis_off()
    if lay.title_xy is not None:
        x, y = lay.title_xy
        fig.text(x / lay.width, y / lay.height, title, ha="center", va="center", fontsize=TITLE_PT, color=cbs.ink)
    if lay.bar is not None:
        _draw_colorbar(fig, lay, lay.bar, style, rng, label)
    return fig


def _thin(rgba: np.ndarray, long_px: float) -> np.ndarray:
    """Every n-th pixel of a picture far larger than drawn (still twice the drawn size)."""
    step = int(max(rgba.shape[:2]) // max(2.0 * long_px, 1.0))
    return rgba[::step, ::step] if step >= 2 else rgba


def _draw_colorbar(fig, lay: FigureLayout, bar: Box, style: FrameStyle, rng: ColorRange, label: str) -> None:
    import matplotlib
    from matplotlib import cm, colors

    cbs = style.colorbar_style
    cmap = matplotlib.colormaps[style.colormap]
    if style.levels:
        cmap = cmap.resampled(style.levels)
    sm = cm.ScalarMappable(norm=colors.Normalize(rng.vmin, rng.vmax), cmap=cmap)
    cax = fig.add_axes(bar.fraction(lay.width, lay.height))
    cb = fig.colorbar(sm, cax=cax, orientation="vertical" if cbs.vertical else "horizontal",
                      ticklocation=cbs.position)   # ticks and label on the outer side
    axis = cax.yaxis if cbs.vertical else cax.xaxis
    ink, tick_pt = cbs.ink, cbs.font_size * TICK_RATIO
    cb.set_label(label, fontsize=cbs.font_size, color=ink, fontfamily=cbs.font_family)
    cax.tick_params(colors=ink, labelsize=tick_pt)
    # The ticks of this range exist now and are reused when drawn; later ticks copy the first one.
    # (tick_params(labelfontfamily=...) would need matplotlib 3.8.)
    for tick in axis.get_major_ticks():
        tick.label1.set_fontfamily(cbs.font_family)
        tick.label2.set_fontfamily(cbs.font_family)
    axis.get_offset_text().set(fontsize=tick_pt, color=ink, fontfamily=cbs.font_family)
    cb.outline.set_edgecolor(ink)


def figure_rgba(fig) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba()).copy()


def _save_options(fmt: str, jpeg_quality: int) -> dict[str, Any]:
    return {"pil_kwargs": {"quality": int(jpeg_quality)}} if fmt == ".jpg" else {}


# -- exporters ----------------------------------------------------------------------------------


def export_frames(folder: str | Path, provider: FieldProvider, images: Any, frames: Sequence[int], style: FrameStyle,
                  units: Units, fmt: str = ".png", prefix: str | None = None, callbacks: Callbacks | None = None,
                  first_number: int = 0, max_side: int = 0, jpeg_quality: int = 92) -> list[Path]:
    """One file per frame; returns the written paths.

    Frame ``k`` is numbered ``k + first_number`` in the file name and the title: 0 (the image
    index, the API convention) or 1 (the GUI, which shows the first image as frame 1). Pictures
    are ``style.dpi`` sharp with their long edge capped at ``max_side`` pixels (0: no cap; for PDF
    and SVG the cap applies to the embedded picture); ``jpeg_quality`` (1-100) is used for JPEG.
    """
    cb = coerce(callbacks)
    fmt = fmt if fmt.startswith(".") else "." + fmt
    if fmt not in IMAGE_FORMATS:
        raise ValueError(f"image format must be one of {IMAGE_FORMATS}")
    if not 1 <= int(jpeg_quality) <= 100:
        raise ValueError(f"jpeg_quality must be in [1, 100], got {jpeg_quality}")
    style = style_for_format(style, fmt)
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    spec = ALL_FIELDS[style.field]
    rng_all = color_range(provider, style, units, frames) if style.range_mode == "sequence" else None
    label = colorbar_label(style.field, units)
    options = _save_options(fmt, jpeg_quality)
    written = []
    for i, k in enumerate(frames):
        check_stop(cb)
        rng = rng_all or color_range(provider, style, units, [k])
        rgba = compose(provider, images, k, style, units, rng)
        if style.crop_to_roi:
            x0, y0, x1, y1 = _crop_box(provider)
            rgba = rgba[y0:y1, x0:x1]
        fig = render_figure(rgba, style, rng, label, f"{spec.label}, frame {k + first_number}", max_side=max_side)
        p = out / f"{prefix or style.field}_{k + first_number:04d}{fmt}"
        fig.savefig(p, dpi=fig.dpi, **options)
        written.append(p)
        cb.progress((i + 1) / len(frames), f"image {i + 1}/{len(frames)}")
    return written


def animation_dpi(shape: tuple[int, int]) -> float:
    """DPI that draws a picture of ``shape`` pixels at its own size (at least :data:`ANIMATION_MIN_DPI`)."""
    return max(ANIMATION_MIN_DPI, max(int(s) for s in shape) / PICTURE_IN)


def export_animation(path: str | Path, provider: FieldProvider, images: Any, frames: Sequence[int], style: FrameStyle,
                     units: Units, fps: float = 5.0, loop: bool = True, max_side: int = 1024,
                     callbacks: Callbacks | None = None, first_number: int = 0) -> Path:
    """GIF (Pillow) or MP4 (imageio-ffmpeg) with one colour range for every frame.

    Frames show the pictures at their own pixel size (:func:`animation_dpi`, ``style.dpi`` is not
    used), the long edge of the animation capped at ``max_side`` pixels (0: no cap). Titles number
    frame ``k`` as ``k + first_number``, as in :func:`export_frames`.
    """
    cb = coerce(callbacks)
    p = Path(path)
    if p.suffix.lower() not in ANIMATION_FORMATS:
        raise ValueError(f"animation format must be one of {ANIMATION_FORMATS}")
    style = style_for_format(style, p.suffix.lower())
    gif = p.suffix.lower() == ".gif"
    spec = ALL_FIELDS[style.field]
    rng = color_range(provider, style, units, frames)
    label = colorbar_label(style.field, units)
    pictures = []
    for i, k in enumerate(frames):
        check_stop(cb)
        rgba = compose(provider, images, k, style, units, rng)
        if style.crop_to_roi:
            x0, y0, x1, y1 = _crop_box(provider)
            rgba = rgba[y0:y1, x0:x1]
        frame_style = replace(style, dpi=animation_dpi(rgba.shape[:2]))
        fig = render_figure(rgba, frame_style, rng, label, f"{spec.label}, frame {k + first_number}",
                            max_side=max_side)
        picture = figure_rgba(fig)[..., :3]
        pictures.append(_gif_frame(picture) if gif else picture)   # GIF: one byte per pixel from here on
        cb.progress(0.9 * (i + 1) / len(frames), f"frame {i + 1}/{len(frames)}")
    p.parent.mkdir(parents=True, exist_ok=True)
    if gif:
        pictures[0].save(p, save_all=True, append_images=pictures[1:], duration=int(1000 / max(fps, 0.1)),
                         loop=0 if loop else 1, optimize=False)
    else:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError("MP4 needs the imageio and imageio-ffmpeg packages; save as GIF instead") from exc
        even = [a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2] for a in pictures]
        imageio.mimsave(p, even, fps=fps, codec="libx264", quality=8)
    cb.progress(1.0, f"saved {p.name}")
    return p


def _gif_frame(rgb: np.ndarray):
    from PIL import Image

    return Image.fromarray(rgb).convert("P", palette=Image.Palette.ADAPTIVE, colors=255)


def replace_dpi(style: FrameStyle, dpi: int) -> FrameStyle:
    return replace(style, dpi=dpi)
