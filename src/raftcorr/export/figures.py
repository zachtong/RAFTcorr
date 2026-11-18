"""Pictures of results: single frames (PNG, TIFF, PDF, SVG) and animations (GIF, MP4).

Frames are composed with the same code as the GUI canvas (:mod:`raftcorr.viz`), then framed with
matplotlib for the colour bar, units and title. One colour range is used for every frame of an
animation (``range_mode="sequence"``) unless asked otherwise, so colours mean the same in every frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..callbacks import Callbacks, check_stop, coerce
from ..viz.colormaps import discrete_lut, lut
from ..viz.fields import ALL_FIELDS, FieldProvider, Units
from ..viz.render import ColorRange, auto_range, colorize, composite, grey_rgba

IMAGE_FORMATS = (".png", ".tif", ".tiff", ".pdf", ".svg", ".jpg")
ANIMATION_FORMATS = (".gif", ".mp4")


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
    dpi: int = 150
    fill_trimmed_edges: bool = False   # fill the edge-trimmed strain band (pictures only)


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
        bg = np.empty((h, w, 4), np.uint8)
        bg[...] = 255
    else:
        bg = grey_rgba(np.asarray(images[k_bg]))
    if not provider.has_frame(style.field, frame):
        return bg
    fill = style.fill_trimmed_edges
    if style.configuration == "deformed":
        vals, pl = provider.deformed(style.field, frame, fill_trimmed=fill)
    else:
        vals, pl = provider.values(style.field, frame, fill_trimmed=fill)
        if not spec.is_strain:
            vals = np.where(provider.series.mask, vals, np.nan)
    vals = vals * units.factor(spec)
    table = discrete_lut(style.colormap, style.levels) if style.levels else lut(style.colormap)
    ov = colorize(vals, rng, style.colormap, alpha=style.opacity, table=table)
    if pl.step > 1:
        ov = np.repeat(np.repeat(ov, pl.step, axis=0), pl.step, axis=1)
    return composite(bg, ov, int(round(pl.x0)), int(round(pl.y0)))


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


def render_figure(rgba: np.ndarray, style: FrameStyle, rng: ColorRange, label: str, title: str):
    """A matplotlib figure of the composed frame with colour bar and title.

    A bare ``Figure`` draws through the Agg / PDF / SVG canvases directly, so no pyplot backend is
    selected: this runs in worker threads beside the Qt application.
    """
    import matplotlib
    from matplotlib import cm, colors
    from matplotlib.figure import Figure

    h, w = rgba.shape[:2]
    scale = 7.0 / max(h, w)
    cb_w = 1.3 if style.colorbar else 0.0
    fig = Figure(figsize=(w * scale + cb_w + 0.3, h * scale + (0.5 if style.title else 0.15)), dpi=style.dpi)
    ax = fig.add_axes([0.02, 0.02, (w * scale) / (w * scale + cb_w + 0.3), 0.96 - (0.07 if style.title else 0)])
    ax.imshow(rgba, interpolation="nearest")
    ax.set_axis_off()
    if style.title:
        ax.set_title(title, fontsize=11)
    if style.colorbar:
        cmap = matplotlib.colormaps[style.colormap]
        if style.levels:
            cmap = cmap.resampled(style.levels)
        sm = cm.ScalarMappable(norm=colors.Normalize(rng.vmin, rng.vmax), cmap=cmap)
        cax = fig.add_axes([ax.get_position().x1 + 0.02, 0.12, 0.025, 0.72])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(label, fontsize=10.5)
        cb.ax.tick_params(labelsize=10)
    return fig


def figure_rgba(fig) -> np.ndarray:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba()).copy()


def export_frames(folder: str | Path, provider: FieldProvider, images: Any, frames: Sequence[int], style: FrameStyle,
                  units: Units, fmt: str = ".png", prefix: str | None = None, callbacks: Callbacks | None = None,
                  first_number: int = 0) -> list[Path]:
    """One file per frame; returns the written paths.

    Frame ``k`` is numbered ``k + first_number`` in the file name and the title: 0 (the image
    index, the API convention) or 1 (the GUI, which shows the first image as frame 1).
    """
    cb = coerce(callbacks)
    fmt = fmt if fmt.startswith(".") else "." + fmt
    if fmt not in IMAGE_FORMATS:
        raise ValueError(f"image format must be one of {IMAGE_FORMATS}")
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    spec = ALL_FIELDS[style.field]
    rng_all = color_range(provider, style, units, frames) if style.range_mode == "sequence" else None
    label = f"{style.field} ({units.unit(spec)})"
    written = []
    for i, k in enumerate(frames):
        check_stop(cb)
        rng = rng_all or color_range(provider, style, units, [k])
        rgba = compose(provider, images, k, style, units, rng)
        if style.crop_to_roi:
            x0, y0, x1, y1 = _crop_box(provider)
            rgba = rgba[y0:y1, x0:x1]
        fig = render_figure(rgba, style, rng, label, f"{spec.label}, frame {k + first_number}")
        p = out / f"{prefix or style.field}_{k + first_number:04d}{fmt}"
        fig.savefig(p, dpi=style.dpi)
        written.append(p)
        cb.progress((i + 1) / len(frames), f"image {i + 1}/{len(frames)}")
    return written


def export_animation(path: str | Path, provider: FieldProvider, images: Any, frames: Sequence[int], style: FrameStyle,
                     units: Units, fps: float = 5.0, loop: bool = True, max_side: int = 1200,
                     callbacks: Callbacks | None = None, first_number: int = 0) -> Path:
    """GIF (Pillow) or MP4 (imageio-ffmpeg) with one colour range for every frame.

    Titles number frame ``k`` as ``k + first_number``, as in :func:`export_frames`.
    """
    cb = coerce(callbacks)
    p = Path(path)
    if p.suffix.lower() not in ANIMATION_FORMATS:
        raise ValueError(f"animation format must be one of {ANIMATION_FORMATS}")
    spec = ALL_FIELDS[style.field]
    rng = color_range(provider, style, units, frames)
    label = f"{style.field} ({units.unit(spec)})"
    pictures = []
    for i, k in enumerate(frames):
        check_stop(cb)
        rgba = compose(provider, images, k, style, units, rng)
        if style.crop_to_roi:
            x0, y0, x1, y1 = _crop_box(provider)
            rgba = rgba[y0:y1, x0:x1]
        h, w = rgba.shape[:2]
        if max(h, w) > max_side:
            step = int(np.ceil(max(h, w) / max_side))
            rgba = rgba[::step, ::step]
        fig = render_figure(rgba, replace_dpi(style, 100), rng, label, f"{spec.label}, frame {k + first_number}")
        pictures.append(figure_rgba(fig)[..., :3])
        cb.progress(0.9 * (i + 1) / len(frames), f"frame {i + 1}/{len(frames)}")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".gif":
        from PIL import Image

        ims = [Image.fromarray(a).convert("P", palette=Image.Palette.ADAPTIVE, colors=255) for a in pictures]
        ims[0].save(p, save_all=True, append_images=ims[1:], duration=int(1000 / max(fps, 0.1)), loop=0 if loop else 1,
                    optimize=False)
    else:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise RuntimeError("MP4 needs the imageio and imageio-ffmpeg packages; save as GIF instead") from exc
        even = [a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2] for a in pictures]
        imageio.mimsave(p, even, fps=fps, codec="libx264", quality=8)
    cb.progress(1.0, f"saved {p.name}")
    return p


def replace_dpi(style: FrameStyle, dpi: int) -> FrameStyle:
    from dataclasses import replace

    return replace(style, dpi=dpi)
