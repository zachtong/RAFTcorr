"""Exported pictures and animations: colour bar style, margin, resolution cap, JPEG quality,
hidden background (white / black / transparent) and untracked-point fill (raftcorr.export.figures)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from raftcorr.export.figure_layout import TICK_RATIO, ColorbarStyle, effective_dpi, figure_layout
from raftcorr.export.figures import (
    FrameStyle,
    animation_dpi,
    compose,
    export_animation,
    export_frames,
    figure_rgba,
    render_figure,
    style_for_format,
)
from raftcorr.viz.fields import FieldProvider, Units
from raftcorr.viz.render import ColorRange, colorize

from .test_export_scientific import GRID, make_series, make_strain

RNG = ColorRange(-2.0, 2.0)
UNTRACKED = (2, 3)            # u[1, 2, 3] is NaN inside the ROI (make_series)
HOLE = (6, 12)                # outside the ROI


@pytest.fixture
def provider() -> FieldProvider:
    return FieldProvider(make_series(), make_strain())


@pytest.fixture
def images() -> list[np.ndarray]:
    rng = np.random.default_rng(5)
    return [rng.integers(40, 200, (GRID.image_height, GRID.image_width)).astype(np.uint8) for _ in range(3)]


def pixel(rgba: np.ndarray, r: int, c: int) -> tuple[int, ...]:
    """The composed pixel of grid point ``(r, c)``."""
    return tuple(int(v) for v in rgba[GRID.y0 + r, GRID.x0 + c])


def bare(**kw) -> FrameStyle:
    return FrameStyle(field="u", configuration="reference", background="none", title=False, **kw)


# ------------------------------------------------------------------------------ hidden background
@pytest.mark.parametrize("fill, expected", [("white", (255, 255, 255, 255)), ("black", (0, 0, 0, 255)),
                                            ("transparent", (0, 0, 0, 0))])
def test_hidden_background_fill(provider, fill, expected):
    rgba = compose(provider, None, 1, bare(hidden_background=fill), Units(), RNG)
    assert tuple(int(v) for v in rgba[0, 0]) == expected            # outside the grid
    assert pixel(rgba, *HOLE) == expected                             # inside the grid, outside the ROI


def test_transparent_fill_keeps_the_colour_and_spends_the_opacity_on_alpha(provider):
    style = bare(hidden_background="transparent", opacity=0.6)
    rgba = compose(provider, None, 1, style, Units(), RNG)
    u = float(provider.series.u[1, 0, 0])
    want = colorize(np.array([[u]]), RNG, style.colormap, alpha=0.6)[0, 0]
    assert pixel(rgba, 0, 0) == tuple(int(v) for v in want)
    assert want[3] == round(255 * 0.6)


def test_a_shown_background_image_ignores_the_hidden_fill(provider, images):
    style = replace(bare(hidden_background="transparent"), background="auto")
    rgba = compose(provider, images, 1, style, Units(), RNG)
    assert rgba[..., 3].min() == 255 and tuple(rgba[0, 0, :3]) == (images[0][0, 0],) * 3


def test_formats_without_alpha_get_white_instead_of_transparent():
    style = bare(hidden_background="transparent")
    for ext in (".png", ".tif", ".tiff"):
        assert style_for_format(style, ext) is style
    for ext in (".jpg", ".pdf", ".svg", ".gif", ".mp4"):
        assert style_for_format(style, ext).hidden_background == "white"
    assert style_for_format(bare(hidden_background="black"), ".jpg").hidden_background == "black"


# ------------------------------------------------------------------------------ untracked fill
def test_hide_untracked_off_fills_untracked_points_inside_the_roi(provider):
    hidden = compose(provider, None, 1, bare(), Units(), RNG)
    filled = compose(provider, None, 1, bare(hide_untracked=False), Units(), RNG)
    assert pixel(hidden, *UNTRACKED) == (255, 255, 255, 255)          # left as background
    r, c = UNTRACKED
    neighbours = {pixel(filled, r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))}
    assert pixel(filled, *UNTRACKED) in neighbours                    # the nearest tracked value
    assert pixel(filled, *HOLE) == (255, 255, 255, 255)               # outside the ROI: never filled
    assert np.array_equal(np.delete(hidden.reshape(-1, 4), (GRID.y0 + r) * GRID.image_width + GRID.x0 + c, 0),
                          np.delete(filled.reshape(-1, 4), (GRID.y0 + r) * GRID.image_width + GRID.x0 + c, 0))


def test_hide_untracked_off_fills_strain_on_its_coarser_grid():
    strain = make_strain()
    strain.components["exx"][1, 0, 0] = np.nan                        # inside the ROI (mask[0, 0])
    prov = FieldProvider(make_series(), strain)
    style = FrameStyle(field="exx", background="none", title=False, hide_untracked=False)
    filled = compose(prov, None, 1, style, Units(), ColorRange(-1, 1))
    hidden = compose(prov, None, 1, replace(style, hide_untracked=True), Units(), ColorRange(-1, 1))
    assert pixel(hidden, 0, 0) == (255, 255, 255, 255)
    assert pixel(filled, 0, 0) != (255, 255, 255, 255) and pixel(filled, 1, 1) == pixel(filled, 0, 0)


def test_hide_untracked_only_changes_the_reference_configuration(provider):
    style = replace(bare(hide_untracked=False), configuration="deformed")
    assert np.array_equal(compose(provider, None, 1, style, Units(), RNG),
                          compose(provider, None, 1, replace(style, hide_untracked=True), Units(), RNG))


# ------------------------------------------------------------------------------ colour bar style
def picture_rgba(h: int = 60, w: int = 80) -> np.ndarray:
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 1] = 128
    rgba[..., 3] = 255
    return rgba


@pytest.mark.parametrize("position", ["right", "left", "top", "bottom"])
def test_colorbar_position(position):
    style = bare(colorbar_style=ColorbarStyle(position=position))
    fig = render_figure(picture_rgba(), style, RNG, "u (px)", "")
    pic, bar = (a.get_position() for a in fig.axes)
    side = {"right": bar.x0 > pic.x1, "left": bar.x1 < pic.x0, "top": bar.y0 > pic.y1, "bottom": bar.y1 < pic.y0}
    assert side[position]
    vertical = position in ("left", "right")
    assert (bar.height > bar.width) == vertical
    axis = fig.axes[1].yaxis if vertical else fig.axes[1].xaxis
    assert axis.get_label_position() == position and axis.get_ticks_position() == position   # outer side


def test_colorbar_font_and_background_reach_the_figure():
    style = bare(colorbar_style=ColorbarStyle(font_size=16, font_family="serif", background="black"))
    fig = render_figure(picture_rgba(), style, RNG, "u (px)", "")
    img = figure_rgba(fig)                                            # draws: ticks exist from here on
    label = fig.axes[1].yaxis.label
    assert label.get_text() == "u (px)" and label.get_fontsize() == 16 and label.get_color() == "white"
    assert label.get_fontfamily() == ["serif"]
    ticks = [t for t in fig.axes[1].yaxis.get_ticklabels() if t.get_text()]
    assert ticks and all(t.get_fontsize() == pytest.approx(16 * TICK_RATIO) for t in ticks)
    assert all(t.get_fontfamily() == ["serif"] for t in ticks)
    assert tuple(img[0, 0]) == (0, 0, 0, 255) and tuple(img[-1, -1]) == (0, 0, 0, 255)


def test_default_colorbar_is_black_on_white():
    fig = render_figure(picture_rgba(), bare(), RNG, "u (px)", "")
    img = figure_rgba(fig)
    assert fig.axes[1].yaxis.label.get_color() == "black"
    assert tuple(img[0, 0]) == (255, 255, 255, 255)


def test_bar_thickness_is_a_fraction_of_the_picture_edge():
    for position, thickness in (("right", 0.1), ("bottom", 0.2)):
        fig = render_figure(picture_rgba(), bare(colorbar_style=ColorbarStyle(position=position, thickness=thickness)),
                            RNG, "u", "")
        pic, bar = (a.get_position() for a in fig.axes)
        if position == "right":
            assert bar.width == pytest.approx(thickness * pic.width)
        else:
            assert bar.height == pytest.approx(thickness * pic.height)


def test_margin_surrounds_the_content_in_its_colour():
    plain = render_figure(picture_rgba(), bare(), RNG, "u", "")
    framed = render_figure(picture_rgba(), bare(margin=0.1, margin_color="black"), RNG, "u", "")
    long_edge = max(plain.get_figwidth(), plain.get_figheight())
    assert framed.get_figwidth() == pytest.approx(plain.get_figwidth() + 0.2 * long_edge)
    assert framed.get_figheight() == pytest.approx(plain.get_figheight() + 0.2 * long_edge)
    img = figure_rgba(framed)
    m = int(0.1 * long_edge * framed.dpi)
    assert tuple(img[m // 2, m // 2]) == (0, 0, 0, 255)               # margin
    assert tuple(img[m + 3, m + 3]) == (255, 255, 255, 255)            # content background


def test_transparent_figure_keeps_only_the_colour_bar_strip_opaque():
    rgba = np.zeros((60, 80, 4), np.uint8)                             # nothing drawn: all transparent
    style = bare(hidden_background="transparent", margin=0.05)
    fig = render_figure(rgba, style, RNG, "u", "")
    img = figure_rgba(fig)
    lay = figure_layout(rgba.shape[:2], style.colorbar_style, False, style.margin)

    def alpha(x_in: float, y_in: float) -> int:                       # inches from the bottom-left corner
        return int(img[int((lay.height - y_in) * fig.dpi), int(x_in * fig.dpi), 3])

    strip, pic = lay.strip, lay.picture
    assert alpha(0.01, 0.01) == 0                                     # margin
    assert alpha(pic.x + pic.w / 2, pic.y + pic.h / 2) == 0           # picture
    assert alpha(pic.x - 0.05, pic.y + pic.h / 2) == 0                # padding around the picture
    assert alpha(strip.x + strip.w - 0.05, strip.y + 0.05) == 255     # the colour bar strip


@pytest.mark.parametrize("bad", [dict(position="middle"), dict(background="grey"), dict(font_family="Comic"),
                                 dict(font_size=0), dict(thickness=0)])
def test_colorbar_style_is_validated(bad):
    with pytest.raises(ValueError):
        ColorbarStyle(**bad)


@pytest.mark.parametrize("bad", [dict(hidden_background="grey"), dict(margin_color="red"), dict(margin=-0.1)])
def test_frame_style_is_validated(bad):
    with pytest.raises(ValueError):
        FrameStyle(**bad)


# ------------------------------------------------------------------------------ files
def written_size(folder: Path, fmt: str) -> tuple[int, int]:
    (path,) = sorted(folder.glob(f"*{fmt}"))
    with Image.open(path) as im:
        return im.size


def test_resolution_cap_bounds_the_long_edge(tmp_path, provider):
    style = bare(dpi=150)
    export_frames(tmp_path / "full", provider, None, [1], style, Units())
    export_frames(tmp_path / "cap", provider, None, [1], style, Units(), max_side=300)
    export_frames(tmp_path / "loose", provider, None, [1], style, Units(), max_side=20000)
    lay = figure_layout(GRID.image_shape, style.colorbar_style, False)
    full = written_size(tmp_path / "full", ".png")
    assert np.allclose(full, (lay.width * 150, lay.height * 150), atol=1)   # the size the DPI gives
    assert max(written_size(tmp_path / "cap", ".png")) == 300
    assert written_size(tmp_path / "loose", ".png") == full
    assert effective_dpi(150, lay, 0) == 150 and effective_dpi(150, lay, 300) < 150


def test_jpeg_quality(tmp_path, provider, images):
    style = replace(bare(), background="auto")
    export_frames(tmp_path / "hi", provider, images, [1], style, Units(), fmt=".jpg", jpeg_quality=98)
    export_frames(tmp_path / "lo", provider, images, [1], style, Units(), fmt=".jpg", jpeg_quality=60)
    hi, lo = (next((tmp_path / d).glob("*.jpg")).stat().st_size for d in ("hi", "lo"))
    assert hi > lo * 1.3
    with pytest.raises(ValueError):
        export_frames(tmp_path / "bad", provider, images, [1], style, Units(), fmt=".jpg", jpeg_quality=0)


def picture_corner(style: FrameStyle, size: tuple[int, int]) -> tuple[int, int]:
    """(x, y) of a written pixel near the picture's top-left corner (image pixel (0, 0), off the grid)."""
    lay = figure_layout(GRID.image_shape, style.colorbar_style, style.title, style.margin)
    sx, sy = size[0] / lay.width, size[1] / lay.height
    return int((lay.picture.x + 0.05) * sx), int((lay.height - lay.picture.y - lay.picture.h + 0.05) * sy)


@pytest.mark.parametrize("fmt, mode, corner", [(".png", "RGBA", (0, 0, 0, 0)), (".tif", "RGBA", (0, 0, 0, 0)),
                                               (".jpg", "RGB", (255, 255, 255))])
def test_transparent_fill_in_files(tmp_path, provider, fmt, mode, corner):
    style = bare(hidden_background="transparent")
    export_frames(tmp_path, provider, None, [1], style, Units(), fmt=fmt)
    (path,) = tmp_path.glob(f"*{fmt}")
    with Image.open(path) as im:
        assert im.mode == mode
        got = im.getpixel(picture_corner(style, im.size))
    if fmt == ".jpg":
        assert all(v >= 245 for v in got)                             # JPEG noise around pure white
    else:
        assert got[3] == 0


def test_black_fill_in_files(tmp_path, provider):
    style = bare(hidden_background="black")
    export_frames(tmp_path, provider, None, [1], style, Units())
    with Image.open(next(tmp_path.glob("*.png"))) as im:
        assert im.getpixel(picture_corner(style, im.size)) == (0, 0, 0, 255)


def test_animation_resolution_cap_and_white_for_transparent(tmp_path, provider):
    style = bare(hidden_background="transparent")
    capped = export_animation(tmp_path / "capped.gif", provider, None, [0, 1], style, Units(), max_side=200)
    full = export_animation(tmp_path / "full.gif", provider, None, [0, 1], style, Units(), max_side=0)
    lay = figure_layout(GRID.image_shape, style.colorbar_style, False)
    with Image.open(capped) as im:
        assert max(im.size) == 200 and im.n_frames == 2
    with Image.open(full) as im:
        dpi = animation_dpi(GRID.image_shape)
        assert dpi == 100 and np.allclose(im.size, (lay.width * dpi, lay.height * dpi), atol=1)
        assert im.convert("RGB").getpixel(picture_corner(style, im.size)) == (255, 255, 255)   # not transparent


def test_animation_full_resolution_draws_large_images_at_their_own_size():
    assert animation_dpi((50, 60)) == 100
    assert animation_dpi((2000, 3500)) == pytest.approx(500)
