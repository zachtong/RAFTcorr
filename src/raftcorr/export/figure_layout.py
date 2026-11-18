"""Where everything goes in an exported picture: the colour bar style and the figure geometry.

A picture is the composed frame (its long edge drawn 7 inches long) inside a padded content box,
with the colour bar strip on one side and an optional title band on top; an optional margin
surrounds the content box. Every length here is in inches, so the written size is ``width x height``
inches at the figure's DPI, and lowering the DPI (a resolution cap) scales the whole picture
uniformly without changing its layout.

The defaults keep the look RAFTcorr's pictures had before the colour bar became configurable: a
bar on the right, black text on white, no margin.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Where the colour bar goes relative to the picture.
COLORBAR_POSITIONS: tuple[str, ...] = ("right", "left", "top", "bottom")
#: Background of the colour bar strip (and of the figure around the picture); text uses the other.
COLORBAR_BACKGROUNDS: tuple[str, ...] = ("black", "white")
#: Generic matplotlib families: always resolvable, never a missing font.
FONT_FAMILIES: tuple[str, ...] = ("sans-serif", "serif", "monospace")
#: Colour of the margin around the content.
MARGIN_COLORS: tuple[str, ...] = ("white", "black")

PICTURE_IN = 7.0          # long edge of the picture
PAD_IN = 0.15             # padding around the picture inside the content box
GAP_IN = 0.12             # between the picture and the colour bar
TITLE_IN = 0.4            # height of the title band
TITLE_PT = 11.0
BAR_LENGTH = 0.8          # colour bar length, as a fraction of the picture edge beside it
TICK_RATIO = 0.95         # tick labels relative to the colour bar label
_TEXT_ROOM_V = 6.0        # room for ticks + label beside a vertical bar, in units of the font size
_TEXT_ROOM_H = 4.5        # ... below / above a horizontal bar


@dataclass(frozen=True)
class ColorbarStyle:
    """Appearance of the colour bar of exported pictures and animations.

    ``font_size``: label size in points (tick labels are :data:`TICK_RATIO` of it);
    ``thickness``: bar thickness as a fraction of the picture edge perpendicular to the bar (the
    width for a bar on the left / right), so the bar scales with the picture.
    """

    position: str = "right"
    font_size: float = 11.0
    font_family: str = "sans-serif"
    thickness: float = 0.03
    background: str = "white"

    def __post_init__(self) -> None:
        _check_choice("position", self.position, COLORBAR_POSITIONS)
        _check_choice("background", self.background, COLORBAR_BACKGROUNDS)
        _check_choice("font_family", self.font_family, FONT_FAMILIES)
        if not 1.0 <= float(self.font_size) <= 72.0:
            raise ValueError(f"font_size must be in [1, 72] points, got {self.font_size}")
        if not 0.0 < float(self.thickness) <= 0.5:
            raise ValueError(f"thickness must be in (0, 0.5], got {self.thickness}")

    @property
    def vertical(self) -> bool:
        return self.position in ("left", "right")

    @property
    def ink(self) -> str:
        """Colour of the text and ticks: the contrast of the background."""
        return "white" if self.background == "black" else "black"


@dataclass(frozen=True)
class Box:
    """A rectangle in inches from the figure's bottom-left corner."""

    x: float
    y: float
    w: float
    h: float

    def fraction(self, width: float, height: float) -> list[float]:
        """``[left, bottom, width, height]`` as fractions of a ``width x height`` figure."""
        return [self.x / width, self.y / height, self.w / width, self.h / height]


@dataclass(frozen=True)
class FigureLayout:
    """Geometry of one picture: figure size, content box, picture, colour bar, title."""

    width: float
    height: float
    content: Box              # inside the margin: the figure background
    picture: Box
    bar: Box | None           # the colour bar axes (None without a colour bar)
    strip: Box | None         # the bar with its ticks and label: kept opaque on a transparent figure
    title_xy: tuple[float, float] | None

    @property
    def long_edge(self) -> float:
        return max(self.width, self.height)


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def figure_layout(shape: tuple[int, int], colorbar: ColorbarStyle | None, title: bool,
                  margin: float = 0.0) -> FigureLayout:
    """Layout of a picture of ``shape`` (rows, columns) pixels.

    ``colorbar`` None draws no colour bar; ``margin`` is the margin width as a fraction of the
    content's long edge.
    """
    h, w = (max(1, int(s)) for s in shape)
    scale = PICTURE_IN / max(h, w)
    pw, ph = w * scale, h * scale
    side = colorbar.position if colorbar is not None else None
    thick = room = strip = 0.0
    if colorbar is not None:
        thick = colorbar.thickness * (pw if colorbar.vertical else ph)
        room = (_TEXT_ROOM_V if colorbar.vertical else _TEXT_ROOM_H) * float(colorbar.font_size) / 72.0
        strip = GAP_IN + thick + room
    vertical = side in ("left", "right")
    title_h = TITLE_IN if title else 0.0
    cw = 2 * PAD_IN + pw + (strip if vertical else 0.0)
    ch = 2 * PAD_IN + ph + title_h + (strip if side in ("top", "bottom") else 0.0)
    m = max(0.0, float(margin)) * max(cw, ch)
    picture = Box(m + PAD_IN + (strip if side == "left" else 0.0),
                  m + PAD_IN + (strip if side == "bottom" else 0.0), pw, ph)
    bar, band = _bar_boxes(picture, side, thick, room)
    title_xy = (picture.x + pw / 2, m + ch - PAD_IN - title_h / 2) if title else None
    return FigureLayout(cw + 2 * m, ch + 2 * m, Box(m, m, cw, ch), picture, bar, band, title_xy)


def _bar_boxes(p: Box, side: str | None, thick: float, room: float) -> tuple[Box | None, Box | None]:
    """The colour bar axes and its strip (bar + ticks + label) beside picture ``p``."""
    if side is None:
        return None, None
    if side in ("left", "right"):
        length = BAR_LENGTH * p.h
        y = p.y + (p.h - length) / 2
        if side == "right":
            x = p.x + p.w + GAP_IN
            return Box(x, y, thick, length), Box(x - GAP_IN / 2, p.y, GAP_IN / 2 + thick + room, p.h)
        x = p.x - GAP_IN - thick
        return Box(x, y, thick, length), Box(x - room, p.y, room + thick + GAP_IN / 2, p.h)
    length = BAR_LENGTH * p.w
    x = p.x + (p.w - length) / 2
    if side == "top":
        y = p.y + p.h + GAP_IN
        return Box(x, y, length, thick), Box(p.x, y - GAP_IN / 2, p.w, GAP_IN / 2 + thick + room)
    y = p.y - GAP_IN - thick
    return Box(x, y, length, thick), Box(p.x, y - room, p.w, room + thick + GAP_IN / 2)


def effective_dpi(dpi: float, layout: FigureLayout, max_side: int = 0) -> float:
    """``dpi``, lowered so the written long edge is at most ``max_side`` pixels (0: no cap).

    Raster writers truncate ``inches x dpi``; the half pixel lands a capped picture exactly on
    ``max_side``.
    """
    if max_side and max_side > 0:
        return min(float(dpi), (int(max_side) + 0.5) / layout.long_edge)
    return float(dpi)
