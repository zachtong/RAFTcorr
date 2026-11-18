"""Qt-free visualisation: colormaps, colour mapping, field catalogue (shared by GUI and exports)."""

from .colormaps import COLORMAPS, DEFAULT_COLORMAP, DIVERGING, SEQUENTIAL, discrete_lut, is_diverging, lut
from .fields import ALL_FIELDS, DISPLACEMENT_FIELDS, STRAIN_FIELDS, VELOCITY_FIELDS, FieldProvider, FieldSpec, Placement, Units
from .render import ColorRange, auto_range, colorize, composite, format_tick, grey_rgba, mask_rgba, nice_ticks

__all__ = [
    "ALL_FIELDS",
    "COLORMAPS",
    "ColorRange",
    "DEFAULT_COLORMAP",
    "DISPLACEMENT_FIELDS",
    "DIVERGING",
    "FieldProvider",
    "FieldSpec",
    "Placement",
    "SEQUENTIAL",
    "STRAIN_FIELDS",
    "Units",
    "VELOCITY_FIELDS",
    "auto_range",
    "colorize",
    "composite",
    "discrete_lut",
    "format_tick",
    "grey_rgba",
    "is_diverging",
    "lut",
    "mask_rgba",
    "nice_ticks",
]
