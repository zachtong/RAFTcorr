"""Rendering of the displayed field (pyALDIC's VizController role, RAFTcorr dense fields).

RAFTcorr results are dense pixel fields, so there is no node-to-pixel interpolation: values come
from :class:`raftcorr.viz.fields.FieldProvider` (reference configuration, or the deformed one
through the cached inverse map), are coloured with :mod:`raftcorr.viz.render`, and the last
QPixmaps are cached per (frame, field, configuration, colour settings).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QImage, QPixmap
from scipy import ndimage

from raftcorr.engine.mapping import sample_bilinear
from raftcorr.gui.app_state import DISPLAY_FIELDS
from raftcorr.viz.fields import ALL_FIELDS
from raftcorr.viz.render import ColorRange, colorize

_PIXMAP_CACHE = 48


def apply_colormap(values: NDArray, vmin: float, vmax: float, cmap: str) -> NDArray[np.uint8]:
    """``(h, w, 4)`` RGBA for ``values`` (NaN transparent); pyALDIC's helper name."""
    return colorize(values, ColorRange(float(vmin), float(vmax)), cmap)


@dataclass
class RenderedField:
    pixmap: QPixmap
    x0: float
    y0: float
    step: int
    cmap: str
    vmin: float
    vmax: float
    label: str


@dataclass
class _Shown:
    name: str
    values: NDArray
    x0: float
    y0: float
    step: int
    factor: float
    unit: str


def resolve_range(state, values: NDArray) -> tuple[float, float]:
    """Auto = :func:`raftcorr.viz.render.auto_range` of the displayed values (written back to the
    state so switching to Fixed starts from what is on screen); fixed = the user's range."""
    if not state.color_auto:
        return float(state.color_min), float(state.color_max)
    from raftcorr.viz.render import auto_range

    rng = auto_range(values, state.colormap)
    state.color_min, state.color_max = rng.vmin, rng.vmax
    return rng.vmin, rng.vmax


def _fill_untracked(values: NDArray, inside: NDArray) -> NDArray:
    """Nearest valid value for NaN points inside ``inside`` ('Hide untracked regions' off)."""
    bad = inside & ~np.isfinite(values)
    if not bad.any() or not np.isfinite(values).any():
        return values
    _, (ii, jj) = ndimage.distance_transform_edt(~np.isfinite(values), return_indices=True)
    out = values.copy()
    out[bad] = values[ii[bad], jj[bad]]
    return out


class VizController:
    """Cached rendering of the main window's field."""

    def __init__(self) -> None:
        self._pixmaps: OrderedDict[tuple, RenderedField] = OrderedDict()
        self.shown: _Shown | None = None

    def clear_all(self) -> None:
        self._pixmaps.clear()
        self.shown = None

    def clear_pixmap_cache(self) -> None:
        self._pixmaps.clear()

    def invalidate_masks(self) -> None:
        self._pixmaps.clear()

    def render_field(self, state) -> RenderedField | None:
        """The current frame's field as a pixmap with its placement and colour range."""
        prov = state.provider
        if prov is None:
            return None
        key = DISPLAY_FIELDS.get(state.display_field, "u")
        spec = ALL_FIELDS.get(key)
        frame = state.current_frame
        if spec is None or frame >= prov.series.n_frames or not prov.has_frame(key, frame):
            self.shown = None
            return None
        if key == "residual" and prov.series.residual is None:
            self.shown = None
            return None
        units = state.units()
        factor = units.factor(spec)
        deformed = bool(state.show_deformed) and frame > 0
        if deformed:
            vals, pl = prov.deformed(key, frame)
        else:
            vals, pl = prov.values(key, frame)
            vals = np.where(prov.series.mask, vals, np.nan)
            if not state.hide_untracked:
                vals = _fill_untracked(np.asarray(vals, dtype=np.float32), prov.series.mask)
        vals = np.asarray(vals, dtype=np.float32)
        if not np.isfinite(vals).any():  # nothing tracked on this frame: draw nothing
            self.shown = None
            return None
        shown_vals = vals * factor if factor != 1.0 else vals
        vmin, vmax = resolve_range(state, shown_vals)
        cmap = state.colormap
        label = self._label(state.display_field, units.unit(spec))
        self.shown = _Shown(label.split(" (")[0], vals, pl.x0, pl.y0, pl.step, factor, units.unit(spec))
        ck = (id(prov.series), frame, key, deformed, cmap, vmin, vmax, factor, state.hide_untracked)
        hit = self._pixmaps.get(ck)
        if hit is not None:
            self._pixmaps.move_to_end(ck)
            return hit
        rgba = colorize(shown_vals, ColorRange(vmin, vmax), cmap)
        h, w = rgba.shape[:2]
        qimg = QImage(np.ascontiguousarray(rgba).data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        out = RenderedField(QPixmap.fromImage(qimg), float(pl.x0), float(pl.y0), int(pl.step), cmap, vmin, vmax,
                            label)
        self._pixmaps[ck] = out
        while len(self._pixmaps) > _PIXMAP_CACHE:
            self._pixmaps.popitem(last=False)
        return out

    @staticmethod
    def _label(display_field: str, unit: str) -> str:
        """Colorbar title: the names the strain window and the export use (U, V, Magnitude, Residual)."""
        from raftcorr.gui.widgets.strain_field_selector import field_label

        if display_field == "residual":
            name = QCoreApplication.translate("FieldSelector", "Residual")
        else:
            name = field_label(display_field)
        return f"{name} ({unit})"

    def value_at(self, x: float, y: float) -> tuple[str, float, str] | None:
        """(field name, value, unit) under image point ``(x, y)``, or None."""
        s = self.shown
        if s is None or not np.isfinite(x) or not np.isfinite(y):
            return None
        v = sample_bilinear(np.asarray(s.values, dtype=np.float64), np.array([(x - s.x0) / s.step]),
                            np.array([(y - s.y0) / s.step]))[0]
        if not np.isfinite(v):
            return None
        return s.name, float(v) * s.factor, s.unit
