"""What the canvas shows for the current state: background, field overlay, ROI, colour bar, readout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..engine.mapping import sample_bilinear
from ..viz.colormaps import discrete_lut, lut
from ..viz.fields import ALL_FIELDS, Placement
from ..viz.render import ColorRange, auto_range, colorize, grey_rgba
from .state import AppState

ROI_FILL = (74, 108, 247)
ROI_EDGE = (255, 255, 255)


@dataclass
class Layer:
    rgba: np.ndarray | None
    x0: float = 0.0
    y0: float = 0.0
    step: float = 1.0


@dataclass
class Shown:
    """The field values currently on the canvas (for the cursor readout)."""

    key: str | None = None
    values: np.ndarray | None = None
    placement: Placement | None = None
    factor: float = 1.0
    unit: str = ""


class ViewController:
    def __init__(self, state: AppState, editing=None) -> None:
        self.state = state
        #: callable -> bool: is a mask being edited right now (set by the main window from the canvas tool)
        self.editing = editing or (lambda: False)
        self.shown = Shown()
        self.last_range: ColorRange | None = None
        self._seq_range: dict[tuple, ColorRange] = {}

    def invalidate_ranges(self) -> None:
        self._seq_range.clear()

    # -- background -------------------------------------------------------------------------------
    def background(self, editing: bool | None = None) -> np.ndarray | None:
        """The image behind the field. While a mask is edited it is always the frame being edited:
        a mask lives in the coordinates of its own frame."""
        st = self.state
        if st.sequence is None:
            return None
        v = st.view
        if self.editing() if editing is None else editing:
            return grey_rgba(st.sequence[st.current_frame])
        has_field = st.provider is not None and v.show_overlay
        if v.background == "none" and has_field:
            h, w = st.sequence.shape
            out = np.empty((h, w, 4), np.uint8)
            out[...] = 255  # white, as in exported pictures
            return out
        if v.background == "reference":
            k = 0
        elif v.background == "frame" or not has_field:
            k = st.current_frame
        else:  # auto: match the configuration
            k = 0 if v.configuration == "reference" else st.current_frame
        return grey_rgba(st.sequence[k])

    # -- ROI --------------------------------------------------------------------------------------
    def roi_layer(self, editing: bool) -> np.ndarray | None:
        st = self.state
        if st.sequence is None or not (st.view.show_roi or editing):
            return None
        k = st.current_frame
        m = st.roi(k)
        if m is None:
            return None
        out = np.zeros(m.shape + (4,), np.uint8)
        if editing or st.provider is None:
            out[m] = (*ROI_FILL, 80)
        edge = m & ~ndimage.binary_erosion(m, iterations=1, border_value=0)
        out[edge] = (*ROI_EDGE, 230) if not editing else (*ROI_FILL, 255)
        return out

    # -- field ------------------------------------------------------------------------------------
    def overlay(self, editing: bool | None = None) -> Layer:
        st = self.state
        if self.editing() if editing is None else editing:  # the field would sit on the wrong image while a mask is edited
            self.shown = Shown()
            self.last_range = None
            return Layer(None)
        self.shown = Shown()
        self.last_range = None
        prov = st.provider
        v = st.view
        if prov is None or not v.show_overlay or st.sequence is None:
            return Layer(None)
        key = v.field
        spec = ALL_FIELDS.get(key)
        k = st.current_frame
        if spec is None or k >= prov.series.n_frames or not prov.has_frame(key, k):
            return Layer(None)
        try:
            if v.configuration == "deformed":
                vals, pl = prov.deformed(key, k)
            else:
                vals, pl = prov.values(key, k)
                if not spec.is_strain:
                    vals = np.where(prov.series.mask, vals, np.nan)
        except KeyError:
            return Layer(None)
        factor = st.units.factor(spec)
        shown_vals = vals * factor if factor != 1.0 else vals
        d = v.display(key)
        if d.auto:
            if v.range_scope == "sequence":
                rng = self._sequence_range(key, factor, d.colormap)
            else:
                rng = auto_range(shown_vals, d.colormap)
        else:
            rng = ColorRange(d.vmin, d.vmax)
        table = discrete_lut(d.colormap, d.levels) if d.levels else lut(d.colormap)
        rgba = colorize(shown_vals, rng, d.colormap, alpha=v.opacity, table=table)
        self.last_range = rng
        self.shown = Shown(key, vals, pl, factor, st.units.unit(spec))
        return Layer(rgba, pl.x0, pl.y0, pl.step)

    def _sequence_range(self, key: str, factor: float, cmap: str) -> ColorRange:
        st = self.state
        ck = (key, factor, cmap, id(st.series), id(st.strain))
        hit = self._seq_range.get(ck)
        if hit is not None:
            return hit
        prov = st.provider
        samples = []
        for k in range(prov.series.n_frames):
            if prov.has_frame(key, k):
                vals, _ = prov.values(key, k)
                samples.append(np.asarray(vals)[::3, ::3].ravel() * factor)
        rng = auto_range(np.concatenate(samples) if samples else np.array([]), cmap)
        self._seq_range[ck] = rng
        return rng

    def value_at(self, x: float, y: float) -> float | None:
        s = self.shown
        if s.values is None or s.placement is None or not np.isfinite(x):
            return None
        pl = s.placement
        val = sample_bilinear(np.asarray(s.values, dtype=np.float64), np.array([(x - pl.x0) / pl.step]),
                              np.array([(y - pl.y0) / pl.step]))[0]
        return None if not np.isfinite(val) else float(val) * s.factor
