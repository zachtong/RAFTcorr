"""Right sidebar: display, units, strain, probes."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QSizePolicy,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config import PAPER_STRAIN, STRAIN_MEASURES, StrainConfig
from ..post.probes import METRICS
from ..viz.colormaps import COLORMAPS
from ..viz.fields import ALL_FIELDS, DISPLACEMENT_FIELDS, VELOCITY_FIELDS, Units
from .state import AppState
from .widgets import IntField, NoWheelCombo, Section, hint, muted, row

UNIT_CHOICES = ("px", "nm", "µm", "mm", "cm", "m", "in")
MEASURE_LABELS = {"green_lagrange": "Green-Lagrange", "infinitesimal": "Infinitesimal (small strain)",
                  "euler_almansi": "Euler-Almansi"}


class SciSpin(QDoubleSpinBox):
    """Double spin box with enough decimals for strains and displacements."""

    def __init__(self) -> None:
        super().__init__()
        self.setDecimals(6)
        self.setRange(-1e9, 1e9)
        self.setKeyboardTracking(False)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class DisplayPanel(Section):
    def __init__(self, state: AppState) -> None:
        super().__init__("Display")
        self.state = state
        self.field = NoWheelCombo()
        self.field.setToolTip("Field shown on the canvas (1-9 to switch quickly)")
        self.add_layout(row("Field", self.field, stretch_last=True))
        self.ref_btn = QRadioButton("Reference")
        self.def_btn = QRadioButton("Deformed")
        self.ref_btn.setToolTip("Draw values at the frame-0 position of each material point")
        self.def_btn.setToolTip("Draw values where the material is in the current frame (D)")
        g = QButtonGroup(self)
        g.addButton(self.ref_btn)
        g.addButton(self.def_btn)
        self.add_layout(row("Drawn on", self.ref_btn, self.def_btn))
        self.background = NoWheelCombo()
        self.background.addItem("matching (frame 0 / current)", "auto")
        self.background.addItem("current frame", "frame")
        self.background.addItem("frame 0", "reference")
        self.background.addItem("none", "none")
        self.background.setToolTip("Image behind the field. 'Matching' shows frame 0 under the reference "
                                   "configuration and the current frame under the deformed one.")
        self.add_layout(row("Image", self.background, stretch_last=True))
        self.cmap = NoWheelCombo()
        self.cmap.addItems(list(COLORMAPS))
        self.levels = NoWheelCombo()
        for n in (0, 5, 8, 10, 12, 16, 20):
            self.levels.addItem("continuous" if n == 0 else f"{n} bands", n)
        self.add_layout(row("Colours", self.cmap, self.levels))
        self.auto = QCheckBox("Auto range")
        self.scope = NoWheelCombo()
        self.scope.addItem("this frame", "frame")
        self.scope.addItem("all frames", "sequence")
        self.scope.setToolTip("Auto range from the current frame or from the whole sequence (fixed while playing)")
        self.add_layout(row(self.auto, self.scope))
        self.vmin = SciSpin()
        self.vmax = SciSpin()
        self.add_layout(row("min", self.vmin, "max", self.vmax))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.add_layout(row("Opacity", self.opacity, stretch_last=True))
        self.show_roi = QCheckBox("Show ROI outline")
        self.show_overlay = QCheckBox("Show field (F)")
        self.add_layout(row(self.show_overlay, self.show_roi))
        self.readout = muted("")
        self.add(self.readout)
        for w, sig in ((self.field, "currentIndexChanged"), (self.ref_btn, "toggled"), (self.background, "currentIndexChanged"),
                       (self.cmap, "currentTextChanged"), (self.levels, "currentIndexChanged"), (self.auto, "toggled"),
                       (self.scope, "currentIndexChanged"), (self.vmin, "valueChanged"), (self.vmax, "valueChanged"),
                       (self.opacity, "valueChanged"), (self.show_roi, "toggled"), (self.show_overlay, "toggled")):
            getattr(w, sig).connect(self._apply)
        state.view_changed.connect(self.sync)
        state.strain_changed.connect(self._fill_fields)
        state.results_changed.connect(self._fill_fields)
        self._fill_fields()

    def _fill_fields(self) -> None:
        st = self.state
        self.field.blockSignals(True)
        self.field.clear()
        specs = st.provider.available() if st.provider is not None else list(DISPLACEMENT_FIELDS) + list(VELOCITY_FIELDS)
        for s in specs:
            self.field.addItem(s.label, s.key)
        self.field.blockSignals(False)
        self.sync()

    def _apply(self, *_args) -> None:
        st = self.state
        key = self.field.currentData()
        if key is None:
            return
        v = st.view
        changed_field = key != v.field
        v = replace(v, field=key, configuration="deformed" if self.def_btn.isChecked() else "reference",
                    background=self.background.currentData(), opacity=self.opacity.value() / 100.0,
                    show_roi=self.show_roi.isChecked(), show_overlay=self.show_overlay.isChecked(),
                    range_scope=self.scope.currentData())
        if not changed_field:
            d = v.display(key)
            lo, hi = float(self.vmin.value()), float(self.vmax.value())
            auto = self.auto.isChecked()
            if not auto and hi <= lo:
                hi = lo + max(abs(lo) * 1e-3, 1e-6)
            v = v.with_display(key, colormap=self.cmap.currentText(), auto=auto, vmin=lo if not auto else d.vmin,
                               vmax=hi if not auto else d.vmax, levels=int(self.levels.currentData() or 0))
        st.set_view(v)

    def set_auto_range(self, vmin: float, vmax: float) -> None:
        """The canvas reports the range it used; shown in the (disabled) min / max boxes."""
        if self.auto.isChecked():
            for w, val in ((self.vmin, vmin), (self.vmax, vmax)):
                w.blockSignals(True)
                w.setValue(val)
                w.blockSignals(False)

    def sync(self) -> None:
        v = self.state.view
        d = v.display(v.field)
        ws = (self.field, self.ref_btn, self.def_btn, self.background, self.cmap, self.levels, self.auto, self.scope,
              self.vmin, self.vmax, self.opacity, self.show_roi, self.show_overlay)
        for w in ws:
            w.blockSignals(True)
        idx = self.field.findData(v.field)
        if idx >= 0:
            self.field.setCurrentIndex(idx)
        self.ref_btn.setChecked(v.configuration == "reference")
        self.def_btn.setChecked(v.configuration == "deformed")
        self.background.setCurrentIndex(max(0, self.background.findData(v.background)))
        self.cmap.setCurrentText(d.colormap)
        self.levels.setCurrentIndex(max(0, self.levels.findData(d.levels)))
        self.auto.setChecked(d.auto)
        self.scope.setCurrentIndex(max(0, self.scope.findData(v.range_scope)))
        self.scope.setEnabled(d.auto)
        if not d.auto:
            self.vmin.setValue(d.vmin)
            self.vmax.setValue(d.vmax)
        self.vmin.setEnabled(not d.auto)
        self.vmax.setEnabled(not d.auto)
        self.opacity.setValue(int(round(v.opacity * 100)))
        self.show_roi.setChecked(v.show_roi)
        self.show_overlay.setChecked(v.show_overlay)
        for w in ws:
            w.blockSignals(False)
        spec = ALL_FIELDS.get(v.field)
        self.readout.setText("" if spec is None else f"Units: {self.state.units.unit(spec)}")


class UnitsPanel(Section):
    def __init__(self, state: AppState) -> None:
        super().__init__("Units", expanded=False)
        self.state = state
        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setDecimals(6)
        self.pixel_size.setRange(1e-9, 1e9)
        self.pixel_size.setValue(1.0)
        self.unit = NoWheelCombo()
        self.unit.addItems(list(UNIT_CHOICES))
        self.add_layout(row("1 px =", self.pixel_size, self.unit))
        self.rate_on = QCheckBox("Frame rate")
        self.rate = QDoubleSpinBox()
        self.rate.setDecimals(3)
        self.rate.setRange(1e-6, 1e9)
        self.rate.setValue(1.0)
        self.rate.setSuffix(" fps")
        self.add_layout(row(self.rate_on, self.rate))
        self.add(muted("Displacements and velocities are shown and exported in these units; "
                       "exports also keep the pixel values."))
        for w, sig in ((self.pixel_size, "valueChanged"), (self.unit, "currentTextChanged"), (self.rate_on, "toggled"),
                       (self.rate, "valueChanged")):
            getattr(w, sig).connect(self._apply)
        state.units_changed.connect(self.sync)
        self.sync()

    def _apply(self, *_a) -> None:
        unit = self.unit.currentText()
        self.state.set_units(Units(pixel_size=float(self.pixel_size.value()), length_unit=unit,
                                   frame_rate=float(self.rate.value()) if self.rate_on.isChecked() else None))
        self.pixel_size.setEnabled(unit != "px")
        self.rate.setEnabled(self.rate_on.isChecked())

    def sync(self) -> None:
        u = self.state.units
        for w in (self.pixel_size, self.unit, self.rate_on, self.rate):
            w.blockSignals(True)
        self.unit.setCurrentText(u.length_unit)
        self.pixel_size.setValue(u.pixel_size)
        self.rate_on.setChecked(u.frame_rate is not None)
        if u.frame_rate:
            self.rate.setValue(u.frame_rate)
        for w in (self.pixel_size, self.unit, self.rate_on, self.rate):
            w.blockSignals(False)
        self.pixel_size.setEnabled(u.length_unit != "px")
        self.rate.setEnabled(u.frame_rate is not None)


class StrainPanel(Section):
    compute = Signal()

    def __init__(self, state: AppState) -> None:
        super().__init__("Strain")
        self.state = state
        self.measure = NoWheelCombo()
        for k in STRAIN_MEASURES:
            self.measure.addItem(MEASURE_LABELS[k], k)
        self.add_layout(row("Measure", self.measure, stretch_last=True))
        self.gauge = IntField()
        self.gauge.setRange(3, 1001)
        self.gauge.setSingleStep(2)
        self.gauge.setSuffix(" px")
        self.gauge.setToolTip("Virtual strain gauge side Lg (odd): the window of each local fit")
        self.weight = NoWheelCombo()
        self.weight.addItem("uniform", "uniform")
        self.weight.addItem("Gaussian (σ = Lg/4)", "gaussian")
        self.add_layout(row("Gauge", self.gauge, self.weight))
        self.order = NoWheelCombo()
        self.order.addItem("plane", 1)
        self.order.addItem("quadratic", 2)
        self.step = IntField()
        self.step.setRange(1, 64)
        self.step.setSuffix(" px")
        self.step.setToolTip("Spacing of the strain points (1 = every pixel)")
        self.add_layout(row("Fit", self.order, "step", self.step))
        self.presmooth = QDoubleSpinBox()
        self.presmooth.setRange(0.0, 50.0)
        self.presmooth.setSingleStep(0.5)
        self.presmooth.setSuffix(" px")
        self.presmooth.setToolTip("Gaussian smoothing of the displacement before the fit (0 = none)")
        self.crack = QCheckBox("Never across cracks / holes")
        self.crack.setToolTip("Fit each connected part of the valid region separately")
        self.add_layout(row("Pre-smooth σ", self.presmooth))
        self.add(self.crack)
        paper = QPushButton("Paper settings")
        paper.setToolTip("Green-Lagrange, unweighted plane fit, Lg = 41 px, no smoothing")
        paper.clicked.connect(lambda: state.set_strain_config(PAPER_STRAIN))
        self.go = QPushButton("Compute strain")
        self.go.setProperty("role", "primary")
        self.go.clicked.connect(self.compute)
        self.add_layout(row(self.go, paper))
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setVisible(False)
        self.add(self.bar)
        self.status = muted("Run the correlation first.")
        self.add(self.status)
        for w, sig in ((self.measure, "currentIndexChanged"), (self.gauge, "valueChanged"), (self.weight, "currentIndexChanged"),
                       (self.order, "currentIndexChanged"), (self.step, "valueChanged"), (self.presmooth, "valueChanged"),
                       (self.crack, "toggled")):
            getattr(w, sig).connect(self._apply)
        state.strain_config_changed.connect(self.sync)
        state.results_changed.connect(self._status)
        state.strain_changed.connect(self._status)
        self.sync()
        self._status()

    def _apply(self, *_a) -> None:
        g = self.gauge.value()
        if g % 2 == 0:  # the gauge has a centre pixel: show the size that is used
            g += 1
            self.gauge.blockSignals(True)
            self.gauge.setValue(g)
            self.gauge.blockSignals(False)
        try:
            cfg = replace(self.state.strain_config, measure=self.measure.currentData(), gauge_px=g,
                          weighting=self.weight.currentData(), poly_order=int(self.order.currentData()),
                          step=self.step.value(), smoothing_sigma=float(self.presmooth.value()),
                          connectivity_aware=self.crack.isChecked())
        except ValueError as exc:
            self.state.emit_log(str(exc), "warning")
            return
        self.state.set_strain_config(cfg)

    def sync(self) -> None:
        c = self.state.strain_config
        ws = (self.measure, self.gauge, self.weight, self.order, self.step, self.presmooth, self.crack)
        for w in ws:
            w.blockSignals(True)
        self.measure.setCurrentIndex(max(0, self.measure.findData(c.measure)))
        self.gauge.setValue(c.gauge_px)
        self.weight.setCurrentIndex(max(0, self.weight.findData(c.weighting)))
        self.order.setCurrentIndex(max(0, self.order.findData(c.poly_order)))
        self.step.setValue(c.step)
        self.presmooth.setValue(c.smoothing_sigma)
        self.crack.setChecked(c.connectivity_aware)
        for w in ws:
            w.blockSignals(False)
        self._status()

    def _status(self) -> None:
        st = self.state
        self.go.setEnabled(st.series is not None and st.run_state == "idle")
        if st.series is None:
            self.status.setText("Run the correlation first.")
        elif st.strain is None:
            self.status.setText("Not computed yet.")
        else:
            s = st.strain.settings
            paper = (s.measure, s.weighting, s.poly_order, s.smoothing_sigma) == ("green_lagrange", "uniform", 1, 0.0)
            changed = StrainConfig.from_dict({**self.state.strain_config.to_dict()}) != StrainConfig(
                measure=s.measure, gauge_px=s.gauge_px, weighting=s.weighting, poly_order=s.poly_order, step=s.step,
                gaussian_sigma=s.gaussian_sigma, smoothing_sigma=s.smoothing_sigma, connectivity_aware=s.connectivity_aware,
                condition_max=self.state.strain_config.condition_max, min_coverage=self.state.strain_config.min_coverage)
            self.status.setText(f"Computed: {MEASURE_LABELS[s.measure]}, Lg {s.gauge_px} px, {s.weighting}, "
                                f"order {s.poly_order}" + (" (paper settings)" if paper else "")
                                + ("\nSettings changed since: compute again." if changed else ""))

    def set_progress(self, f: float | None) -> None:
        self.bar.setVisible(f is not None)
        if f is not None:
            self.bar.setValue(int(1000 * f))


class ProbesPanel(Section):
    tool_selected = Signal(str)
    export_csv = Signal()

    def __init__(self, state: AppState) -> None:
        super().__init__("Probes")
        self.state = state
        self.add(muted("Place probes on the canvas; their time series appear in the Chart tab below."))
        grid = QGridLayout()
        grid.setSpacing(4)
        self.tools = QButtonGroup(self)
        self.tools.setExclusive(False)
        for i, (key, text, tip) in enumerate((
            ("probe_point", "• Point", "Point probe: click"),
            ("probe_line", "╱ Line", "Line probe / extensometer: drag"),
            ("probe_rect", "▭ Rect", "Rectangular area: drag"),
            ("probe_circle", "◯ Circle", "Circular area: drag from the centre"),
            ("probe_polygon", "⬠ Polygon", "Polygonal area: click vertices, double-click to close"),
        )):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setProperty("tool", key)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.tools.addButton(b)
            grid.addWidget(b, i // 3, i % 3)
        self.tools.buttonClicked.connect(self._tool_clicked)
        self.add_layout(grid)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Probe", "Type", "Where"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setMaximumHeight(150)
        self.add(self.table)
        self.metric_combo = NoWheelCombo()
        self.metric_combo.addItems(list(METRICS))
        self.metric_combo.setToolTip("How line and area probes reduce their points to one value")
        rm = QPushButton("Remove")
        rm.clicked.connect(self._remove)
        clr = QPushButton("Clear")
        clr.clicked.connect(state.clear_probes)
        csv = QPushButton("CSV…")
        csv.setToolTip("Export the time series of every probe")
        csv.clicked.connect(self.export_csv)
        self.add_layout(row("Areas/lines:", self.metric_combo))
        self.add_layout(row(rm, clr, csv))
        state.probes_changed.connect(self.refresh)
        self.refresh()

    def _tool_clicked(self, btn) -> None:
        on = btn.isChecked()
        for b in self.tools.buttons():
            if b is not btn:
                b.setChecked(False)
        self.tool_selected.emit(btn.property("tool") if on else "pan")

    def set_tool(self, tool: str) -> None:
        for b in self.tools.buttons():
            b.setChecked(b.property("tool") == tool)

    def refresh(self) -> None:
        ps = self.state.probes
        self.table.setRowCount(len(ps))
        for i, p in enumerate(ps):
            it = QTableWidgetItem(p.label)
            from PySide6.QtGui import QColor

            it.setForeground(QColor(p.color))
            self.table.setItem(i, 0, it)
            self.table.setItem(i, 1, QTableWidgetItem(p.kind))
            self.table.setItem(i, 2, QTableWidgetItem(p.describe()))

    def _remove(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        ps = self.state.probes
        if not rows and ps:
            rows = [len(ps) - 1]
        for r in rows:
            if 0 <= r < len(ps):
                self.state.remove_probe(ps[r].id)


class RightSidebar(QWidget):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.setObjectName("sidebarContents")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        self.display = DisplayPanel(state)
        self.units = UnitsPanel(state)
        self.strain = StrainPanel(state)
        self.probes = ProbesPanel(state)
        for w in (self.display, self.units, self.strain, self.probes):
            lay.addWidget(w)
        lay.addStretch(1)
