"""RAFTcorr parameter panels in the pyALDIC style: PARAMETERS and ADVANCED (tiling).

Defaults: 20 iterations, raw displacements (no smoothing). Labels are Title Case like the rest
of the main window; the label column fits the longest label of the current language (pyALDIC's
rule), and checkboxes that belong to a row sit in the input column.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from raftcorr.gui.app_state import AppState

#: The label column is as wide as its longest label in the current language, never narrower
#: than the English layout and, past about 28 characters of the label font, wrapped rather than
#: squeezing the inputs (pyALDIC's ParamPanel).
_LABEL_MIN_WIDTH = 120
_LABEL_MAX_CHARS = 28
_LABEL_PADDING = 6
_ROW_SPACING = 6        # between a label and its input


def _row(label: str, widget: QWidget, layout: QVBoxLayout, tip: str = "") -> QLabel:
    row = QHBoxLayout()
    row.setSpacing(_ROW_SPACING)
    lbl = QLabel(label)
    if tip:
        lbl.setToolTip(tip)
        widget.setToolTip(tip)
    row.addWidget(lbl)
    row.addWidget(widget)
    layout.addLayout(row)
    return lbl


def _fit_label_column(labels: list[QLabel], aligned: list[QLayout]) -> None:
    """Give the labels the width of the longest one; indent *aligned* rows to the input column."""
    if not labels:
        return
    labels[0].ensurePolished()
    metrics = labels[0].fontMetrics()
    widest = max(metrics.horizontalAdvance(lbl.text()) for lbl in labels)
    cap = max(_LABEL_MIN_WIDTH, _LABEL_MAX_CHARS * metrics.averageCharWidth())
    width = min(max(_LABEL_MIN_WIDTH, widest + _LABEL_PADDING), cap)
    for lbl in labels:
        lbl.setWordWrap(True)
        lbl.setFixedWidth(width)
    for item in aligned:
        item.setContentsMargins(width + _ROW_SPACING, 0, 0, 0)


class _WheelGuard(QWidget):
    """Base: spin boxes and combos ignore the wheel unless focused (no accidental edits)."""

    def _guard(self, *widgets: QWidget) -> None:
        for w in widgets:
            w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            w.installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)


class ParamPanel(_WheelGuard):
    """Iterations, displacement smoothing, device."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        state = AppState.instance()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 4, 0)

        self._iters = QSpinBox()
        self._iters.setRange(1, 100)
        self._iters.setValue(state.iters)
        labels = [_row(self.tr("Iterations"), self._iters, layout,
                       self.tr("Refinement iterations of the network (default: 20)."))]

        self._smooth = QDoubleSpinBox()
        self._smooth.setRange(0.0, 50.0)
        self._smooth.setSingleStep(0.5)
        self._smooth.setDecimals(1)
        self._smooth.setSuffix(" px")
        self._smooth.setSpecialValueText(self.tr("Off", "Displacement smoothing"))
        self._smooth.setValue(state.smoothing_sigma)
        # "Smoothing" here (the results); the strain window's "Pre-fit smoothing" only feeds its fit.
        labels.append(_row(self.tr("Smoothing"), self._smooth, layout, self.tr(
            "Gaussian smoothing (σ in pixels) of the tracked displacement, kept in the results.\n"
            "Off by default: smoothing coarsens the spatial resolution.")))

        self._device = QComboBox()
        self._device.addItem(self.tr("Automatic"), "auto")
        self._device.addItem("GPU (CUDA)", "cuda")
        self._device.addItem("CPU", "cpu")
        self._device.setCurrentIndex(max(0, self._device.findData(state.device)))
        labels.append(_row(self.tr("Device"), self._device, layout,
                           self.tr("Automatic uses the GPU when PyTorch sees one.")))
        _fit_label_column(labels, [])

        self._iters.valueChanged.connect(lambda v: state.set_param("iters", int(v)))
        self._smooth.valueChanged.connect(lambda v: state.set_param("smoothing_sigma", float(v)))
        self._device.currentIndexChanged.connect(lambda _i: state.set_param("device", self._device.currentData()))
        state.params_changed.connect(self._sync)
        self._guard(self._iters, self._smooth, self._device)

    def _sync(self) -> None:
        state = AppState.instance()
        for w, v in ((self._iters, state.iters), (self._smooth, state.smoothing_sigma)):
            if w.value() != v:
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)
        idx = self._device.findData(state.device)
        if idx >= 0 and idx != self._device.currentIndex():
            self._device.blockSignals(True)
            self._device.setCurrentIndex(idx)
            self._device.blockSignals(False)


class AdvancedPanel(_WheelGuard):
    """Tiling: displacement margin, memory budget, seam blending, correlation implementation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        state = AppState.instance()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 4, 0)

        self._margin = QSpinBox()
        self._margin.setRange(0, 4096)
        self._margin.setSuffix(" px")
        self._margin.setValue(state.margin_px if state.margin_px is not None else 128)
        self._margin.setEnabled(state.margin_px is not None)
        labels = [_row(self.tr("Margin"), self._margin, layout,
                       self.tr("Must cover the largest displacement of the region of interest."))]

        # In the input column under its row, as pyALDIC's refinement checkboxes.
        self._margin_auto = QCheckBox(self.tr("Use the Model Reach"))
        self._margin_auto.setChecked(state.margin_px is None)
        self._margin_auto.setToolTip(self.tr(
            "Extra image context around the region of interest so that displaced points stay inside\n"
            "the window given to the network. Automatic: the model's reach (32 s px)."))
        auto_row = QHBoxLayout()
        auto_row.addWidget(self._margin_auto)
        layout.addLayout(auto_row)

        self._mem = QDoubleSpinBox()
        self._mem.setRange(0.05, 1.0)
        self._mem.setSingleStep(0.05)
        self._mem.setDecimals(2)
        self._mem.setValue(state.memory_fraction)
        labels.append(_row(self.tr("GPU Memory"), self._mem, layout,
                           self.tr("Fraction of the free GPU memory one window may use.\n"
                                   "Large regions are cut into tiles to fit it.")))

        self._blend = QSpinBox()
        self._blend.setRange(0, 512)
        self._blend.setSuffix(" px")
        self._blend.setValue(state.blend_px)
        labels.append(_row(self.tr("Tile Blending"), self._blend, layout,
                           self.tr("Width of the feathered overlap between neighbouring tiles.")))

        self._corr = QComboBox()
        self._corr.addItem(self.tr("Automatic"), None)
        self._corr.addItem(self.tr("On Demand"), "tile")
        self._corr.addItem(self.tr("Stored Volume"), "volume")
        self._corr.setCurrentIndex(max(0, self._corr.findData(state.corr_mode)))
        labels.append(_row(self.tr("Correlation"), self._corr, layout,
                           self.tr("How the correlation lookup is computed; the results are identical.\n"
                                   "Automatic: stored volume for s8 (fast), on demand for s1-s4\n"
                                   "(memory linear in the image size).")))
        _fit_label_column(labels, [auto_row])

        self._margin_auto.toggled.connect(self._margin_changed)
        self._margin.valueChanged.connect(self._margin_changed)
        self._mem.valueChanged.connect(lambda v: state.set_param("memory_fraction", float(v)))
        self._blend.valueChanged.connect(lambda v: state.set_param("blend_px", int(v)))
        self._corr.currentIndexChanged.connect(lambda _i: state.set_param("corr_mode", self._corr.currentData()))
        state.params_changed.connect(self._sync)
        self._guard(self._margin, self._mem, self._blend, self._corr)

    def _sync(self) -> None:
        """Follow changes made elsewhere (project loading), without writing them back.

        A widget is set only when it differs from the state, as in :meth:`ParamPanel._sync`: each
        keystroke in a spin box sets the parameter and comes back here through ``params_changed``,
        and setting the box being typed in again would reformat its text under the cursor.
        """
        state = AppState.instance()
        auto = state.margin_px is None
        widgets = (self._margin_auto, self._margin, self._mem, self._blend, self._corr)
        for w in widgets:
            w.blockSignals(True)
        try:
            if self._margin_auto.isChecked() != auto:
                self._margin_auto.setChecked(auto)
            if not auto and self._margin.value() != state.margin_px:   # automatic: keep the last number
                self._margin.setValue(state.margin_px)
            self._margin.setEnabled(not auto)
            if abs(self._mem.value() - state.memory_fraction) > 1e-9:
                self._mem.setValue(state.memory_fraction)
            if self._blend.value() != state.blend_px:
                self._blend.setValue(state.blend_px)
            idx = self._corr.findData(state.corr_mode)
            if idx >= 0 and idx != self._corr.currentIndex():
                self._corr.setCurrentIndex(idx)
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _margin_changed(self, *_a) -> None:
        auto = self._margin_auto.isChecked()
        self._margin.setEnabled(not auto)
        AppState.instance().set_param("margin_px", None if auto else int(self._margin.value()))
