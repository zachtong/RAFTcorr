"""RAFTcorr parameter panels in the pyALDIC style: PARAMETERS and ADVANCED (tiling).

Defaults are the paper's: 20 iterations, raw displacements (no smoothing).
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from raftcorr.gui.app_state import AppState

_LABEL_W = 120


def _row(label: str, widget: QWidget, layout: QVBoxLayout, tip: str = "") -> QHBoxLayout:
    row = QHBoxLayout()
    lbl = QLabel(label)
    lbl.setFixedWidth(_LABEL_W)
    if tip:
        lbl.setToolTip(tip)
        widget.setToolTip(tip)
    row.addWidget(lbl)
    row.addWidget(widget)
    layout.addLayout(row)
    return row


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
        _row(self.tr("Iterations"), self._iters, layout,
             self.tr("Refinement iterations of the network (default: 20)."))

        self._smooth = QDoubleSpinBox()
        self._smooth.setRange(0.0, 50.0)
        self._smooth.setSingleStep(0.5)
        self._smooth.setDecimals(1)
        self._smooth.setSuffix(" px")
        self._smooth.setSpecialValueText(self.tr("off (raw)"))
        self._smooth.setValue(state.smoothing_sigma)
        _row(self.tr("Smoothing σ"), self._smooth, layout,
             self.tr("Gaussian smoothing of the displacement (standard deviation).\n"
                     "Off by default: smoothing coarsens the spatial resolution."))

        self._device = QComboBox()
        self._device.addItem(self.tr("Automatic"), "auto")
        self._device.addItem("GPU (CUDA)", "cuda")
        self._device.addItem("CPU", "cpu")
        self._device.setCurrentIndex(max(0, self._device.findData(state.device)))
        _row(self.tr("Device"), self._device, layout,
             self.tr("Automatic uses the GPU when PyTorch sees one."))

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

        self._margin_auto = QCheckBox(self.tr("Margin = model reach"))
        self._margin_auto.setChecked(state.margin_px is None)
        self._margin_auto.setToolTip(self.tr(
            "Extra image context around the region of interest so that displaced points stay inside\n"
            "the window given to the network. Automatic: the model's reach (32 s px)."))
        layout.addWidget(self._margin_auto)
        self._margin = QSpinBox()
        self._margin.setRange(0, 4096)
        self._margin.setSuffix(" px")
        self._margin.setValue(state.margin_px if state.margin_px is not None else 128)
        self._margin.setEnabled(state.margin_px is not None)
        _row(self.tr("Margin"), self._margin, layout,
             self.tr("Must cover the largest displacement of the region of interest."))

        self._mem = QDoubleSpinBox()
        self._mem.setRange(0.05, 1.0)
        self._mem.setSingleStep(0.05)
        self._mem.setDecimals(2)
        self._mem.setValue(state.memory_fraction)
        _row(self.tr("GPU memory"), self._mem, layout,
             self.tr("Fraction of the free GPU memory one window may use.\n"
                     "Large regions are cut into tiles to fit it."))

        self._blend = QSpinBox()
        self._blend.setRange(0, 512)
        self._blend.setSuffix(" px")
        self._blend.setValue(state.blend_px)
        _row(self.tr("Tile blending"), self._blend, layout,
             self.tr("Width of the feathered overlap between neighbouring tiles."))

        self._corr = QComboBox()
        self._corr.addItem(self.tr("Automatic"), None)
        self._corr.addItem(self.tr("On demand"), "tile")
        self._corr.addItem(self.tr("Stored volume"), "volume")
        self._corr.setCurrentIndex(max(0, self._corr.findData(state.corr_mode)))
        _row(self.tr("Correlation"), self._corr, layout,
             self.tr("How the correlation lookup is computed; the results are identical.\n"
                     "Automatic: stored volume for s8 (fast), on demand for s1-s4\n"
                     "(memory linear in the image size)."))

        self._margin_auto.toggled.connect(self._margin_changed)
        self._margin.valueChanged.connect(self._margin_changed)
        self._mem.valueChanged.connect(lambda v: state.set_param("memory_fraction", float(v)))
        self._blend.valueChanged.connect(lambda v: state.set_param("blend_px", int(v)))
        self._corr.currentIndexChanged.connect(lambda _i: state.set_param("corr_mode", self._corr.currentData()))
        self._guard(self._margin, self._mem, self._blend, self._corr)

    def _margin_changed(self, *_a) -> None:
        auto = self._margin_auto.isChecked()
        self._margin.setEnabled(not auto)
        AppState.instance().set_param("margin_px", None if auto else int(self._margin.value()))
