"""Strain-only parameter panel for the post-processing window.

Edits a :class:`raftcorr.config.StrainConfig` (RAFTcorr's virtual strain
gauge, VSG):

* ``measure`` -- strain type: infinitesimal, Eulerian-Almansi or
  Green-Lagrangian;
* ``gauge_px`` -- VSG size ``Lg`` (odd, px): side of the square window of
  each local fit;
* ``poly_order`` -- fit: plane (1st order) or quadratic (2nd order);
* ``weighting`` -- uniform or Gaussian (``gaussian_sigma``; Auto = Lg / 4);
* ``step`` -- output spacing of the strain points (px);
* ``smoothing_sigma`` -- Gaussian pre-smoothing of the displacement (px);
* ``connectivity_aware`` -- never fit a gauge across a crack or a hole.

"Paper settings" restores :data:`raftcorr.config.PAPER_STRAIN`
(Green-Lagrangian, unweighted plane fit, Lg = 41 px, no smoothing).
Fields the panel does not show (``condition_max``, ``min_coverage``) keep the
values of the configuration last loaded with :meth:`set_config`.

Tracks a dirty flag so the window can show a "Stale" hint until the user
explicitly recomputes.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from raftcorr.config import PAPER_STRAIN, StrainConfig
from raftcorr.gui.widgets.double_spin import LocaleSafeDoubleSpinBox
from raftcorr.gui.widgets.info_icon import InfoIcon

#: Strain measures in combo order (pyALDIC's order: infinitesimal, Eulerian,
#: Green-Lagrangian).
_MEASURES: tuple[str, ...] = ("infinitesimal", "euler_almansi", "green_lagrange")
_WEIGHTINGS: tuple[str, ...] = ("uniform", "gaussian")
_ORDERS: tuple[int, ...] = (1, 2)

_VSG_MIN, _VSG_MAX = 3, 401
_STEP_MAX = 64
_SIGMA_MAX = 50.0


def _label_with_info(text: str, tip: str) -> QWidget:
    """A form label followed by an ⓘ icon carrying *tip*."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(2)
    row.addWidget(QLabel(text))
    row.addWidget(InfoIcon(tip))
    row.addStretch()
    widget = QWidget()
    widget.setLayout(row)
    return widget


class StrainParamPanel(QWidget):
    """Compose measure / VSG / fit / weighting / smoothing editors with a dirty flag."""

    params_dirty = Signal()
    #: The configuration in the controls changed (by the user or Paper settings).
    config_changed = Signal(object)

    def __init__(self, config: StrainConfig | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._base = config if config is not None else StrainConfig()

        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # --- Method: polynomial order of the local fit ---
        self._method_combo = QComboBox()
        self._method_combo.addItem(self.tr("Plane fitting"), 1)
        self._method_combo.addItem(self.tr("Quadratic fitting"), 2)
        self._method_combo.setToolTip(self.tr(
            "Polynomial fitted to each displacement component over the VSG "
            "window. Its slopes at the centre are the displacement gradients.\n"
            "• Plane fitting (1st order): the standard VSG (paper setting).\n"
            "• Quadratic fitting (2nd order): follows strain gradients inside "
            "the window better, at the cost of more noise."
        ))
        layout.addRow(self.tr("Method"), self._method_combo)

        # --- VSG size (virtual strain gauge side, pixels) ---
        self._vsg_spin = QSpinBox()
        self._vsg_spin.setRange(_VSG_MIN, _VSG_MAX)
        self._vsg_spin.setSingleStep(2)
        self._vsg_spin.setSuffix(" px")
        vsg_tip = self.tr(
            "VSG (Virtual Strain Gauge) size Lg is the side, in pixels, of the "
            "square window around each point over which the displacement is "
            "fitted. Strain is then taken from the fit's slopes.\n\n"
            "• Larger VSG → smoother strain, lower spatial resolution.\n"
            "• Smaller VSG → sharper strain, more noise.\n"
            "• Odd, so that the window has a centre pixel (default: 41 px)."
        )
        layout.addRow(_label_with_info(self.tr("VSG size"), vsg_tip), self._vsg_spin)

        # --- Weighting ---
        self._weight_combo = QComboBox()
        self._weight_combo.addItem(self.tr("Uniform"), "uniform")
        self._weight_combo.addItem(self.tr("Gaussian"), "gaussian")
        self._weight_combo.setToolTip(self.tr(
            "Weight of the points inside the VSG window: all equal (uniform, "
            "paper setting), or decreasing with the distance from the centre "
            "(Gaussian)."
        ))
        layout.addRow(self.tr("Weighting"), self._weight_combo)

        # --- Gaussian sigma (Gaussian weighting only; 0 = Auto = Lg / 4) ---
        self._sigma_spin = LocaleSafeDoubleSpinBox()
        self._sigma_spin.setRange(0.0, 200.0)
        self._sigma_spin.setDecimals(1)
        self._sigma_spin.setSingleStep(0.5)
        self._sigma_spin.setSuffix(" px")
        self._sigma_spin.setSpecialValueText(self.tr("Auto (Lg / 4)"))
        self._sigma_spin.setToolTip(self.tr(
            "Standard deviation of the Gaussian weight. Auto uses a quarter "
            "of the VSG size."
        ))
        layout.addRow(self.tr("Gaussian σ"), self._sigma_spin)

        # --- Output spacing ---
        self._step_spin = QSpinBox()
        self._step_spin.setRange(1, _STEP_MAX)
        self._step_spin.setSuffix(" px")
        self._step_spin.setToolTip(self.tr(
            "Spacing of the strain points: 1 computes strain at every pixel; "
            "larger values are faster and lighter for long sequences."
        ))
        layout.addRow(self.tr("Output spacing"), self._step_spin)

        # --- Displacement pre-smoothing ---
        self._smooth_spin = LocaleSafeDoubleSpinBox()
        self._smooth_spin.setRange(0.0, _SIGMA_MAX)
        self._smooth_spin.setDecimals(1)
        self._smooth_spin.setSingleStep(0.5)
        self._smooth_spin.setSuffix(" px")
        self._smooth_spin.setSpecialValueText(self.tr("Off", "Strain smoothing preset"))
        self._smooth_spin.setToolTip(self.tr(
            "Gaussian smoothing (σ in pixels) of the displacement before the "
            "fit. Off (0) differentiates the measured displacement as it is "
            "(paper setting)."
        ))
        layout.addRow(self.tr("Displacement smoothing"), self._smooth_spin)

        # --- Crack-aware fitting ---
        self._crack_check = QCheckBox(self.tr("Never fit across cracks or holes"))
        self._crack_check.setToolTip(self.tr(
            "Fit each connected part of the tracked region separately, so no "
            "VSG window ever spans a crack, a hole or the gap between two "
            "regions."
        ))
        # Full-width row: the sentence does not fit in the field column.
        layout.addRow(self._crack_check)

        # --- Strain type ---
        self._type_combo = QComboBox()
        self._type_combo.addItem(self.tr("Infinitesimal"), "infinitesimal")
        self._type_combo.addItem(self.tr("Eulerian"), "euler_almansi")
        self._type_combo.addItem(self.tr("Green-Lagrangian"), "green_lagrange")
        self._type_combo.setToolTip(self.tr(
            "Strain measure from the deformation gradient F = I + ∇u:\n"
            "• Infinitesimal: (∇u + ∇uᵀ) / 2 (small strains only).\n"
            "• Eulerian (Euler-Almansi): (I − (F Fᵀ)⁻¹) / 2.\n"
            "• Green-Lagrangian: (Fᵀ F − I) / 2 (paper setting)."
        ))
        layout.addRow(self.tr("Strain type"), self._type_combo)

        # --- Paper settings ---
        self._paper_btn = QPushButton(self.tr("Paper settings"))
        self._paper_btn.setToolTip(self.tr(
            "Green-Lagrangian strain, unweighted plane fit, VSG 41 px, every "
            "pixel, no smoothing, never across cracks (the settings of the "
            "RAFTcorr paper)."
        ))
        self._paper_btn.clicked.connect(self.apply_paper_settings)
        layout.addRow("", self._paper_btn)

        self._dirty = False
        self._loading = False
        self._show_config(self._base)

        # Wire dirty propagation
        self._method_combo.currentIndexChanged.connect(self._on_edited)
        self._vsg_spin.valueChanged.connect(self._on_vsg_value_changed)
        self._weight_combo.currentIndexChanged.connect(self._on_weighting_changed)
        self._sigma_spin.valueChanged.connect(self._on_edited)
        self._step_spin.valueChanged.connect(self._on_edited)
        self._smooth_spin.valueChanged.connect(self._on_edited)
        self._crack_check.toggled.connect(self._on_edited)
        self._type_combo.currentIndexChanged.connect(self._on_edited)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_config(self) -> StrainConfig:
        """The configuration in the controls (always valid: the controls
        only offer valid values)."""
        sigma = float(self._sigma_spin.value())
        return replace(
            self._base,
            measure=str(self._type_combo.currentData()),
            gauge_px=int(self._vsg_spin.value()) | 1,
            weighting=str(self._weight_combo.currentData()),
            poly_order=int(self._method_combo.currentData()),
            step=int(self._step_spin.value()),
            gaussian_sigma=sigma if sigma > 0 else None,
            smoothing_sigma=float(self._smooth_spin.value()),
            connectivity_aware=self._crack_check.isChecked(),
        )

    def set_config(self, config: StrainConfig) -> None:
        """Show *config* without marking the panel dirty or emitting signals."""
        self._base = config
        self._show_config(config)

    def apply_paper_settings(self) -> None:
        """Set every control to :data:`PAPER_STRAIN` (dirty if anything changed)."""
        before = self.get_config()
        paper = replace(PAPER_STRAIN, condition_max=self._base.condition_max,
                        min_coverage=self._base.min_coverage)
        self._show_config(paper)
        if self.get_config() != before:
            self._mark_dirty()

    def is_paper_settings(self) -> bool:
        """True when the controls hold the paper's settings."""
        c = self.get_config()
        p = PAPER_STRAIN
        return (c.measure, c.gauge_px, c.weighting, c.poly_order, c.step, c.smoothing_sigma,
                c.connectivity_aware) == (p.measure, p.gauge_px, p.weighting, p.poly_order, p.step,
                                          p.smoothing_sigma, p.connectivity_aware)

    def is_dirty(self) -> bool:
        """Return True if any parameter changed since the last
        :meth:`mark_clean` call."""
        return self._dirty

    def mark_clean(self) -> None:
        """Reset the dirty flag (typically after a successful compute)."""
        self._dirty = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _controls(self) -> tuple[QWidget, ...]:
        return (self._method_combo, self._vsg_spin, self._weight_combo, self._sigma_spin,
                self._step_spin, self._smooth_spin, self._crack_check, self._type_combo)

    def _show_config(self, c: StrainConfig) -> None:
        self._loading = True
        for w in self._controls():
            w.blockSignals(True)
        try:
            self._method_combo.setCurrentIndex(max(0, self._method_combo.findData(int(c.poly_order))))
            self._vsg_spin.setValue(int(c.gauge_px))
            self._weight_combo.setCurrentIndex(max(0, self._weight_combo.findData(c.weighting)))
            self._sigma_spin.setValue(float(c.gaussian_sigma) if c.gaussian_sigma else 0.0)
            self._step_spin.setValue(int(c.step))
            self._smooth_spin.setValue(float(c.smoothing_sigma))
            self._crack_check.setChecked(bool(c.connectivity_aware))
            self._type_combo.setCurrentIndex(max(0, self._type_combo.findData(c.measure)))
        finally:
            for w in self._controls():
                w.blockSignals(False)
            self._loading = False
        self._sigma_spin.setEnabled(c.weighting == "gaussian")

    def _on_vsg_value_changed(self, value: int) -> None:
        """Snap even inputs to the next odd integer, then mark dirty."""
        if value % 2 == 0:
            self._vsg_spin.blockSignals(True)
            self._vsg_spin.setValue(min(_VSG_MAX, value + 1))
            self._vsg_spin.blockSignals(False)
        self._on_edited()

    def _on_weighting_changed(self, *_args: object) -> None:
        self._sigma_spin.setEnabled(self._weight_combo.currentData() == "gaussian")
        self._on_edited()

    def _on_edited(self, *_args: object) -> None:
        if self._loading:
            return
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        self._dirty = True
        self.params_dirty.emit()
        self.config_changed.emit(self.get_config())


__all__ = ["StrainParamPanel"]
