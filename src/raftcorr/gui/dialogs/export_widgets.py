"""Widgets of the export dialog: field labels, per-field rows, run controls (pyALDIC's).

Split from :mod:`raftcorr.gui.dialogs.export_dialog`, which re-exports :class:`VizExportHint` and
:class:`FieldImageConfig`. Texts are translated in the ``ExportDialog`` context like pyALDIC's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import QT_TRANSLATE_NOOP, QCoreApplication, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from raftcorr.export.scientific import PhysicalUnits
from raftcorr.gui.app_state import AppState
from raftcorr.gui.support.colormaps import COLORMAP_NAMES
from raftcorr.gui.theme_manager import themed
from raftcorr.gui.widgets.double_spin import LocaleSafeDoubleSpinBox
from raftcorr.gui.widgets.range_mode import AutoFixedSelector
from raftcorr.viz.fields import Units

# ---------------------------------------------------------------------------
# Friendly UI labels — match the main window / strain window naming
# ---------------------------------------------------------------------------

from raftcorr.gui.support.fields import STRAIN_FIELDS, strain_symbol  # noqa: E402

_EXPORT_LABELS: dict[str, str] = {
    "disp_u":                 "U",
    "disp_v":                 "V",
    "disp_magnitude":         QT_TRANSLATE_NOOP("ExportDialog", "Magnitude"),
    "residual":               QT_TRANSLATE_NOOP("ExportDialog", "Residual"),
    **{name: strain_symbol(name) for name in STRAIN_FIELDS},
}


def field_label(key: str) -> str:
    """Return the user-facing label for a canonical field name."""
    # Symbols (U, εxx) come back unchanged; words (Magnitude, Residual) are translated.
    return QCoreApplication.translate("ExportDialog", _EXPORT_LABELS.get(key, key))


#: Column widths of the field rows (export, field, colormap, range mode, min / max), shared by
#: the header labels of the Images and Animation tabs.
HEADER_WIDTHS: tuple[int, ...] = (50, 100, 100, 120, 95)

DEFAULT_ENABLED = {"disp_u", "disp_v", "strain_exx", "strain_eyy", "strain_exy"}

#: Older names that share a field's colour state.
_STATE_ALIASES: dict[str, tuple[str, ...]] = {"disp_magnitude": ("disp_magnitude", "disp_mag")}


def color_state(field_name: str):
    """The main window's colour state of *field_name* (created with defaults if never shown)."""
    state = AppState.instance()
    known = getattr(state, "_field_states", {})
    for alias in _STATE_ALIASES.get(field_name, (field_name,)):
        if alias in known:
            return state.get_field_state(alias)
    return state.get_field_state(field_name)


# ---------------------------------------------------------------------------
# Pure-data configuration types (no Qt, fully testable)
# ---------------------------------------------------------------------------

@dataclass
class VizExportHint:
    """Snapshot of the current window's visualisation settings.

    ``hidden_bg_color``, ``fill_trimmed_edges`` and ``hide_untracked`` are accepted for the
    callers ported from pyALDIC; RAFTcorr pictures always leave untracked points transparent and
    draw on white when the background image is hidden.
    """
    colormap: str = "turbo"
    auto_range: bool = True
    vmin: float = 0.0
    vmax: float = 1.0
    show_deformed: bool = False
    show_background: bool = True
    hidden_bg_color: str = "white"
    fill_trimmed_edges: bool = False
    hide_untracked: bool = True
    overlay_alpha: float = 0.7
    use_physical_units: bool = False
    pixel_size: float = 1.0
    pixel_unit: str = "mm"
    frame_rate: float = 1.0


@dataclass
class FieldImageConfig:
    """Per-field image/animation export settings."""
    field_name: str
    enabled: bool
    colormap: str
    auto_range: bool
    vmin: float
    vmax: float
    bg_alpha: float = 0.7


# ---------------------------------------------------------------------------
# Per-field row widget (Images Tab + Animation Tab)
# ---------------------------------------------------------------------------

class FieldRow(QWidget):
    """One row in the field list (Images or Animation tab)."""

    def __init__(self, field_name: str, hint: VizExportHint, has_data: bool,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._field_name = field_name
        enabled_default = has_data and (field_name in DEFAULT_ENABLED)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(6)

        self._check = QCheckBox()
        self._check.setChecked(enabled_default)
        self._check.setEnabled(has_data)
        self._check.setFixedWidth(HEADER_WIDTHS[0])
        row.addWidget(self._check)

        # Show user-facing label, not internal variable name
        name_lbl = QLabel(field_label(field_name))
        name_lbl.setFixedWidth(HEADER_WIDTHS[1])
        name_lbl.setToolTip(field_name)   # internal name visible on hover
        themed(name_lbl,
               "color: {TEXT_PRIMARY}; font-size: 11px;" if has_data
               else "color: {TEXT_MUTED}; font-size: 11px;")
        row.addWidget(name_lbl)

        # Per-field color state of the main window (WYSIWYG: export matches the live view).
        fs = color_state(field_name)

        self._cmap_combo = QComboBox()
        names = list(COLORMAP_NAMES)
        if fs.colormap not in names:   # e.g. the residual's magma
            names.append(fs.colormap)
        self._cmap_combo.addItems(names)
        self._cmap_combo.setCurrentText(fs.colormap)
        self._cmap_combo.setEnabled(has_data)
        self._cmap_combo.setFixedWidth(HEADER_WIDTHS[2])
        row.addWidget(self._cmap_combo)

        self._auto_check = AutoFixedSelector()
        self._auto_check.setChecked(fs.auto)
        self._auto_check.setEnabled(has_data)
        self._auto_check.toggled.connect(self._on_auto_changed)
        self._auto_check.setFixedWidth(HEADER_WIDTHS[3])
        row.addWidget(self._auto_check)

        self._vmin_spin = self._range_spin(fs.vmin, has_data and not fs.auto)
        row.addWidget(self._vmin_spin)
        self._vmax_spin = self._range_spin(fs.vmax, has_data and not fs.auto)
        row.addWidget(self._vmax_spin)

        opacity_lbl = QLabel(QCoreApplication.translate("ExportDialog", "Opacity"))
        opacity_lbl.setToolTip(QCoreApplication.translate(
            "ExportDialog", "Field opacity (0 = transparent, 1 = fully opaque)"))
        row.addWidget(opacity_lbl)
        self._alpha_spin = LocaleSafeDoubleSpinBox()
        self._alpha_spin.setRange(0.0, 1.0)
        self._alpha_spin.setSingleStep(0.05)
        self._alpha_spin.setValue(hint.overlay_alpha)
        self._alpha_spin.setDecimals(2)
        self._alpha_spin.setFixedWidth(65)
        self._alpha_spin.setEnabled(has_data)
        row.addWidget(self._alpha_spin)

        row.addStretch()

    @staticmethod
    def _range_spin(value: float, enabled: bool) -> LocaleSafeDoubleSpinBox:
        spin = LocaleSafeDoubleSpinBox()
        spin.setRange(-1e9, 1e9)
        spin.setDecimals(6)   # strains are often below 1e-3
        spin.setValue(float(value))
        spin.setFixedWidth(HEADER_WIDTHS[4])
        spin.setEnabled(enabled)
        return spin

    def _on_auto_changed(self, auto: bool) -> None:
        self._vmin_spin.setEnabled(not auto)
        self._vmax_spin.setEnabled(not auto)

    def get_config(self) -> FieldImageConfig:
        return FieldImageConfig(
            field_name=self._field_name,
            enabled=self._check.isEnabled() and self._check.isChecked(),
            colormap=self._cmap_combo.currentText(),
            auto_range=self._auto_check.isChecked(),
            vmin=self._vmin_spin.value(),
            vmax=self._vmax_spin.value(),
            bg_alpha=self._alpha_spin.value(),
        )

    def set_enabled_field(self, on: bool) -> None:
        """Tick / untick the row (only when the field has data)."""
        if self._check.isEnabled():
            self._check.setChecked(on)

    @property
    def field_name(self) -> str:
        return self._field_name

    def get_appearance(self) -> dict:
        """Colormap / range / opacity as a plain dict (for the preview panel)."""
        return dict(
            colormap=self._cmap_combo.currentText(),
            auto=self._auto_check.isChecked(),
            vmin=self._vmin_spin.value(),
            vmax=self._vmax_spin.value(),
            opacity=self._alpha_spin.value(),
        )

    def set_appearance(self, colormap=None, auto=None, vmin=None, vmax=None, opacity=None) -> None:
        """Push values into the row's widgets, blocking signals to avoid loops.

        Lets the Preview tab edit a field's appearance while this row stays the
        single source of truth that export reads via :meth:`get_config`.
        """
        for widget, value in ((self._vmin_spin, vmin), (self._vmax_spin, vmax), (self._alpha_spin, opacity)):
            if value is not None:
                widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(False)
        if colormap is not None:
            self._cmap_combo.blockSignals(True)
            if self._cmap_combo.findText(colormap) < 0:
                self._cmap_combo.addItem(colormap)
            self._cmap_combo.setCurrentText(colormap)
            self._cmap_combo.blockSignals(False)
        if auto is not None:
            self._auto_check.blockSignals(True)
            self._auto_check.setChecked(auto)
            self._auto_check.blockSignals(False)
            self._vmin_spin.setEnabled(not auto)
            self._vmax_spin.setEnabled(not auto)


def build_field_rows_scroll(fields: tuple[str, ...], hint: VizExportHint,
                            available: Callable[[str], bool]) -> tuple[QScrollArea, list[FieldRow]]:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setMinimumHeight(150)
    scroll.setMaximumHeight(220)

    container = QWidget()
    c_layout = QVBoxLayout(container)
    c_layout.setSpacing(0)
    c_layout.setContentsMargins(0, 0, 0, 0)

    rows: list[FieldRow] = []
    for fname in fields:
        row_w = FieldRow(fname, hint, available(fname))
        c_layout.addWidget(row_w)
        rows.append(row_w)

    scroll.setWidget(container)
    return scroll, rows


def make_select_all_none(checks: dict[str, QCheckBox]) -> tuple[QPushButton, QPushButton]:
    """Return (All button, None button) wired to the given checkbox dict."""
    all_btn = QPushButton(QCoreApplication.translate("ExportDialog", "All"))
    all_btn.setMinimumWidth(60)  # grows for longer translations
    none_btn = QPushButton(QCoreApplication.translate("ExportDialog", "None"))
    none_btn.setMinimumWidth(60)
    all_btn.clicked.connect(lambda: [c.setChecked(True) for c in checks.values() if c.isEnabled()])
    none_btn.clicked.connect(lambda: [c.setChecked(False) for c in checks.values()])
    return all_btn, none_btn


class RunControls:
    """Progress bar, progress label, Cancel, the export button and the status label of a tab."""

    def __init__(self, dialog: QWidget, layout: QVBoxLayout, run_text: str,
                 on_run: Callable[[], None], on_cancel: Callable[[], None]) -> None:
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.progress_lbl = QLabel("")
        self.progress_lbl.setVisible(False)
        layout.addWidget(self.progress_lbl)

        btn_row = QHBoxLayout()
        self.cancel_btn = QPushButton(dialog.tr("Cancel Export"))
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(on_cancel)
        btn_row.addWidget(self.cancel_btn)
        btn_row.addStretch()
        self.run_btn = QPushButton(run_text)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(on_run)
        btn_row.addWidget(self.run_btn)
        layout.addLayout(btn_row)

        self.status_lbl = QLabel("")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status_lbl)
        self.running = False

    def set_running(self, running: bool, has_folder: bool) -> None:
        self.running = running
        self.run_btn.setEnabled(has_folder and not running)
        self.cancel_btn.setVisible(running)
        self.cancel_btn.setEnabled(running)
        self.progress_lbl.setVisible(running)
        if running:
            self.progress.setVisible(True)
            self.progress.setValue(0)

    def set_status(self, text: str, color: str) -> None:
        self.status_lbl.setText(text)
        themed(self.status_lbl, f"color: {{{color}}}; font-size: 11px;")


def add_check_rows(layout: QVBoxLayout, checks: dict[str, QCheckBox], per_row: int) -> None:
    """Lay the checkboxes out *per_row* to a row."""
    boxes = list(checks.values())
    for start in range(0, len(boxes), per_row):
        row = QHBoxLayout()
        for chk in boxes[start:start + per_row]:
            row.addWidget(chk)
        row.addStretch()
        layout.addLayout(row)


def check_group(title: str, checks: dict[str, QCheckBox], per_row: int) -> QGroupBox:
    """A titled group of checkboxes with "Select: All / None" buttons."""
    group = QGroupBox(title)
    vl = QVBoxLayout(group)
    sel_row = QHBoxLayout()
    sel_row.addWidget(QLabel(QCoreApplication.translate("ExportDialog", "Select:")))
    all_btn, none_btn = make_select_all_none(checks)
    sel_row.addWidget(all_btn)
    sel_row.addWidget(none_btn)
    sel_row.addStretch()
    vl.addLayout(sel_row)
    add_check_rows(vl, checks, per_row)
    return group


def field_header() -> QHBoxLayout:
    """Column titles above the field rows, aligned with :data:`HEADER_WIDTHS`."""
    header_row = QHBoxLayout()
    texts = (QCoreApplication.translate("ExportDialog", "Export"), QCoreApplication.translate("ExportDialog", "Field"), QCoreApplication.translate("ExportDialog", "Colormap"), QCoreApplication.translate("ExportDialog", "Auto"), QCoreApplication.translate("ExportDialog", "Min"), QCoreApplication.translate("ExportDialog", "Max"),
             "α")   # α — math symbol, keep literal
    for text, width in zip(texts, HEADER_WIDTHS + (HEADER_WIDTHS[4], 55)):
        lbl = QLabel(text)
        lbl.setFixedWidth(width)
        themed(lbl, "color: {TEXT_SECONDARY}; font-size: 10px; font-weight: bold;")
        header_row.addWidget(lbl)
    header_row.addStretch()
    return header_row


class UnitsBox(QGroupBox):
    """PHYSICAL UNITS: on / off, pixel size and length unit, frame rate."""

    changed = Signal()

    def __init__(self, hint: VizExportHint, parent: QWidget | None = None) -> None:
        super().__init__(QCoreApplication.translate("ExportDialog", "PHYSICAL UNITS"), parent)
        form = QFormLayout(self)
        form.setSpacing(4)

        self.enable_check = QCheckBox(QCoreApplication.translate("ExportDialog", "Enable physical units"))
        self.enable_check.setChecked(hint.use_physical_units)
        self.enable_check.setToolTip(QCoreApplication.translate("ExportDialog", 
            "Scale displacement values by pixel size and show physical units "
            "on colorbar labels. Strain is dimensionless and unaffected."))
        form.addRow(self.enable_check)

        phys_row = QHBoxLayout()
        self.pixel_size_spin = LocaleSafeDoubleSpinBox()
        self.pixel_size_spin.setRange(1e-6, 1e6)
        self.pixel_size_spin.setDecimals(6)
        self.pixel_size_spin.setValue(hint.pixel_size if hint.pixel_size > 0 else 1.0)
        self.pixel_size_spin.setFixedWidth(100)
        phys_row.addWidget(self.pixel_size_spin)
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["nm", "μm", "mm", "cm", "m", "inch"])
        self.unit_combo.setCurrentText(hint.pixel_unit)
        self.unit_combo.setFixedWidth(70)
        phys_row.addWidget(self.unit_combo)
        phys_row.addWidget(QLabel(QCoreApplication.translate("ExportDialog", "/ pixel")))
        phys_row.addStretch()
        form.addRow(QCoreApplication.translate("ExportDialog", "Pixel size"), phys_row)

        fr_row = QHBoxLayout()
        self.frame_rate_spin = LocaleSafeDoubleSpinBox()
        self.frame_rate_spin.setRange(1e-3, 1e6)
        self.frame_rate_spin.setValue(hint.frame_rate if hint.frame_rate > 0 else 1.0)
        self.frame_rate_spin.setDecimals(2)
        self.frame_rate_spin.setFixedWidth(100)
        fr_row.addWidget(self.frame_rate_spin)
        fr_row.addWidget(QLabel(QCoreApplication.translate("ExportDialog", "fps")))
        fr_row.addStretch()
        form.addRow(QCoreApplication.translate("ExportDialog", "Frame rate"), fr_row)

        # Physical units drive the colorbar label + value scaling (the preview listens).
        self.enable_check.stateChanged.connect(self.changed)
        self.pixel_size_spin.valueChanged.connect(self.changed)
        self.unit_combo.currentIndexChanged.connect(self.changed)

    def units(self) -> Units:
        """Units of the pictures and the report (pixels when off)."""
        if not self.enable_check.isChecked():
            return Units()
        return Units(pixel_size=float(self.pixel_size_spin.value()), length_unit=self.unit_combo.currentText(),
                     frame_rate=float(self.frame_rate_spin.value()))

    def physical_units(self) -> PhysicalUnits:
        """Calibration recorded with the data files (the data stay in pixels)."""
        u = self.units()
        if not u.physical:
            return PhysicalUnits()
        return PhysicalUnits(scale=u.pixel_size, unit=u.length_unit, frame_rate=u.frame_rate)


class FrameRangeBox(QGroupBox):
    """FRAME RANGE: all frames, or From / to in 1-based frame numbers (frame 1 = reference)."""

    def __init__(self, n_frames: int, parent: QWidget | None = None) -> None:
        super().__init__(QCoreApplication.translate("ExportDialog", "FRAME RANGE"), parent)
        vl = QVBoxLayout(self)
        self.all_check = QCheckBox(QCoreApplication.translate("ExportDialog", "All frames"))
        self.all_check.setChecked(True)
        vl.addWidget(self.all_check)

        self._n = max(1, int(n_frames))
        tip = QCoreApplication.translate("ExportDialog", "Numbered like the frame slider: frame 1 is the reference image.\n"
                  "File names use the same numbers: frame 1 is written as _0001.")
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel(QCoreApplication.translate("ExportDialog", "From", "Frame range: starting frame")))
        self.start_spin = QSpinBox()
        self.start_spin.setRange(1, self._n)
        self.start_spin.setValue(1)
        range_row.addWidget(self.start_spin)
        range_row.addWidget(QLabel(QCoreApplication.translate("ExportDialog", "to", "Frame range: ending frame")))
        self.end_spin = QSpinBox()
        self.end_spin.setRange(1, self._n)
        self.end_spin.setValue(self._n)
        range_row.addWidget(self.end_spin)
        range_row.addStretch()
        vl.addLayout(range_row)
        for spin in (self.start_spin, self.end_spin):
            spin.setEnabled(False)
            spin.setToolTip(tip)
        self.all_check.toggled.connect(self._on_all_changed)

    def _on_all_changed(self, all_on: bool) -> None:
        self.start_spin.setEnabled(not all_on)
        self.end_spin.setEnabled(not all_on)

    def frames(self) -> range:
        """The chosen frames, 0-based."""
        if self.all_check.isChecked():
            return range(self._n)
        a, b = self.start_spin.value() - 1, self.end_spin.value() - 1
        return range(min(a, b), max(a, b) + 1)
