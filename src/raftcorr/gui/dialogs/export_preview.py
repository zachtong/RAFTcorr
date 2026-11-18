"""The Preview & Colorbar tab of the export dialog: one frame drawn exactly as the Images tab writes it.

pyALDIC's "Preview & Colorbar" tab: a field / frame picker, the picture
(raftcorr.export.figures.compose + render_figure, the path export_frames takes), the field's
appearance -- colormap, range, opacity -- two-way synced with the field's Images-tab row (the single
source of truth that export reads), and the COLORBAR STYLE shared by every picture, animation and
report image: colour bar position, font, thickness and background, the margin, and what replaces a
hidden background image. Texts are translated in the ``ExportDialog`` context like pyALDIC's.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QCoreApplication, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from raftcorr.export.figure_layout import FONT_FAMILIES, ColorbarStyle
from raftcorr.export.figures import FrameStyle, _crop_box, color_range, compose, figure_rgba, render_figure
from raftcorr.gui.dialogs.export_widgets import FieldImageConfig, FieldRow, field_label
from raftcorr.gui.support.colormaps import COLORMAP_NAMES
from raftcorr.gui.theme_manager import themed
from raftcorr.gui.widgets.double_spin import LocaleSafeDoubleSpinBox
from raftcorr.gui.widgets.range_mode import AutoFixedSelector
from raftcorr.viz.fields import ALL_FIELDS, FieldProvider, Units

_PREVIEW_DPI = 80


def _checkerboard(h: int, w: int, size: int = 12) -> np.ndarray:
    """Grey checkerboard, so a transparent preview does not read as white."""
    yy, xx = np.mgrid[0:h, 0:w]
    odd = ((yy // size) + (xx // size)) % 2
    return np.where(odd[..., None], 153, 204).astype(np.uint8).repeat(3, 2)


def _on_checkerboard(rgba: np.ndarray) -> np.ndarray:
    """``rgba`` flattened over a checkerboard where it is not opaque (as an image editor shows it)."""
    if rgba[..., 3].min() == 255:
        return rgba
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3] * a + _checkerboard(*rgba.shape[:2]) * (1.0 - a)
    return np.dstack([(rgb + 0.5).astype(np.uint8), np.full(rgba.shape[:2], 255, np.uint8)])


class ColorbarStyleBox(QGroupBox):
    """COLORBAR STYLE: colour bar position / font / thickness / background, margin, hidden background."""

    changed = Signal()

    def __init__(self, hidden_background: str = "white", parent: QWidget | None = None) -> None:
        super().__init__(QCoreApplication.translate("ExportDialog", "COLORBAR STYLE"), parent)
        default = ColorbarStyle()
        self.form = QFormLayout(self)

        self.position_combo = QComboBox()
        for text, value in ((QCoreApplication.translate("ExportDialog", "Right"), "right"),
                            (QCoreApplication.translate("ExportDialog", "Left"), "left"),
                            (QCoreApplication.translate("ExportDialog", "Top"), "top"),
                            (QCoreApplication.translate("ExportDialog", "Bottom"), "bottom")):
            self.position_combo.addItem(text, value)
        self._select(self.position_combo, default.position)
        self.form.addRow(QCoreApplication.translate("ExportDialog", "Position"), self.position_combo)

        self.font_spin = QSpinBox()
        self.font_spin.setRange(6, 32)
        self.font_spin.setValue(int(round(default.font_size)))
        self.form.addRow(QCoreApplication.translate("ExportDialog", "Font size"), self.font_spin)

        self.font_combo = QComboBox()
        for family in FONT_FAMILIES:   # family names stay literal
            self.font_combo.addItem(family, family)
        self._select(self.font_combo, default.font_family)
        self.form.addRow(QCoreApplication.translate("ExportDialog", "Font family"), self.font_combo)

        self.thickness_spin = self._fraction_spin(0.02, 0.25, default.thickness)
        self.thickness_spin.setToolTip(QCoreApplication.translate(
            "ExportDialog", "Bar thickness as a fraction of the picture's width (bar on the left or right) "
                            "or height (bar on top or at the bottom)."))
        self.form.addRow(QCoreApplication.translate("ExportDialog", "Bar thickness"), self.thickness_spin)

        self.background_combo = QComboBox()
        for text, value in ((QCoreApplication.translate("ExportDialog", "Black"), "black"),
                            (QCoreApplication.translate("ExportDialog", "White"), "white")):
            self.background_combo.addItem(text, value)
        self._select(self.background_combo, default.background)
        self.background_combo.setToolTip(QCoreApplication.translate(
            "ExportDialog", "Colour behind the colorbar and around the picture; text and ticks are drawn "
                            "in the opposite colour."))
        self.form.addRow(QCoreApplication.translate("ExportDialog", "Background"), self.background_combo)

        self.margin_spin = self._fraction_spin(0.0, 0.30, 0.0)
        self.margin_spin.setToolTip(QCoreApplication.translate(
            "ExportDialog", "Add a blank border around the exported content, as a fraction of "
                            "the long edge (0 = none)."))
        self.form.addRow(QCoreApplication.translate("ExportDialog", "Margin"), self.margin_spin)

        self.margin_color_combo = QComboBox()
        for text, value in ((QCoreApplication.translate("ExportDialog", "White"), "white"),
                            (QCoreApplication.translate("ExportDialog", "Black"), "black")):
            self.margin_color_combo.addItem(text, value)
        self.form.addRow(QCoreApplication.translate("ExportDialog", "Margin color"), self.margin_color_combo)

        self.hidden_bg_combo = QComboBox()
        for text, value in ((QCoreApplication.translate("ExportDialog", "White"), "white"),
                            (QCoreApplication.translate("ExportDialog", "Black"), "black"),
                            (QCoreApplication.translate("ExportDialog", "Transparent"), "transparent")):
            self.hidden_bg_combo.addItem(text, value)
        self._select(self.hidden_bg_combo, hidden_background)
        self.hidden_bg_combo.setToolTip(QCoreApplication.translate(
            "ExportDialog", "Fill used where the background image would have been, when "
                            "'Show background image' is off.\n"
                            "Transparency is kept for PNG and TIFF; the other formats get white instead."))
        self.form.addRow(QCoreApplication.translate("ExportDialog", "Hidden background"), self.hidden_bg_combo)

        for combo in (self.position_combo, self.font_combo, self.background_combo, self.margin_color_combo,
                      self.hidden_bg_combo):
            combo.currentIndexChanged.connect(self.changed)
        for spin in (self.font_spin, self.thickness_spin, self.margin_spin):
            spin.valueChanged.connect(self.changed)

    @staticmethod
    def _select(combo: QComboBox, value: str) -> None:
        i = combo.findData(value)
        if i >= 0:
            combo.setCurrentIndex(i)

    @staticmethod
    def _fraction_spin(lo: float, hi: float, value: float) -> LocaleSafeDoubleSpinBox:
        spin = LocaleSafeDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(0.01)
        spin.setDecimals(2)
        spin.setValue(value)
        return spin

    def colorbar_style(self) -> ColorbarStyle:
        return ColorbarStyle(position=self.position_combo.currentData(), font_size=float(self.font_spin.value()),
                             font_family=self.font_combo.currentData(), thickness=float(self.thickness_spin.value()),
                             background=self.background_combo.currentData())

    def margin(self) -> float:
        return float(self.margin_spin.value())

    def margin_color(self) -> str:
        return str(self.margin_color_combo.currentData())

    def hidden_background(self) -> str:
        return str(self.hidden_bg_combo.currentData())


class ExportPreview(QWidget):
    """Preview of one exported frame + the previewed field's appearance + the colour bar style.

    ``image_rows`` are the Images-tab rows (what is previewed and edited), ``other_rows`` the
    Animation-tab rows ("Apply to all fields" reaches them too); ``style_for`` turns a row's
    config into the FrameStyle the Images tab would write (the dialog reads :attr:`look`, the
    COLORBAR STYLE box, for it), ``units`` gives the export units and ``frames_for(key)`` the
    frames the Images tab would write for a field. ``hidden_background`` starts the box's
    "Hidden background".
    """

    def __init__(self, provider: FieldProvider, images: Any, image_rows: list[FieldRow], other_rows: list[FieldRow],
                 style_for: Callable[[FieldImageConfig], FrameStyle], units: Callable[[], Units],
                 frames_for: Callable[[str], list[int]], hidden_background: str = "white",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._provider = provider     # GUI-thread caches
        self._images = images
        self._image_rows = image_rows
        self._other_rows = other_rows
        self._style_for = style_for
        self._units = units
        self._frames_for = frames_for
        self._range_cache: dict[tuple, Any] = {}
        n_frames = provider.series.n_frames

        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        self._label = QLabel(QCoreApplication.translate("ExportDialog", "Open this tab to render a preview."))
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(420, 340)
        themed(self._label, "background: {BG_CANVAS}; color: {TEXT_MUTED}; border: 1px solid {BORDER};")
        left.addWidget(self._label, 1)

        pick_row = QHBoxLayout()
        pick_row.addWidget(QLabel(QCoreApplication.translate("ExportDialog", "Field")))
        self.field_combo = QComboBox()
        self.field_combo.currentIndexChanged.connect(self._on_field_changed)
        pick_row.addWidget(self.field_combo, 1)
        left.addLayout(pick_row)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel(QCoreApplication.translate("ExportDialog", "Frame")))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, max(0, n_frames - 1))
        self.frame_slider.setValue(min(1, n_frames - 1))   # the first deformed frame
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        frame_row.addWidget(self.frame_slider, 1)
        self._frame_lbl = QLabel(str(self.frame_slider.value() + 1))   # 1-based, like the frame slider
        self._frame_lbl.setFixedWidth(52)
        frame_row.addWidget(self._frame_lbl)
        left.addLayout(frame_row)
        layout.addLayout(left, 1)

        self.look = ColorbarStyleBox(hidden_background)
        self.look.changed.connect(self.schedule)
        refresh = QPushButton(QCoreApplication.translate("ExportDialog", "Refresh preview"))
        refresh.clicked.connect(self.render)
        self.look.form.addRow(refresh)

        right = QVBoxLayout()
        right.addWidget(self._build_appearance())
        right.addWidget(self.look)
        right.addStretch()
        layout.addLayout(right)

        # Debounced re-render so dragging a slider is smooth.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(220)
        self._timer.timeout.connect(self.render)

    def _build_appearance(self) -> QGroupBox:
        appear_group = QGroupBox(QCoreApplication.translate("ExportDialog", "FIELD APPEARANCE"))
        aform = QFormLayout(appear_group)
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(list(COLORMAP_NAMES))
        self._cmap_combo.currentIndexChanged.connect(self._on_appearance_changed)
        aform.addRow(QCoreApplication.translate("ExportDialog", "Colormap"), self._cmap_combo)

        self._auto_check = AutoFixedSelector()
        self._auto_check.toggled.connect(self._on_appearance_changed)
        aform.addRow(QCoreApplication.translate("ExportDialog", "Range"), self._auto_check)

        self._vmin_spin = FieldRow._range_spin(0.0, True)
        self._vmin_spin.valueChanged.connect(self._on_appearance_changed)
        aform.addRow(QCoreApplication.translate("ExportDialog", "Min"), self._vmin_spin)

        self._vmax_spin = FieldRow._range_spin(1.0, True)
        self._vmax_spin.valueChanged.connect(self._on_appearance_changed)
        for spin in (self._vmin_spin, self._vmax_spin):   # the table rows' width is fixed; here they fill the column
            spin.setMinimumWidth(0)
            spin.setMaximumWidth(16777215)
        aform.addRow(QCoreApplication.translate("ExportDialog", "Max"), self._vmax_spin)

        self._opacity_spin = LocaleSafeDoubleSpinBox()
        self._opacity_spin.setRange(0.0, 1.0)
        self._opacity_spin.setSingleStep(0.05)
        self._opacity_spin.setDecimals(2)
        self._opacity_spin.valueChanged.connect(self._on_appearance_changed)
        aform.addRow(QCoreApplication.translate("ExportDialog", "Opacity"), self._opacity_spin)

        apply_all = QPushButton(QCoreApplication.translate("ExportDialog", "Apply to all fields"))
        apply_all.setToolTip(QCoreApplication.translate("ExportDialog", "Apply this field's colormap, opacity and auto-range to every "
                                 "enabled field (each field keeps its own min/max)."))
        apply_all.clicked.connect(self._apply_appearance_to_all)
        aform.addRow(apply_all)
        return appear_group

    # -- public ------------------------------------------------------------------------------

    def activate(self) -> None:
        """The tab was opened: follow the enabled fields, then draw."""
        self._refresh_fields()
        self._load_appearance()
        self.schedule()

    def schedule(self) -> None:
        """Draw again shortly (settings changed)."""
        if self.isVisible():
            self._timer.start()

    def render(self) -> None:
        """Draw the previewed frame now."""
        try:
            self._render()
        except Exception as exc:  # never let a preview error break the dialog
            self._label.setText(QCoreApplication.translate("ExportDialog", "Preview failed: ") + str(exc))

    def preview_image(self) -> np.ndarray | None:
        """The previewed frame as RGBA, drawn like the Images tab writes it (None: nothing to show)."""
        row = self._selected_row()
        if row is None:
            return None
        frame = int(self.frame_slider.value())
        style = self._style_for(row.get_config())
        if not self._provider.has_frame(style.field, frame):
            return None
        units = self._units()
        rng = self._range(style, units, frame)
        rgba = compose(self._provider, self._images, frame, style, units, rng)
        if style.crop_to_roi:   # the crop export_frames applies
            x0, y0, x1, y1 = _crop_box(self._provider)
            rgba = rgba[y0:y1, x0:x1]
        label = f"{style.field} ({units.unit(ALL_FIELDS[style.field])})"   # the label export_frames writes
        fig = render_figure(rgba, replace(style, dpi=_PREVIEW_DPI), rng, label, "")
        return figure_rgba(fig)

    # -- internals ---------------------------------------------------------------------------

    def _range(self, style: FrameStyle, units: Units, frame: int):
        if (style.vmin is not None and style.vmax is not None) or style.range_mode == "frame":
            return color_range(self._provider, style, units, [frame])
        frames = self._frames_for(style.field)
        key = (style.field, style.colormap, units, tuple(frames))
        if key not in self._range_cache:
            self._range_cache[key] = color_range(self._provider, style, units, frames or [frame])
        return self._range_cache[key]

    def _render(self) -> None:
        if self.field_combo.currentData() is None:
            self._label.setText(QCoreApplication.translate("ExportDialog", "Enable a field on the Images tab to preview."))
            return
        img = self.preview_image()
        if img is None:
            self._label.setText(QCoreApplication.translate("ExportDialog", "No data for this field/frame."))
            return
        rgba = np.ascontiguousarray(_on_checkerboard(img))
        h, w = rgba.shape[:2]
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
        pix = QPixmap.fromImage(qimg).scaled(self._label.width(), self._label.height(),
                                             Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation)
        self._label.setPixmap(pix)

    def _on_frame_changed(self, value: int) -> None:
        self._frame_lbl.setText(str(value + 1))
        self.schedule()

    def _selected_row(self) -> FieldRow | None:
        """The Images-tab row of the field currently being previewed."""
        name = self.field_combo.currentData()
        if name is None:
            return None
        return next((r for r in self._image_rows if r.field_name == name), None)

    def _on_field_changed(self) -> None:
        self._load_appearance()
        self.schedule()

    def _load_appearance(self) -> None:
        """Load the selected field's colormap/range/opacity into the panel."""
        row = self._selected_row()
        if row is None:
            return
        a = row.get_appearance()
        widgets = (self._cmap_combo, self._auto_check, self._vmin_spin, self._vmax_spin, self._opacity_spin)
        for w in widgets:
            w.blockSignals(True)
        if self._cmap_combo.findText(a["colormap"]) < 0:
            self._cmap_combo.addItem(a["colormap"])
        self._cmap_combo.setCurrentText(a["colormap"])
        self._auto_check.setChecked(a["auto"])
        self._vmin_spin.setValue(a["vmin"])
        self._vmax_spin.setValue(a["vmax"])
        self._opacity_spin.setValue(a["opacity"])
        for w in widgets:
            w.blockSignals(False)
        self._vmin_spin.setEnabled(not a["auto"])
        self._vmax_spin.setEnabled(not a["auto"])

    def _on_appearance_changed(self) -> None:
        """Push the panel's appearance edits back to the Images-tab row."""
        row = self._selected_row()
        if row is None:
            return
        auto = self._auto_check.isChecked()
        row.set_appearance(colormap=self._cmap_combo.currentText(), auto=auto, vmin=self._vmin_spin.value(),
                           vmax=self._vmax_spin.value(), opacity=self._opacity_spin.value())
        self._vmin_spin.setEnabled(not auto)
        self._vmax_spin.setEnabled(not auto)
        self.schedule()

    def _apply_appearance_to_all(self) -> None:
        """Push colormap / opacity / auto-range to every enabled field (image and animation
        rows).  Per-field min/max are left untouched because different fields have different
        value scales."""
        cmap = self._cmap_combo.currentText()
        auto = self._auto_check.isChecked()
        opacity = self._opacity_spin.value()
        for row in [*self._image_rows, *self._other_rows]:
            if row.get_config().enabled:
                row.set_appearance(colormap=cmap, auto=auto, opacity=opacity)
        self.schedule()

    def _refresh_fields(self) -> None:
        """Repopulate the field picker from the currently enabled image fields."""
        prev = self.field_combo.currentData()
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        for row in self._image_rows:
            cfg = row.get_config()
            if cfg.enabled:
                self.field_combo.addItem(field_label(cfg.field_name), cfg.field_name)
        if prev is not None:
            i = self.field_combo.findData(prev)
            if i >= 0:
                self.field_combo.setCurrentIndex(i)
        self.field_combo.blockSignals(False)
