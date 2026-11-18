"""Export dialog for RAFTcorr results -- pyALDIC's tab layout on RAFTcorr's exporters.

Tabs:
  - Data Tab:      Parameters JSON (always) + NPZ / MATLAB v7.3 / HDF5 with per-variable
                   checkboxes (raftcorr.export.export_results), and per-frame CSV tables.
  - Images Tab:    Per-field rows -> one picture per frame (PNG / TIFF / JPEG / PDF / SVG).
  - Animation Tab: Per-field rows -> GIF (MP4 when imageio-ffmpeg is installed) animations.
  - Report Tab:    Run parameters + field statistics + sample images -> self-contained HTML.
  - Preview Tab:   One frame drawn exactly as the Images tab writes it, plus the field's
                   colormap / range / opacity (two-way synced with its Images-tab row).

UX design (pyALDIC):
  - All export actions are QPushButton (non-AcceptRole); dialog stays open.
  - Timestamp is refreshed on every export click -> exports do not collide; files that do
    exist already are only replaced after asking.
  - "Open Folder" button enabled after the first successful export.
  - Last-used folder persisted via QSettings.
  - 1-based frame numbers in spinboxes (frame 1 = the reference image).
  - Progress bar + status label + Cancel inside each tab; every export runs in a QThread.
  - Field labels match the main window / strain window naming (U, V, εxx, ...); each row
    starts from that field's colour state in the main window (WYSIWYG).
  - Pictures: drawn on the deformed or the reference configuration, over the matching
    photo (deformed: each frame's own; reference: frame 1) or on white.

Pictures come from raftcorr.export.figures (the renderer of the GUI canvas, framed by matplotlib
with the colour bar), data from raftcorr.export.scientific; see export_jobs for file names.

Entry points:
  - MainWindow "Export Results" button  (right_sidebar._on_export)
  - StrainWindow "Export Results" button (strain_window._on_export_strain)
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, QUrl
from PySide6.QtGui import QDesktopServices, QShowEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from raftcorr.export.figures import FrameStyle
from raftcorr.gui.app_state import AppState
from raftcorr.gui.dialogs import export_jobs as jobs
from raftcorr.gui.dialogs.export_preview import ExportPreview
from raftcorr.gui.dialogs.export_widgets import (
    FieldImageConfig,
    FieldRow,
    FrameRangeBox,
    RunControls,
    UnitsBox,
    VizExportHint,
    add_check_rows,
    build_field_rows_scroll,
    check_group,
    field_header,
    field_label,
)
from raftcorr.gui.support.fields import provider_key
from raftcorr.gui.theme_manager import themed
from raftcorr.gui.window_geometry import fit_dialog_to_screen
from raftcorr.i18n import tr_args
from raftcorr.viz.fields import FieldProvider, Units

__all__ = ["ExportDialog", "FieldImageConfig", "VizExportHint", "mp4_available"]

_SETTINGS_KEY_FOLDER = "ExportDialog/last_folder"
_FROM_STATE = object()  # "take it from AppState" default of the constructor


# ---------------------------------------------------------------------------
# Known exportable fields
# ---------------------------------------------------------------------------

_DISP_FIELDS: tuple[str, ...] = ("disp_u", "disp_v", "disp_magnitude", "residual")
_STRAIN_FIELDS: tuple[str, ...] = (
    "strain_exx", "strain_eyy", "strain_exy", "strain_principal_max", "strain_principal_min",
    "strain_maxshear", "strain_von_mises", "strain_rotation", "strain_principal_angle", "strain_area_change",
)
_ALL_FIELDS: tuple[str, ...] = _DISP_FIELDS + _STRAIN_FIELDS

_DATA_DEFAULT_SELECTED = {"disp_u", "disp_v", "disp_magnitude", "strain_exx", "strain_eyy", "strain_exy"}

#: GUI field name -> variable name in the data files (raftcorr.export.scientific).
_DATA_VARIABLE: dict[str, str] = {
    "disp_u": "U", "disp_v": "V", "disp_magnitude": "Magnitude", "residual": "Residual",
    **{name: provider_key(name) for name in _STRAIN_FIELDS},
}


_IMAGE_FORMATS: tuple[tuple[str, str], ...] = (
    ("PNG", ".png"), ("TIFF", ".tif"), ("JPEG", ".jpg"), ("PDF", ".pdf"), ("SVG", ".svg"))


# How long closing the dialog waits for a running export to notice the stop request. Export
# checks it between frames, so this only has to cover one frame's render; a worker that
# outlasts it is cut loose instead (see ExportDialog._shutdown_workers).
_WORKER_STOP_WAIT_MS = 5000

# Workers that outlived their dialog. Holding a reference here keeps the Python object alive
# until the thread actually finishes -- dropping it while the thread runs is exactly the crash
# this list exists to prevent.
_ORPHANED_WORKERS: list = []


def mp4_available() -> bool:
    """MP4 export needs imageio with its ffmpeg plugin."""
    return all(importlib.util.find_spec(m) is not None for m in ("imageio", "imageio_ffmpeg"))


def _still_running(worker) -> bool:
    """True while *worker* must be kept alive; False once it is safe to drop."""
    try:
        return worker.isRunning()
    except RuntimeError:      # C++ side already deleted
        return False


# ---------------------------------------------------------------------------
# Main dialog
# ---------------------------------------------------------------------------

class ExportDialog(QDialog):
    """Export dialog: Data, Images, Animation, Report, Preview.

    Timestamp is refreshed on every export action so repeated clicks
    produce distinct output files rather than overwriting.

    ``results`` is the :class:`~raftcorr.results.DisplacementSeries`; ``strain`` and ``sequence``
    (the :class:`~raftcorr.io.images.ImageSequence` behind the pictures) default to AppState's
    when it holds these results. ``roi_mask`` / ``per_frame_rois`` are accepted for the pyALDIC
    callers: RAFTcorr results carry their own ROI.
    """

    def __init__(self, results, image_folder: Path | None, hint: VizExportHint,
                 image_files: list[str] | None = None,
                 roi_mask=None,
                 per_frame_rois: dict | None = None,
                 parent: QWidget | None = None,
                 *, strain: Any = _FROM_STATE, sequence: Any = _FROM_STATE) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Export Results"))
        self.setMinimumWidth(780)
        # Fit-to-screen guard: this modal, taskbar-less dialog must never open
        # off-screen (it would be unreachable and freeze the app).  Clamped
        # once on first show via showEvent.
        self._screen_fitted = False

        state = AppState.instance()
        own = state.results is results
        self._results = results
        self._strain = (state.strain_results if own else None) if strain is _FROM_STATE else strain
        self._images = self._resolve_images(state.sequence if own else None, sequence, image_files)
        self._hint = hint
        self._prefix = jobs.make_prefix(image_folder)
        self._image_folder = image_folder
        self._has_strain = self._strain is not None
        # What the dialog took from the application, to notice when a new run or project replaces it.
        self._results_from_state = own
        self._strain_from_state = self._strain is not None and self._strain is state.strain_results
        self._provider = FieldProvider(results, self._strain)   # GUI-thread caches (preview, frame lists)
        self._workers: dict[str, jobs.ExportWorker | None] = {"data": None, "img": None, "anim": None, "report": None}
        self._controls: dict[str, RunControls] = {}

        root = QVBoxLayout(self)
        root.setSpacing(8)

        # --- Shared folder row ---
        folder_lbl = QLabel(self.tr("OUTPUT FOLDER"))
        themed(folder_lbl, "color: {TEXT_SECONDARY}; font-size: 11px; font-weight: bold;")
        root.addWidget(folder_lbl)

        path_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText(self.tr("Select output folder…"))
        self._folder_edit.textChanged.connect(self._on_folder_changed)
        path_row.addWidget(self._folder_edit, 1)

        # A minimum, not a fixed width: the label must fit at any display
        # scaling and in every language (a fixed 90 px cut "Open Folder").
        browse_btn = QPushButton(self.tr("Browse…"))
        browse_btn.setMinimumWidth(90)
        browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(browse_btn)

        self._open_folder_btn = QPushButton(self.tr("Open Folder"))
        self._open_folder_btn.setMinimumWidth(90)
        self._open_folder_btn.setEnabled(False)
        self._open_folder_btn.clicked.connect(self._on_open_folder)
        path_row.addWidget(self._open_folder_btn)
        root.addLayout(path_row)

        self._units_box = UnitsBox(hint)
        self._units_box.changed.connect(self._schedule_preview)
        root.addWidget(self._units_box)

        # --- Tabs ---
        self._tabs = QTabWidget()
        root.addWidget(self._tabs, 1)
        self._tabs.addTab(self._build_data_tab(), self.tr("Data"))
        self._tabs.addTab(self._build_images_tab(), self.tr("Images"))
        self._tabs.addTab(self._build_animation_tab(), self.tr("Animation"))
        self._tabs.addTab(self._build_report_tab(), self.tr("Report"))
        self._preview_tab_index = self._tabs.addTab(self._build_preview_tab(), self.tr("Preview"))
        # Render the preview lazily the first time the tab is opened.
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Last-used folder (the buttons follow it once every tab is built).
        last = QSettings().value(_SETTINGS_KEY_FOLDER, "", type=str)
        if last and Path(last).is_dir():
            self._folder_edit.setText(last)
        self._on_folder_changed(self._folder_edit.text())

        # --- Close button ---
        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.reject)
        root.addWidget(close_box)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _resolve_images(self, state_sequence, sequence, image_files):
        """The frames behind the pictures, or None (pictures then go on white)."""
        seq = state_sequence if sequence is _FROM_STATE else sequence
        if seq is None and image_files:
            try:
                from raftcorr.io.images import ImageSequence

                seq = ImageSequence(list(image_files))
            except Exception as exc:  # unreadable or missing images: pictures without them
                AppState.instance().log_message.emit(
                    tr_args(self.tr("Export: images not readable (%1); pictures go on white."), exc), "warn")
                seq = None
        if seq is not None and tuple(seq.shape) != tuple(self._results.grid.image_shape):
            AppState.instance().log_message.emit(
                self.tr("Export: the images do not match the results' image size; pictures go on white."), "warn")
            seq = None
        return seq

    def _available(self, name: str) -> bool:
        """True when the results hold *name*."""
        if name == "residual":
            return self._results.residual is not None
        if name in _STRAIN_FIELDS:
            return self._strain is not None and provider_key(name) in self._strain.components
        return True

    def _missing_tip(self, name: str) -> str:
        if name == "residual":
            return self.tr("This run has no intensity residual.")
        return self.tr("Run Compute Strain first.")

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 (Qt API)
        super().showEvent(event)
        # Keep this modal, taskbar-less dialog fully on screen so its title bar
        # and Close button always stay reachable.  Only on first show: the
        # layout (and frameGeometry) is realised once the dialog is shown.
        if not self._screen_fitted:
            self._screen_fitted = True
            fit_dialog_to_screen(self, self.parentWidget())

    def _field_checks(self, names: tuple[str, ...], selected: set[str]) -> dict[str, QCheckBox]:
        checks: dict[str, QCheckBox] = {}
        for key in names:
            chk = QCheckBox(field_label(key))
            has = self._available(key)
            chk.setEnabled(has)
            chk.setChecked(has and key in selected)
            if not has:
                chk.setToolTip(self._missing_tip(key))
            checks[key] = chk
        return checks

    def _build_data_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        # FORMAT group
        fmt_group = QGroupBox(self.tr("FORMAT"))
        fmt_layout = QVBoxLayout(fmt_group)
        self._npz_check = QCheckBox(self.tr("NumPy Archive (.npz)"))
        self._npz_check.setChecked(True)
        self._mat_check = QCheckBox(self.tr("MATLAB v7.3 (.mat)"))
        self._mat_check.setChecked(True)
        self._mat_check.setToolTip(self.tr("An HDF5-based MAT file: load it in MATLAB with load()."))
        self._h5_check = QCheckBox(self.tr("HDF5 (.h5)"))
        self._h5_check.setChecked(False)
        self._csv_check = QCheckBox(self.tr("CSV (per frame)"))
        self._csv_check.setChecked(False)
        self._csv_check.setToolTip(self.tr(
            "One table per computed frame, one row per point of the region of interest.\n"
            "Dense fields make large files: a 1000 x 1000 px region is a million rows per frame."))
        for chk in (self._npz_check, self._mat_check, self._h5_check, self._csv_check):
            fmt_layout.addWidget(chk)
        layout.addWidget(fmt_group)

        self._data_disp_checks = self._field_checks(_DISP_FIELDS, _DATA_DEFAULT_SELECTED)
        layout.addWidget(check_group(self.tr("DISPLACEMENT"), self._data_disp_checks, 4))

        self._data_strain_checks = self._field_checks(_STRAIN_FIELDS, _DATA_DEFAULT_SELECTED)
        strain_group = check_group(self.tr("STRAIN"), self._data_strain_checks, 5)
        strain_group.setEnabled(self._has_strain)
        if not self._has_strain:
            strain_group.setToolTip(self.tr("Run Compute Strain first."))
        layout.addWidget(strain_group)

        params_note = QLabel(
            self.tr("✓ Parameters file (JSON) always exported") + "\n"
            + self.tr("✓ Grid coordinates (X, Y), ROI and computed-frame flags always included") + "\n"
            + self.tr("Displacements are written in pixels (+u right, +v down); the pixel size and "
                      "frame rate above are recorded with them."))
        params_note.setWordWrap(True)
        themed(params_note, "color: {TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(params_note)

        self._controls["data"] = RunControls(self, layout, self.tr("Export Data"), self._on_export_data,
                                             lambda: self._on_cancel("data"))
        self._export_data_btn = self._controls["data"].run_btn
        layout.addStretch()
        return w

    def _render_as_row(self) -> tuple[QHBoxLayout, QComboBox, QCheckBox]:
        """Where the field is drawn and whether an image sits behind it: one control each."""
        row = QHBoxLayout()
        geom = QComboBox()
        geom.addItem(self.tr("Deformed frame"), True)
        geom.addItem(self.tr("Reference frame"), False)
        geom.setCurrentIndex(0 if self._hint.show_deformed else 1)
        geom.setToolTip(self.tr(
            "Deformed: the field is drawn where each point is in that frame, over the frame's own "
            "photo.\nReference: drawn at the points' positions in frame 1, over frame 1."))
        row.addWidget(geom)
        bg = QCheckBox(self.tr("Show background image"))
        has_images = self._images is not None
        bg.setChecked(self._hint.show_background and has_images)
        bg.setEnabled(has_images)
        bg.setToolTip(self.tr("Uncheck to export the field on its own, on white, with no speckle image behind it.")
                      if has_images else self.tr("The images of this run are not available."))
        row.addWidget(bg)
        row.addStretch()
        return row, geom, bg

    def _build_images_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        layout.addLayout(field_header())
        scroll, self._img_field_rows = build_field_rows_scroll(_ALL_FIELDS, self._hint, self._available)
        layout.addWidget(scroll)

        # IMAGE SETTINGS group
        img_group = QGroupBox(self.tr("IMAGE SETTINGS"))
        img_form = QFormLayout(img_group)
        img_form.setSpacing(6)

        fmt_row = QHBoxLayout()
        self._img_fmt_combo = QComboBox()
        for name, ext in _IMAGE_FORMATS:   # file-format names stay literal per glossary
            self._img_fmt_combo.addItem(name, ext)
        fmt_row.addWidget(self._img_fmt_combo)
        fmt_row.addStretch()
        img_form.addRow(self.tr("Format"), fmt_row)

        dpi_row = QHBoxLayout()
        self._img_dpi_spin = QSpinBox()
        self._img_dpi_spin.setRange(72, 600)
        self._img_dpi_spin.setValue(150)
        self._img_dpi_spin.setFixedWidth(70)
        self._img_dpi_spin.setToolTip(self.tr(
            "Resolution of the written figure: about 8.6 x 7 inches at this many dots per inch."))
        dpi_row.addWidget(self._img_dpi_spin)
        dpi_row.addStretch()
        img_form.addRow(self.tr("DPI"), dpi_row)

        range_row = QHBoxLayout()
        self._img_range_combo = QComboBox()
        self._img_range_combo.addItem(self.tr("Each frame"), "frame")
        self._img_range_combo.addItem(self.tr("Whole sequence"), "sequence")
        self._img_range_combo.setToolTip(self.tr(
            "How Auto picks the colour range: from each frame's own values (like the main window),\n"
            "or one range from every exported frame, so a colour means the same value in every image."))
        self._img_range_combo.currentIndexChanged.connect(self._schedule_preview)
        range_row.addWidget(self._img_range_combo)
        range_row.addStretch()
        img_form.addRow(self.tr("Auto range"), range_row)

        self._img_colorbar_check = QCheckBox(self.tr("Include colorbar"))
        self._img_colorbar_check.setChecked(True)
        self._img_colorbar_check.setToolTip(self.tr("Add a vertical colorbar to the right of each image."))
        self._img_colorbar_check.toggled.connect(self._schedule_preview)
        img_form.addRow(self._img_colorbar_check)

        config_row, self._img_geom_combo, self._img_bg_check = self._render_as_row()
        self._img_geom_combo.currentIndexChanged.connect(self._schedule_preview)
        self._img_bg_check.toggled.connect(self._schedule_preview)
        img_form.addRow(self.tr("Render as"), config_row)

        self._img_crop_check = QCheckBox(self.tr("Crop to ROI"))
        self._img_crop_check.setToolTip(self.tr("Cut the picture to the region of interest plus a small margin."))
        self._img_crop_check.toggled.connect(self._schedule_preview)
        img_form.addRow(self._img_crop_check)
        layout.addWidget(img_group)

        self._img_range_box = FrameRangeBox(self._results.n_frames)
        layout.addWidget(self._img_range_box)
        self._controls["img"] = RunControls(self, layout, self.tr("Export Images"), self._on_export_images,
                                            lambda: self._on_cancel("img"))
        self._export_img_btn = self._controls["img"].run_btn
        return w

    def _build_animation_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(8)
        layout.addLayout(field_header())
        scroll, self._anim_field_rows = build_field_rows_scroll(_ALL_FIELDS, self._hint, self._available)
        layout.addWidget(scroll)

        # ANIMATION SETTINGS group
        anim_group = QGroupBox(self.tr("ANIMATION SETTINGS"))
        anim_form = QFormLayout(anim_group)
        anim_form.setSpacing(6)

        fmt_row = QHBoxLayout()
        self._anim_fmt_combo = QComboBox()
        # File-format names stay literal per glossary.
        self._anim_fmt_combo.addItem("MP4", ".mp4")
        self._anim_fmt_combo.addItem("GIF", ".gif")
        if not mp4_available():
            item = self._anim_fmt_combo.model().item(0)
            item.setEnabled(False)
            tip = self.tr("MP4 needs the imageio and imageio-ffmpeg packages "
                          "(pip install imageio imageio-ffmpeg). Save as GIF instead.")
            item.setToolTip(tip)
            self._anim_fmt_combo.setToolTip(tip)
            self._anim_fmt_combo.setCurrentIndex(1)
        fmt_row.addWidget(self._anim_fmt_combo)
        fmt_row.addStretch()
        anim_form.addRow(self.tr("Format"), fmt_row)

        fps_row = QHBoxLayout()
        self._anim_fps_spin = QSpinBox()
        self._anim_fps_spin.setRange(1, 60)
        self._anim_fps_spin.setValue(10)
        self._anim_fps_spin.setFixedWidth(60)
        fps_row.addWidget(self._anim_fps_spin)
        fps_row.addStretch()
        anim_form.addRow(self.tr("FPS"), fps_row)

        # Frame step: keep every Nth frame (faster + smaller, choppier). FPS
        # above is the pre-decimation timeline, so playback stays real-time.
        step_row = QHBoxLayout()
        self._anim_step_spin = QSpinBox()
        self._anim_step_spin.setRange(1, 20)
        self._anim_step_spin.setValue(1)
        self._anim_step_spin.setFixedWidth(60)
        self._anim_step_spin.setToolTip(self.tr(
            "Export every Nth frame (1 = every frame). Higher is faster and "
            "smaller\nbut looks choppier. Playback duration is preserved (the "
            "FPS above is the pre-decimation rate)."))
        step_row.addWidget(self._anim_step_spin)
        step_row.addStretch()
        anim_form.addRow(self.tr("Frame step"), step_row)

        self._anim_colorbar_check = QCheckBox(self.tr("Include colorbar"))
        self._anim_colorbar_check.setChecked(True)
        self._anim_colorbar_check.setToolTip(self.tr("Add a vertical colorbar to the right of each frame."))
        anim_form.addRow(self._anim_colorbar_check)

        anim_config_row, self._anim_geom_combo, self._anim_bg_check = self._render_as_row()
        anim_form.addRow(self.tr("Render as"), anim_config_row)

        self._anim_crop_check = QCheckBox(self.tr("Crop to ROI"))
        self._anim_crop_check.setToolTip(self.tr("Cut the frames to the region of interest plus a small margin."))
        anim_form.addRow(self._anim_crop_check)

        range_note = QLabel(self.tr(
            "Auto range: one range for the whole animation, so a colour means the same value in every frame."))
        range_note.setWordWrap(True)
        themed(range_note, "color: {TEXT_MUTED}; font-size: 11px;")
        anim_form.addRow(range_note)
        layout.addWidget(anim_group)

        self._anim_range_box = FrameRangeBox(self._results.n_frames)
        layout.addWidget(self._anim_range_box)
        self._controls["anim"] = RunControls(self, layout, self.tr("Export Animation"), self._on_export_animation,
                                             lambda: self._on_cancel("anim"))
        self._export_anim_btn = self._controls["anim"].run_btn
        return w

    def _build_report_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        content_group = QGroupBox(self.tr("CONTENT"))
        content_vl = QVBoxLayout(content_group)
        self._report_params_check = QCheckBox(self.tr("Parameter summary table"))
        self._report_params_check.setChecked(True)
        self._report_stats_check = QCheckBox(self.tr("Field statistics (min/max/mean/std per frame)"))
        self._report_stats_check.setChecked(True)
        self._report_images_check = QCheckBox(self.tr("Sample field images"))
        self._report_images_check.setChecked(True)
        self._report_images_check.setToolTip(self.tr(
            "Drawn like the Images tab: same colormap, range, opacity and configuration."))
        for chk in (self._report_params_check, self._report_stats_check, self._report_images_check):
            content_vl.addWidget(chk)

        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel(self.tr("Sample every")))
        self._report_sample_spin = QSpinBox()
        self._report_sample_spin.setRange(1, 100)
        self._report_sample_spin.setValue(5)
        self._report_sample_spin.setFixedWidth(60)
        sample_row.addWidget(self._report_sample_spin)
        sample_row.addWidget(QLabel(self.tr("frames", "Report: sample every N frames")))
        sample_row.addStretch()
        content_vl.addLayout(sample_row)
        layout.addWidget(content_group)

        # FIELDS group
        fields_group = QGroupBox(self.tr("FIELDS"))
        fields_vl = QVBoxLayout(fields_group)
        for title, names, attr, per_row in ((self.tr("Displacement:"), _DISP_FIELDS, "_report_disp_checks", 4),
                                            (self.tr("Strain:"), _STRAIN_FIELDS, "_report_strain_checks", 5)):
            lbl = QLabel(title)
            themed(lbl, "color: {TEXT_SECONDARY}; font-size: 11px; font-weight: bold;")
            fields_vl.addWidget(lbl)
            checks = self._field_checks(names, _DATA_DEFAULT_SELECTED)
            setattr(self, attr, checks)
            add_check_rows(fields_vl, checks, per_row)
        layout.addWidget(fields_group)

        format_note = QLabel(self.tr("Format: HTML (self-contained, view in any browser)"))
        themed(format_note, "color: {TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(format_note)

        self._controls["report"] = RunControls(self, layout, self.tr("Generate Report"), self._on_generate_report,
                                               lambda: self._on_cancel("report"))
        self._generate_report_btn = self._controls["report"].run_btn
        layout.addStretch()
        return w

    # -- Preview tab -------------------------------------------------------

    def _build_preview_tab(self) -> QWidget:
        """WYSIWYG preview of one exported frame + the field's appearance."""
        self._preview = ExportPreview(
            self._provider, self._images, self._img_field_rows, self._anim_field_rows,
            style_for=lambda cfg: self._image_style(cfg, self._img_settings()), units=self._units,
            frames_for=lambda key: self._frames_for(key, "img"))
        return self._preview

    def _on_tab_changed(self, idx: int) -> None:
        if idx == getattr(self, "_preview_tab_index", -1):
            self._preview.activate()

    def _schedule_preview(self) -> None:
        if hasattr(self, "_preview"):
            self._preview.schedule()

    # ------------------------------------------------------------------
    # Settings -> jobs
    # ------------------------------------------------------------------

    def _folder(self) -> str:
        return self._folder_edit.text().strip()

    def _names(self) -> jobs.OutputNames:
        """Output names with a fresh timestamp (one per export click)."""
        return jobs.OutputNames(Path(self._folder()), self._prefix, jobs.make_timestamp())

    def _units(self) -> Units:
        return self._units_box.units()

    def _frame_window(self, prefix: str) -> range:
        """0-based frames of a tab's FRAME RANGE."""
        return (self._img_range_box if prefix == "img" else self._anim_range_box).frames()

    def _frames_for(self, key: str, prefix: str, step: int = 1) -> list[int]:
        """Computed frames of provider field *key* in a tab's frame range."""
        return [k for k in self._frame_window(prefix)[::max(1, step)] if self._provider.has_frame(key, k)]

    def _img_settings(self) -> dict:
        return dict(deformed=bool(self._img_geom_combo.currentData()), background=self._img_bg_check.isChecked(),
                    colorbar=self._img_colorbar_check.isChecked(), crop=self._img_crop_check.isChecked(),
                    range_mode=self._img_range_combo.currentData(), dpi=self._img_dpi_spin.value())

    def _anim_settings(self) -> dict:
        return dict(deformed=bool(self._anim_geom_combo.currentData()), background=self._anim_bg_check.isChecked(),
                    colorbar=self._anim_colorbar_check.isChecked(), crop=self._anim_crop_check.isChecked(),
                    range_mode="sequence", dpi=100)

    def _image_style(self, cfg: FieldImageConfig, settings: dict) -> FrameStyle:
        fixed = not cfg.auto_range
        return FrameStyle(
            field=provider_key(cfg.field_name),
            configuration="deformed" if settings["deformed"] else "reference",
            background="auto" if settings["background"] and self._images is not None else "none",
            colormap=cfg.colormap, opacity=float(cfg.bg_alpha),
            vmin=float(cfg.vmin) if fixed else None, vmax=float(cfg.vmax) if fixed else None,
            range_mode=settings["range_mode"], colorbar=settings["colorbar"], title=False,
            crop_to_roi=settings["crop"], dpi=int(settings["dpi"]))

    def _field_outputs(self, rows: list[FieldRow], settings: dict, prefix: str,
                       step: int = 1) -> tuple[list[jobs.FieldOutput], str]:
        """The enabled fields with computed frames in range, or ([], reason)."""
        out = []
        for row in rows:
            cfg = row.get_config()
            if not cfg.enabled:
                continue
            if not cfg.auto_range and not cfg.vmax > cfg.vmin:
                return [], tr_args(self.tr("Min must be below Max for %1."), field_label(cfg.field_name))
            style = self._image_style(cfg, settings)
            frames = self._frames_for(style.field, prefix, step)
            if frames:
                out.append(jobs.FieldOutput(cfg.field_name, field_label(cfg.field_name), style, tuple(frames)))
        if not out:
            return [], self.tr("Nothing to export: enable at least one field with computed frames in the frame range.")
        return out, ""

    def data_job(self) -> jobs.DataJob | str:
        """The Data tab's job, or the reason it cannot run."""
        formats = tuple(ext for chk, ext in ((self._npz_check, ".npz"), (self._mat_check, ".mat"),
                                             (self._h5_check, ".h5")) if chk.isChecked())
        if not formats and not self._csv_check.isChecked():
            return self.tr("Select at least one format.")
        variables = tuple(_DATA_VARIABLE[k] for k, chk in {**self._data_disp_checks, **self._data_strain_checks}.items()
                          if chk.isEnabled() and chk.isChecked())
        if not variables:
            return self.tr("Select at least one variable to export.")
        return jobs.DataJob(self._names(), self._results, self._strain, formats=formats,
                            csv=self._csv_check.isChecked(), variables=variables,
                            units=self._units_box.physical_units())

    def images_job(self) -> jobs.ImagesJob | str:
        fields, reason = self._field_outputs(self._img_field_rows, self._img_settings(), "img")
        if not fields:
            return reason
        return jobs.ImagesJob(self._names(), self._results, self._strain, self._images, tuple(fields),
                              units=self._units(), fmt=self._img_fmt_combo.currentData())

    def animation_job(self) -> jobs.AnimationJob | str:
        step = self._anim_step_spin.value()
        fields, reason = self._field_outputs(self._anim_field_rows, self._anim_settings(), "anim", step)
        if not fields:
            return reason
        return jobs.AnimationJob(self._names(), self._results, self._strain, self._images, tuple(fields),
                                 units=self._units(), ext=self._anim_fmt_combo.currentData(),
                                 fps=self._anim_fps_spin.value() / step)

    def report_job(self) -> jobs.ReportJob | str:
        wanted = [k for k, chk in {**self._report_disp_checks, **self._report_strain_checks}.items()
                  if chk.isEnabled() and chk.isChecked()]
        settings = self._img_settings()
        rows = {r.field_name: r for r in self._img_field_rows}
        fields = []
        for name in wanted:
            cfg = replace(rows[name].get_config(), enabled=True)
            if not cfg.auto_range and not cfg.vmax > cfg.vmin:
                return tr_args(self.tr("Min must be below Max for %1."), field_label(name))
            fields.append(jobs.FieldOutput(name, field_label(name), self._image_style(cfg, settings), ()))
        if not fields and not self._report_params_check.isChecked():
            return self.tr("Nothing to export: select a field or the parameter table.")
        return jobs.ReportJob(self._names(), self._results, self._strain, self._images, tuple(fields),
                              units=self._units(), include_parameters=self._report_params_check.isChecked(),
                              include_statistics=self._report_stats_check.isChecked(),
                              include_images=self._report_images_check.isChecked(),
                              sample_every=self._report_sample_spin.value())

    # ------------------------------------------------------------------
    # Slots — folder
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        # Start from the current text, or fall back to the image folder.
        start = self._folder()
        if not start and self._image_folder:
            start = str(self._image_folder)
        folder = QFileDialog.getExistingDirectory(self, self.tr("Select Output Folder"), start)
        if folder:
            self._folder_edit.setText(folder)
            QSettings().setValue(_SETTINGS_KEY_FOLDER, folder)

    def _on_folder_changed(self, text: str) -> None:
        has_folder = bool(text.strip())
        for ctl in self._controls.values():
            ctl.run_btn.setEnabled(has_folder and not ctl.running)

    def _on_open_folder(self) -> None:
        folder = self._folder()
        if folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    # ------------------------------------------------------------------
    # Slots — exports
    # ------------------------------------------------------------------

    def _on_export_data(self) -> None:
        self._start("data", self.data_job())

    def _on_export_images(self) -> None:
        self._start("img", self.images_job())

    def _on_export_animation(self) -> None:
        self._start("anim", self.animation_job())

    def _on_generate_report(self) -> None:
        self._start("report", self.report_job())

    def _confirm_overwrite(self, existing: list[Path]) -> bool:
        """Ask before replacing files; True to go ahead."""
        if not existing:
            return True
        shown = "\n".join(p.name for p in existing[:8])
        if len(existing) > 8:
            shown += "\n" + tr_args(self.tr("… and %1 more"), len(existing) - 8)
        answer = QMessageBox.question(
            self, self.tr("Replace Existing Files?"),
            tr_args(self.tr("These already exist in the output folder and will be replaced:\n\n%1"), shown),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def _results_replaced(self) -> bool:
        """True when the application's results (or strain) are no longer the ones shown here."""
        state = AppState.instance()
        if self._results_from_state and state.results is not self._results:
            return True
        return self._strain_from_state and state.strain_results is not self._strain

    def _start(self, tab: str, job) -> None:
        ctl = self._controls[tab]
        if not self._folder() or ctl.running:
            return
        if isinstance(job, str):   # a reason why it cannot run
            ctl.set_status(job, "DANGER")
            return
        if self._results_replaced():
            ctl.set_status(self.tr("The results changed after this dialog was opened (a new run, strain "
                                   "or project). Close it and open Export again to export them."), "DANGER")
            return
        if not self._confirm_overwrite(job.existing()):
            ctl.set_status(self.tr("Nothing exported: the existing files were kept."), "WARNING")
            return
        ctl.set_status("", "TEXT_MUTED")
        worker = jobs.ExportWorker(job, parent=self)
        worker.progress.connect(self._on_worker_progress)
        worker.succeeded.connect(self._on_worker_succeeded)
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)
        previous = self._workers[tab]
        if previous is not None and not _still_running(previous):
            previous.deleteLater()   # finished: release it instead of keeping it as a child
        self._workers[tab] = worker
        ctl.set_running(True, True)
        ctl.progress_lbl.setText(self.tr("Starting…"))
        worker.start()

    def _tab_of(self, worker) -> str | None:
        return next((tab for tab, w in self._workers.items() if w is worker and w is not None), None)

    def _on_cancel(self, tab: str) -> None:
        worker = self._workers.get(tab)
        if worker is not None and _still_running(worker):
            worker.request_stop()
            ctl = self._controls[tab]
            ctl.cancel_btn.setEnabled(False)
            ctl.progress_lbl.setText(self.tr("Stopping…"))

    def _on_worker_progress(self, fraction: float, item: str, done: int, total: int) -> None:
        tab = self._tab_of(self.sender())
        if tab is None:
            return
        ctl = self._controls[tab]
        ctl.progress.setValue(int(round(fraction * 100)))
        if not ctl.cancel_btn.isEnabled():   # stopping: keep saying so
            return
        if tab in ("img", "anim"):
            ctl.progress_lbl.setText(tr_args(self.tr("Rendering %1 (%2/%3)"), item, done, total))
        else:
            ctl.progress_lbl.setText(tr_args(self.tr("Writing %1 (%2%)"), item, int(round(fraction * 100))))

    def _finish(self, tab: str) -> RunControls:
        ctl = self._controls[tab]
        ctl.set_running(False, bool(self._folder()))
        return ctl

    def _on_worker_succeeded(self, paths: list) -> None:
        tab = self._tab_of(self.sender())
        if tab is None:
            return
        ctl = self._finish(tab)
        ctl.progress.setValue(100)
        folder = self._folder()
        n = len(paths)
        if tab == "data":
            ctl.set_status(tr_args(self.tr("Exported %n file(s) → %1", "", n), folder), "SUCCESS")
        elif tab == "img":
            ctl.set_status(tr_args(self.tr("Exported %n image(s) → %1", "", n), folder), "SUCCESS")
        elif tab == "anim":
            if not paths:
                # Finishing with nothing written is a failure, not a success with a count of zero.
                ctl.set_status(self.tr("No animation was written. See the log for details."), "DANGER")
                return
            ctl.set_status(tr_args(self.tr("Exported %n animation(s) → %1", "", n), folder), "SUCCESS")
        else:
            ctl.set_status(tr_args(self.tr("Report saved → %1"), paths[0] if paths else folder), "SUCCESS")
        self._open_folder_btn.setEnabled(True)

    def _on_worker_failed(self, msg: str) -> None:
        tab = self._tab_of(self.sender())
        if tab is None:
            return
        self._finish(tab).set_status(tr_args(self.tr("Error: %1"), msg), "DANGER")
        AppState.instance().log_message.emit(tr_args(self.tr("Export failed: %1"), msg), "error")

    def _on_worker_cancelled(self, paths: list) -> None:
        tab = self._tab_of(self.sender())
        if tab is None:
            return
        n = len(paths)
        self._finish(tab).set_status(
            self.tr("Export cancelled: %n file(s) written before stopping.", "", n), "WARNING")
        if paths:
            self._open_folder_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def is_busy(self) -> bool:
        """True while any export of this dialog is running."""
        return any(w is not None and _still_running(w) for w in self._workers.values())

    def _shutdown_workers(self) -> None:
        """Stop and join running export threads before this dialog goes away.

        The workers are QThreads parented to the dialog, so letting the dialog be destroyed
        mid-export tears down a running thread and aborts the whole process. Ask the worker
        to stop (export honours the stop request between frames) and wait for it. Should it
        outlast the wait, cut it loose instead of blocking the UI: detach its signals so it
        cannot call back into a dialog that is going away, drop the parent link so Qt will not
        destroy it, and keep a reference until it finishes on its own.
        """
        for tab, worker in list(self._workers.items()):
            if worker is None:
                continue
            self._workers[tab] = None
            try:
                if not worker.isRunning():
                    continue
                worker.request_stop()
                if worker.wait(_WORKER_STOP_WAIT_MS):
                    continue
                for signal in (worker.progress, worker.succeeded, worker.failed, worker.cancelled):
                    try:
                        signal.disconnect()
                    except (RuntimeError, TypeError):
                        pass
                worker.setParent(None)
                _ORPHANED_WORKERS.append(worker)
            except RuntimeError:
                # Underlying C++ object already gone; nothing left to join.
                continue
        _ORPHANED_WORKERS[:] = [w for w in _ORPHANED_WORKERS if _still_running(w)]

    def reject(self) -> None:
        """Close button / Esc — never leave an export thread running."""
        self._shutdown_workers()
        super().reject()

    def accept(self) -> None:
        self._shutdown_workers()
        super().accept()

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        """Window close (X) — same teardown as the Close button."""
        self._shutdown_workers()
        super().closeEvent(event)
