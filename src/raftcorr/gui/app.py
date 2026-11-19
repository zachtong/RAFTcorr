"""RAFTcorr GUI application entry point (the pyALDIC main window, RAFTcorr engine)."""

import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import shiboken6
from PySide6.QtCore import QCoreApplication, Qt, QThread, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QWidget,
)

from raftcorr.callbacks import StopRequested
from raftcorr.i18n import tr_args
from raftcorr.gui.app_state import AppState, RunState
from raftcorr.models.zoo import ModelUnavailableError
from raftcorr.gui.controllers.image_controller import ImageController
from raftcorr.gui.controllers.pipeline_controller import PipelineController
from raftcorr.gui.controllers.roi_controller import ROIController
from raftcorr.gui.controllers.viz_controller import VizController
from raftcorr.gui.icons import icon_app
from raftcorr.gui.panels.canvas_area import CanvasArea
from raftcorr.gui.panels.left_sidebar import LeftSidebar
from raftcorr.gui.panels.right_sidebar import RightSidebar
from raftcorr.gui.theme_manager import (
    apply_title_bar,
    connect_theme,
    startup_theme,
    theme_manager,
)

PROJECT_FILTER = "RAFTcorr project (*.raftcorr)"


class _Task(QThread):
    """Runs ``fn(callbacks)`` off the GUI thread (project files, model survey)."""

    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()                    # Cancel of the progress dialog: not an error
    model_unavailable = Signal(str, str)    # (model key, reason): the Models window takes over

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn
        self.stop_requested = False

    def run(self) -> None:
        from raftcorr.callbacks import FunctionCallbacks

        cb = FunctionCallbacks(progress=lambda f, m="": self.progress.emit(float(f), str(m)),
                               should_stop=lambda: self.stop_requested)
        try:
            self.done.emit(self._fn(cb))
        except StopRequested:
            self.cancelled.emit()
        except ModelUnavailableError as exc:
            self.model_unavailable.emit(exc.spec.key, exc.short)
        except Exception as exc:  # report to the GUI thread
            logging.getLogger("raftcorr").exception("background task failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    """Three-column main window: left sidebar | canvas | right sidebar."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RAFTcorr")
        self.setWindowIcon(icon_app())
        self.setMinimumSize(1120, 760)
        apply_title_bar(self)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # State and controllers
        self._state = AppState.instance()
        self._image_ctrl = ImageController(self._state)
        self._roi_ctrl: ROIController | None = None
        self._task: _Task | None = None

        self._left_sidebar = LeftSidebar(self._image_ctrl)
        layout.addWidget(self._left_sidebar, stretch=0)

        self._viz_ctrl = VizController()
        self._canvas_area = CanvasArea(self._image_ctrl, viz_ctrl=self._viz_ctrl)
        layout.addWidget(self._canvas_area, stretch=1)

        self._pipeline_ctrl = PipelineController(self._state, self._image_ctrl)
        self._right_sidebar = RightSidebar(self._pipeline_ctrl)
        layout.addWidget(self._right_sidebar, stretch=0)

        # Lazy strain post-processing window (singleton until MainWindow closes).
        self._strain_window = None
        self._right_sidebar.open_strain_window_requested.connect(self._on_open_strain_window)
        self._state.run_state_changed.connect(self._on_run_state_changed)
        self._run_warning_box: QMessageBox | None = None
        self._state.run_warning.connect(self._on_run_warning)

        # ROI toolbar
        roi_tb = self._left_sidebar.roi_toolbar
        roi_tb.draw_requested.connect(self._on_draw_requested)
        roi_tb.clear_requested.connect(self._on_roi_clear)
        roi_tb.import_requested.connect(self._on_roi_import)
        roi_tb.save_requested.connect(self._on_roi_save)
        roi_tb.invert_requested.connect(self._on_roi_invert)
        roi_tb.batch_import_requested.connect(self._on_batch_import)
        self._canvas_area.canvas.drawing_finished.connect(roi_tb.deactivate)

        # Model suggestion (quick s8 survey)
        self._left_sidebar.workflow_panel.suggest_requested.connect(self._on_suggest_model)

        # Per-frame ROI editing from the image list
        self._left_sidebar._image_list.roi_edit_requested.connect(self._on_roi_edit_for_frame)
        self._left_sidebar._image_list.roi_import_for_frames.connect(self._on_roi_import_for_frames)

        self._state.images_changed.connect(self._init_roi_controller)
        self._state.current_frame_changed.connect(self._on_frame_changed_for_roi)
        self._state.roi_changed.connect(self._on_roi_changed_reload)
        self._state.results_changed.connect(self._viz_ctrl.clear_all)
        # A model that is neither installed nor downloadable: the Models window, not an error box.
        self._state.model_unavailable.connect(self._open_models)

        self._build_menu_bar()

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------
    def _build_menu_bar(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu(self.tr("&File"))

        def add(menu, text, slot, shortcut=None):
            act = QAction(text, self)
            if shortcut is not None:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            menu.addAction(act)
            return act

        add(file_menu, self.tr("Open Image Folder…"), self._on_open_images, QKeySequence("Ctrl+I"))
        file_menu.addSeparator()
        add(file_menu, self.tr("Open Project…"), self._on_open_project, QKeySequence.StandardKey.Open)
        add(file_menu, self.tr("Save Project"), lambda: self._on_save_project(False), QKeySequence.StandardKey.Save)
        add(file_menu, self.tr("Save Project As…"), lambda: self._on_save_project(True), QKeySequence("Ctrl+Shift+S"))
        file_menu.addSeparator()
        add(file_menu, self.tr("Export Results…"), self._right_sidebar._on_export, QKeySequence("Ctrl+E"))
        from raftcorr.gui import file_association

        if file_association.is_supported():
            file_menu.addSeparator()
            act = add(file_menu, self.tr("Associate .raftcorr files with RAFTcorr…"), self._on_register_association)
            act.setToolTip(self.tr("Register .raftcorr so double-clicking a project file opens RAFTcorr "
                                   "(current user only, no admin rights needed)."))
        file_menu.addSeparator()
        add(file_menu, self.tr("Quit"), self.close, QKeySequence.StandardKey.Quit)

        from raftcorr.i18n import SUPPORTED_LANGUAGES, LanguageManager

        settings_menu = menu_bar.addMenu(self.tr("&Settings"))
        from raftcorr.gui.theme import current_theme

        theme_menu = settings_menu.addMenu(self.tr("Theme"))
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        self._theme_actions: dict[str, QAction] = {}
        for name, label in (("dark", self.tr("Dark", "Color theme")), ("light", self.tr("Light", "Color theme"))):
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(name == current_theme())
            act.triggered.connect(lambda _checked=False, n=name: theme_manager().set_theme(n))
            theme_group.addAction(act)
            theme_menu.addAction(act)
            self._theme_actions[name] = act
        connect_theme(self._on_theme_changed)

        language_menu = settings_menu.addMenu(self.tr("Language"))
        current = LanguageManager.saved_preference()
        for code, display_name in SUPPORTED_LANGUAGES.items():
            act = QAction(display_name, self)
            act.setCheckable(True)
            act.setChecked(code == current)
            act.triggered.connect(lambda _checked=False, c=code: self._on_language_selected(c))
            language_menu.addAction(act)

        help_menu = menu_bar.addMenu(self.tr("&Help"))
        add(help_menu, self.tr("Models…"), lambda: self._open_models())
        add(help_menu, self.tr("Self-test"), self._on_self_test)
        add(help_menu, self.tr("About RAFTcorr"), self._on_about)

    # ------------------------------------------------------------------
    # Images, projects, models
    # ------------------------------------------------------------------
    def _on_open_images(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, self.tr("Select Image Folder"),
                                                  str(self._state.image_folder or ""))
        if folder:
            self._image_ctrl.load_folder(folder)

    def _run_task(self, title: str, fn, on_done) -> None:
        """Run ``fn(callbacks)`` in the background behind a progress dialog."""
        if self._task is not None:
            return
        dlg = QProgressDialog(title + "…", self.tr("Cancel"), 0, 1000, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(300)
        task = _Task(fn)
        self._task = task
        dlg.canceled.connect(lambda: setattr(task, "stop_requested", True))
        task.progress.connect(lambda f, m: (dlg.setValue(int(1000 * f)), dlg.setLabelText(m or title)))

        def _finish() -> None:
            self._task = None
            dlg.close()

        task.done.connect(lambda r: (_finish(), on_done(r)))
        task.cancelled.connect(_finish)
        task.model_unavailable.connect(lambda key, why: (_finish(), self._state.model_unavailable.emit(key, why)))
        task.failed.connect(lambda msg: (_finish(), QMessageBox.critical(self, title, msg),
                                         self._state.log_message.emit(msg, "error")))
        task.start()

    def _on_save_project(self, ask: bool) -> None:
        state = self._state
        if not state.image_files:
            QMessageBox.information(self, self.tr("Save Project"), self.tr(
                "Load the images first: a project keeps the images, the regions of interest, the "
                "settings and the results together."))
            return
        path = state.project_path
        if ask or path is None:
            start = str(path or (state.image_folder or Path.home()) / "project.raftcorr")
            name, _ = QFileDialog.getSaveFileName(self, self.tr("Save Project"), start, PROJECT_FILTER)
            if not name:
                return
            path = Path(name).with_suffix(".raftcorr")
        from raftcorr.gui.project_bridge import project_from_state
        from raftcorr.io.project import save_project

        include_results = self._ask_include_results()
        if include_results is None:
            return
        project = project_from_state(state, include_results=include_results)

        def done(p) -> None:
            state.project_path = Path(p)
            self.setWindowTitle(f"RAFTcorr - {Path(p).name}")
            state.log_message.emit(tr_args(self.tr("Project saved: %1"), p), "success")

        self._run_task(self.tr("Saving the project"), lambda cb: save_project(path, project, cb), done)

    def _ask_include_results(self) -> bool | None:
        """pyALDIC's question before saving: keep the results (size given) or settings only.

        True / False, or None when the user cancels; True without asking when there are none.
        """
        if self._state.results is None:
            return True
        from raftcorr.gui.project_bridge import estimated_result_bytes
        from raftcorr.gui.support.locale_format import format_bytes

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(self.tr("Include Results?"))
        box.setText(self.tr("Include the computed results in this project?"))
        box.setInformativeText(tr_args(self.tr(
            "Including results (about %1 uncompressed) lets you reopen the project without "
            "recomputing. Choose No to save a small settings-only file for sharing."),
            format_bytes(estimated_result_bytes(self._state))))
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                               | QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QMessageBox.StandardButton.Yes)
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Cancel:
            return None
        return choice == QMessageBox.StandardButton.Yes

    def _on_open_project(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, self.tr("Open Project"), "", PROJECT_FILTER)
        if name:
            self.open_project_path(name)

    def open_project_path(self, path: str) -> None:
        from raftcorr.io.project import load_project

        self._run_task(self.tr("Opening the project"), lambda cb: load_project(path),
                       lambda prj: self._apply_project(prj, Path(path)))

    def _apply_project(self, prj, path: Path) -> None:
        from raftcorr.gui.project_bridge import apply_project

        folder = Path(prj.image_folder) if prj.image_folder else None
        if folder is None or not folder.is_dir():
            QMessageBox.warning(self, self.tr("Images not found"), self.tr(
                "The image folder of this project was not found. Choose its new location."))
            chosen = QFileDialog.getExistingDirectory(self, self.tr("Image folder of the project"), str(path.parent))
            if not chosen:
                return
            folder = Path(chosen)
        try:
            apply_project(self._state, self._image_ctrl, prj, folder, path)
        except Exception as exc:
            QMessageBox.critical(self, self.tr("Cannot open the project"), str(exc))
            return
        self._init_roi_controller()
        self.setWindowTitle(f"RAFTcorr - {path.name}")
        for w in prj.warnings:
            self._state.log_message.emit(w, "warn")
        self._state.log_message.emit(tr_args(self.tr("Project opened: %1"), path), "success")

    def _on_register_association(self) -> None:
        """Register the .raftcorr file type (Windows, current user)."""
        from raftcorr.gui import file_association

        try:
            file_association.register_association()
        except Exception as e:  # noqa: BLE001 - report any failure to the user
            QMessageBox.warning(self, self.tr("File Association Failed"),
                                self.tr("Could not register .raftcorr files: ") + str(e))
            return
        QMessageBox.information(self, self.tr("File Association"), self.tr(
            "Done. Double-clicking a .raftcorr file will now open RAFTcorr and restore that project."))

    def _on_suggest_model(self) -> None:
        state = self._state
        if len(state.image_files) < 2 or state.sequence is None:
            state.log_message.emit(self.tr("Load at least two images first."), "warn")
            return
        if state.run_state in (RunState.RUNNING, RunState.PAUSED) or self._task is not None:
            state.log_message.emit(self.tr("Wait for the running task to finish."), "warn")
            return
        from raftcorr.config import RunConfig
        from raftcorr.models.advisor import sample_frames, survey

        seq = state.sequence
        roi = state.roi_mask if state.roi_mask is not None else np.ones(seq.shape, bool)
        pairs = [(0, k) for k in sample_frames(len(seq))]
        cfg = RunConfig(model="s8", device=state.device, iters=state.iters)
        cache = self._pipeline_ctrl._loaded_cache

        def job(cb):
            from raftcorr.engine.pipeline import Pipeline
            from raftcorr.gui.support.model_weights import ensure_weights
            from raftcorr.models import load_model

            p = ensure_weights("s8", cb, lambda f, text: cb.progress(f, text) if text else None)
            key = (str(p), cfg.device, None)
            loaded = cache.get(key)
            if loaded is None:
                cache.clear()
                loaded = load_model(p, cfg.device)
                cache[key] = loaded
            return survey(Pipeline(cfg, cb, loaded), seq, roi, pairs, cb)

        def done(advice) -> None:
            from raftcorr.models.zoo import MODELS

            frames = ", ".join(str(k + 1) for k, _ in advice.per_frame)   # 1-based, as on screen
            text = tr_args(self.tr(
                "Suggested: RAFTcorr-%1. Largest displacement of the region of interest about %2 px (99th percentile "
                "over frames %3); RAFTcorr-%1 reaches %4 px."),
                advice.key, f"{advice.max_displacement_px:.1f}", frames, MODELS[advice.key].lookup_radius_px)
            if advice.beyond_reach:
                text += " " + self.tr("This exceeds even the reach of RAFTcorr-s8: use incremental tracking.")
            state.model_advice = text
            state.log_message.emit(state.model_advice, "info")
            state.set_param("model", advice.key)

        self._run_task(self.tr("Measuring the motion with RAFTcorr-s8"), job, done)

    def _open_models(self, missing_key: str | None = None, detail: str = "") -> None:
        """The Models window; opened by a run or a survey whose model could not be had, it says
        which one and why."""
        from raftcorr.gui.dialogs.models_dialog import ModelsDialog

        ModelsDialog(self, missing_key, detail).exec()

    def _on_self_test(self) -> None:
        from raftcorr.selftest import run_selftest

        def done(rep) -> None:
            self._state.log_message.emit(rep.text, "success" if rep.ok else "error")
            (QMessageBox.information if rep.ok else QMessageBox.warning)(self, self.tr("Self-test"), rep.text)

        self._run_task(self.tr("Self-test"), lambda cb: run_selftest(self._state.device, cb), done)

    def _on_about(self) -> None:
        from raftcorr._version import __version__

        QMessageBox.about(self, self.tr("About RAFTcorr"), tr_args(self.tr(
            "<b>RAFTcorr %1</b><br>Learned digital image correlation with the "
            "RAFTcorr-s1, -s2, -s4 and -s8 models.<br><br>Displacements: reference to deformed, +u right, "
            "+v down, in pixels, frame-1 (Lagrangian) coordinates.<br><br>"
            "Interface derived from pyALDIC.<br>Zixiang (Zach) Tong, The University of Texas at Austin."),
            __version__))

    def _init_roi_controller(self) -> None:
        """Create the ROI controller matching the loaded image dimensions."""
        if not self._state.image_files or self._state.sequence is None:
            return
        h, w = self._state.sequence.shape
        self._roi_ctrl = ROIController((h, w))
        self._canvas_area.canvas.set_roi_controller(self._roi_ctrl)

    def closeEvent(self, event) -> None:  # noqa: N802
        worker = self._pipeline_ctrl._worker
        if worker is not None and worker.isRunning():
            worker.request_stop()
            worker.wait(60000)
        if self._task is not None and self._task.isRunning():
            self._task.stop_requested = True
            self._task.wait(30000)
        self._close_run_warning()
        if self._strain_window is not None:
            self._strain_window.close()
            self._strain_window = None
        self._state.set_results(None)
        self._pipeline_ctrl.cleanup()
        super().closeEvent(event)


    def _on_theme_changed(self, name: str) -> None:
        """Keep the Theme menu's check on the theme in use."""
        action = self._theme_actions.get(name)
        if action is not None:
            action.setChecked(True)

    def _on_language_selected(self, lang_code: str) -> None:
        """Persist the chosen language and prompt for a restart.

        Phase-1 strategy: write the preference, show an info dialog,
        wait for the next app launch. A live runtime switch will arrive
        once every widget implements retranslate_ui() + changeEvent().
        """
        from raftcorr.i18n import LanguageManager, SUPPORTED_LANGUAGES

        app = QApplication.instance()
        lang_mgr: LanguageManager | None = getattr(
            app, "_pyaldic_lang_mgr", None)
        if lang_mgr is not None:
            lang_mgr.load(lang_code)

        from raftcorr.i18n import tr_args

        QMessageBox.information(
            self,
            self.tr("Language changed"),
            tr_args(
                self.tr(
                    "Language set to %1. Please restart RAFTcorr for all "
                    "widgets to pick up the new language."),
                SUPPORTED_LANGUAGES.get(lang_code, lang_code),
            ),
        )

    def _enter_roi_editing(self) -> None:
        """Switch to ROI editing mode — show ROI overlay, hide field overlay."""
        self._state.roi_editing = True
        self._state.display_changed.emit()
        # Auto-scroll the left sidebar to the ROI section so the
        # controls are immediately visible, no matter how far the user
        # had scrolled through the settings.
        self._left_sidebar.focus_roi_section()

    def _on_roi_edit_for_frame(self, frame: int) -> None:
        """Enter ROI editing mode for a specific frame.

        Single source of truth: ``current_frame`` *is* the editing frame.
        Switching frames here keeps the image list selection, progress bar,
        and canvas display all in sync with what the user is editing.
        """
        state = self._state
        if not state.image_files:
            return
        if self._roi_ctrl is None:
            self._init_roi_controller()
        if self._roi_ctrl is None:
            return
        state.set_current_frame(frame)
        self._load_roi_buffer_for_current_frame()
        state.roi_editing = True
        state.display_changed.emit()

    def _load_roi_buffer_for_current_frame(self) -> None:
        """Mirror per_frame_rois[current_frame] into the ROI controller buffer.

        Called on entry to ROI editing and whenever current_frame changes
        during editing.
        """
        if self._roi_ctrl is None:
            return
        state = self._state
        existing = state.per_frame_rois.get(state.current_frame)
        if existing is not None:
            self._roi_ctrl.mask = existing.copy()
        else:
            self._roi_ctrl.clear()

    def _on_frame_changed_for_roi(self, _frame: int) -> None:
        """Reload the ROI controller buffer for the new current frame.

        Only matters during ROI editing -- the canvas always paints the
        overlay from per_frame_rois[current_frame] regardless.  But the
        in-memory buffer must mirror the new frame so the next stamp
        operation (draw / invert / clear) starts from the right base.
        """
        if self._state.roi_editing:
            self._load_roi_buffer_for_current_frame()

    def _on_roi_changed_reload(self) -> None:
        """Refresh the working ROI buffer when per_frame_rois mutates.

        Only runs during ROI editing -- outside editing mode the buffer
        is irrelevant and must be left alone (otherwise batch import
        paths that the user isn't actively editing would stomp the
        in-memory buffer).
        """
        if self._state.roi_editing and self._roi_ctrl is not None:
            self._load_roi_buffer_for_current_frame()

    def _on_draw_requested(self, shape: str, mode: str) -> None:
        """Activate one-shot drawing mode on the canvas.

        The toolbar Draw button always edits the *currently displayed* frame.
        To edit a different frame, navigate there first (image list click,
        arrow keys, or per-frame Edit button).
        """
        self._load_roi_buffer_for_current_frame()
        self._enter_roi_editing()
        canvas = self._canvas_area.canvas
        canvas.set_drawing_mode(mode)
        canvas.set_tool(shape)

    def _on_roi_clear(self) -> None:
        """Clear the ROI mask for the currently edited frame.

        Brush refinement only lives on frame 0; if the user clears
        frame 0's ROI, the painted brush region becomes meaningless
        (it was scoped to the now-deleted ROI), so cascade-delete it.
        """
        if self._roi_ctrl is not None:
            self._enter_roi_editing()
            self._roi_ctrl.clear()
            self._canvas_area.canvas.update_roi_overlay()
            state = self._state
            state.set_frame_roi(state.current_frame, None)

    def _on_roi_import(self, path: str) -> None:
        """Import a mask file into the ROI controller for the current editing frame."""
        if self._roi_ctrl is None:
            return
        self._enter_roi_editing()
        try:
            self._roi_ctrl.import_mask(path)
            self._canvas_area.canvas.update_roi_overlay()
            state = self._state
            state.set_frame_roi(
                state.current_frame, self._roi_ctrl.mask.copy()
            )
        except IOError:
            pass

    def _no_own_roi_hint(self) -> str | None:
        """Explain an empty ROI buffer when the frame itself is the reason.

        The ROI toolbar acts on the current frame's *own* mask, while the
        solver falls back to frame 1's.  Reopening a session lands on
        whichever frame was last displayed, so a user who left off browsing
        results finds the toolbar dead on a frame that never had its own ROI
        -- "mask is empty" then points at the mask instead of the frame.

        Returns a ready-to-log explanation in that case, or None when the
        Region of Interest is genuinely undefined everywhere.
        """
        state = self._state
        if state.current_frame == 0:
            return None
        if state.per_frame_rois.get(state.current_frame) is not None:
            return None
        if state.per_frame_rois.get(0) is None:
            return None
        from raftcorr.i18n import tr_args
        return tr_args(
            self.tr(
                "Frame %1 has no Region of Interest of its own — frame 1's is "
                "used for computation. Switch to frame 1 to edit it, or import "
                "a mask to give this frame its own."
            ),
            state.current_frame + 1,
        )

    def _on_roi_save(self) -> None:
        """Save the current ROI mask to a PNG file."""
        if self._roi_ctrl is None:
            self._state.log_message.emit(
                self.tr("No Region of Interest to save — load images first."),
                "warn",
            )
            return
        if not self._roi_ctrl.mask.any():
            self._state.log_message.emit(
                self._no_own_roi_hint()
                or self.tr("Region of Interest mask is empty."),
                "warn",
            )
            return
        self._enter_roi_editing()
        # An anchored suggestion, not a bare name: a relative path is resolved
        # against the working directory, which for a frozen app launched from a
        # shortcut or a file association is not the user's project folder.
        folder = self._state.image_folder
        suggested = str(Path(folder) / "roi_mask.png") if folder else "roi_mask.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save Region of Interest Mask"),
            suggested,
            self.tr("PNG Images") + " (*.png);;"
            + self.tr("All Files") + " (*)",
        )
        if not path:
            return
        try:
            self._roi_ctrl.save_mask(path)
            from raftcorr.i18n import tr_args
            self._state.log_message.emit(
                tr_args(self.tr("Mask saved to %1"), path), "success"
            )
        except IOError as e:
            self._state.log_message.emit(f"Save failed: {e}", "error")

    def _on_roi_invert(self) -> None:
        """Invert the ROI mask for the currently edited frame."""
        if self._roi_ctrl is None:
            self._state.log_message.emit(
                self.tr("No Region of Interest to invert — load images first."),
                "warn"
            )
            return
        # On a frame with no ROI of its own the buffer is empty, and inverting
        # it would hand that frame the whole image as a brand-new per-frame
        # Region of Interest -- silently changing what gets correlated.
        hint = self._no_own_roi_hint()
        if hint is not None:
            self._state.log_message.emit(hint, "warn")
            return
        self._enter_roi_editing()
        self._roi_ctrl.invert()
        self._canvas_area.canvas.update_roi_overlay()
        state = self._state
        state.set_frame_roi(
            state.current_frame, self._roi_ctrl.mask.copy()
        )

    def _on_batch_import(self) -> None:
        """Open the batch mask import dialog and load assigned masks."""
        state = self._state
        if not state.image_files:
            state.log_message.emit(self.tr("Load images first."), "warn")
            return

        # Ensure ROI controller is initialized so we can get image dimensions
        if self._roi_ctrl is None:
            self._init_roi_controller()
        if self._roi_ctrl is None:
            return
        img_shape = self._roi_ctrl.shape

        # Unlike pyALDIC, RAFTcorr reads the mask of EVERY frame: on a reference frame it is
        # the ROI; on any other frame it removes the points that move into the regions it
        # leaves out (a crack, a hole). So every frame is offered.
        required_frames = None

        from raftcorr.gui.dialogs.batch_import_dialog import BatchImportDialog
        from raftcorr.i18n import tr_args

        dialog = BatchImportDialog(
            state.image_files,
            parent=self,
            required_frames=required_frames,
            img_shape=img_shape,
        )
        accepted = dialog.exec() == BatchImportDialog.DialogCode.Accepted
        masks = dialog.load_masks(img_shape) if accepted else {}
        # Delete it: a dialog given a parent outlives the call that made it,
        # and this one is reached on both paths out of here.
        dialog.deleteLater()
        if not accepted:
            return

        for frame_idx, mask in masks.items():
            state.per_frame_rois[frame_idx] = mask
            state.log_message.emit(
                tr_args(
                    self.tr("  Imported mask for frame %1"), frame_idx + 1   # 1-based, as on screen
                ),
                "info",
            )
        n_masks = len(masks)
        state.log_message.emit(
            self.tr("Batch import: %n mask(s) loaded", "", n_masks),
            "success",
        )
        state.roi_changed.emit()

    def _on_roi_import_for_frames(self, mapping: dict) -> None:
        """Import mask files for specific frames (from context menu)."""
        if self._roi_ctrl is None:
            self._init_roi_controller()
        if self._roi_ctrl is None:
            return

        from raftcorr.gui.support.io_compat import read_mask_as_bool

        img_shape = self._roi_ctrl.shape
        state = self._state
        count = 0
        for frame_idx, path in mapping.items():
            try:
                mask = read_mask_as_bool(path, target_shape=img_shape)
                state.per_frame_rois[frame_idx] = mask
                count += 1
            except (FileNotFoundError, IOError) as e:
                state.log_message.emit(f"Failed to read: {e}", "warn")

        if count:
            state.log_message.emit(
                self.tr("Imported Region of Interest for %n frame(s)", "",
                        count),
                "success",
            )
            state.roi_changed.emit()

    def _on_open_strain_window(self) -> None:
        """Show the strain post-processing window (lazy singleton)."""
        if self._state.results is None:
            self._state.log_message.emit(
                QCoreApplication.translate(
                    "App",
                    "Run DIC first -- no displacement results to post-process.",
                ),
                "warn",
            )
            return
        if self._strain_window is None:
            from raftcorr.gui.strain_window import StrainWindow
            self._strain_window = StrainWindow(self._state, parent=None)
        self._strain_window.show()
        self._strain_window.raise_()
        self._strain_window.activateWindow()

    def _on_run_state_changed(self, new_state) -> None:
        """Auto-open the strain post-processing window when a run completes.

        Previously this prompted the user with a Yes/No dialog, which added
        an extra click for an almost-always-yes answer. The strain window is
        non-modal and the user can close it immediately if they only need
        displacement, so auto-opening is strictly less friction.
        """
        from raftcorr.gui.app_state import RunState
        if new_state == RunState.RUNNING:
            # A new run makes the last run's tracking warning stale.
            self._close_run_warning()
            return
        if new_state != RunState.DONE or self._state.results is None:
            return
        self._on_open_strain_window()

    def _on_run_warning(
        self, title: str, text: str, informative: str, details: str,
    ) -> None:
        """Show a tracking-quality warning that blocks nothing.

        Modeless ``show()``: a ``QMessageBox`` is modal by default, and
        ``open()`` would still lock the window it sits over; ``exec()``
        would also run a nested event loop, so a run that finishes
        unattended (or headless) would hang on it. The box sits over the
        strain window when that is showing (it opens on DONE), replaces the
        previous run's box, and deletes itself once closed. The texts
        arrive translated.
        """
        self._close_run_warning()
        box = QMessageBox(_warning_parent(self._strain_window, self))
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        box.setInformativeText(informative)
        box.setDetailedText(details)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        box.setWindowModality(Qt.WindowModality.NonModal)
        self._run_warning_box = box
        box.show()

    def _close_run_warning(self) -> None:
        """Close the tracking warning still open, if any (it deletes
        itself; it is already gone if the user or its parent closed it)."""
        box, self._run_warning_box = self._run_warning_box, None
        if box is not None and shiboken6.isValid(box):
            box.close()


def _warning_parent(
    strain_window: QWidget | None, main_window: QWidget,
) -> QWidget:
    """The window a run warning belongs over: the strain window when it is
    showing (the user is looking there), else the main window."""
    if strain_window is not None and strain_window.isVisible():
        return strain_window
    return main_window


def user_data_dir() -> Path:
    """Per-user writable directory for logs and other app-owned state.

    Deliberately not next to the executable: a onedir bundle is routinely
    unzipped into Program Files, onto a network share, or anywhere else the
    user has no write permission -- and a macOS app bundle must not change
    once signed. Each platform's own place, never a bare folder in the home
    directory: %LOCALAPPDATA% on Windows, Application Support on macOS,
    $XDG_DATA_HOME on Linux. packaging/rthook_pyaldic.py mirrors this.
    """
    home = Path(os.path.expanduser("~"))
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or home)
    elif sys.platform == "darwin":
        base = home / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or home / ".local" / "share")
    return base / "RAFTcorr"



# Path of the log file, once one has been opened. None when not frozen.
_LOG_FILE: Path | None = None


def _configure_logging() -> None:
    """Send the logging stream to a file when there is no console.

    A windowed PyInstaller build has ``sys.stdout is None``, which makes every
    ``print`` a silent no-op and leaves ``logging``'s last-resort handler with
    nowhere to write. Without a file the application has no way at all to
    report a startup failure -- which is precisely when it fails.
    """
    global _LOG_FILE
    if not getattr(sys, "frozen", False):
        return
    try:
        log_dir = user_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "RAFTcorr.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            handlers=[logging.FileHandler(log_file, encoding="utf-8")],
        )
    except OSError:
        return  # a read-only profile is not a reason to refuse to start
    _LOG_FILE = log_file


def _report_crash_dialog(exc_type, exc_value, tb_str: str) -> None:
    """Show the traceback in a dialog -- a windowed build has no other channel."""
    if QApplication.instance() is None:
        return
    from raftcorr.i18n import tr_args

    box = QMessageBox(
        QMessageBox.Icon.Critical,
        QCoreApplication.translate("Application", "RAFTcorr has hit an error"),
        QCoreApplication.translate(
            "Application",
            "An unexpected error occurred. The application may not behave "
            "correctly from here on, so saving your project and restarting "
            "is recommended.",
        ),
    )
    if _LOG_FILE is not None:
        box.setInformativeText(
            tr_args(
                QCoreApplication.translate(
                    "Application", "Details were written to %1"
                ),
                str(_LOG_FILE),
            )
        )
    box.setDetailedText(f"{exc_type.__name__}: {exc_value}" + chr(10) * 2 + tb_str)
    box.exec()


def _global_exception_hook(exc_type, exc_value, exc_tb):
    """Catch unhandled exceptions so the GUI doesn't silently crash."""
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(f"\n{'='*60}", flush=True)
    print("UNHANDLED EXCEPTION — this would normally crash the GUI:", flush=True)
    print(tb_str, flush=True)
    print(f"{'='*60}\n", flush=True)

    # The log file is the only channel that survives a windowed build.
    logging.getLogger("raftcorr").critical("Unhandled exception: %s", tb_str)

    # Also try to log to GUI console if available
    try:
        state = AppState.instance()
        state.log_message.emit(f"CRASH: {exc_type.__name__}: {exc_value}", "error")
        state.log_message.emit(tb_str, "error")
    except Exception:
        pass

    # ...and a dialog, for failures that happen before the console widget
    # exists, or that leave it unreachable.
    try:
        _report_crash_dialog(exc_type, exc_value, tb_str)
    except Exception:
        pass


def main() -> None:
    """Launch the GUI application."""
    import multiprocessing

    multiprocessing.freeze_support()
    _configure_logging()
    sys.excepthook = _global_exception_hook

    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("RAFTcorr")
    app.setApplicationName("RAFTcorr")
    app.setStyle("Fusion")  # required for QSS to work correctly
    theme_manager(app).set_theme(startup_theme(), persist=False)

    from raftcorr.gui.gc_guard import MainThreadGarbageCollector

    # Workers must never destroy Qt objects: the cyclic collector runs in this thread only.
    app._raftcorr_gc = MainThreadGarbageCollector(parent=app)  # type: ignore[attr-defined]

    from raftcorr.gui.support.matplotlib_fonts import configure_matplotlib_fonts
    from raftcorr.i18n import LanguageManager

    configure_matplotlib_fonts()
    lang_mgr = LanguageManager(app)
    lang_mgr.load(LanguageManager.resolve_language())
    app._pyaldic_lang_mgr = lang_mgr  # type: ignore[attr-defined]

    window = MainWindow()
    window.show()

    target = _path_from_argv(sys.argv)
    if target is not None:
        from PySide6.QtCore import QTimer

        if target.suffix.lower() == ".raftcorr":
            QTimer.singleShot(0, lambda: window.open_project_path(str(target)))
        else:
            QTimer.singleShot(0, lambda: window._image_ctrl.load_folder(str(target)))
    sys.exit(app.exec())


def _path_from_argv(argv: list[str]) -> Path | None:
    """The first existing ``.raftcorr`` project or image folder on the command line."""
    for arg in argv[1:]:
        p = Path(arg)
        if p.suffix.lower() == ".raftcorr" and p.is_file():
            return p
        if p.is_dir():
            return p
    return None


if __name__ == "__main__":
    main()
