"""Top-level strain post-processing window (pyALDIC's, on RAFTcorr results).

Independent ``QMainWindow`` that consumes the displacement series of the main
window (``state.results``, a :class:`raftcorr.results.DisplacementSeries`),
runs :class:`StrainController` on demand (RAFTcorr's virtual strain gauge,
:func:`raftcorr.post.strain.compute_strain`), publishes the
:class:`raftcorr.results.StrainSeries` with ``state.set_strain_results`` and
renders the fields through ``state.provider`` (a
:class:`raftcorr.viz.fields.FieldProvider`) with a *private*
:class:`FieldRenderer` cache.

Decoupling contracts (as in pyALDIC):

* Owns its own ``_strain_current_frame`` -- never mutates
  ``state.current_frame``.
* Owns its own colour/geometry controls -- never reads or writes
  ``state.colormap`` / ``state.color_min`` / ``state.color_max`` /
  ``state.display_field``. Colour settings are remembered per field in the
  state's per-field store (shared with the main window on purpose).
* Never modifies the displacement series; strain goes through
  :class:`StrainController` only.

Field routing:

* ``disp_u``, ``disp_v``, ``disp_magnitude``, ``velocity`` (speed) -- from the
  displacement series; available before Compute Strain.
* ``strain_*`` -- require a completed compute.

Frames: result frame ``k`` is image ``k`` (frame 0 is the reference); the
navigator shows them 1-based.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QEvent, QTimer, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from raftcorr.config import StrainConfig
from raftcorr.gui.app_state import AppState
from raftcorr.gui.controllers.image_controller import ImageController
from raftcorr.gui.controllers.strain_controller import (
    StrainController,
    StrainWorker,
    settings_summary,
)
from raftcorr.gui.panels.strain_canvas import (
    FieldImage,
    FieldRenderer,
    ShownField,
    StrainCanvas,
    percentile_range,
)
from raftcorr.gui.theme_manager import set_icon, themed
from raftcorr.gui.widgets.collapsible_section import CollapsibleSection
from raftcorr.gui.widgets.colorbar_overlay import ColorbarOverlay
from raftcorr.gui.widgets.console_log import ConsoleLog
from raftcorr.gui.widgets.physical_units_widget import PhysicalUnitsWidget
from raftcorr.gui.widgets.strain_field_selector import (
    DISP_FIELD_NAMES,
    StrainFieldSelector,
    field_colorbar_label,
    field_key,
)
from raftcorr.gui.widgets.strain_navigator import StrainNavigator
from raftcorr.gui.widgets.strain_param_panel import StrainParamPanel
from raftcorr.gui.widgets.strain_viz_panel import StrainVizPanel
from raftcorr.i18n import tr_args
from raftcorr.viz.fields import ALL_FIELDS as PROVIDER_FIELDS

try:
    from raftcorr.gui.icons import icon_maximize, icon_zoom_in, icon_zoom_out
    _HAS_ICONS = True
except ImportError:  # pragma: no cover
    _HAS_ICONS = False

logger = logging.getLogger(__name__)

__all__ = ["DISP_FIELD_NAMES", "FieldImage", "StrainWindow", "initial_window_size"]


def initial_window_size(avail_width: int, avail_height: int) -> tuple[int, int]:
    """Clamp the preferred 1280x800 default to the usable screen area.

    A fixed 800 px height overflows small laptop screens (1366x768 /
    1280x800 minus menu bar, dock/taskbar and title bar), and on macOS a
    window whose bottom edge opens off-screen cannot be shrunk because
    the resize handle is unreachable and the title bar cannot be dragged
    above the menu bar. The margins keep the whole frame, including the
    resize edges, on screen.
    """
    return (
        max(640, min(1280, avail_width - 40)),
        max(480, min(800, avail_height - 80)),
    )


class StrainWindow(QMainWindow):
    """Independent strain post-processing window."""

    def __init__(
        self,
        state: AppState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Strain Post-Processing"))
        # Use the shared app icon so the Strain window is recognisable
        # in the taskbar / Alt-Tab alongside the main window.
        from raftcorr.gui.icons import icon_app
        self.setWindowIcon(icon_app())
        # Match the main window's OS title bar so the two windows share one
        # visual frame.
        from raftcorr.gui.theme_manager import apply_title_bar
        apply_title_bar(self)
        # Default size: 1280x800 clamped to the current screen's usable area
        # so the bottom edge (frame slider, resize handle) never opens
        # off-screen. The right column is wrapped in a QScrollArea so even
        # smaller screens still reach every control.
        screen = self.screen() or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()
        self.resize(*initial_window_size(avail.width(), avail.height()))

        self._state = state
        self._strain_ctrl = StrainController(state)
        self._renderer = FieldRenderer()   # PRIVATE -- isolated from MainWindow
        self._image_ctrl = ImageController(state)
        self._strain_current_frame: int = 0
        # Cache the last auto-computed range so Auto→Manual switch is seamless
        self._last_rendered_vmin: float = 0.0
        self._last_rendered_vmax: float = 1.0
        # Guard flag to prevent save-back loop when we're loading AppState → StrainVizPanel
        self._loading_field_state: bool = False
        # What the canvas shows, for the cursor read-out
        self._shown: ShownField | None = None
        # Background shape the view was last fitted to
        self._fitted_shape: tuple[int, int] | None = None
        # What the background shows (sequence, shape, frame, settings)
        self._bg_key: tuple | None = None
        self._strain_worker: StrainWorker | None = None
        self._state_connected = False

        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        root.addLayout(self._build_left_pane(), 1)
        root.addWidget(self._build_right_pane(), 0)

        # The field view becomes the first tab; probe analysis is the second.
        self._tabs = QTabWidget(self)
        self._tabs.addTab(central, self.tr("Strain Field"))
        self._analysis_tab = self._build_analysis_tab()
        self.setCentralWidget(self._tabs)
        # The field view renders only while it is the page on show; a render
        # asked for meanwhile is made once, on return.
        self._render_pending = False
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # Track external pipeline runs and shared display settings.
        self._connect_state()

        # Load default color state for the initially selected field.
        self._load_field_state_into_panel()

        self._sync_slider_range()
        self._render_current()
        # The window can be opened with results already present (session
        # reload, or auto-open when a run finishes) -- results_changed fired
        # before this window existed, so set the Export enabled state now.
        self._refresh_export_enabled()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_left_pane(self) -> QVBoxLayout:
        """Zoom toolbar + canvas (with colorbar overlay) + frame navigator."""
        left = QVBoxLayout()
        left.setSpacing(0)

        # --- Zoom toolbar (matches main window toolbar style) ---
        zoom_bar = QWidget()
        zoom_bar.setFixedHeight(36)
        themed(zoom_bar,
            "background: {BG_PANEL}; "
            "border-bottom: 1px solid {BORDER};"
        )
        zoom_layout = QHBoxLayout(zoom_bar)
        zoom_layout.setContentsMargins(8, 2, 8, 2)
        zoom_layout.setSpacing(4)
        btn_fit = QPushButton(self.tr("Fit"))
        btn_fit.setToolTip(self.tr("Fit image to viewport"))
        btn_fit.setFixedWidth(60)
        if _HAS_ICONS:
            set_icon(btn_fit, icon_maximize)
            btn_fit.setText("")
            btn_fit.setFixedWidth(28)
        btn_100 = QPushButton(self.tr("100%"))
        btn_100.setToolTip(self.tr("Zoom to 100% (1:1)"))
        btn_100.setFixedWidth(60)
        btn_zin = QPushButton("+")
        btn_zin.setToolTip(self.tr("Zoom in"))
        btn_zin.setFixedWidth(28)
        if _HAS_ICONS:
            set_icon(btn_zin, icon_zoom_in)
            btn_zin.setText("")
        btn_zout = QPushButton(self.tr("–"))
        btn_zout.setToolTip(self.tr("Zoom out"))
        btn_zout.setFixedWidth(28)
        if _HAS_ICONS:
            set_icon(btn_zout, icon_zoom_out)
            btn_zout.setText("")
        for b in (btn_fit, btn_100, btn_zin, btn_zout):
            zoom_layout.addWidget(b)
        # Why no field is drawn (strain not computed, frame not tracked ...)
        self._hint_label = QLabel("")
        themed(self._hint_label,
            "color: {WARNING}; font-size: 11px; font-style: italic; "
            "background: transparent; border: none; padding-left: 8px;"
        )
        zoom_layout.addWidget(self._hint_label)
        zoom_layout.addStretch()
        # Cursor read-out: position and the value of the field under the cursor.
        self._readout = QLabel("")
        themed(self._readout,
            "color: {TEXT_SECONDARY}; font-size: 11px; "
            "background: transparent; border: none;"
        )
        zoom_layout.addWidget(self._readout)
        left.addWidget(zoom_bar)

        self._canvas = StrainCanvas()
        left.addWidget(self._canvas, 1)
        # Colorbar overlaid on the canvas viewport (same pattern as main window)
        self._colorbar = ColorbarOverlay(self._canvas.viewport())
        self._canvas.viewport().installEventFilter(self)
        self._canvas.scene_hover.connect(self._on_scene_hover)
        self._canvas.hover_left.connect(lambda: self._readout.setText(""))

        btn_fit.clicked.connect(self._canvas.fit_to_view)
        btn_100.clicked.connect(self._canvas.zoom_to_100)
        btn_zin.clicked.connect(self._canvas.zoom_in)
        btn_zout.clicked.connect(self._canvas.zoom_out)

        self._frame_nav = StrainNavigator()
        self._frame_nav.frame_changed.connect(self._on_frame_nav_changed)
        left.addWidget(self._frame_nav)
        return left

    def _build_right_pane(self) -> QScrollArea:
        """Scrollable column of collapsible sections and the action buttons."""
        right_container = QWidget()
        # Ignored horizontal sizePolicy lets the QScrollArea below constrain
        # width to its viewport even when a collapsed section reports an
        # unusual sizeHint -- same pattern as left_sidebar.
        right_container.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred,
        )
        right = QVBoxLayout(right_container)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(6)

        # Strain parameters (first thing users tweak -- expanded).
        self._params_section = CollapsibleSection(
            self.tr("STRAIN PARAMETERS"), expanded=True,
        )
        self._param_panel = StrainParamPanel(self._state.strain_config)
        self._param_panel.params_dirty.connect(self._on_params_dirty)
        self._param_panel.config_changed.connect(self._on_config_edited)
        self._params_section.add_widget(self._param_panel)
        right.addWidget(self._params_section)

        # Less used fit options, folded like the main window's ADVANCED section.
        self._advanced_section = CollapsibleSection(self.tr("ADVANCED"), expanded=False)
        self._advanced_section.add_widget(self._param_panel.advanced_widget())
        right.addWidget(self._advanced_section)

        # Action buttons stay OUTSIDE collapsible sections so users can
        # trigger Compute / Export even when STRAIN PARAMETERS is folded.
        self._compute_btn = QPushButton(self.tr("Compute Strain"))
        self._compute_btn.setProperty("class", "btn-primary")
        self._compute_btn.setFixedHeight(40)
        self._compute_btn.clicked.connect(self._on_compute_clicked)
        right.addWidget(self._compute_btn)

        # Cancel button: hidden until a compute is running (mirrors the main
        # window's cancel). Aborts at the next frame boundary and keeps the
        # previous strain result.
        self._cancel_btn = QPushButton(self.tr("Cancel"))
        self._cancel_btn.setProperty("class", "btn-danger")
        self._cancel_btn.setFixedHeight(30)
        self._cancel_btn.setToolTip(
            self.tr("Cancel the running strain computation. "
                    "The previous strain result is kept.")
        )
        from raftcorr.gui.icons import icon_stop
        set_icon(self._cancel_btn, icon_stop)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        right.addWidget(self._cancel_btn)

        self._export_strain_btn = QPushButton(self.tr("Export Results"))
        self._export_strain_btn.setFixedHeight(30)
        self._export_strain_btn.setToolTip(self.tr(
            "Export displacement and strain results to NPZ / MAT / CSV / PNG"
        ))
        self._export_strain_btn.setEnabled(False)
        self._export_strain_btn.clicked.connect(self._on_export_strain)
        right.addWidget(self._export_strain_btn)

        # Progress bar (hidden until compute starts)
        self._strain_progress = QProgressBar()
        self._strain_progress.setRange(0, 1000)
        self._strain_progress.setValue(0)
        self._strain_progress.setTextVisible(False)
        self._strain_progress.setFixedHeight(8)
        self._strain_progress.setVisible(False)
        right.addWidget(self._strain_progress)

        self._strain_progress_label = QLabel("")
        themed(self._strain_progress_label,
            "color: {TEXT_SECONDARY}; font-size: 11px;"
        )
        self._strain_progress_label.setVisible(False)
        right.addWidget(self._strain_progress_label)

        self._stale_label = QLabel("")
        self._stale_label.setWordWrap(True)
        themed(self._stale_label,
            "color: {WARNING}; font-size: 11px; font-style: italic;"
        )
        right.addWidget(self._stale_label)

        # Field selector (switched often -- expanded).
        self._field_section = CollapsibleSection(self.tr("FIELD"), expanded=True)
        self._field_selector = StrainFieldSelector()
        self._field_selector.field_changed.connect(self._on_field_changed)
        self._field_section.add_widget(self._field_selector)
        right.addWidget(self._field_section)

        # Visualization controls (touched on every render -- expanded).
        self._viz_section = CollapsibleSection(self.tr("VISUALIZATION"), expanded=True)
        self._viz_panel = StrainVizPanel()
        # Start from the main window's choice, then stay independent -- the
        # same contract as this panel's colormap, opacity and geometry.
        self._viz_panel.set_hidden_bg_color(self._state.hidden_bg_color)
        self._viz_panel.set_hide_untracked(self._state.hide_untracked)
        self._viz_panel.viz_changed.connect(self._on_viz_panel_changed)
        self._viz_panel.auto_disabled.connect(self._on_auto_range_disabled)
        self._viz_section.add_widget(self._viz_panel)
        right.addWidget(self._viz_section)

        # Physical units (rarely changed mid-session -- collapsed).
        self._units_section = CollapsibleSection(self.tr("PHYSICAL UNITS"), expanded=False)
        self._physical_units = PhysicalUnitsWidget()
        self._units_section.add_widget(self._physical_units)
        right.addWidget(self._units_section)

        # Console / log (diagnostics -- collapsed by default).
        self._log_section = CollapsibleSection(self.tr("LOG"), expanded=False)
        self._console = ConsoleLog()
        self._log_section.add_widget(self._console)
        right.addWidget(self._log_section)

        right.addStretch(1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setWidget(right_container)
        # 320 content + ~20 for the scrollbar gutter. Fixed so the canvas
        # keeps its proportional share of the window.
        right_scroll.setFixedWidth(340)
        return right_scroll

    def _build_analysis_tab(self) -> QWidget | None:
        """The probe Analysis tab, or a placeholder when it cannot be built.

        The tab is additive: whatever goes wrong while importing or wiring it
        must not take the field view down with it.
        """
        tab = None
        try:
            from raftcorr.gui.panels.analysis import AnalysisTab

            tab = AnalysisTab(self._state, self)
            # One frame across both tabs; the chart's cursor and a click on
            # it move the field view too.
            tab.frame_requested.connect(self.set_strain_frame)
            tab.set_frame(self._strain_current_frame)
            tab.set_default_field(self._field_selector.current_field())
            tab.set_parameters_provider(self._strain_parameters)
            tab.set_field_renderer(self.reference_field_image)
        except Exception:
            logger.exception("Strain window: the Analysis tab is not available")
            if tab is not None:
                tab.deleteLater()
            placeholder = QLabel(self.tr("Analysis is not available"))
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            themed(placeholder, "color: {TEXT_SECONDARY}; font-size: 13px;")
            self._tabs.addTab(placeholder, self.tr("Analysis"))
            self._log(self.tr("Analysis is not available (see the application log)."), "warn")
            return None
        self._tabs.addTab(tab, self.tr("Analysis"))
        return tab

    def _analysis_call(self, method: str, *args: object) -> None:
        """Forward to the Analysis tab; its failure never breaks this view."""
        tab = self._analysis_tab
        if tab is None:
            return
        try:
            getattr(tab, method)(*args)
        except Exception:
            logger.exception("Strain window: AnalysisTab.%s failed", method)

    def _state_connections(self) -> list[tuple[object, Callable]]:
        st = self._state
        return [
            (st.results_changed, self._on_state_results_changed),
            (st.strain_changed, self._on_state_strain_changed),
            (st.display_changed, self._on_state_display_changed),
            (st.physical_units_changed, self._renderer.clear),
            (st.params_changed, self._on_state_params_changed),
        ]

    def _connect_state(self) -> None:
        """Connect to the state's signals once (closeEvent disconnects, so a
        closed window kept for re-opening does no work in the background)."""
        if self._state_connected:
            return
        for sig, slot in self._state_connections():
            sig.connect(slot)
        self._state_connected = True

    def _disconnect_state(self) -> None:
        if not self._state_connected:
            return
        for sig, slot in self._state_connections():
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        self._state_connected = False

    # ------------------------------------------------------------------
    # Public accessors (used by tests + integration)
    # ------------------------------------------------------------------

    def _refresh_export_enabled(self) -> None:
        """Export is available whenever results exist (displacement is
        exportable even before strain is computed)."""
        self._export_strain_btn.setEnabled(self._state.results is not None)

    def strain_current_frame(self) -> int:
        return self._strain_current_frame

    def set_strain_frame(self, idx: int) -> None:
        n = self._strain_frame_count()
        clamped = max(0, min(int(idx), max(0, n - 1)))
        if clamped == self._strain_current_frame:
            return
        self._strain_current_frame = clamped
        self._frame_nav.set_state(n, clamped)
        self._render_current()
        self._analysis_call("set_frame", clamped)

    def current_field(self) -> str:
        return self._field_selector.current_field()

    def set_current_field(self, name: str) -> None:
        self._field_selector.set_current_field(name)

    def is_stale(self) -> bool:
        return self._param_panel.is_dirty()

    def is_computing(self) -> bool:
        worker = self._strain_worker
        return worker is not None and worker.isRunning()

    def param_panel(self) -> StrainParamPanel:
        return self._param_panel

    def viz_panel(self) -> StrainVizPanel:
        return self._viz_panel

    def canvas(self) -> StrainCanvas:
        return self._canvas

    def shown_field(self) -> ShownField | None:
        """The numbers behind the drawn field (None when nothing is drawn)."""
        return self._shown

    def _strain_parameters(self) -> dict[str, str]:
        """The settings of the strain on show, for an export header.

        Read from the strain series itself (it records its settings), so they
        are right after a project is reopened too. Empty without strain.
        """
        strain = self._state.strain_results
        return {} if strain is None else settings_summary(strain.settings)

    def trigger_compute(self) -> None:
        """Synchronous compute -- used by tests (blocking, no progress bar)."""
        if self._state.results is None:
            return
        config = self._param_panel.get_config()
        try:
            self._strain_ctrl.compute_and_store(config)
        except Exception as exc:
            logger.exception("strain computation failed")
            self._log(
                tr_args(self.tr("Strain compute failed: %1: %2"), type(exc).__name__, str(exc)),
                "error",
            )
            return
        self._param_panel.mark_clean()
        self._stale_label.setText("")
        self._log(self.tr("Strain computation complete."), "success")
        self._export_strain_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Compute slots
    # ------------------------------------------------------------------

    def _on_compute_clicked(self) -> None:
        series = self._state.results
        if series is None:
            self._log(
                self.tr("Strain window: no displacement results to post-process."),
                "warn",
            )
            return
        # Guard against double-click while running
        if self.is_computing():
            return
        config = self._param_panel.get_config()
        self._state.strain_config = config
        self._compute_btn.setEnabled(False)
        self._strain_progress.setValue(0)
        self._strain_progress.setVisible(True)
        self._strain_progress_label.setText(self.tr("Starting…"))
        self._strain_progress_label.setVisible(True)
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setVisible(True)

        worker = StrainWorker(self._strain_ctrl, config, series)
        worker.progress.connect(self._on_strain_progress)
        worker.completed.connect(self._on_strain_finished)
        worker.error.connect(self._on_strain_error)
        worker.cancelled.connect(self._on_strain_cancelled)
        self._strain_worker = worker
        worker.start()

    def _on_cancel_clicked(self) -> None:
        worker = self._strain_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            self._cancel_btn.setEnabled(False)
            self._strain_progress_label.setText(self.tr("Canceling…"))

    def _on_strain_progress(self, fraction: float, message: str) -> None:
        self._strain_progress.setValue(int(fraction * 1000))
        self._strain_progress_label.setText(message)

    def _on_strain_finished(self, strain) -> None:
        worker = self._strain_worker
        self._cancel_btn.setVisible(False)
        self._compute_btn.setEnabled(True)
        if worker is None or not self._strain_ctrl.store(worker.series, worker.config, strain):
            # A new run replaced the displacement meanwhile: this strain
            # belongs to displacements no longer shown.
            self._strain_progress.setVisible(False)
            self._strain_progress_label.setVisible(False)
            self._log(self.tr("The displacement results changed during the strain "
                              "computation; its result was discarded."), "warn")
            return
        self._strain_progress.setValue(1000)
        self._strain_progress_label.setText(self.tr("Complete"))
        self._param_panel.mark_clean()
        self._stale_label.setText("")
        self._log(self.tr("Strain computation complete."), "success")
        self._export_strain_btn.setEnabled(True)
        QTimer.singleShot(2000, self, self._hide_progress)

    def _hide_progress(self) -> None:
        if self.is_computing():
            return
        self._strain_progress.setVisible(False)
        self._strain_progress_label.setVisible(False)

    def _on_strain_cancelled(self) -> None:
        # User aborted mid-run: discard the partial recompute and keep the
        # previous strain result untouched.
        self._cancel_btn.setVisible(False)
        self._strain_progress.setVisible(False)
        self._strain_progress_label.setVisible(False)
        self._compute_btn.setEnabled(True)
        self._log(self.tr("Strain computation canceled."), "warn")

    def _on_strain_error(self, message: str) -> None:
        self._cancel_btn.setVisible(False)
        self._strain_progress.setVisible(False)
        self._strain_progress_label.setVisible(False)
        self._compute_btn.setEnabled(True)
        self._log(tr_args(self.tr("Strain compute failed: %1"), message), "error")
        # Surface the failure in a modal dialog. The LOG panel is collapsed by
        # default, so a log-only error was effectively silent -- the user could
        # unknowingly keep the previous result and export a stale field.
        QMessageBox.warning(self, self.tr("Strain Computation Failed"), message)

    def _on_export_strain(self) -> None:
        """Open the export dialog pre-filled with this window's viz settings."""
        if self._state.results is None:
            return
        try:
            from raftcorr.gui.dialogs.export_dialog import ExportDialog, VizExportHint

            viz = self._viz_panel.get_state()
            wanted = dict(
                colormap=str(viz["colormap"]),
                auto_range=bool(viz["use_percentile"]),
                vmin=float(viz["vmin"]),
                vmax=float(viz["vmax"]),
                show_deformed=bool(viz["show_deformed"]),
                show_background=bool(viz["show_background"]),
                hidden_bg_color=str(viz["hidden_bg_color"]),
                hide_untracked=bool(viz["hide_untracked"]),
                fill_trimmed_edges=bool(viz["fill_trimmed_edges"]),
                overlay_alpha=float(viz["alpha"]),
                use_physical_units=self._state.use_physical_units,
                pixel_size=self._state.pixel_size,
                pixel_unit=self._state.pixel_unit,
                frame_rate=self._state.frame_rate,
            )
            known = getattr(VizExportHint, "__dataclass_fields__", wanted)
            hint = VizExportHint(**{k: v for k, v in wanted.items() if k in known})
            dlg = ExportDialog(
                self._state.results,
                self._state.image_folder,
                hint,
                image_files=self._state.image_files,
                roi_mask=self._state.roi_mask,
                per_frame_rois=self._state.per_frame_rois or None,
                parent=self,
            )
        except Exception as exc:
            logger.exception("Strain window: the export dialog could not be opened")
            message = f"{type(exc).__name__}: {exc}"
            self._log(tr_args(self.tr("Export is not available: %1"), message), "error")
            QMessageBox.warning(self, self.tr("Export Results"),
                                tr_args(self.tr("Export is not available: %1"), message))
            return
        # All exports happen inside the dialog; exec() blocks until user clicks Close.
        dlg.exec()
        # Delete it: a dialog given a parent outlives the call that made it.
        dlg.deleteLater()

    # ------------------------------------------------------------------
    # Parameter / field / display slots
    # ------------------------------------------------------------------

    def _on_params_dirty(self) -> None:
        self._stale_label.setText(self.tr("⚠ Params changed -- click Compute Strain"))

    def _on_config_edited(self, config: StrainConfig) -> None:
        """Panel edits are the session's strain settings (saved with a project)."""
        self._state.strain_config = config

    def _on_state_params_changed(self) -> None:
        """A project was opened (or settings set elsewhere): show its strain settings."""
        cfg = self._state.strain_config
        if isinstance(cfg, StrainConfig) and cfg != self._param_panel.get_config():
            self._param_panel.set_config(cfg)

    def _on_field_changed(self, name: str) -> None:
        """Switch the active display field: restore its remembered color state first."""
        self._analysis_call("set_default_field", name)
        self._load_field_state_into_panel()
        self._render_current()

    def _on_viz_panel_changed(self) -> None:
        """User changed a setting in StrainVizPanel -- save to AppState, then re-render.

        Saves the current StrainVizPanel state to AppState's per-field store so
        that switching away and back restores the user's settings. The guard flag
        prevents the resulting display_changed signal from causing a second render.
        """
        self._loading_field_state = True
        try:
            self._save_panel_state_to_app()
        finally:
            self._loading_field_state = False
        self._render_current()

    def _on_state_display_changed(self) -> None:
        """AppState display settings changed (possibly from the main window).

        If we triggered this ourselves (via _save_panel_state_to_app), skip the
        reload to avoid a redundant render. Otherwise reload the current field's
        color state so changes made in the main window are reflected here too.
        """
        if self._loading_field_state:
            return
        self._load_field_state_into_panel()
        self._render_current()

    def _load_field_state_into_panel(self) -> None:
        """Read AppState's stored color state for the current field and push it
        into StrainVizPanel controls (no signals fired, so no save-back loop)."""
        fs = self._state.get_field_state(self._field_selector.current_field())
        self._viz_panel.load_field_state(auto=fs.auto, vmin=fs.vmin, vmax=fs.vmax, colormap=fs.colormap)

    def _save_panel_state_to_app(self) -> None:
        """Persist the current StrainVizPanel state into AppState's per-field store.

        Also emits display_changed so the main window reacts if it is currently
        showing the same field (disp_u / disp_v share their state).
        """
        viz = self._viz_panel.get_state()
        fs = self._state.get_field_state(self._field_selector.current_field())
        fs.colormap = str(viz["colormap"])
        fs.auto = bool(viz["use_percentile"])
        if not fs.auto:
            fs.vmin = float(viz["vmin"])
            fs.vmax = float(viz["vmax"])
        self._state.display_changed.emit()

    def _on_auto_range_disabled(self) -> None:
        """User switched to manual range: populate spinboxes with the last rendered range."""
        self._viz_panel.set_range(self._last_rendered_vmin, self._last_rendered_vmax)

    def _on_state_results_changed(self) -> None:
        self._renderer.clear()
        self._bg_key = None  # new images or results: load the background again
        self._frame_nav.stop_playback()
        self._sync_slider_range()
        self._render_current()
        # Export follows the presence of results (displacement and/or strain).
        self._refresh_export_enabled()

    def _on_state_strain_changed(self) -> None:
        self._renderer.clear(strain_only=True)
        self._render_current()

    def _on_frame_nav_changed(self, value: int) -> None:
        self._strain_current_frame = int(value)
        self._render_current()
        self._analysis_call("set_frame", self._strain_current_frame)

    def _on_tab_changed(self, _index: int) -> None:
        if self._render_pending and not self._analysis_shown():
            self._render_pending = False
            self._render_current()

    def _analysis_shown(self) -> bool:
        tabs = getattr(self, "_tabs", None)
        return (tabs is not None and self._analysis_tab is not None
                and tabs.currentWidget() is self._analysis_tab)

    # ------------------------------------------------------------------
    # Frames and background
    # ------------------------------------------------------------------

    def _strain_frame_count(self) -> int:
        """Frames of the results (reference included), else of the images."""
        result = self._state.results
        if result is not None:
            return int(result.n_frames)
        return len(self._state.image_files)

    def _sync_slider_range(self) -> None:
        n = self._strain_frame_count()
        max_idx = max(0, n - 1)
        if self._strain_current_frame > max_idx:
            self._strain_current_frame = max_idx
        self._frame_nav.set_state(n, self._strain_current_frame)
        self._analysis_call("set_frame", self._strain_current_frame)

    def _image_shape(self) -> tuple[int, int] | None:
        seq = self._state.sequence
        if seq is not None:
            return tuple(int(v) for v in seq.shape)  # type: ignore[return-value]
        result = self._state.results
        if result is not None:
            return result.grid.image_shape
        return None

    def _load_background(self, frame: int, deformed: bool, show_background: bool, hidden_bg: str) -> None:
        """The image behind the field: the current frame on the deformed view,
        the reference otherwise; a solid fill when hidden or not loaded."""
        idx = frame if deformed and frame >= 1 else 0
        key = (id(self._state.sequence), self._image_shape(), idx, show_background, hidden_bg)
        if key == self._bg_key:
            return
        self._bg_key = key
        shown = False
        if show_background and self._state.image_files and self._state.sequence is not None:
            try:
                self._canvas.set_image(self._image_ctrl.read_image_rgb(idx))
                shown = True
            except (IndexError, FileNotFoundError, ValueError, OSError):
                logger.debug("strain window: image %d not readable", idx, exc_info=True)
        if not shown:
            shape = self._image_shape()
            if shape is not None:
                self._canvas.set_blank(shape[0], shape[1], hidden_bg if not show_background else "white")
        self._fit_on_new_shape()

    def _fit_on_new_shape(self) -> None:
        rect = self._canvas.scene().sceneRect()
        shape = (int(rect.height()), int(rect.width()))
        if shape != self._fitted_shape and not rect.isEmpty() and self.isVisible():
            self._fitted_shape = shape
            self._canvas.fit_to_view()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_current(self) -> None:
        if self._analysis_shown():
            self._render_pending = True
            return
        try:
            self._render_field_view()
        except Exception as exc:
            logger.exception("strain window: render failed")
            self._log(f"Render error: {type(exc).__name__}: {exc}", "error")
            self._clear_field()

    def _render_field_view(self) -> None:
        viz = self._viz_panel.get_state()
        frame = self._strain_current_frame
        field_name = self._field_selector.current_field()
        deformed = bool(viz["show_deformed"])
        self._load_background(frame, deformed, bool(viz["show_background"]), str(viz["hidden_bg_color"]))
        if self._state.results is None or self._state.provider is None:
            self._clear_field()
            self._hint_label.setText(self.tr("No displacement results."))
            return
        out = self._field_image(
            field_name, frame,
            deformed=deformed,
            cmap=str(viz["colormap"]),
            alpha=float(viz["alpha"]),
            hide_untracked=bool(viz["hide_untracked"]),
            fill_trimmed=bool(viz["fill_trimmed_edges"]),
            range_of=self._resolve_range,
        )
        self._update_trim_readout(frame)
        if out is None:
            self._clear_field()
            self._hint_label.setText(self._missing_reason(field_name, frame))
            return
        image, shown = out
        self._hint_label.setText("")
        self._shown = shown
        # Cache so that Auto→Manual switch can populate spinboxes with the
        # exact same range that was just rendered.
        self._last_rendered_vmin = image.vmin
        self._last_rendered_vmax = image.vmax
        self._canvas.show_field(image)
        vp = self._canvas.viewport()
        self._colorbar.setGeometry(0, 0, vp.width(), vp.height())
        self._colorbar.update_params(image.cmap, image.vmin, image.vmax, image.label)
        self._colorbar.setVisible(True)

    def _clear_field(self) -> None:
        self._canvas.clear_overlay()
        self._colorbar.setVisible(False)
        self._shown = None

    def _missing_reason(self, field_name: str, frame: int) -> str:
        """Why *field_name* has nothing to draw at *frame* (shown on the toolbar)."""
        result = self._state.results
        if result is None:
            return self.tr("No displacement results.")
        if not 0 <= frame < result.n_frames or not bool(result.done[frame]):
            return tr_args(self.tr("Frame %1 was not computed."), frame + 1)
        try:
            spec = PROVIDER_FIELDS[field_key(field_name)]
        except (KeyError, ValueError):
            return ""
        if spec.is_strain:
            strain = self._state.strain_results
            if strain is None:
                return self.tr("Click Compute Strain to show this field.")
            return tr_args(self.tr("No strain for frame %1: click Compute Strain."), frame + 1)
        if spec.kind == "velocity" and len(result.completed_frames) < 2:
            return self.tr("Velocity needs at least two computed frames.")
        return tr_args(self.tr("No valid values in frame %1."), frame + 1)

    def reference_field_image(self, field_name: str, frame: int) -> FieldImage | None:
        """*field_name* at *frame* in the reference configuration.

        What the Analysis tab lays under its probes: probe coordinates are
        frame-0 pixels, so the field is drawn where they are, whatever this
        tab's own view shows. Colormap and range are the field's own (the
        state the Strain Field tab and the main window share); opacity and
        "Hide untracked regions" are this window's.
        """
        if self._state.results is None:
            return None
        fs = self._state.get_field_state(field_name)
        viz = self._viz_panel.get_state()

        def range_of(values: NDArray[np.float32]) -> tuple[float, float]:
            if not fs.auto:
                return float(fs.vmin), float(fs.vmax)
            return percentile_range(values, str(fs.colormap))

        out = self._field_image(
            field_name, frame,
            deformed=False,
            cmap=str(fs.colormap),
            alpha=float(viz["alpha"]),
            hide_untracked=bool(viz["hide_untracked"]),
            fill_trimmed=bool(viz["fill_trimmed_edges"]),
            range_of=range_of,
        )
        return None if out is None else out[0]

    def _field_image(
        self,
        field_name: str,
        frame: int,
        *,
        deformed: bool,
        cmap: str,
        alpha: float,
        range_of: Callable[[NDArray[np.float32]], tuple[float, float]],
        hide_untracked: bool = True,
        fill_trimmed: bool = False,
    ) -> tuple[FieldImage, ShownField] | None:
        """Render *field_name* at *frame*: the one path both tabs draw through.

        *deformed* asks for the deformed configuration; the reference is used
        for frame 0. Physical units scale displacement and velocity only.
        """
        try:
            key = field_key(field_name)
        except ValueError:
            return None
        spec = PROVIDER_FIELDS[key]
        units = self._state.units()
        unit = units.unit(spec)
        st = self._state
        return self._renderer.render(
            st.provider, key, frame,
            deformed=deformed,
            cmap=cmap,
            alpha=alpha,
            range_of=range_of,
            factor=units.factor(spec),
            unit="" if unit == "-" else unit,
            label=field_colorbar_label(field_name, st.use_physical_units, st.pixel_unit, st.frame_rate),
            hide_untracked=hide_untracked,
            fill_trimmed=fill_trimmed,
        )

    def _update_trim_readout(self, frame: int) -> None:
        """'Trimmed: N points (M%)' of the shown frame's strain; hidden without a trim."""
        strain = self._state.strain_results
        trimmed = None if strain is None or not 0 <= frame < len(strain.done) else strain.trimmed_in(frame)
        if trimmed is None:
            self._param_panel.set_trim_readout(0, 0)
            return
        n_trimmed = int(trimmed.sum())
        n_kept = int(np.isfinite(strain.components["exx"][frame]).sum())
        self._param_panel.set_trim_readout(n_trimmed, n_trimmed + n_kept)

    def _resolve_range(self, values: NDArray[np.float32]) -> tuple[float, float]:
        """Compute (vmin, vmax) from the local viz panel state."""
        viz = self._viz_panel.get_state()
        if not viz["use_percentile"]:
            return float(viz["vmin"]), float(viz["vmax"])
        return percentile_range(values, str(viz["colormap"]))

    def _on_scene_hover(self, sx: float, sy: float) -> None:
        # scene pixel (c, r) covers [c, c + 1): image coordinates are scene - 0.5
        x, y = sx - 0.5, sy - 0.5
        text = f"x {x:.1f}  y {y:.1f} px"
        shown = self._shown
        if shown is not None:
            val = shown.value_at(x, y)
            if val is not None:
                text += f"   {shown.name} = {val:.5g}" + (f" {shown.unit}" if shown.unit else "")
        self._readout.setText(text)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, message: str, level: str = "info") -> None:
        """Append a message to the local console."""
        self._console.append_log(message, level)

    # ------------------------------------------------------------------
    # Qt events
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # Reconnect signals: they are disconnected in closeEvent so re-opening
        # after a new run does not receive stale connections.
        self._connect_state()
        # Clear the cache so a previous session's overlay never bleeds through
        # when the user re-opens after running new data.
        self._renderer.clear()
        self._bg_key = None
        self._sync_slider_range()
        # Re-render now that the viewport has its real size, and fit the
        # image once the layout has settled.
        self._render_current()
        self._fitted_shape = None
        QTimer.singleShot(0, self, self._fit_on_new_shape)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        """Update colorbar geometry when the canvas viewport is resized."""
        if obj is self._canvas.viewport() and event.type() == QEvent.Type.Resize:
            if self._colorbar.isVisible():
                vp = self._canvas.viewport()
                self._colorbar.setGeometry(0, 0, vp.width(), vp.height())
        return super().eventFilter(obj, event)

    def closeEvent(self, event) -> None:  # noqa: N802
        worker = self._strain_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(30_000)
        self._frame_nav.stop_playback()
        self._disconnect_state()
        super().closeEvent(event)
