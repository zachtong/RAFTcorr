"""The main window: layout, menus, shortcuts, canvas tools, background tasks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .._version import __version__
from ..io.images import ImageSequence, lexicographic_differs, list_images
from ..post.probes import Probe, color_for
from ..roi.masks import list_mask_files, load_mask, match_mask_files, save_mask
from ..roi.shapes import Circle, Polygon, Rect, apply_shape, brush_stroke
from ..viz.fields import ALL_FIELDS
from . import theme
from .canvas import ImageCanvas
from .chart import ProbeChart
from .panels_left import LeftSidebar
from .panels_right import RightSidebar
from .state import AppState
from .view_controller import ViewController
from .widgets import ColorBar, FrameNavigator, LogView
from .workers import TaskWorker

ORG, APP = "RAFTcorr", "RAFTcorr"


def _scroll(widget: QWidget, name: str, width: int) -> QScrollArea:
    sa = QScrollArea()
    sa.setObjectName("sidebarScroll")
    sa.setWidgetResizable(True)
    sa.setWidget(widget)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    sa.setMinimumWidth(width)
    sa.setMaximumWidth(width + 80)
    widget.setMinimumWidth(width - 14)  # room for the vertical scroll bar
    return sa


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None) -> None:
        super().__init__()
        self.state = state or AppState()
        self.view = ViewController(self.state)
        self.settings = QSettings(ORG, APP)
        self._worker: TaskWorker | None = None
        self._strain_worker: TaskWorker | None = None
        self._loaded_cache: dict[tuple, Any] = {}
        self._brush_base: np.ndarray | None = None
        self._temp_dirs: list[str] = []
        self.setWindowTitle(f"RAFTcorr {__version__}")
        self.resize(1560, 960)
        self.setAcceptDrops(True)
        self._build()
        self.view.editing = lambda: self.canvas.tool in ("rect", "circle", "polygon", "brush")
        self._build_menus()
        self._connect()
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render)
        self._plan_timer = QTimer(self)
        self._plan_timer.setSingleShot(True)
        self._plan_timer.timeout.connect(self._update_plan_preview)
        self.request_render()

    # ======================================================================================== layout
    def _build(self) -> None:
        st = self.state
        self.left = LeftSidebar(st)
        self.right = RightSidebar(st)
        self.canvas = ImageCanvas()
        self.colorbar = ColorBar()
        self.colorbar.setVisible(False)
        self.nav = FrameNavigator()

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(0)
        top.addWidget(self.canvas, 1)
        top.addWidget(self.colorbar)
        cl.addLayout(top, 1)
        cl.addWidget(self.nav)
        left = _scroll(self.left, "sidebar", 340)
        left.setObjectName("sidebar")
        right = _scroll(self.right, "rightSidebar", 350)
        right.setObjectName("rightSidebar")
        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)
        bl.addWidget(left)
        bl.addWidget(center, 1)
        bl.addWidget(right)
        self.setCentralWidget(body)

        self.logview = LogView()
        self.chart = ProbeChart(st, metric_source=lambda: self.right.probes.metric_combo.currentText())
        self.tabs = QTabWidget()
        self.tabs.addTab(self.logview, "Log")
        self.tabs.addTab(self.chart, "Chart")
        dock = QDockWidget("Log and chart", self)
        dock.setObjectName("bottomDock")
        dock.setWidget(self.tabs)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                         | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.dock = dock
        self.resizeDocks([dock], [230], Qt.Orientation.Vertical)

        self.pos_label = QLabel("")
        self.value_label = QLabel("")
        self.zoom_label = QLabel("")
        self.device_label = QLabel("")
        sb = self.statusBar()
        sb.addWidget(self.pos_label)
        sb.addWidget(self.value_label, 1)
        sb.addPermanentWidget(self.zoom_label)
        sb.addPermanentWidget(self.device_label)
        self._show_device()

    def _build_menus(self) -> None:
        mb = self.menuBar()

        def act(menu, text: str, slot: Callable, shortcut: str | QKeySequence | None = None, tip: str = "") -> QAction:
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            if tip:
                a.setStatusTip(tip)
            a.triggered.connect(slot)
            menu.addAction(a)
            return a

        f = mb.addMenu("&File")
        act(f, "Open image folder…", self.open_folder, QKeySequence.StandardKey.Open)
        act(f, "Open project…", self.open_project, "Ctrl+Shift+O")
        self.recent_menu = f.addMenu("Open recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        act(f, "Save project", self.save_project, QKeySequence.StandardKey.Save)
        act(f, "Save project as…", lambda: self.save_project(ask=True), "Ctrl+Shift+S")
        f.addSeparator()
        act(f, "Export results…", self.export_results, "Ctrl+E")
        act(f, "Export images / animation…", self.export_images, "Ctrl+Shift+E")
        act(f, "Export probe table (CSV)…", self.export_probes)
        f.addSeparator()
        act(f, "Quit", self.close, QKeySequence.StandardKey.Quit)

        e = mb.addMenu("&Edit")
        self.undo_act = act(e, "Undo ROI edit", self.state.undo_roi, QKeySequence.StandardKey.Undo)
        self.redo_act = act(e, "Redo ROI edit", self.state.redo_roi, "Ctrl+Y")

        v = mb.addMenu("&View")
        act(v, "Fit image", self.canvas.fit, "Ctrl+0")
        act(v, "Actual size (100 %)", lambda: self.canvas.set_zoom(1.0), "Ctrl+1")
        act(v, "Toggle field overlay\tF", lambda: self.state.update_view(show_overlay=not self.state.view.show_overlay))
        act(v, "Toggle reference / deformed\tD", self._toggle_configuration)
        v.addSeparator()
        act(v, "Light theme", lambda: self.set_theme("light"))
        act(v, "Dark theme", lambda: self.set_theme("dark"))
        v.addSeparator()
        v.addAction(self.dock.toggleViewAction())
        dbg = QAction("Show debug messages", self, checkable=True)
        dbg.toggled.connect(lambda on: setattr(self.logview, "show_debug", on))
        v.addAction(dbg)

        r = mb.addMenu("&Analysis")
        act(r, "Run correlation", self.run_correlation, "F5")
        act(r, "Stop", self.stop_task, "Shift+F5")
        act(r, "Compute strain", self.compute_strain, "F6")
        act(r, "Suggest a model", self.suggest_model)

        h = mb.addMenu("&Help")
        act(h, "Keyboard shortcuts", self.show_shortcuts, "F1")
        act(h, "Self-test (synthetic pair)", self.self_test)
        act(h, "About RAFTcorr", self.show_about)

        # Bare keys (Space, arrows, letters, digits) are handled in keyPressEvent rather than as
        # window shortcuts, so that a focused button, slider or combo box keeps its own keys.
        self._bare_keys: dict[int, Callable[[], None]] = {
            Qt.Key.Key_Space: self.nav.toggle_play,
            Qt.Key.Key_Right: lambda: self.nav.step(1),
            Qt.Key.Key_Left: lambda: self.nav.step(-1),
            Qt.Key.Key_Home: lambda: self.nav.request(0),
            Qt.Key.Key_End: lambda: self.nav.request(self.state.n_frames - 1),
            Qt.Key.Key_Escape: lambda: self.set_tool("pan"),
            Qt.Key.Key_R: lambda: self._tool_key("rect"),
            Qt.Key.Key_C: lambda: self._tool_key("circle"),
            Qt.Key.Key_P: lambda: self._tool_key("polygon"),
            Qt.Key.Key_B: lambda: self._tool_key("brush"),
            Qt.Key.Key_F: lambda: self.state.update_view(show_overlay=not self.state.view.show_overlay),
            Qt.Key.Key_D: self._toggle_configuration,
        }
        for i in range(1, 10):
            self._bare_keys[int(Qt.Key.Key_0) + i] = lambda i=i: self._field_key(i - 1)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Reached only when the focused widget did not use the key."""
        slot = self._bare_keys.get(event.key())
        if slot is not None and not (event.modifiers() & (Qt.KeyboardModifier.ControlModifier
                                                          | Qt.KeyboardModifier.AltModifier)):
            slot()
            event.accept()
            return
        super().keyPressEvent(event)

    def _connect(self) -> None:
        st = self.state
        L, R = self.left, self.right
        L.images.open_folder.connect(self.open_folder)
        L.images.sort_changed.connect(self._resort)
        L.roi.tool_selected.connect(self.set_tool)
        L.roi.import_mask.connect(self.import_mask)
        L.roi.import_frame_masks.connect(self.import_frame_masks)
        L.roi.save_mask.connect(self.save_current_mask)
        L.roi.radius.valueChanged.connect(lambda r: setattr(self.canvas, "brush_radius", float(r)))
        L.model.suggest.connect(self.suggest_model)
        L.run.run.connect(self.run_correlation)
        L.run.stop.connect(self.stop_task)
        R.strain.compute.connect(self.compute_strain)
        R.probes.tool_selected.connect(self.set_tool)
        R.probes.export_csv.connect(self.export_probes)
        R.probes.metric_combo.currentTextChanged.connect(lambda _m: self.chart.refresh())
        self.nav.frame_requested.connect(st.set_frame)
        self.chart.frame_clicked.connect(st.set_frame)
        self.canvas.shape_finished.connect(self._shape_finished)
        self.canvas.brush_stroke.connect(self._brush)
        self.canvas.hovered.connect(self._hover)
        self.canvas.zoom_changed.connect(lambda z: self.zoom_label.setText(f"{z * 100:.0f} %"))
        for sig in (st.frame_changed, st.roi_changed, st.view_changed, st.results_changed, st.strain_changed,
                    st.images_changed, st.units_changed):
            sig.connect(self.request_render)
        st.results_changed.connect(self.view.invalidate_ranges)
        st.strain_changed.connect(self.view.invalidate_ranges)
        st.frame_changed.connect(self.nav.set_frame)
        st.images_changed.connect(self._images_changed)
        st.results_changed.connect(self._results_changed)
        st.probes_changed.connect(self.request_render)
        st.log.connect(self.logview.append_line)
        st.project_path_changed.connect(
            lambda p: self.setWindowTitle(f"RAFTcorr - {Path(p).name}" if p else f"RAFTcorr {__version__}"))
        st.run_state_changed.connect(self._run_state_changed)
        for sig in (st.roi_changed, st.run_config_changed, st.images_changed):
            sig.connect(lambda *_a: self._plan_timer.start(150))

    # ======================================================================================= helpers
    def log(self, msg: str, level: str = "info") -> None:
        self.state.emit_log(msg, level)

    def error(self, title: str, msg: str) -> None:
        self.log(f"{title}: {msg}", "error")
        QMessageBox.critical(self, title, msg)

    def _show_device(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                self.device_label.setText(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device_label.setText("CPU only (no CUDA device)")
        except Exception:
            self.device_label.setText("PyTorch not available")

    def set_theme(self, name: str) -> None:
        theme.set_current(name)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(theme.stylesheet())
        self.settings.setValue("theme", name)
        self.colorbar.update()
        self.chart.refresh()

    # ====================================================================================== rendering
    def request_render(self, *_a) -> None:
        self._render_timer.start(0)

    def _render(self) -> None:
        st = self.state
        editing = self.canvas.tool in ("rect", "circle", "polygon", "brush")
        self.canvas.set_background(self.view.background(editing))
        lay = self.view.overlay(editing)
        self.canvas.set_overlay(lay.rgba, lay.x0, lay.y0, lay.step)
        self.canvas.set_roi_overlay(self.view.roi_layer(editing))
        self.canvas.set_probes(st.probes, mapper=self._probe_mapper())
        rng = self.view.last_range
        spec = ALL_FIELDS.get(st.view.field)
        if rng is not None and spec is not None:
            d = st.view.display(st.view.field)
            self.colorbar.set_scale(d.colormap, rng.vmin, rng.vmax, f"{st.view.field} ({st.units.unit(spec)})", d.levels)
            self.colorbar.setVisible(True)
            self.right.display.set_auto_range(rng.vmin, rng.vmax)
        else:
            self.colorbar.setVisible(False)

    def _probe_mapper(self):
        """Reference -> current position of probe points in the deformed configuration."""
        st = self.state
        s = st.series
        k = st.current_frame
        if s is None or st.view.configuration != "deformed" or k >= s.n_frames or not s.done[k]:
            return None
        from ..engine.mapping import sample_bilinear

        uv = s.field(k)
        g = s.grid

        def mapper(pts: np.ndarray) -> np.ndarray:
            d = sample_bilinear(uv, pts[:, 0], pts[:, 1], g.x0, g.y0)
            d = np.where(np.isfinite(d), d, 0.0)
            return pts + d.T

        return mapper

    def _hover(self, x: float, y: float) -> None:
        if not np.isfinite(x):
            self.pos_label.setText("")
            self.value_label.setText("")
            return
        self.pos_label.setText(f"x {x:.1f}  y {y:.1f} px")
        val = self.view.value_at(x, y)
        s = self.view.shown
        self.value_label.setText("" if val is None else f"{s.key} = {val:.5g} {s.unit}")

    def _update_plan_preview(self) -> None:
        """How the next run will cut the ROI into windows, without loading the model."""
        st = self.state
        label = self.left.run.plan
        if st.sequence is None:
            label.setText("")
            return
        from ..engine.backend import min_input_side
        from ..engine.memory import device_budget_bytes, estimate_peak_bytes, max_pixels_for_budget
        from ..engine.tiling import plan_tiles
        from ..models.zoo import MODELS
        from ..nn.corr import default_corr_mode
        from ..roi.masks import bbox_of

        cfg = st.run_config
        spec = MODELS.get(str(cfg.model))
        if spec is None:
            label.setText("")
            return
        s = spec.downsample_factor
        roi = st.roi(0)
        bbox = bbox_of(roi) if roi is not None else (0, 0, st.sequence.shape[1], st.sequence.shape[0])
        mode = cfg.corr_mode or default_corr_mode(s)
        try:
            import torch

            dev = "cuda" if (cfg.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
            max_px = cfg.tiling.max_tile_pixels or max_pixels_for_budget(
                device_budget_bytes(dev, cfg.tiling.memory_fraction), s, mode)
            margin = spec.lookup_radius_px if cfg.tiling.margin_px is None else cfg.tiling.margin_px
            plan = plan_tiles(st.sequence.shape, bbox, margin, max_px, min_input_side(s), cfg.tiling.blend_px)
            peak = estimate_peak_bytes(plan.max_window_pixels, s, mode) / 1024**3
            label.setText(f"Plan: {plan.describe()}; about {peak:.1f} GB per window on {dev.upper()}")
        except Exception as exc:  # e.g. not enough memory for one window
            label.setText(f"Plan: {exc}")

    # ========================================================================================= images
    def open_folder(self) -> None:
        start = str(self.state.image_folder or self.settings.value("last_folder", str(Path.home())))
        folder = QFileDialog.getExistingDirectory(self, "Open image folder", start)
        if folder:
            self.load_folder(Path(folder))

    def load_folder(self, folder: Path, natural: bool | None = None) -> bool:
        natural = self.left.images.natural.isChecked() if natural is None else natural
        try:
            files = list_images(folder, natural=natural)
            if not files:
                raise FileNotFoundError(f"no images (.png .jpg .tif .bmp) in {folder}")
            seq = ImageSequence(files, cache_size=12)
        except Exception as exc:
            self.error("Cannot open the folder", str(exc))
            return False
        self.state.natural_sort = natural
        self.state.set_sequence(seq, folder)
        self.settings.setValue("last_folder", str(folder))
        self.remember(folder)
        self.log(f"opened {len(seq)} frames from {folder} ({seq.shape[1]} x {seq.shape[0]} px, {seq.scale.description})")
        if not natural and lexicographic_differs(files):
            self.log("the file names are numbered: natural sort would order them differently", "warning")
        return True

    # -- recent items and drag and drop ---------------------------------------------------------------
    MAX_RECENT = 10

    def remember(self, path: Path) -> None:
        items = [str(path)] + [p for p in self.recent() if p != str(path)]
        self.settings.setValue("recent", items[: self.MAX_RECENT])

    def recent(self) -> list[str]:
        v = self.settings.value("recent", [])
        return [v] if isinstance(v, str) else list(v or [])

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        items = [p for p in self.recent() if Path(p).exists()]
        if not items:
            a = self.recent_menu.addAction("(nothing yet)")
            a.setEnabled(False)
            return
        for p in items:
            self.recent_menu.addAction(p, lambda p=p: self.open_path(Path(p)))

    def open_path(self, p: Path) -> None:
        """Open an image folder or a project file."""
        if p.is_dir():
            self.load_folder(p)
        elif p.suffix.lower() == ".raftcorr" and p.is_file():
            from .project_io import load_project_into

            self.remember(p)
            load_project_into(self, p)
        elif p.is_file() and p.suffix.lower() in (".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"):
            self.load_folder(p.parent)
        else:
            self.log(f"cannot open {p}", "warning")

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            self.open_path(Path(urls[0].toLocalFile()))
            event.acceptProposedAction()

    def _resort(self, natural: bool) -> None:
        st = self.state
        if st.image_folder is None:
            return
        if st.roi_masks or st.series is not None or st.probes:
            ans = QMessageBox.question(self, "Sort the frames again",
                                       "Changing the frame order reloads the folder and discards the ROIs, "
                                       "masks, probes and results of this session. Continue?")
            if ans != QMessageBox.StandardButton.Yes:
                box = self.left.images.natural
                box.blockSignals(True)
                box.setChecked(not natural)
                box.blockSignals(False)
                return
        self.load_folder(st.image_folder, natural)

    def _images_changed(self) -> None:
        st = self.state
        self.nav.set_count(st.n_frames, st.sequence.names if st.sequence else None)
        self.nav.set_playable(None)
        self.canvas.fit()

    def _results_changed(self) -> None:
        st = self.state
        if st.series is not None:
            self.nav.set_playable(st.series.completed_frames)
        else:
            self.nav.set_playable(None)

    # =========================================================================================== tools
    def set_tool(self, tool: str) -> None:
        self.canvas.set_tool(tool)
        self.left.roi.set_tool(tool)
        self.right.probes.set_tool(tool)
        if tool in ("rect", "circle", "polygon", "brush"):
            self.state.update_view(show_roi=True)
        self.request_render()

    def _tool_key(self, tool: str) -> None:
        if self.state.sequence is not None:
            self.set_tool(tool if self.canvas.tool != tool else "pan")

    def _shape_finished(self, tool: str, pts: list) -> None:
        st = self.state
        if st.sequence is None:
            return
        if tool.startswith("probe_"):
            self._add_probe(tool.removeprefix("probe_"), pts)
            return
        shape = _shape(tool, pts)
        if shape is None:
            return
        k = st.current_frame
        base = st.roi(k)
        base = np.zeros(st.sequence.shape, bool) if base is None else base
        mode = self.left.roi.mode
        if QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier:
            mode = "cut"
        st.set_roi(k, apply_shape(base, shape, mode))

    def _brush(self, pts: list, radius: float, finished: bool) -> None:
        st = self.state
        if st.sequence is None:
            return
        k = st.current_frame
        if self._brush_base is None:
            base = st.roi(k)
            self._brush_base = np.zeros(st.sequence.shape, bool) if base is None else base.copy()
        stroke = brush_stroke(*st.sequence.shape, pts, radius)
        cut = self.left.roi.mode == "cut" or bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier)
        new = (self._brush_base & ~stroke) if cut else (self._brush_base | stroke)
        if finished:
            before = self._brush_base
            self._brush_base = None
            st.roi_history.record(k, None if not before.any() else before)
            st.set_roi(k, new, record=False)
        else:
            from ..viz.render import mask_rgba

            self.canvas.set_roi_overlay(mask_rgba(new, alpha=110))

    def _to_material(self, pts: list) -> list | None:
        """Frame-0 (material) coordinates of points clicked on the canvas.

        In the deformed configuration the canvas shows frame k, so a click at x is the material
        point X with X + U(X) = x (inverse map); elsewhere the click already is in frame-0 pixels.
        Returns None when a point does not sit on tracked material.
        """
        st = self.state
        k = st.current_frame
        if st.view.configuration != "deformed" or st.provider is None or not st.series.done[k] or k == 0:
            return pts
        from ..engine.mapping import sample_bilinear

        X, Y, (x0, y0, _x1, _y1) = st.provider.inverse(k)
        if X.size == 0:
            return None
        xs = np.array([p[0] for p in pts], float) - x0
        ys = np.array([p[1] for p in pts], float) - y0
        mx = sample_bilinear(X, xs, ys)
        my = sample_bilinear(Y, xs, ys)
        if not (np.isfinite(mx).all() and np.isfinite(my).all()):
            return None
        return [(float(a), float(b)) for a, b in zip(mx, my)]

    def _add_probe(self, kind: str, pts: list) -> None:
        st = self.state
        if kind == "circle":
            (cx, cy), (ex, ey) = pts[0], pts[-1]
            pts = [(cx, cy), (ex, ey)]
        elif kind in ("rect", "line"):
            pts = [pts[0], pts[-1]]
        mapped = self._to_material(pts)
        if mapped is None:
            self.log("the probe must lie on tracked material in the deformed view", "warning")
            return
        pts = mapped
        pid = st.next_probe_id()
        try:
            probe = Probe(pid, kind, tuple(pts), color_for(len(st.probes)))
        except ValueError as exc:
            self.log(str(exc), "warning")
            return
        st.add_probe(probe)
        self.tabs.setCurrentWidget(self.chart)
        self.log(f"probe {probe.label}: {probe.describe()}")

    def _toggle_configuration(self) -> None:
        v = self.state.view
        self.state.update_view(configuration="deformed" if v.configuration == "reference" else "reference")

    def _field_key(self, i: int) -> None:
        combo = self.right.display.field
        if i < combo.count():
            combo.setCurrentIndex(i)

    # ========================================================================================= masks
    def import_mask(self) -> None:
        st = self.state
        if st.sequence is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, f"Mask for frame {st.current_frame}", str(st.image_folder or ""),
                                              "Images (*.png *.tif *.tiff *.bmp *.jpg)")
        if not path:
            return
        try:
            m = load_mask(path, st.sequence.shape)
        except Exception as exc:
            self.error("Cannot read the mask", str(exc))
            return
        st.set_roi(st.current_frame, m)
        self.log(f"mask of frame {st.current_frame} from {Path(path).name}: {int(m.sum()):,} px")

    def import_frame_masks(self) -> None:
        st = self.state
        if st.sequence is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder with one mask per frame", str(st.image_folder or ""))
        if not folder:
            return
        files = list_mask_files(folder)
        strategy = "by_number"
        mapping = match_mask_files(files, st.n_frames, strategy)
        if len(mapping) < min(len(files), st.n_frames) // 2:
            strategy = "sequential"
            mapping = match_mask_files(files, st.n_frames, strategy)
        ans = QMessageBox.question(
            self, "Import masks",
            f"{len(files)} mask files, {len(mapping)} matched to frames "
            f"({'by the number in the name' if strategy == 'by_number' else 'in sorted order'}).\nImport them?")
        if ans != QMessageBox.StandardButton.Yes:
            return
        for k, p in sorted(mapping.items()):
            try:
                st.set_roi(k, load_mask(p, st.sequence.shape))
            except Exception as exc:
                self.log(f"{p.name}: {exc}", "warning")
        self.log(f"imported {len(mapping)} masks from {folder}")

    def save_current_mask(self) -> None:
        st = self.state
        m = st.roi(st.current_frame)
        if m is None:
            self.log("this frame has no mask to save", "warning")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save mask", str((st.image_folder or Path.home()) / f"mask_{st.current_frame:04d}.png"),
                                              "PNG (*.png)")
        if path:
            save_mask(path, m)
            self.log(f"saved {path}")

    # ====================================================================================== tasks
    def _start(self, worker: TaskWorker, on_done: Callable[[Any], None], label: str) -> None:
        st = self.state
        if self._worker is not None:
            self.log(f"{label}: another task is still running; wait for it or stop it", "warning")
            return
        self._worker = worker
        worker.progress.connect(lambda f, m: st.progress.emit(f, m))
        worker.log.connect(st.emit_log)
        worker.done.connect(on_done)
        worker.failed.connect(lambda msg, tb: (self.log(tb, "debug"), self.error(f"{label} failed", msg)))
        worker.stopped.connect(lambda: self.log(f"{label} stopped", "warning"))
        worker.finished.connect(self._task_finished)
        st.set_run_state("running")
        worker.start()

    def _task_finished(self) -> None:
        self._worker = None
        self.state.set_run_state("idle")
        self.right.strain.set_progress(None)
        self.right.strain._status()

    def stop_task(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self.state.set_run_state("stopping")

    def _run_state_changed(self, s: str) -> None:
        busy = s != "idle"
        self.right.strain.go.setEnabled(not busy and self.state.series is not None)

    def loaded_model(self, cfg, callbacks) -> Any:
        """Load (or reuse) the model of ``cfg`` on its device."""
        from ..models import load_model, resolve_weights

        path = resolve_weights(cfg.model, callbacks=callbacks)
        key = (str(path), cfg.device, cfg.corr_mode)
        hit = self._loaded_cache.get(key)
        if hit is None:
            self._loaded_cache.clear()  # one model at a time on the GPU
            hit = load_model(path, cfg.device, cfg.corr_mode, callbacks)
            self._loaded_cache[key] = hit
        return hit

    def run_correlation(self) -> None:
        st = self.state
        if self._worker is not None:
            self.log("another task is running; wait for it or stop it", "warning")
            return
        if st.sequence is None or st.n_frames < 2:
            self.error("Nothing to run", "Open a folder with at least two images first.")
            return
        roi = st.roi(0)
        if roi is None:
            ans = QMessageBox.question(self, "No ROI", "No ROI on frame 0. Track the whole image?")
            if ans != QMessageBox.StandardButton.Yes:
                return
        cfg = st.run_config
        seq = st.sequence
        masks = {k: m for k, m in st.roi_masks.items() if k != 0}
        signature = st.run_signature()
        storage = self._result_storage(roi)

        def job(cb):
            from ..engine.pipeline import Pipeline

            loaded = self.loaded_model(cfg, cb)
            pipe = Pipeline(cfg, cb, loaded)
            return pipe.run(seq, roi, frame_masks=masks, storage_dir=storage)

        def done(series) -> None:
            if int(series.done.sum()) <= 1 and st.series is not None:
                self.log("the run stopped before any frame was computed; the previous results are kept", "warning")
                return
            st.set_strain(None)
            st.set_results(series)
            st._last_run_signature = signature
            st.stale_changed.emit(st.is_stale)
            n = int(series.done.sum()) - 1
            self.log(f"correlation finished: {n} frames in {series.info.seconds:.1f} s ({series.info.model_name})")
            if st.view.field not in ("u", "v", "magnitude"):
                st.update_view(field="u")
            if st.current_frame == 0 and series.n_frames > 1:
                st.set_frame(series.completed_frames[-1] if n > 0 else 0)
            quality = [note for note in series.info.notes if note.startswith("tracking quality")]
            if quality:
                self.left.run.set_notice("Finished with a tracking-quality warning; see the log and the "
                                         "'intensity residual' field.")
                text = quality[0][0].upper() + quality[0][1:]
                QMessageBox.warning(self, "Tracking quality",
                                    f"{text}.\n\nShow the field 'intensity residual' to see where.")

        self.log(f"running {cfg.model}, {cfg.tracking.mode}, {cfg.iters} iterations…")
        self._start(TaskWorker(job, "correlation"), done, "Correlation")

    RAM_RESULTS_LIMIT = 2 * 1024**3

    def _result_storage(self, roi) -> str | None:
        """A temporary folder for memory-mapped results when they would not fit comfortably in RAM."""
        import tempfile

        from ..roi.masks import bbox_of

        st = self.state
        b = bbox_of(roi) if roi is not None else (0, 0, st.sequence.shape[1], st.sequence.shape[0])
        nbytes = st.n_frames * b[2] * b[3] * (4 + 4 + 2)
        if nbytes <= self.RAM_RESULTS_LIMIT:
            return None
        d = tempfile.mkdtemp(prefix="raftcorr_results_")
        self._temp_dirs.append(d)
        self.log(f"results of {nbytes / 1024**3:.1f} GB are kept on disk in {d} (memory-mapped)")
        return d

    def compute_strain(self) -> None:
        st = self.state
        if self._worker is not None:
            self.log("another task is running; wait for it or stop it", "warning")
            return
        if st.series is None:
            self.log("run the correlation before computing strain", "warning")
            return
        series, cfg = st.series, st.strain_config

        def job(cb):
            from ..post.strain import compute_strain

            return compute_strain(series, cfg, callbacks=cb)

        def done(strain) -> None:
            st.set_strain(strain)
            self.log(f"strain computed ({cfg.measure}, Lg {cfg.gauge_px} px, {cfg.weighting})")
            if not ALL_FIELDS[st.view.field].is_strain:
                st.update_view(field="exx")

        self.right.strain.set_progress(0.0)
        w = TaskWorker(job, "strain")
        w.progress.connect(lambda f, _m: self.right.strain.set_progress(f))
        self._start(w, done, "Strain")

    def suggest_model(self) -> None:
        st = self.state
        if self._worker is not None:
            self.log("another task is running; wait for it or stop it", "warning")
            return
        if st.sequence is None or st.n_frames < 2:
            self.log("open an image sequence of at least two frames first", "warning")
            return
        from ..config import RunConfig
        from ..models.advisor import sample_frames, survey

        roi = st.roi(0) if st.roi(0) is not None else np.ones(st.sequence.shape, bool)
        pairs = [(0, k) for k in sample_frames(st.n_frames)]
        base = replace(st.run_config, model="s8", corr_mode=None)
        seq = st.sequence

        def job(cb):
            from ..engine.pipeline import Pipeline

            pipe = Pipeline(RunConfig(model="s8", device=base.device, iters=base.iters, tiling=base.tiling), cb,
                            self.loaded_model(base, cb))
            return survey(pipe, seq, roi, pairs, cb)

        def done(advice) -> None:
            self.left.model.advice.setText(f"Suggested: RAFTcorr-{advice.key}. {advice.reason}.")
            self.log(f"model survey: {advice.reason}")
            ans = QMessageBox.question(self, "Suggested model",
                                       f"RAFTcorr-{advice.key} is suggested:\n{advice.reason}.\n\nUse it?")
            if ans == QMessageBox.StandardButton.Yes:
                st.set_run_config(replace(st.run_config, model=advice.key))

        self._start(TaskWorker(job, "survey"), done, "Model survey")

    def self_test(self) -> None:
        from ..selftest import run_selftest

        if self._worker is not None:
            return

        def done(report) -> None:
            self.log(report.text, "info" if report.ok else "error")
            (QMessageBox.information if report.ok else QMessageBox.warning)(self, "Self-test", report.text)

        self._start(TaskWorker(lambda cb: run_selftest(device=self.state.run_config.device, callbacks=cb), "self-test"),
                    done, "Self-test")

    # ====================================================================================== project
    def open_project(self) -> None:
        from .project_io import open_project_dialog

        open_project_dialog(self)

    def save_project(self, ask: bool = False) -> None:
        from .project_io import save_project_dialog

        save_project_dialog(self, ask)

    def export_results(self) -> None:
        from .export_dialogs import export_results_dialog

        export_results_dialog(self)

    def export_images(self) -> None:
        from .export_dialogs import export_images_dialog

        export_images_dialog(self)

    def export_probes(self) -> None:
        from .export_dialogs import export_probes_dialog

        export_probes_dialog(self)

    # ======================================================================================== help
    def show_shortcuts(self) -> None:
        QMessageBox.information(self, "Keyboard shortcuts", SHORTCUTS)

    def show_about(self) -> None:
        QMessageBox.about(self, "About RAFTcorr", ABOUT.format(version=__version__))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._worker is not None:
            ans = QMessageBox.question(self, "Quit", "A computation is running. Stop it and quit?")
            if ans != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._worker.request_stop()
            if not self._worker.wait(60000):
                self.log("the background task did not stop within a minute", "warning")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.state.set_results(None)  # release memory maps before removing their files
        import gc
        import shutil

        gc.collect()
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
        super().closeEvent(event)


def _shape(tool: str, pts: list):
    if tool == "rect":
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        return Rect(x0, y0, x1, y1)
    if tool == "circle":
        (cx, cy), (ex, ey) = pts[0], pts[-1]
        return Circle(cx, cy, float(np.hypot(ex - cx, ey - cy)))
    if tool == "polygon" and len(pts) >= 3:
        return Polygon(tuple(pts))
    return None


SHORTCUTS = """Files
  Ctrl+O  open an image folder      Ctrl+Shift+O  open a project
  Ctrl+S  save the project          Ctrl+E  export results

Frames
  Left / Right  previous / next     Home / End  first / last
  Space  play / pause

Region of interest (on the current frame)
  R  rectangle   C  circle   P  polygon   B  brush   Esc  stop drawing
  Hold Alt while drawing to cut instead of add
  Polygon: click the vertices, then double-click, Enter or right-click
  Ctrl+Z / Ctrl+Y  undo / redo

View
  wheel  zoom   middle drag  pan   Ctrl+0  fit   Ctrl+1  100 %
  F  field on / off   D  reference / deformed   1-9  choose the field

Analysis
  F5  run   Shift+F5  stop   F6  compute strain"""

ABOUT = """<b>RAFTcorr {version}</b><br>
Learned digital image correlation with the RAFTcorr-s1, -s2, -s4 and -s8 models.<br><br>
Displacements are reference -> deformed, +u right, +v down, in pixels, in frame-0
(Lagrangian) coordinates. Strain from a virtual strain gauge.<br><br>
Zixiang (Zach) Tong, The University of Texas at Austin."""
