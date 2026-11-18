"""Left sidebar: the workflow (images -> ROI -> model -> analysis settings -> run)."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QSizePolicy,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config import TilingConfig, TrackingConfig
from ..models.zoo import MODELS, available_models, load_card
from .state import AppState
from .widgets import IntField, NoWheelCombo, Section, hint, muted, row

TRACKING_LABELS = {
    "accumulative": "Accumulative (every frame vs frame 0)",
    "incremental": "Incremental (every frame vs the previous)",
    "every_n": "New reference every N frames",
    "custom": "Custom reference frames",
}


class ImagesPanel(Section):
    open_folder = Signal()
    sort_changed = Signal(bool)

    def __init__(self, state: AppState) -> None:
        super().__init__("1  Images")
        self.state = state
        b = QPushButton("Open image folder…")
        b.setToolTip("Folder with the image sequence; frame 0 (first file) is the reference (Ctrl+O)")
        b.clicked.connect(self.open_folder)
        self.add(b)
        self.info = muted("No images loaded.")
        self.add(self.info)
        self.natural = QCheckBox("Natural sort (img2 before img10)")
        self.natural.setChecked(True)
        self.natural.toggled.connect(self.sort_changed)
        self.add(self.natural)
        self.list = QListWidget()
        self.list.setMinimumHeight(110)
        self.list.setMaximumHeight(210)
        self.list.currentRowChanged.connect(lambda r: r >= 0 and state.set_frame(r))
        self.add(self.list)
        state.images_changed.connect(self.refresh)
        state.frame_changed.connect(self._select)
        state.roi_changed.connect(lambda _f: self._mark_rois())
        state.results_changed.connect(self._mark_rois)

    def refresh(self) -> None:
        st = self.state
        self.list.blockSignals(True)
        self.list.clear()
        if st.sequence is None:
            self.info.setText("No images loaded.")
        else:
            h, w = st.sequence.shape
            folder = st.image_folder.name if st.image_folder else ""
            self.info.setText(f"{folder}: {st.n_frames} frames, {w} x {h} px, {st.sequence.scale.description}")
            self.info.setToolTip(str(st.image_folder))
            for i, name in enumerate(st.sequence.names):
                self.list.addItem(QListWidgetItem(f"{i:3d}  {name}"))
        self.list.blockSignals(False)
        self._mark_rois()
        self._select(st.current_frame)

    def _mark_rois(self) -> None:
        st = self.state
        series = st.series
        names = st.sequence.names if st.sequence else []
        for i in range(min(self.list.count(), len(names))):
            it = self.list.item(i)
            name = st.sequence.names[i] if st.sequence else ""
            tags = []
            if i == 0:
                tags.append("reference")
            if i in st.roi_masks:
                tags.append("ROI" if i == 0 else "mask")
            if series is not None and i < series.n_frames and series.done[i] and i > 0:
                tags.append("✓")
            it.setText(f"{i:3d}  {name}" + (f"   [{', '.join(tags)}]" if tags else ""))

    def _select(self, k: int) -> None:
        if 0 <= k < self.list.count() and self.list.currentRow() != k:
            self.list.blockSignals(True)
            self.list.setCurrentRow(k)
            self.list.blockSignals(False)


class RoiPanel(Section):
    tool_selected = Signal(str)
    import_mask = Signal()
    import_frame_masks = Signal()
    save_mask = Signal()

    def __init__(self, state: AppState) -> None:
        super().__init__("2  Region of interest")
        self.state = state
        self.add(muted("Draw on frame 0. A mask on another frame excludes the regions it leaves out "
                       "(cracks, holes) and is the ROI when that frame is a reference."))
        grid = QGridLayout()
        grid.setSpacing(4)
        self.tools = QButtonGroup(self)
        self.tools.setExclusive(False)
        for i, (key, text, tip) in enumerate((
            ("rect", "▭ Rect", "Rectangle: drag (R)"),
            ("circle", "◯ Circle", "Circle: drag from the centre (C)"),
            ("polygon", "⬠ Polygon", "Polygon: click vertices, double-click / Enter / right-click to close (P)"),
            ("brush", "✎ Brush", "Brush: paint with the mouse (B)"),
        )):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setProperty("tool", key)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.tools.addButton(b)
            grid.addWidget(b, i // 2, i % 2)
        self.tools.buttonClicked.connect(self._tool_clicked)
        self.add_layout(grid)
        self.mode_add = QRadioButton("Add")
        self.mode_cut = QRadioButton("Cut")
        self.mode_add.setChecked(True)
        self.mode_add.setToolTip("New shapes are added to the ROI")
        self.mode_cut.setToolTip("New shapes are removed from the ROI (hold Alt to cut once)")
        self.radius = IntField()
        self.radius.setRange(1, 500)
        self.radius.setValue(20)
        self.radius.setSuffix(" px")
        self.radius.setToolTip("Brush radius")
        self.add_layout(row(self.mode_add, self.mode_cut, QLabel("brush"), self.radius))
        acts = QGridLayout()
        acts.setSpacing(4)
        for i, (text, slot, tip) in enumerate((
            ("Undo", state.undo_roi, "Undo the last ROI edit (Ctrl+Z)"),
            ("Redo", state.redo_roi, "Redo (Ctrl+Y)"),
            ("Invert", self._invert, "Invert the mask of the current frame"),
            ("Clear", self._clear, "Remove the mask of the current frame"),
            ("Import…", self.import_mask, "Load a mask image (white = inside) for the current frame"),
            ("Import set…", self.import_frame_masks, "Load masks for several frames from a folder"),
            ("Save…", self.save_mask, "Save the mask of the current frame as a PNG"),
        )):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            acts.addWidget(b, i // 2, i % 2)
        self.add_layout(acts)
        self.status = muted("")
        self.add(self.status)
        state.roi_changed.connect(lambda _f: self._update_status())
        state.frame_changed.connect(lambda _f: self._update_status())
        state.images_changed.connect(self._update_status)

    @property
    def mode(self) -> str:
        return "cut" if self.mode_cut.isChecked() else "add"

    def _tool_clicked(self, btn) -> None:
        key = btn.property("tool")
        on = btn.isChecked()
        for b in self.tools.buttons():
            if b is not btn:
                b.setChecked(False)
        self.tool_selected.emit(key if on else "pan")

    def set_tool(self, tool: str) -> None:
        for b in self.tools.buttons():
            b.setChecked(b.property("tool") == tool)

    def _invert(self) -> None:
        st = self.state
        if st.sequence is None:
            return
        m = st.roi(st.current_frame)
        import numpy as np

        base = np.zeros(st.sequence.shape, bool) if m is None else m
        st.set_roi(st.current_frame, ~base)

    def _clear(self) -> None:
        st = self.state
        if st.roi(st.current_frame) is not None:
            st.set_roi(st.current_frame, None)

    def _update_status(self) -> None:
        st = self.state
        if st.sequence is None:
            self.status.setText("")
            return
        k = st.current_frame
        m = st.roi(k)
        what = "ROI" if k == 0 else "mask"
        if m is None:
            txt = f"Frame {k}: no {what}." + (" Draw one to start." if k == 0 else "")
        else:
            from ..roi.masks import bbox_of

            x, y, w, h = bbox_of(m)
            txt = f"Frame {k} {what}: {int(m.sum()):,} px, box {w} x {h} at ({x}, {y})"
        others = sorted(f for f in st.roi_masks if f != 0)
        if others:
            txt += f"\nMasks on frames: {', '.join(map(str, others[:12]))}{' …' if len(others) > 12 else ''}"
        self.status.setText(txt)


class ModelPanel(Section):
    suggest = Signal()

    def __init__(self, state: AppState) -> None:
        super().__init__("3  Model")
        self.state = state
        self.group = QButtonGroup(self)
        avail = available_models()
        self.buttons: dict[str, QRadioButton] = {}
        for key, spec in MODELS.items():
            path = avail.get(key)
            card = load_card(path) if path else None
            b = QRadioButton(f"RAFTcorr-{key}   reach {spec.lookup_radius_px} px")
            tip = [f"{spec.name}: {spec.summary}.",
                   f"Encoder downsampling s = {spec.downsample_factor}; nominal lookup radius 32 s = {spec.lookup_radius_px} px."]
            if card is not None:
                if card.spatial_resolution_px:
                    tip.append(f"Spatial resolution about {card.spatial_resolution_px:g} px.")
                if card.translation_capture_px:
                    tip.append(f"Uniform translation captured up to about {card.translation_capture_px:g} px.")
                if card.rotation_capture_deg:
                    tip.append(f"Rotation captured up to about {card.rotation_capture_deg:g} deg.")
            if path is None:
                tip.append("Weights not installed.")
                b.setEnabled(False)
            b.setToolTip("\n".join(tip))
            self.group.addButton(b)
            self.buttons[key] = b
            self.add(b)
            self.add(muted("    " + spec.summary))
        self.group.buttonClicked.connect(self._picked)
        self.device = NoWheelCombo()
        self.device.addItems(["auto", "cuda", "cpu"])
        self.device.setToolTip("auto = the GPU when one is available")
        self.device.currentTextChanged.connect(lambda d: state.set_run_config(replace(state.run_config, device=d)))
        sug = QPushButton("Suggest a model")
        sug.setToolTip("Measure the displacement quickly with s8 and pick the finest model that reaches it")
        sug.clicked.connect(self.suggest)
        self.add_layout(row("Device", self.device, sug))
        self.advice = muted("")
        self.add(self.advice)
        state.run_config_changed.connect(self.sync)
        self.sync()

    def _picked(self, btn) -> None:
        key = next(k for k, b in self.buttons.items() if b is btn)
        self.state.set_run_config(replace(self.state.run_config, model=key))

    def sync(self) -> None:
        cfg = self.state.run_config
        b = self.buttons.get(str(cfg.model))
        if b is not None and not b.isChecked():
            b.setChecked(True)
        if self.device.currentText() != cfg.device:
            self.device.blockSignals(True)
            self.device.setCurrentText(cfg.device)
            self.device.blockSignals(False)


class SettingsPanel(Section):
    def __init__(self, state: AppState) -> None:
        super().__init__("4  Analysis settings")
        self.state = state
        self.mode = NoWheelCombo()
        for k, label in TRACKING_LABELS.items():
            self.mode.addItem(label, k)
        self.mode.setToolTip("Accumulative is preferred whenever the motion stays within the model's reach: "
                             "incremental tracking adds up the error of every step.")
        self.every = IntField()
        self.every.setRange(1, 10000)
        self.every.setValue(10)
        self.keys = QLineEdit()
        self.keys.setPlaceholderText("reference frames, e.g. 0, 20, 45")
        self.add_layout(row("Tracking", self.mode, stretch_last=True))
        self.every_row = QWidget()
        self.every_row.setLayout(row("every", self.every, QLabel("frames")))
        self.add(self.every_row)
        self.add(self.keys)
        self.iters = IntField()
        self.iters.setRange(1, 100)
        self.iters.setToolTip("Refinement iterations of the network (the paper uses 20)")
        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0.0, 50.0)
        self.smooth.setSingleStep(0.5)
        self.smooth.setSuffix(" px")
        self.smooth.setToolTip("Gaussian smoothing of the displacement (0 = raw, as in the paper). "
                               "Smoothing lengthens the spatial resolution.")
        self.add_layout(row("Iterations", self.iters, hint("The paper evaluates every model with 20 iterations.")))
        self.add_layout(row("Smoothing σ", self.smooth))
        adv = Section("Tiling (advanced)", expanded=False)
        self.margin_auto = QCheckBox("auto")
        self.margin_auto.setToolTip("Margin = the model's reach (32 s px)")
        self.margin = IntField()
        self.margin.setRange(0, 4096)
        self.margin.setSuffix(" px")
        adv.add_layout(row("Margin", self.margin, self.margin_auto,
                           hint("Extra context around the ROI so that displaced points stay inside the "
                                "window. Must cover the largest displacement.")))
        self.memfrac = QDoubleSpinBox()
        self.memfrac.setRange(0.05, 1.0)
        self.memfrac.setSingleStep(0.05)
        self.memfrac.setToolTip("Fraction of the free GPU memory one window may use")
        adv.add_layout(row("Memory", self.memfrac, QLabel("of free")))
        self.blend = IntField()
        self.blend.setRange(0, 512)
        self.blend.setSuffix(" px")
        self.blend.setToolTip("Feathered overlap between neighbouring tiles")
        adv.add_layout(row("Blend", self.blend))
        self.corr = NoWheelCombo()
        self.corr.addItem("auto (volume for s8, on demand otherwise)", None)
        self.corr.addItem("on demand (tile)", "tile")
        self.corr.addItem("stored volume", "volume")
        self.corr.setToolTip("Correlation implementation; identical results, different speed and memory")
        adv.add(self.corr)
        self.add(adv)
        for w, sig in ((self.mode, "currentIndexChanged"), (self.every, "valueChanged"), (self.keys, "editingFinished"),
                       (self.iters, "valueChanged"), (self.smooth, "valueChanged"), (self.margin, "valueChanged"),
                       (self.margin_auto, "toggled"), (self.memfrac, "valueChanged"), (self.blend, "valueChanged"),
                       (self.corr, "currentIndexChanged")):
            getattr(w, sig).connect(self._apply)
        state.run_config_changed.connect(self.sync)
        self.sync()

    def _apply(self, *_args) -> None:
        st = self.state
        try:
            keys = tuple(int(t) for t in self.keys.text().replace(";", ",").split(",") if t.strip())
        except ValueError:
            st.emit_log(f"reference frames must be whole numbers separated by commas: {self.keys.text()!r}", "warning")
            keys = st.run_config.tracking.key_frames
        bad = [k for k in keys if st.sequence is not None and not 0 <= k <= st.n_frames - 2]
        if bad:
            st.emit_log(f"reference frames {bad} are outside 0..{st.n_frames - 2} and are ignored", "warning")
            keys = tuple(k for k in keys if k not in bad)
        try:
            cfg = replace(
                st.run_config,
                iters=self.iters.value(),
                smoothing_sigma=float(self.smooth.value()),
                corr_mode=self.corr.currentData(),
                tracking=TrackingConfig(mode=self.mode.currentData(), every_n=self.every.value(), key_frames=keys),
                tiling=TilingConfig(margin_px=None if self.margin_auto.isChecked() else self.margin.value(),
                                    memory_fraction=float(self.memfrac.value()), blend_px=self.blend.value()),
            )
        except ValueError as exc:
            st.emit_log(str(exc), "warning")
            return
        st.set_run_config(cfg)
        self._visibility()

    def _visibility(self) -> None:
        mode = self.mode.currentData()
        self.every_row.setVisible(mode == "every_n")
        self.keys.setVisible(mode == "custom")
        self.margin.setEnabled(not self.margin_auto.isChecked())

    def sync(self) -> None:
        cfg = self.state.run_config
        widgets = (self.mode, self.every, self.keys, self.iters, self.smooth, self.margin, self.margin_auto,
                   self.memfrac, self.blend, self.corr)
        for w in widgets:
            w.blockSignals(True)
        self.mode.setCurrentIndex(max(0, self.mode.findData(cfg.tracking.mode)))
        self.every.setValue(cfg.tracking.every_n)
        self.keys.setText(", ".join(str(k) for k in cfg.tracking.key_frames))
        self.iters.setValue(cfg.iters)
        self.smooth.setValue(cfg.smoothing_sigma)
        self.margin_auto.setChecked(cfg.tiling.margin_px is None)
        spec = MODELS.get(str(cfg.model))
        self.margin.setValue(cfg.tiling.margin_px if cfg.tiling.margin_px is not None
                             else (spec.lookup_radius_px if spec else 128))
        self.memfrac.setValue(cfg.tiling.memory_fraction)
        self.blend.setValue(cfg.tiling.blend_px)
        self.corr.setCurrentIndex(max(0, self.corr.findData(cfg.corr_mode)))
        for w in widgets:
            w.blockSignals(False)
        self._visibility()


class RunPanel(Section):
    run = Signal()
    stop = Signal()

    def __init__(self, state: AppState) -> None:
        super().__init__("5  Run")
        self.state = state
        self.run_btn = QPushButton("Run correlation")
        self.run_btn.setProperty("role", "primary")
        self.run_btn.setToolTip("Track every frame (F5)")
        self.run_btn.clicked.connect(self.run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setProperty("role", "danger")
        self.stop_btn.setToolTip("Stop after the current step; computed frames are kept")
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        lay = row(self.run_btn, self.stop_btn, stretch_last=False)
        lay.setStretch(0, 1)
        self.add_layout(lay)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.add(self.bar)
        self.status = muted("Load images and draw an ROI on frame 0.")
        self.add(self.status)
        self.plan = muted("")
        self.add(self.plan)
        self.stale = QLabel("Results are out of date: the ROI, model or settings changed since the last run.")
        self.stale.setProperty("role", "warning")
        self.stale.setWordWrap(True)
        self.stale.setVisible(False)
        self.add(self.stale)
        state.run_state_changed.connect(self._run_state)
        state.progress.connect(self._progress)
        state.stale_changed.connect(self.stale.setVisible)
        state.roi_changed.connect(lambda _f: self._inputs_changed())
        state.images_changed.connect(self._inputs_changed)
        state.run_config_changed.connect(self._inputs_changed)

    def set_notice(self, text: str) -> None:
        """A message about the last run that stays until the inputs change."""
        self._notice = text
        self.status.setText(text)

    def _inputs_changed(self) -> None:
        self._notice = ""
        self._ready()

    def _ready(self) -> None:
        st = self.state
        if st.run_state != "idle":
            return
        if getattr(self, "_notice", ""):
            self.status.setText(self._notice)
            return
        if st.sequence is None:
            self.status.setText("Load images first.")
        elif st.n_frames < 2:
            self.status.setText("The folder needs at least two images.")
        elif st.roi(0) is None:
            self.status.setText("Draw an ROI on frame 0 (or press Run to use the whole image).")
        else:
            self.status.setText("Ready.")

    def _run_state(self, s: str) -> None:
        running = s != "idle"
        self.run_btn.setEnabled(not running)
        self.stop_btn.setEnabled(s == "running")
        if s == "stopping":
            self.status.setText("Stopping after the current step…")
        if not running:
            self._ready()

    def _progress(self, f: float, msg: str) -> None:
        self.bar.setValue(int(1000 * max(0.0, min(1.0, f))))
        if msg:
            self.status.setText(msg)


class LeftSidebar(QWidget):
    def __init__(self, state: AppState) -> None:
        super().__init__()
        self.setObjectName("sidebarContents")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)
        self.images = ImagesPanel(state)
        self.roi = RoiPanel(state)
        self.model = ModelPanel(state)
        self.settings = SettingsPanel(state)
        self.run = RunPanel(state)
        for w in (self.images, self.roi, self.model, self.settings, self.run):
            lay.addWidget(w)
        lay.addStretch(1)
