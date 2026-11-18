"""Dialogs for exporting results, pictures / animations, and probe tables."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from ..export.scientific import PhysicalUnits, available_variables, export_results
from ..viz.fields import ALL_FIELDS, Units
from .workers import TaskWorker

if TYPE_CHECKING:
    from .main_window import MainWindow


def to_physical(u: Units) -> PhysicalUnits:
    return PhysicalUnits(scale=u.scale, unit=u.length_unit, frame_rate=u.frame_rate)


def from_physical(p: PhysicalUnits) -> Units:
    return Units(pixel_size=p.scale, length_unit=p.unit, frame_rate=p.frame_rate)


def _default_dir(win: "MainWindow") -> Path:
    st = win.state
    if st.project_path is not None:
        return st.project_path.parent
    return st.image_folder or Path.home()


def _path_row(edit: QLineEdit, pick) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.addWidget(edit, 1)
    b = QPushButton("Browse…")
    b.clicked.connect(pick)
    lay.addWidget(b)
    return lay


# ================================================================================ scientific data
class ResultsDialog(QDialog):
    FORMATS = (("HDF5 (.h5)", ".h5"), ("MATLAB v7.3 (.mat)", ".mat"), ("NumPy (.npz)", ".npz"))

    def __init__(self, win: "MainWindow") -> None:
        super().__init__(win)
        self.setWindowTitle("Export results")
        self.win = win
        st = win.state
        lay = QVBoxLayout(self)
        fmt_box = QGroupBox("Format")
        fl = QHBoxLayout(fmt_box)
        self.fmt = QButtonGroup(self)
        for i, (label, ext) in enumerate(self.FORMATS):
            rb = QRadioButton(label)
            rb.setProperty("ext", ext)
            self.fmt.addButton(rb, i)
            fl.addWidget(rb)
        self.fmt.button(0).setChecked(True)
        self.fmt.buttonClicked.connect(self._ext_changed)
        lay.addWidget(fmt_box)
        var_box = QGroupBox("Variables")
        grid = QGridLayout(var_box)
        self.checks: dict[str, QCheckBox] = {}
        for i, name in enumerate(available_variables(st.strain, st.series)):
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.checks[name] = cb
            grid.addWidget(cb, i // 6, i % 6)
        allb, noneb = QPushButton("All"), QPushButton("None")
        allb.clicked.connect(lambda: [c.setChecked(True) for c in self.checks.values()])
        noneb.clicked.connect(lambda: [c.setChecked(False) for c in self.checks.values()])
        row = QHBoxLayout()
        row.addWidget(allb)
        row.addWidget(noneb)
        row.addStretch(1)
        lay.addWidget(var_box)
        lay.addLayout(row)
        note = QLabel("Displacements are written in px (+u right, +v down, frame-0 coordinates); the pixel size, "
                      "length unit and frame rate are stored with them. NaN marks points outside the ROI or not tracked.")
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        lay.addWidget(note)
        self.path = QLineEdit(str(_default_dir(win) / "raftcorr_results.h5"))
        lay.addLayout(_path_row(self.path, self._pick))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    @property
    def ext(self) -> str:
        return self.fmt.checkedButton().property("ext")

    def _ext_changed(self, *_a) -> None:
        p = Path(self.path.text())
        self.path.setText(str(p.with_suffix(self.ext)))

    def _pick(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export results", self.path.text(), f"*{self.ext}")
        if path:
            self.path.setText(str(Path(path).with_suffix(self.ext)))

    def variables(self) -> list[str]:
        return [n for n, c in self.checks.items() if c.isChecked()]


def export_results_dialog(win: "MainWindow") -> None:
    st = win.state
    if st.series is None:
        win.error("Nothing to export", "Run the correlation first.")
        return
    dlg = ResultsDialog(win)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return
    path = Path(dlg.path.text())
    variables = dlg.variables()
    if not variables:
        win.error("Nothing to export", "Choose at least one variable.")
        return
    overwrite = False
    if path.exists():
        if QMessageBox.question(win, "Overwrite?", f"{path.name} exists. Replace it?") != QMessageBox.StandardButton.Yes:
            return
        overwrite = True
    series, strain, units = st.series, st.strain, to_physical(st.units)
    win._start(TaskWorker(lambda cb: export_results(path, series, strain, variables, units, overwrite, cb), "export"),
               lambda p: win.log(f"exported {len(variables)} variables to {p}"), "Export")


# ======================================================================================= pictures
class PicturesDialog(QDialog):
    def __init__(self, win: "MainWindow") -> None:
        super().__init__(win)
        self.setWindowTitle("Export images / animation")
        st = win.state
        lay = QVBoxLayout(self)
        g = QGridLayout()
        g.addWidget(QLabel("Field"), 0, 0)
        self.field = QComboBox()
        for spec in st.provider.available():
            self.field.addItem(spec.label, spec.key)
        self.field.setCurrentIndex(max(0, self.field.findData(st.view.field)))
        g.addWidget(self.field, 0, 1, 1, 3)
        g.addWidget(QLabel("Frames"), 1, 0)
        done = st.series.completed_frames
        self.first = QSpinBox()
        self.last = QSpinBox()
        self.every = QSpinBox()
        for sb in (self.first, self.last):
            sb.setRange(0, st.series.n_frames - 1)
        self.first.setValue(done[0] if done else 0)
        self.last.setValue(done[-1] if done else 0)
        self.every.setRange(1, 1000)
        g.addWidget(self.first, 1, 1)
        g.addWidget(self.last, 1, 2)
        g.addWidget(QLabel("every"), 1, 3)
        g.addWidget(self.every, 1, 4)
        g.addWidget(QLabel("Range"), 2, 0)
        self.range = QComboBox()
        self.range.addItem("one range for all frames", "sequence")
        self.range.addItem("each frame its own range", "frame")
        self.range.addItem("the range on screen", "manual")
        d = st.view.display(st.view.field)
        if not d.auto:
            self.range.setCurrentIndex(2)
        g.addWidget(self.range, 2, 1, 1, 4)
        self.colorbar = QCheckBox("colour bar")
        self.colorbar.setChecked(True)
        self.title = QCheckBox("title")
        self.title.setChecked(True)
        self.crop = QCheckBox("crop to the ROI")
        g.addWidget(self.colorbar, 3, 1)
        g.addWidget(self.title, 3, 2)
        g.addWidget(self.crop, 3, 3)
        g.addWidget(QLabel("Output"), 4, 0)
        self.kind = QComboBox()
        for label, ext in (("PNG images", ".png"), ("TIFF images", ".tif"), ("PDF figures", ".pdf"),
                           ("SVG figures", ".svg"), ("GIF animation", ".gif"), ("MP4 animation", ".mp4")):
            self.kind.addItem(label, ext)
        g.addWidget(self.kind, 4, 1, 1, 2)
        self.dpi = QSpinBox()
        self.dpi.setRange(50, 1200)
        self.dpi.setValue(200)
        self.dpi.setSuffix(" dpi")
        g.addWidget(self.dpi, 4, 3)
        self.fps = QDoubleSpinBox()
        self.fps.setRange(0.1, 60)
        self.fps.setValue(5)
        self.fps.setSuffix(" fps")
        g.addWidget(self.fps, 4, 4)
        lay.addLayout(g)
        self.folder = _default_dir(win)  # images go to folder / raftcorr_figures, animations to folder / name
        self.path = QLineEdit()
        lay.addLayout(_path_row(self.path, self._pick))
        note = QLabel("Pictures use the display settings (configuration, colour map, opacity, background, units).")
        note.setProperty("role", "muted")
        note.setWordWrap(True)
        lay.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.field.currentIndexChanged.connect(self._kind_changed)
        self.path.editingFinished.connect(self._path_edited)
        self._kind_changed()

    @property
    def ext(self) -> str:
        return self.kind.currentData()

    @property
    def animation(self) -> bool:
        return self.ext in (".gif", ".mp4")

    def _kind_changed(self, *_a) -> None:
        """The path follows the output kind and the field; the base folder never changes here."""
        self.fps.setEnabled(self.animation)
        if self.animation:
            self.path.setText(str(self.folder / f"raftcorr_{self.field.currentData()}{self.ext}"))
        else:
            self.path.setText(str(self.folder / "raftcorr_figures"))

    def _path_edited(self) -> None:
        p = Path(self.path.text())
        self.folder = p.parent

    def _pick(self) -> None:
        if self.animation:
            path, _ = QFileDialog.getSaveFileName(self, "Animation", self.path.text(), f"*{self.ext}")
            if path:
                p = Path(path).with_suffix(self.ext)
                self.folder = p.parent
                self.path.setText(str(p))
        else:
            path = QFileDialog.getExistingDirectory(self, "Folder for the images", str(self.folder))
            if path:
                self.path.setText(path)
                self.folder = Path(path).parent


def export_images_dialog(win: "MainWindow") -> None:
    from dataclasses import replace

    from ..export.figures import FrameStyle, export_animation, export_frames

    st = win.state
    if st.provider is None:
        win.error("Nothing to export", "Run the correlation first.")
        return
    dlg = PicturesDialog(win)
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return
    key = dlg.field.currentData()
    d = st.view.display(key)
    rmode = dlg.range.currentData()
    vmin = vmax = None
    if rmode == "manual":  # the range on screen: the manual one, or the automatic one now shown
        if not d.auto:
            vmin, vmax = d.vmin, d.vmax
        elif key == st.view.field and win.view.last_range is not None:
            vmin, vmax = win.view.last_range.vmin, win.view.last_range.vmax
        else:
            rmode = "sequence"
    style = FrameStyle(field=key, configuration=st.view.configuration, background=st.view.background,
                       colormap=d.colormap, levels=d.levels, opacity=st.view.opacity,
                       vmin=vmin, vmax=vmax,
                       range_mode="frame" if rmode == "frame" else "sequence", colorbar=dlg.colorbar.isChecked(),
                       title=dlg.title.isChecked(), crop_to_roi=dlg.crop.isChecked(), dpi=dlg.dpi.value())
    frames = [k for k in range(dlg.first.value(), dlg.last.value() + 1, dlg.every.value())
              if st.provider.has_frame(key, k)]
    if not frames:
        win.error("Nothing to export", "No computed frames in that range for this field.")
        return
    from ..viz.fields import FieldProvider

    prov = FieldProvider(st.series, st.strain)  # a private cache: the worker must not share the GUI's
    seq, units, out, fps = st.sequence, st.units, Path(dlg.path.text()), dlg.fps.value()
    if dlg.animation:
        job = lambda cb: export_animation(out, prov, seq, frames, style, units, fps=fps, callbacks=cb)  # noqa: E731
        done = lambda p: win.log(f"saved animation {p}")  # noqa: E731
    else:
        job = lambda cb: export_frames(out, prov, seq, frames, replace(style), units, dlg.ext, callbacks=cb)  # noqa: E731
        done = lambda ps: win.log(f"saved {len(ps)} images in {out}")  # noqa: E731
    win._start(TaskWorker(job, "pictures"), done, "Picture export")


# ======================================================================================= probes
def export_probes_dialog(win: "MainWindow") -> None:
    from ..export.tables import probe_table_csv
    from ..post.probes import extensometer, probe_series

    st = win.state
    if st.provider is None or not st.probes:
        win.error("Nothing to export", "Run the correlation and place probes first.")
        return
    path, _ = QFileDialog.getSaveFileName(win, "Probe table", str(_default_dir(win) / "raftcorr_probes.csv"), "*.csv")
    if not path:
        return
    metric = win.right.probes.metric_combo.currentText()
    units = st.units
    cols: dict[str, np.ndarray] = {}
    for spec in st.provider.available():
        f = units.factor(spec)
        unit = units.unit(spec)
        for p in st.probes:
            name = f"{p.label}_{spec.key}" + (f"_{metric}" if p.kind != "point" else "") + f" [{unit}]"
            cols[name] = probe_series(p, st.provider, spec.key, metric) * f
    for p in st.probes:
        if p.kind == "line":
            L, e = extensometer(p, st.series)
            cols[f"{p.label}_length [px]"] = L
            cols[f"{p.label}_extensometer_strain [-]"] = e
    probe_table_csv(path, list(range(st.series.n_frames)), cols, units.frame_rate)
    win.log(f"saved the probe table ({len(cols)} columns) to {path}")
