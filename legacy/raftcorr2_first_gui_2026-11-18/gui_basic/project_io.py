"""Saving and opening project files (.raftcorr) from the GUI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QFileDialog, QMessageBox

from ..io.images import ImageSequence
from ..io.project import PROJECT_EXTENSION, Project, load_project, save_project
from ..post.probes import Probe
from .export_dialogs import from_physical, to_physical
from .state import ViewSettings
from .workers import TaskWorker

if TYPE_CHECKING:
    from .main_window import MainWindow


def project_from_state(win: "MainWindow") -> Project:
    st = win.state
    return Project(
        image_folder=str(st.image_folder) if st.image_folder else None,
        image_files=tuple(st.sequence.names) if st.sequence else (),
        natural_sort=st.natural_sort,
        run_config=st.run_config,
        strain_config=st.strain_config,
        roi_masks=dict(st.roi_masks),
        view={**st.view.to_dict(), "results_stale": st.is_stale},
        probes=[p.to_dict() for p in st.probes],
        units=to_physical(st.units),
        series=st.series,
        strain=st.strain,
    )


def save_project_dialog(win: "MainWindow", ask: bool = False) -> None:
    st = win.state
    path = st.project_path
    if ask or path is None:
        start = str(path or (st.image_folder or Path.home()) / f"project{PROJECT_EXTENSION}")
        name, _ = QFileDialog.getSaveFileName(win, "Save project", start, f"RAFTcorr project (*{PROJECT_EXTENSION})")
        if not name:
            return
        path = Path(name).with_suffix(PROJECT_EXTENSION)
    project = project_from_state(win)

    def done(p) -> None:
        win.remember(Path(p))
        st.project_path = Path(p)
        st.project_path_changed.emit(str(p))
        win.setWindowTitle(f"RAFTcorr - {Path(p).name}")
        win.log(f"saved project {p}")

    win._start(TaskWorker(lambda cb: save_project(path, project, cb), "save project"), done, "Saving the project")


def open_project_dialog(win: "MainWindow") -> None:
    start = str(win.settings.value("last_project_dir", str(Path.home())))
    name, _ = QFileDialog.getOpenFileName(win, "Open project", start, f"RAFTcorr project (*{PROJECT_EXTENSION})")
    if name:
        win.settings.setValue("last_project_dir", str(Path(name).parent))
        load_project_into(win, Path(name))


def load_project_into(win: "MainWindow", path: Path) -> None:
    win._start(TaskWorker(lambda cb: load_project(path), "open project"), lambda prj: apply_project(win, prj, path),
               "Opening the project")


def apply_project(win: "MainWindow", prj: Project, path: Path) -> None:
    st = win.state
    folder = Path(prj.image_folder) if prj.image_folder else None
    if folder is None or not folder.is_dir():
        QMessageBox.warning(win, "Images not found",
                            f"The image folder of this project was not found:\n{prj.image_folder}\n\nChoose its new location.")
        chosen = QFileDialog.getExistingDirectory(win, "Image folder of the project", str(path.parent))
        folder = Path(chosen) if chosen else None
    if folder is None:
        win.error("Project not opened", "The project needs its images.")
        return
    files = [folder / f for f in prj.image_files] if prj.image_files else None
    try:
        seq = ImageSequence(files) if files else ImageSequence.from_folder(folder, natural=prj.natural_sort)
    except Exception as exc:
        win.error("Cannot read the images of the project", str(exc))
        return
    st.natural_sort = prj.natural_sort
    win.left.images.natural.blockSignals(True)
    win.left.images.natural.setChecked(prj.natural_sort)
    win.left.images.natural.blockSignals(False)
    st.set_sequence(seq, folder)
    for k, m in sorted(prj.roi_masks.items()):
        if m.shape == seq.shape:
            st.set_roi(k, m, record=False)
    st.roi_history.clear()
    st.set_run_config(prj.run_config)
    st.set_strain_config(prj.strain_config)
    st.set_units(from_physical(prj.units))
    if prj.series is not None:
        st.set_results(prj.series, prj.strain)
    try:
        st.set_view(ViewSettings.from_dict(prj.view))
    except (TypeError, ValueError, KeyError) as exc:
        win.log(f"view settings of the project ignored ({exc})", "warning")
    if prj.series is not None and prj.view.get("results_stale"):
        st.mark_stale()
    st.probes = [Probe.from_dict(d) for d in prj.probes]
    st.probes_changed.emit()
    st.project_path = path
    st.project_path_changed.emit(str(path))
    win.setWindowTitle(f"RAFTcorr - {path.name}")
    for w in prj.warnings:
        win.log(w, "warning")
    win.log(f"opened project {path}")
