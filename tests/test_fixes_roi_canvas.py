"""Region of interest, canvas and image list: regressions found while documenting the GUI."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

H, W = 48, 64


@pytest.fixture(autouse=True)
def no_modal_boxes(monkeypatch):
    """Record modal boxes instead of blocking; ``question`` answers ``answers[0]`` (default Yes)."""
    shown: list = []
    answers: list = []

    def _box(name):
        def show(*a, **k):
            shown.append((name, a[1:3]))
            if name == "question":
                return answers.pop(0) if answers else QMessageBox.StandardButton.Yes
            return QMessageBox.StandardButton.Ok
        return staticmethod(show)

    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, _box(name))
    shown_answers = (shown, answers)
    return shown_answers


@pytest.fixture(autouse=True)
def fresh_state(qapp):
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.setOrganizationName("RAFTcorr-tests")
    QCoreApplication.setApplicationName("RAFTcorr-tests")
    from raftcorr.gui.app_state import AppState

    AppState._instance = None
    yield
    AppState._instance = None


@pytest.fixture()
def win(qtbot):
    from raftcorr.gui.app import MainWindow

    w = MainWindow()
    w._on_open_strain_window = lambda: None
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    yield w
    w._state.set_results(None)


def _folder(tmp_path: Path, names=("img1", "img2", "img3", "img4", "img5", "img6")) -> Path:
    d = tmp_path / "imgs"
    d.mkdir()
    rng = np.random.default_rng(3)
    for n in names:
        Image.fromarray(rng.integers(0, 255, (H, W), dtype=np.uint8)).save(d / f"{n}.png")
    return d


def _mask_file(path: Path, mask: np.ndarray) -> Path:
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(path)
    return path


def _box_mask(y0=10, y1=30, x0=10, x1=30) -> np.ndarray:
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True          # 400 px
    return m


def _logs(win) -> list:
    out: list = []
    win._state.log_message.connect(lambda msg, lvl: out.append((lvl, msg)))
    return out


# ---------------------------------------------------------------- 1. Invert / Save, stale buffer
def _frame5_with_imported_mask(win, tmp_path) -> np.ndarray:
    win._image_ctrl.load_folder(str(_folder(tmp_path)))
    st = win._state
    # Frame 1 was drawn first: the controller buffer holds frame 1's mask.
    win._on_draw_requested("rect", "add")
    win._roi_ctrl.add_rectangle(0, 0, 40, 40)
    st.set_frame_roi(0, win._roi_ctrl.mask.copy())
    st.roi_editing = False          # e.g. a run started since
    own = _box_mask()
    win._on_roi_import_for_frames({4: str(_mask_file(tmp_path / "m5.png", own))})
    st.set_current_frame(4)
    return own


def test_invert_acts_on_the_current_frames_own_mask(win, tmp_path):
    own = _frame5_with_imported_mask(win, tmp_path)
    win._on_roi_invert()
    np.testing.assert_array_equal(win._state.per_frame_rois[4], ~own)


def test_save_writes_the_current_frames_own_mask(win, tmp_path, monkeypatch):
    own = _frame5_with_imported_mask(win, tmp_path)
    out = tmp_path / "saved.png"
    from raftcorr.gui import app as app_mod

    monkeypatch.setattr(app_mod.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    win._on_roi_save()
    assert out.is_file()
    np.testing.assert_array_equal(np.asarray(Image.open(out)) > 127, own)


def test_invert_after_batch_import_uses_the_imported_mask(win, tmp_path, monkeypatch):
    win._image_ctrl.load_folder(str(_folder(tmp_path)))
    own = _box_mask()
    from raftcorr.gui.dialogs import batch_import_dialog as bid

    class _Dialog:
        DialogCode = bid.BatchImportDialog.DialogCode

        def __init__(self, *a, **k) -> None: ...
        def exec(self): return self.DialogCode.Accepted
        def load_masks(self, shape): return {0: _box_mask(0, 40, 0, 40), 2: own.copy()}
        def deleteLater(self) -> None: ...

    monkeypatch.setattr(bid, "BatchImportDialog", _Dialog)
    win._on_batch_import()
    win._state.set_current_frame(2)
    win._on_roi_invert()
    np.testing.assert_array_equal(win._state.per_frame_rois[2], ~own)


# ---------------------------------------------------------------- 7. Save default path
def test_save_suggests_a_path_outside_the_image_folder(win, tmp_path, monkeypatch):
    _frame5_with_imported_mask(win, tmp_path)
    seen: list = []
    from raftcorr.gui import app as app_mod

    monkeypatch.setattr(app_mod.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: seen.append(a[2]) or ("", "")))
    win._on_roi_save()
    suggested = Path(seen[0])
    folder = Path(win._state.image_folder)
    assert suggested.parent == folder.parent
    assert suggested.name == "imgs_roi_frame5.png"


# ---------------------------------------------------------------- 8. Unreadable mask file
def test_importing_an_unreadable_mask_is_reported(win, tmp_path):
    win._image_ctrl.load_folder(str(_folder(tmp_path)))
    bad = tmp_path / "broken_mask.png"
    bad.write_bytes(b"this is not an image")
    logs = _logs(win)
    win._on_roi_import(str(bad))
    warns = [m for lvl, m in logs if lvl == "warn"]
    assert any("broken_mask.png" in m for m in warns), logs
    assert 0 not in win._state.per_frame_rois


# ---------------------------------------------------------------- 2. Clear tooltip
def test_clear_tooltip_names_the_current_frame_only(win):
    tip = win._left_sidebar.roi_toolbar._btn_clear.toolTip()
    assert "all" not in tip.lower()
    assert "current frame" in tip.lower()


# ---------------------------------------------------------------- 3. Batch Import dialog
def _dialog(qtbot, image_names, mask_dir):
    from raftcorr.gui.dialogs.batch_import_dialog import BatchImportDialog

    dlg = BatchImportDialog([f"C:/imgs/{n}.png" for n in image_names])
    qtbot.addWidget(dlg)
    dlg._load_mask_files(str(mask_dir))
    return dlg


def _mask_dir(tmp_path, names) -> Path:
    d = tmp_path / "masks"
    d.mkdir()
    for n in names:
        _mask_file(d / f"{n}.png", _box_mask())
    return d


def test_available_masks_are_in_natural_order(qtbot, tmp_path):
    d = _mask_dir(tmp_path, ["mask10", "mask2", "mask1"])
    dlg = _dialog(qtbot, ["img1", "img2"], d)
    names = [dlg._mask_list.item(i).text() for i in range(dlg._mask_list.count())]
    assert names == ["mask1.png", "mask2.png", "mask10.png"]


def test_auto_match_pairs_equal_file_numbers(qtbot, tmp_path):
    d = _mask_dir(tmp_path, ["mask1", "mask6", "mask11"])
    dlg = _dialog(qtbot, ["image1", "image6", "image11"], d)
    dlg._auto_match()
    got = {k: Path(v).stem for k, v in dlg.get_assignments().items()}
    assert got == {0: "mask1", 1: "mask6", 2: "mask11"}


def test_auto_match_falls_back_to_the_index_without_image_numbers(qtbot, tmp_path):
    d = _mask_dir(tmp_path, ["mask0", "mask2"])
    dlg = _dialog(qtbot, ["ref", "def_a", "def_b"], d)
    dlg._auto_match()
    got = {k: Path(v).stem for k, v in dlg.get_assignments().items()}
    assert got == {0: "mask0", 2: "mask2"}


def test_batch_tooltips_count_frames_from_one(qtbot, tmp_path):
    from PySide6.QtWidgets import QPushButton

    d = _mask_dir(tmp_path, ["mask1"])
    dlg = _dialog(qtbot, ["img1"], d)
    tips = {b.text(): b.toolTip() for b in dlg.findChildren(QPushButton)}
    assert "frame 0" not in tips["Assign Sequential"]
    assert "frame 1" in tips["Assign Sequential"]
    assert "number" in tips["Auto-Match by Name"]


# ---------------------------------------------------------------- 4. Empty frame-1 ROI
def test_an_empty_frame1_roi_is_refused_before_the_run(win, tmp_path, monkeypatch):
    from raftcorr.gui.app_state import RunState
    from raftcorr.gui.controllers import pipeline_controller as pc
    from raftcorr.gui.controllers.pipeline_messages import PipelineMessages

    started: list = []
    monkeypatch.setattr(pc, "PipelineWorker", lambda *a, **k: started.append(a) or None)
    win._image_ctrl.load_folder(str(_folder(tmp_path)))
    st = win._state
    st.per_frame_rois[0] = np.zeros((H, W), bool)      # as Batch Import stores an all-black mask
    errors: list = []
    st.fatal_error.connect(lambda t, m: errors.append((t, m)))
    win._pipeline_ctrl.start()
    assert errors == [(PipelineMessages().cannot_start_title(), PipelineMessages().empty_roi())]
    assert not started and st.run_state == RunState.IDLE


# ---------------------------------------------------------------- 5. Canvas keys
def _canvas(win, tmp_path):
    win._image_ctrl.load_folder(str(_folder(tmp_path)))
    return win._canvas_area.canvas


def test_escape_cancels_a_picked_but_unstarted_shape(win, tmp_path, qtbot):
    canvas = _canvas(win, tmp_path)
    win._on_draw_requested("rect", "add")
    assert canvas._current_tool == "rect"
    finished: list = []
    canvas.drawing_finished.connect(lambda: finished.append(1))
    qtbot.keyClick(canvas, Qt.Key.Key_Escape)
    assert canvas._current_tool == "select" and finished


def test_escape_cancels_a_shape_in_progress(win, tmp_path, qtbot):
    canvas = _canvas(win, tmp_path)
    win._on_draw_requested("polygon", "add")
    from PySide6.QtCore import QPointF

    canvas._handle_draw_press(QPointF(5, 5))
    canvas._handle_draw_press(QPointF(30, 5))
    qtbot.keyClick(canvas, Qt.Key.Key_Escape)
    assert canvas._draw_state is None and canvas._current_tool == "select"
    assert 0 not in win._state.per_frame_rois


@pytest.mark.parametrize("key", [Qt.Key.Key_Return, Qt.Key.Key_Enter])
def test_enter_closes_a_polygon(win, tmp_path, qtbot, key):
    canvas = _canvas(win, tmp_path)
    win._on_draw_requested("polygon", "add")
    from PySide6.QtCore import QPointF

    for x, y in ((5, 5), (40, 5), (40, 30), (5, 30)):
        canvas._handle_draw_press(QPointF(x, y))
    qtbot.keyClick(canvas, key)
    assert canvas._draw_state is None and canvas._current_tool == "select"
    assert win._state.per_frame_rois[0].sum() > 0


def test_escape_still_leaves_the_seed_tool(win, tmp_path, qtbot):
    """The handler that used to be shadowed by the second ``keyPressEvent``."""
    canvas = _canvas(win, tmp_path)
    canvas.set_tool("seed")
    qtbot.keyClick(canvas, Qt.Key.Key_Escape)
    assert canvas._current_tool == "pan"


def test_picking_a_shape_gives_the_canvas_keyboard_focus(win, tmp_path, qtbot):
    canvas = _canvas(win, tmp_path)
    win.activateWindow()
    win._left_sidebar.roi_toolbar._btn_add.setFocus()
    win._on_draw_requested("rect", "add")
    qtbot.waitUntil(canvas.hasFocus, timeout=2000)


# ---------------------------------------------------------------- 6. Natural Sort
def _sort_folder(tmp_path) -> Path:
    return _folder(tmp_path, names=("img1", "img2", "img10"))


def test_natural_sort_asks_before_discarding_and_no_keeps_everything(win, tmp_path, no_modal_boxes):
    shown, answers = no_modal_boxes
    win._image_ctrl.load_folder(str(_sort_folder(tmp_path)))
    st = win._state
    st.set_frame_roi(0, _box_mask())
    files = list(st.image_files)
    cb = win._left_sidebar._natural_sort_cb
    answers.append(QMessageBox.StandardButton.No)
    cb.setChecked(False)
    assert [n for n, _ in shown] == ["question"]
    assert cb.isChecked() and win._image_ctrl.natural_sort
    assert st.image_files == files and 0 in st.per_frame_rois


def test_natural_sort_yes_reloads_in_the_new_order(win, tmp_path, no_modal_boxes):
    shown, answers = no_modal_boxes
    win._image_ctrl.load_folder(str(_sort_folder(tmp_path)))
    st = win._state
    st.set_frame_roi(0, _box_mask())
    answers.append(QMessageBox.StandardButton.Yes)
    win._left_sidebar._natural_sort_cb.setChecked(False)
    assert [n for n, _ in shown] == ["question"]
    assert [Path(f).stem for f in st.image_files] == ["img1", "img10", "img2"]
    assert not st.per_frame_rois


def test_natural_sort_without_anything_to_lose_does_not_ask(win, tmp_path, no_modal_boxes):
    shown, _ = no_modal_boxes
    win._image_ctrl.load_folder(str(_sort_folder(tmp_path)))
    win._left_sidebar._natural_sort_cb.setChecked(False)
    assert not shown
    assert [Path(f).stem for f in win._state.image_files] == ["img1", "img10", "img2"]


# ================================================== follow-up: image-list mask import pairs in order
def test_image_list_mask_import_pairs_files_in_natural_order(win, tmp_path, monkeypatch):
    """The file dialog may return the files in any order (selection order on some platforms):
    the selected frames take the masks in the natural order of their names."""
    from PySide6.QtWidgets import QFileDialog

    win._image_ctrl.load_folder(str(_folder(tmp_path)))
    files = [str(tmp_path / f"mask{k}.png") for k in (10, 2, 1)]
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: (files, "")))
    got: list = []
    lst = win._left_sidebar._image_list
    lst.roi_import_for_frames.connect(got.append)
    lst._tree.clearSelection()
    for row in (0, 1, 2):
        lst._tree.topLevelItem(row).setSelected(True)
    lst._import_roi_selected()
    assert got == [{0: files[2], 1: files[1], 2: files[0]}]
