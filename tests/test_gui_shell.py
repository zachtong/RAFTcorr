"""The pyALDIC-derived main window on RAFTcorr: state, controllers, rendering, runs, projects."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui

from raftcorr import synthetic as syn  # noqa: E402
from raftcorr.config import RunConfig, TrackingConfig  # noqa: E402
from raftcorr.engine.mapping import Grid  # noqa: E402
from raftcorr.results import DisplacementSeries, RunInfo  # noqa: E402
from raftcorr.schedule import FrameSchedule  # noqa: E402

H, W = 96, 128


@pytest.fixture(autouse=True)
def no_modal_boxes(monkeypatch):
    """Modal message boxes would block an offscreen test forever: record them instead."""
    from PySide6.QtWidgets import QMessageBox

    shown: list = []
    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(
            lambda *a, _n=name, **k: shown.append((_n, a[1:3])) or QMessageBox.StandardButton.Ok))
    return shown


@pytest.fixture(autouse=True)
def fresh_state(qapp):
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.setOrganizationName("RAFTcorr-tests")
    QCoreApplication.setApplicationName("RAFTcorr-tests")
    from raftcorr.gui.app_state import AppState

    AppState._instance = None
    yield
    AppState._instance = None


def _folder(tmp_path: Path, n: int = 3, size=(H, W)) -> Path:
    d = tmp_path / "imgs"
    d.mkdir()
    frames = syn.render_sequence(size[0], size[1], [syn.translation(1.5 * k, -0.5 * k) for k in range(1, n)], seed=4)
    for i, f in enumerate(frames):
        Image.fromarray(np.round(f).astype(np.uint8)).save(d / f"img{i + 1}.png")
    return d


def _series(n: int = 3) -> DisplacementSeries:
    g = Grid(20, 10, 80, 60, H, W)
    X, Y = g.coords()
    u = np.stack([0.01 * k * (X - 60) for k in range(n)]).astype(np.float32)
    v = np.stack([np.full(g.shape, -0.5 * k) for k in range(n)]).astype(np.float32)
    mask = np.ones(g.shape, bool)
    mask[25:35, 30:45] = False
    u[:, ~mask] = np.nan
    v[:, ~mask] = np.nan
    info = RunInfo("test", "RAFTcorr-s4", "0" * 64, 4, 20, "tile", "cpu", RunConfig().to_dict())
    res = np.where(mask, 1.0, np.nan)[None].repeat(n, 0).astype(np.float32)
    return DisplacementSeries(g, mask, u, v, np.ones(n, bool), tuple(f"f{i}" for i in range(n)),
                              FrameSchedule.accumulative(n), info, residual=res)


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


# ---------------------------------------------------------------------------------------- state
def test_run_config_follows_the_pyaldic_fields():
    from raftcorr.gui.app_state import AppState

    st = AppState.instance()
    assert st.run_config().tracking.mode == "accumulative" and st.run_config().iters == 20
    st.tracking_mode, st.inc_ref_mode = "incremental", "every_frame"
    assert st.run_config().tracking.mode == "incremental"
    st.inc_ref_mode, st.inc_ref_interval = "every_n", 7
    assert st.run_config().tracking == TrackingConfig(mode="every_n", every_n=7)
    st.inc_ref_mode, st.inc_custom_refs = "custom", [0, 3]
    assert st.run_config().tracking.key_frames == (0, 3)
    st.model, st.margin_px, st.memory_fraction = "s8", 300, 0.5
    cfg = st.run_config()
    assert cfg.model == "s8" and cfg.tiling.margin_px == 300 and cfg.tiling.memory_fraction == 0.5
    st2 = AppState()
    st2.apply_run_config(cfg)
    assert st2.run_config() == cfg


# ----------------------------------------------------------------------------------- controllers
def test_image_controller_natural_sort_and_frames(tmp_path, qtbot):
    from raftcorr.gui.app_state import AppState
    from raftcorr.gui.controllers.image_controller import ImageController

    d = _folder(tmp_path, n=3)
    (d / "img10.png").write_bytes((d / "img3.png").read_bytes())
    st = AppState.instance()
    ctrl = ImageController(st)
    assert ctrl.natural_sort
    ctrl.load_folder(str(d))
    assert [Path(f).name for f in st.image_files] == ["img1.png", "img2.png", "img3.png", "img10.png"]
    rgb = ctrl.read_image_rgb(1)
    assert rgb.shape == (H, W, 3) and rgb.dtype == np.uint8
    g = ctrl.read_image(1)
    assert 0.0 <= g.min() and g.max() <= 1.0
    ctrl.set_natural_sort(False)
    assert [Path(f).name for f in st.image_files][1] == "img10.png"


def test_image_controller_refuses_mixed_sizes(tmp_path, qtbot):
    from raftcorr.gui.app_state import AppState
    from raftcorr.gui.controllers.image_controller import ImageController

    d = _folder(tmp_path, n=2)
    Image.fromarray(np.zeros((10, 10), np.uint8)).save(d / "img3.png")
    st = AppState.instance()
    errors = []
    st.fatal_error.connect(lambda t, m: errors.append(m))
    ImageController(st).load_folder(str(d))
    assert st.image_files == [] and errors and "img3.png" in errors[0]


def test_roi_controller_shapes_without_opencv():
    from raftcorr.gui.controllers.roi_controller import ROIController

    c = ROIController((50, 60))
    c.add_rectangle(10, 10, 19, 19)
    assert c.mask.sum() == 100
    c.add_circle(40, 25, 5, mode="add")
    c.add_polygon([(0, 40), (10, 40), (10, 49)])
    c.add_rectangle(12, 12, 13, 13, mode="cut")
    assert not c.mask[12, 12] and c.mask[11, 11]
    c.stroke_segment(30, 5, 50, 5, 2)
    assert c.mask[5, 40]
    c.invert()
    assert not c.mask[11, 11]


def test_viz_controller_renders_reference_deformed_and_residual(qtbot):
    from raftcorr.gui.app_state import AppState
    from raftcorr.gui.controllers.viz_controller import VizController

    st = AppState.instance()
    st.set_results(_series())
    st.current_frame = 2
    viz = VizController()
    st.show_deformed = False
    st.display_field = "disp_u"
    img = viz.render_field(st)
    assert img is not None and img.pixmap.width() == 80 and (img.x0, img.y0, img.step) == (20, 10, 1)
    assert img.vmin < img.vmax and img.label.startswith("U")
    name, val, unit = viz.value_at(60.0, 20.0)
    assert name == "U" and abs(val) < 1e-6 and unit == "px"
    st.show_deformed = True
    img_d = viz.render_field(st)
    assert img_d is not None and img_d.y0 <= 10  # moved up by 1 px
    st.display_field = "residual"
    assert viz.render_field(st) is not None
    st.set_physical_units(True, 0.01, 10.0, "mm")
    st.display_field = "disp_v"
    st.show_deformed = False
    viz.clear_all()
    viz.render_field(st)
    assert viz.value_at(60.0, 20.0)[1] == pytest.approx(-0.01) and viz.value_at(60.0, 20.0)[2] == "mm"


def test_hide_untracked_off_fills_the_hole(qtbot):
    from raftcorr.gui.app_state import AppState
    from raftcorr.gui.controllers.viz_controller import VizController

    st = AppState.instance()
    s = _series()
    st.set_results(s)
    st.current_frame, st.show_deformed, st.display_field = 1, False, "disp_v"
    viz = VizController()
    viz.render_field(st)
    assert viz.value_at(20 + 35, 10 + 30) is None  # inside the hole of the ROI
    st.hide_untracked = False
    viz.render_field(st)
    # outside the ROI stays empty even when untracked points are filled
    assert viz.value_at(20 + 35, 10 + 30) is None


# ---------------------------------------------------------------------------------- main window
def test_main_window_builds_with_the_pyaldic_layout(win):
    titles = [a.text() for a in win.menuBar().actions()]
    assert titles[:2] == ["&File", "&Settings"]
    file_items = [a.text() for a in win.menuBar().actions()[0].menu().actions() if a.text()]
    assert "Open Image Folder…" in file_items and "Save Project" in file_items
    assert win._left_sidebar.workflow_panel is not None
    assert not win._left_sidebar.roi_toolbar._btn_refine.isVisible()


def test_loading_and_drawing_an_roi(win, tmp_path, qtbot):
    d = _folder(tmp_path)
    win._image_ctrl.load_folder(str(d))
    st = win._state
    assert len(st.image_files) == 3 and win._roi_ctrl is not None
    win._on_draw_requested("rect", "add")
    assert st.roi_editing
    win._roi_ctrl.add_rectangle(20, 20, 100, 70)
    st.set_frame_roi(0, win._roi_ctrl.mask.copy())
    assert st.roi_mask.sum() == 81 * 51
    win._on_roi_invert()
    assert st.roi_mask[0, 0]


class _FakePipeline:
    """Stands in for raftcorr.engine.pipeline.Pipeline: returns a prepared series."""

    series = None
    gate: threading.Event | None = None
    notes: tuple = ()

    def __init__(self, cfg, cb, loaded) -> None:
        self.cb = cb

    def run(self, seq, roi, frame_masks=None, storage_dir=None):
        from dataclasses import replace

        from raftcorr.callbacks import check_stop

        for i in range(3):
            if _FakePipeline.gate is not None:
                _FakePipeline.gate.wait(5)
            check_stop(self.cb)
            self.cb.progress((i + 1) / 3, f"frame {i + 1}")
            time.sleep(0.01)
        s = _FakePipeline.series
        return replace(s, info=replace(s.info, notes=_FakePipeline.notes))


@pytest.fixture()
def fake_run(monkeypatch):
    import raftcorr.engine.pipeline as pl
    from raftcorr.gui.controllers import pipeline_controller as pc

    monkeypatch.setattr(pl, "Pipeline", _FakePipeline)
    monkeypatch.setattr(pc.PipelineWorker, "_load_model", lambda self, cb: object())
    _FakePipeline.series = _series()
    _FakePipeline.gate = None
    _FakePipeline.notes = ()
    yield _FakePipeline


def _wait_done(qtbot, st):
    from raftcorr.gui.app_state import RunState

    qtbot.waitUntil(lambda: st.run_state not in (RunState.RUNNING, RunState.PAUSED), timeout=15000)


def test_run_publishes_results_and_quality_warning(win, tmp_path, qtbot, fake_run):
    from raftcorr.gui.app_state import RunState

    win._image_ctrl.load_folder(str(_folder(tmp_path)))
    st = win._state
    st.set_frame_roi(0, np.ones((H, W), bool))
    warnings = []
    st.run_warning.connect(lambda *a: warnings.append(a))
    from dataclasses import replace

    series = fake_run.series
    res = np.array(series.residual)
    res[1][np.isfinite(res[1])] = 3.0
    res[2][np.isfinite(res[2])] = 9.0                # a jump on the third frame
    fake_run.series = replace(series, residual=res)
    win._pipeline_ctrl.start()
    _wait_done(qtbot, st)
    assert st.run_state == RunState.DONE and st.results is not None and st.current_frame == 2
    assert len(warnings) == 1
    title, text, informative, details = warnings[0]
    assert title == "Tracking Quality Warning" and "may not be tracked" in text
    assert "from frame 3 on" in informative and "9.0" in informative and "3.0" in informative
    assert "incremental" in informative            # the advice is shown, not hidden
    assert "Frames with a high residual: 3" in details and "Residual field" in details
    win.close()


def test_pause_resume_and_cancel(win, tmp_path, qtbot, fake_run):
    from raftcorr.gui.app_state import RunState

    win._image_ctrl.load_folder(str(_folder(tmp_path)))
    st = win._state
    st.set_frame_roi(0, np.ones((H, W), bool))
    fake_run.gate = threading.Event()
    win._pipeline_ctrl.start()
    win._pipeline_ctrl.pause()
    assert st.run_state == RunState.PAUSED
    win._pipeline_ctrl.resume()
    assert st.run_state == RunState.RUNNING
    win._pipeline_ctrl.stop()
    fake_run.gate.set()
    _wait_done(qtbot, st)
    assert st.results is None and st.run_state == RunState.IDLE


def test_start_without_roi_reports_a_fatal_error(win, tmp_path, qtbot):
    win._image_ctrl.load_folder(str(_folder(tmp_path)))
    errors = []
    win._state.fatal_error.connect(lambda t, m: errors.append(t))
    win._pipeline_ctrl.start()
    assert errors


def test_project_round_trip_through_the_bridge(win, tmp_path, qtbot):
    from raftcorr.gui.app_state import AppState
    from raftcorr.gui.project_bridge import apply_project, project_from_state
    from raftcorr.io.project import load_project, save_project

    d = _folder(tmp_path)
    win._image_ctrl.load_folder(str(d))
    st = win._state
    roi = np.zeros((H, W), bool)
    roi[10:70, 20:100] = True
    st.set_frame_roi(0, roi)
    st.set_frame_roi(2, roi[::-1])
    st.set_param("model", "s2")
    st.set_physical_units(True, 0.02, 25.0, "mm")
    st.set_results(_series())
    st.display_field = "disp_v"
    from raftcorr.analysis.load_data import LoadData, LoadSync, LoadTable

    st.set_load_data(LoadData(LoadTable(("frame", "load"), (np.array([1.0, 2, 3]), np.array([0.0, 5.5, np.nan]))),
                              LoadSync(mode="frame", load_column="load", frame_column="frame"), area_mm2=12.0))
    path = tmp_path / "p.raftcorr"
    save_project(path, project_from_state(st))
    prj = load_project(path)
    AppState._instance = None
    st2 = AppState.instance()
    from raftcorr.gui.controllers.image_controller import ImageController

    apply_project(st2, ImageController(st2), prj, d, path)
    assert st2.model == "s2" and st2.display_field == "disp_v" and st2.use_physical_units
    assert st2.pixel_size == pytest.approx(0.02) and st2.frame_rate == pytest.approx(25.0)
    np.testing.assert_array_equal(st2.per_frame_rois[2], roi[::-1])
    np.testing.assert_allclose(st2.results.v, _series().v, equal_nan=True)
    ld = st2.load_data
    assert ld is not None and ld.area_mm2 == 12.0 and ld.sync.load_column == "load"
    np.testing.assert_array_equal(ld.table.column("load"), [0.0, 5.5, np.nan])


@pytest.mark.gpu
@pytest.mark.weights
def test_real_run_with_s8(win, tmp_path, qtbot, weights, require_cuda):
    weights("s8")
    win._image_ctrl.load_folder(str(_folder(tmp_path, n=3, size=(160, 192))))
    st = win._state
    roi = np.zeros((160, 192), bool)
    roi[40:120, 50:150] = True
    st.set_frame_roi(0, roi)
    st.set_param("model", "s8")
    win._pipeline_ctrl.start()
    _wait_done(qtbot, st)
    assert st.results is not None and st.results.done.all()
    assert abs(np.nanmedian(st.results.u[2][st.results.mask]) - 3.0) < 0.15
    st.set_display_field("disp_u")
    assert win._viz_ctrl.render_field(st) is not None


def test_cli_gui_hands_a_project_path_to_the_application(monkeypatch, tmp_path):
    """The .raftcorr file association runs ``python -m raftcorr gui "<file>"``."""
    import sys

    import raftcorr.gui.app as app
    from raftcorr.cli.main import main as cli_main
    from raftcorr.gui import file_association

    monkeypatch.setattr(sys, "argv", ["raftcorr"])
    seen: dict = {}
    monkeypatch.setattr(app, "main", lambda: seen.setdefault("argv", list(sys.argv)))
    project = tmp_path / "test.raftcorr"
    project.write_bytes(b"")
    assert cli_main(["gui", str(project)]) == 0
    assert app._path_from_argv(seen["argv"]) == project
    assert file_association.open_command().endswith('-m raftcorr gui "%1"')


def test_deleting_a_frame_keeps_rois_probes_and_project(win, tmp_path, qtbot):
    """Removing frames from the list is not loading new images: only the results go."""
    from raftcorr.analysis.probes import PointGeom

    win._image_ctrl.load_folder(str(_folder(tmp_path, n=4)))
    st = win._state
    roi0 = np.zeros((H, W), bool)
    roi0[10:70, 20:100] = True
    roi3 = np.zeros((H, W), bool)
    roi3[20:60, 30:90] = True
    st.set_frame_roi(0, roi0)
    st.set_frame_roi(3, roi3)
    st.probes.add("point", PointGeom(50.0, 40.0))
    st.project_path = tmp_path / "p.raftcorr"
    st.set_results(_series(4))
    names = [Path(f).name for f in st.image_files]

    lst = win._left_sidebar._image_list
    lst._tree.clearSelection()
    lst._tree.topLevelItem(1).setSelected(True)
    lst._delete_selected()

    assert [Path(f).name for f in st.image_files] == [names[0], names[2], names[3]]
    assert len(st.sequence) == 3
    assert set(st.per_frame_rois) == {0, 2}                      # frame 3 is frame 2 now
    np.testing.assert_array_equal(st.per_frame_rois[2], roi3)
    assert len(st.probes) == 1 and st.project_path == tmp_path / "p.raftcorr"
    assert st.results is None and st.strain_results is None      # frame indices changed


def test_new_images_during_a_run_cancel_it_and_drop_its_results(win, tmp_path, qtbot, fake_run):
    from raftcorr.gui.app_state import RunState

    win._image_ctrl.load_folder(str(_folder(tmp_path)))
    st = win._state
    st.set_frame_roi(0, np.ones((H, W), bool))
    fake_run.gate = threading.Event()
    logs: list = []
    st.log_message.connect(lambda m, lvl: logs.append(m))
    win._pipeline_ctrl.start()
    (tmp_path / "x").mkdir()
    win._image_ctrl.load_folder(str(_folder(tmp_path / "x", n=2)))   # other images, mid-run
    fake_run.gate.set()
    qtbot.waitUntil(lambda: win._pipeline_ctrl._worker is None, timeout=15000)
    assert st.results is None and st.run_state == RunState.IDLE
    assert any("images changed" in m for m in logs)


def test_a_stale_worker_finishing_late_leaves_the_current_run_alone(win, qtbot):
    """A cancelled worker's queued finish signal must not take the new worker's place."""
    ctrl = win._pipeline_ctrl

    class _W:
        def wait(self, ms=0):
            return True

        def isRunning(self):
            return False

    current, stale = _W(), _W()
    ctrl._worker = current
    ctrl._on_finished(stale, _series())
    assert ctrl._worker is current and win._state.results is None
    ctrl._worker = None


def test_a_frame_lost_entirely_draws_nothing_and_reads_nothing(qtbot):
    """Done but no finite point (tracking lost everywhere): the deformed view is empty, no error."""
    from raftcorr.gui.app_state import AppState
    from raftcorr.gui.controllers.viz_controller import VizController

    s = _series()
    s.u[2] = np.nan
    s.v[2] = np.nan
    st = AppState.instance()
    st.set_results(s)
    st.current_frame, st.show_deformed, st.display_field = 2, True, "disp_u"
    viz = VizController()
    viz.render_field(st)
    assert viz.value_at(60.0, 30.0) is None


def test_garbage_is_collected_in_the_gui_thread_only(qapp):
    import gc
    import weakref

    from raftcorr.gui.gc_guard import MainThreadGarbageCollector

    was_enabled = gc.isenabled()
    guard = MainThreadGarbageCollector(interval_ms=60_000)
    try:
        assert not gc.isenabled()

        class Node:
            pass

        node = Node()
        node.me = node                      # a reference cycle: only the cyclic collector frees it
        ref = weakref.ref(node)
        del node
        keep = [[] for _ in range(gc.get_threshold()[0] + 10)]   # push generation 0 past its threshold
        assert ref() is not None
        guard.check()
        assert ref() is None
        del keep
    finally:
        guard.stop()
        (gc.enable if was_enabled else gc.disable)()


def test_config_card_follows_the_sidebar(win, tmp_path):
    card = win._canvas_area._config_overlay
    assert not card.isVisible()                       # fewer than two images
    win._image_ctrl.load_folder(str(_folder(tmp_path)))
    st = win._state
    assert card.isVisible()
    assert (card._mode_lbl.text(), card._model_lbl.text(), card._iters_lbl.text()) == \
        ("Accumulative", "RAFTcorr-s4", "20")
    st.tracking_mode, st.inc_ref_mode, st.inc_ref_interval = "incremental", "every_n", 5
    st.set_param("model", "s8")
    st.set_param("iters", 12)
    assert (card._mode_lbl.text(), card._model_lbl.text(), card._iters_lbl.text()) == \
        ("Incremental, every 5 frames", "RAFTcorr-s8", "12")


def test_progress_label_shows_the_frame_as_the_engine_reports_it(win):
    sidebar = win._right_sidebar
    sidebar._on_progress(0.37, "Frame 4/10 (reference 1): 12 s left")
    assert sidebar._progress_label.text() == "37%  —  Frame 4/10"
    sidebar._on_progress(0.40, "loading")                      # keeps the last frame
    assert sidebar._progress_label.text() == "40%  —  Frame 4/10"
