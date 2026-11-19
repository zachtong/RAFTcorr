"""Workflow and parameter panels, projects, the model suggestion and the menus (bug fixes)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui

from raftcorr import synthetic as syn  # noqa: E402
from raftcorr.config import RunConfig, TilingConfig, TrackingConfig  # noqa: E402

H, W = 64, 80


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


def _folder(tmp_path: Path, n: int = 5, name: str = "imgs") -> Path:
    d = tmp_path / name
    d.mkdir()
    frames = syn.render_sequence(H, W, [syn.translation(0.5 * k, 0.0) for k in range(1, n)], seed=3)
    for i, f in enumerate(frames):
        Image.fromarray(np.round(f).astype(np.uint8)).save(d / f"img{i + 1}.png")
    return d


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


@pytest.fixture()
def logs(win):
    out: list[tuple[str, str]] = []
    win._state.log_message.connect(lambda m, lvl: out.append((m, lvl)))
    return out


def _panels(win):
    from raftcorr.gui.widgets.param_panel import AdvancedPanel
    from raftcorr.gui.widgets.workflow_type_panel import WorkflowTypePanel

    return win.findChild(WorkflowTypePanel), win.findChild(AdvancedPanel)


def _sync_run(win):
    """Run ``_run_task`` jobs synchronously, recording their progress messages."""
    from raftcorr.callbacks import FunctionCallbacks

    messages: list[str] = []

    def run(title, fn, on_done):
        on_done(fn(FunctionCallbacks(progress=lambda f, m="": messages.append(m))))

    win._run_task = run
    return messages


# ------------------------------------------------------------ 1. widgets follow an opened project
def test_opening_a_project_updates_every_parameter_widget(win, tmp_path):
    from raftcorr.gui.project_bridge import project_from_state
    from raftcorr.io.project import save_project

    d = _folder(tmp_path, n=12)
    win._image_ctrl.load_folder(str(d))
    st = win._state
    st.set_frame_roi(0, np.ones((H, W), bool))
    st.apply_run_config(RunConfig(
        corr_mode="volume", tracking=TrackingConfig(mode="every_n", every_n=9),
        tiling=TilingConfig(margin_px=300, memory_fraction=0.5, blend_px=10)))
    path = save_project(tmp_path / "p.raftcorr", project_from_state(st, include_results=False))
    st.apply_run_config(RunConfig())                    # back to the defaults before reopening
    _sync_run(win)
    win.open_project_path(str(path))

    wf, adv = _panels(win)
    assert st.inc_ref_interval == 9 and st.margin_px == 300
    assert wf._interval_spin.value() == 9
    assert not adv._margin_auto.isChecked() and adv._margin.isEnabled()
    assert adv._margin.value() == 300
    assert adv._mem.value() == pytest.approx(0.5)
    assert adv._blend.value() == 10
    assert adv._corr.currentData() == "volume"
    assert st.run_config().tiling.margin_px == 300      # the sync wrote nothing back


def test_advanced_panel_follows_the_automatic_margin_and_custom_frames(win, tmp_path):
    win._image_ctrl.load_folder(str(_folder(tmp_path, n=12)))
    st = win._state
    wf, adv = _panels(win)
    st.apply_run_config(RunConfig(tiling=TilingConfig(margin_px=300)))
    assert not adv._margin_auto.isChecked()
    st.apply_run_config(RunConfig(tracking=TrackingConfig(mode="custom", key_frames=(3, 7))))
    assert adv._margin_auto.isChecked() and not adv._margin.isEnabled()
    assert st.margin_px is None
    assert wf._custom_edit.text() == "4, 8" and wf._custom_edit.isVisibleTo(wf)


# ------------------------------------------------------------ 2. custom reference frames are checked
@pytest.mark.parametrize("attr, keys, expected", [("_mem", "0.35", 0.35), ("_blend", "24", 24)])
def test_typing_in_an_advanced_spin_box_is_not_rewritten_by_the_sync(qtbot, attr, keys, expected):
    """Every keystroke sets the parameter and the state answers with params_changed: the sync
    must not set the spin box being typed in again (Qt reformats it and moves the cursor)."""
    from PySide6.QtTest import QTest

    from raftcorr.gui.app_state import AppState
    from raftcorr.gui.widgets.param_panel import AdvancedPanel

    panel = AdvancedPanel()
    qtbot.addWidget(panel)
    panel.show()
    box = getattr(panel, attr)
    box.setFocus()
    box.selectAll()
    QTest.keyClicks(box, keys)
    st = AppState.instance()
    assert box.value() == pytest.approx(expected)
    assert (st.memory_fraction if attr == "_mem" else st.blend_px) == pytest.approx(expected)


def _enter(panel, text):
    panel._custom_edit.setText(text)
    panel._on_custom_refs_changed()


def test_the_last_frame_is_refused_as_a_reference(win, tmp_path, logs):
    win._image_ctrl.load_folder(str(_folder(tmp_path, n=5)))
    st = win._state
    wf, _ = _panels(win)
    st.set_param("tracking_mode", "incremental")
    st.set_param("inc_ref_mode", "custom")
    _enter(wf, "2, 4")                                 # frame 4 of 5 is the last valid reference
    assert st.inc_custom_refs == [1, 3] and not wf.custom_refs_invalid()
    _enter(wf, "3, 5")
    assert st.inc_custom_refs == [1, 3]                # the run keeps the last valid frames
    assert wf.custom_refs_invalid()
    warn = [m for m, lvl in logs if lvl == "warn"]
    assert warn and "1 to 4" in warn[-1] and "5" in warn[-1]
    _enter(wf, "9")
    assert st.inc_custom_refs == [1, 3] and wf.custom_refs_invalid()
    _enter(wf, "3")
    assert st.inc_custom_refs == [2] and not wf.custom_refs_invalid()


def test_a_refused_reference_list_stops_the_run_instead_of_running_the_old_one(win, tmp_path,
                                                                              no_modal_boxes):
    import numpy as np

    win._image_ctrl.load_folder(str(_folder(tmp_path, n=5)))
    st = win._state
    st.set_frame_roi(0, np.ones((H, W), bool))
    wf, _ = _panels(win)
    st.set_param("tracking_mode", "incremental")
    st.set_param("inc_ref_mode", "custom")
    _enter(wf, "2")
    _enter(wf, "9")                                     # refused: frame 4 is the last valid one
    win._pipeline_ctrl.start()
    assert win._pipeline_ctrl._worker is None
    assert no_modal_boxes and "1 to 4" in str(no_modal_boxes[-1])
    st.set_param("inc_ref_mode", "every_frame")         # another mode does not use the list


def test_fewer_images_drop_references_beyond_the_new_last_frame(win, tmp_path, logs):
    win._image_ctrl.load_folder(str(_folder(tmp_path, n=12)))
    st = win._state
    wf, _ = _panels(win)
    st.set_param("tracking_mode", "incremental")
    st.set_param("inc_ref_mode", "custom")
    _enter(wf, "4, 11")
    assert st.inc_custom_refs == [3, 10]
    win._image_ctrl.load_folder(str(_folder(tmp_path, n=5, name="few")))
    assert st.inc_custom_refs == [3]                   # 11 is beyond frame 4, the last valid one
    assert wf._custom_edit.text() == "4"
    assert any("11" in m for m, lvl in logs if lvl == "warn")
    from raftcorr.schedule import FrameSchedule

    cfg = st.run_config().tracking
    FrameSchedule.from_mode(cfg.mode, len(st.image_files), cfg.every_n, cfg.key_frames)  # no error


def test_a_project_with_an_out_of_range_reference_never_reaches_the_run(win, tmp_path, logs):
    win._image_ctrl.load_folder(str(_folder(tmp_path, n=5)))
    st = win._state
    st.apply_run_config(RunConfig(tracking=TrackingConfig(mode="custom", key_frames=(2, 4))))
    assert st.inc_custom_refs == [2]
    assert any(lvl == "warn" for _m, lvl in logs)


def test_the_engine_says_which_numbering_its_key_frames_use():
    from raftcorr.schedule import FrameSchedule

    with pytest.raises(ValueError, match="0-based"):
        FrameSchedule.custom([11], 12)


# ------------------------------------------------------------ 3. model tooltips follow the weights
def test_model_tooltips_are_refreshed_when_the_list_opens(win, monkeypatch):
    from PySide6.QtCore import Qt

    from raftcorr.models import zoo

    wf, _ = _panels(win)
    combo = wf._model
    s4 = combo.findData("s4")
    monkeypatch.setattr(zoo, "available_models", lambda *a, **k: {key: None for key in zoo.MODELS})
    combo.showPopup()
    combo.hidePopup()
    assert "Weights not installed." in combo.itemData(s4, Qt.ItemDataRole.ToolTipRole)
    monkeypatch.setattr(zoo, "available_models",
                        lambda *a, **k: {key: (Path("x.pth") if key == "s4" else None) for key in zoo.MODELS})
    combo.showPopup()
    combo.hidePopup()
    assert "Weights not installed." not in combo.itemData(s4, Qt.ItemDataRole.ToolTipRole)
    assert "Weights not installed." in combo.itemData(combo.findData("s1"), Qt.ItemDataRole.ToolTipRole)


# ------------------------------------------------------------ 4. Suggest a Model
class _FakePipeline:
    """s8 stand-in: frame k moved k px to the right of frame 0."""

    pairs: list = []

    def __init__(self, *a, **k) -> None:
        pass

    def solve_pair(self, ref, deformed, roi):
        r, k = int(ref[0, 0]), int(deformed[0, 0])
        _FakePipeline.pairs.append((r, k))
        field = np.zeros((2, *roi.shape), np.float32)
        field[0] = float(k - r)
        return field, 0, 0


@pytest.fixture()
def fake_survey(win, tmp_path, monkeypatch):
    import raftcorr.engine.pipeline as pipeline_mod
    import raftcorr.gui.support.model_weights as mw
    import raftcorr.models as models

    monkeypatch.setattr(mw, "ensure_weights", lambda *a, **k: Path("s8.pth"))
    monkeypatch.setattr(models, "load_model", lambda *a, **k: object())
    monkeypatch.setattr(pipeline_mod, "Pipeline", _FakePipeline)
    _FakePipeline.pairs = []
    win._image_ctrl.load_folder(str(_folder(tmp_path, n=12)))
    st = win._state
    # Frame k holds the value k, so the fake pipeline can tell which pair it is given.
    seq = [np.full((H, W), float(k), np.float32) for k in range(12)]
    st.sequence = type("Seq", (), {"__len__": lambda s: 12, "__getitem__": lambda s, i: seq[i],
                                   "shape": (H, W)})()
    return _sync_run(win)


def test_accumulative_survey_compares_with_frame_1_and_shows_1_based_progress(win, fake_survey):
    win._on_suggest_model()
    assert all(r == 0 for r, _k in _FakePipeline.pairs)
    assert "Survey: frame 12 against frame 1" in fake_survey
    assert not any("frame 0" in m for m in fake_survey)
    st = win._state
    assert "frames 3, 6, 12" in st.model_advice and st.model == "s1"


def test_incremental_survey_measures_the_steps_of_the_schedule(win, fake_survey):
    st = win._state
    st.apply_run_config(RunConfig(tracking=TrackingConfig(mode="every_n", every_n=5)))
    win._on_suggest_model()
    # Every 5 frames: references 0, 5, 10 (0-based); their spans end at frames 5, 10 and 11.
    assert sorted(_FakePipeline.pairs) == [(0, 5), (5, 10), (10, 11)]
    assert "Survey: frame 6 against frame 1" in fake_survey
    assert "5.0 px" in st.model_advice
    assert "1 → 6, 6 → 11, 11 → 12" in st.model_advice
    assert "reference" in st.model_advice


def test_incremental_every_frame_surveys_consecutive_frames(win, fake_survey):
    st = win._state
    st.apply_run_config(RunConfig(tracking=TrackingConfig(mode="incremental")))
    win._on_suggest_model()
    assert _FakePipeline.pairs and all(k == r + 1 for r, k in _FakePipeline.pairs)
    assert "1.0 px" in st.model_advice


def test_survey_pairs_for_every_mode():
    from raftcorr.models.advisor import sample_frames, survey_pairs
    from raftcorr.schedule import FrameSchedule

    assert survey_pairs(FrameSchedule.accumulative(12)) == [(0, k) for k in sample_frames(12)]
    assert survey_pairs(FrameSchedule.custom([3, 7], 12)) == [(0, 3), (3, 7), (7, 11)]
    steps = survey_pairs(FrameSchedule.incremental(12))
    assert len(steps) == 3 and all(k == r + 1 for r, k in steps) and (10, 11) in steps


def test_survey_progress_text_is_1_based_by_default():
    from raftcorr.callbacks import FunctionCallbacks
    from raftcorr.models.advisor import survey

    seen: list[str] = []
    seq = [np.full((H, W), float(k), np.float32) for k in range(3)]
    survey(_FakePipeline(), seq, np.ones((H, W), bool), [(0, 2)],
           FunctionCallbacks(progress=lambda f, m="": seen.append(m)))
    assert "survey: frame 3 against frame 1" in seen


# ------------------------------------------------------------ 5. exclusive Settings menus
def _menu(win, title):
    for act in win.menuBar().actions():
        for sub in act.menu().actions():
            if sub.menu() is not None and sub.text() == title:
                return sub.menu()
    raise AssertionError(title)


def test_the_language_menu_has_exactly_one_checked_entry(win, monkeypatch):
    monkeypatch.setattr(win, "_on_language_selected", lambda code: None)
    actions = _menu(win, "Language").actions()
    checked = [a for a in actions if a.isChecked()]
    assert len(checked) == 1
    checked[0].trigger()                               # clicking the current one keeps it checked
    assert checked[0].isChecked()
    other = next(a for a in actions if a is not checked[0])
    other.trigger()
    assert [a for a in actions if a.isChecked()] == [other]


def test_the_theme_menu_has_exactly_one_checked_entry(win):
    actions = _menu(win, "Theme").actions()
    checked = [a for a in actions if a.isChecked()]
    assert len(checked) == 1
    checked[0].trigger()
    assert [a for a in actions if a.isChecked()] == checked


# ------------------------------------------------------------ 6. Export Results without results
def test_export_without_results_says_so(win, logs):
    win._right_sidebar._on_export()
    assert any(lvl == "warn" and "export" in m.lower() for m, lvl in logs)


# ------------------------------------------------------------ 7. Cancel while opening a project
def _project_with_results(tmp_path) -> Path:
    from raftcorr.engine.mapping import Grid
    from raftcorr.io.project import Project, save_project
    from raftcorr.results import DisplacementSeries, RunInfo
    from raftcorr.schedule import FrameSchedule

    n = 4
    g = Grid(0, 0, 40, 30, H, W)
    u = np.random.default_rng(1).normal(size=(n, *g.shape)).astype(np.float32)
    info = RunInfo("test", "RAFTcorr-s4", "0" * 64, 4, 20, "tile", "cpu", RunConfig().to_dict())
    series = DisplacementSeries(g, np.ones(g.shape, bool), u, -u, np.ones(n, bool),
                                tuple(f"f{i}" for i in range(n)), FrameSchedule.accumulative(n), info,
                                residual=np.abs(u))
    prj = Project(image_folder=str(tmp_path), image_files=tuple(f"f{i}.png" for i in range(n)),
                  run_config=RunConfig(), series=series)
    return save_project(tmp_path / "r.raftcorr", prj)


def test_loading_a_project_reports_progress_and_can_be_stopped(tmp_path):
    from raftcorr.callbacks import FunctionCallbacks, StopRequested
    from raftcorr.io.project import load_project

    path = _project_with_results(tmp_path)
    fractions: list[float] = []
    prj = load_project(path, FunctionCallbacks(progress=lambda f, m="": fractions.append(f)))
    assert prj.series is not None and fractions[-1] == 1.0 and len(fractions) > 4
    np.testing.assert_array_equal(prj.series.v, -prj.series.u)
    calls = {"n": 0}

    def stop_after_two() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(StopRequested):
        load_project(path, FunctionCallbacks(should_stop=stop_after_two))


def test_open_project_hands_the_dialog_callbacks_to_the_loader(win, tmp_path, monkeypatch):
    from raftcorr.callbacks import FunctionCallbacks, StopRequested

    path = _project_with_results(tmp_path)
    applied: list = []
    monkeypatch.setattr(win, "_apply_project", lambda prj, p: applied.append(prj))

    def run(title, fn, on_done):
        with pytest.raises(StopRequested):
            fn(FunctionCallbacks(should_stop=lambda: True))

    win._run_task = run
    win.open_project_path(str(path))
    assert applied == []


# ------------------------------------------------------------ 8. Tracking Mode tooltip
def test_tracking_mode_tooltip_says_the_reference_is_earlier(win):
    wf, _ = _panels(win)
    tip = wf._tracking_mode.toolTip()
    assert "earlier reference frame" in tip and "later reference" not in tip
