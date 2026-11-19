"""Regression tests of the run, the progress area, the LOG and the strain window.

Each test reproduces one bug found while documenting the application: warnings in the plain
colour, out-of-memory advice naming a control that does not exist, a stale progress line, the
strain window opening after a failed run, the "Hide untracked regions" tooltips and fill, the
strain tooltips, a dead widget and a clipped frame-rate combo.
"""

from __future__ import annotations

import os
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui

from raftcorr.config import RunConfig  # noqa: E402
from raftcorr.engine.mapping import Grid  # noqa: E402
from raftcorr.results import DisplacementSeries, RunInfo, StrainSeries, StrainSettings  # noqa: E402
from raftcorr.schedule import FrameSchedule  # noqa: E402

H, W = 20, 40           # image = grid, so grid indices are pixel indices
CRACK_COL = 20          # a column cut out of the region of interest: two separate parts


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
    from raftcorr.gui.app_state import AppState

    AppState._instance = None
    yield
    AppState._instance = None


def _series(done=(True, True, True)) -> DisplacementSeries:
    """Three frames on a 20 x 40 grid with a crack (column CRACK_COL) cut out of the ROI."""
    n = len(done)
    g = Grid(0, 0, W, H, H, W)
    mask = np.ones(g.shape, bool)
    mask[:, CRACK_COL] = False
    u = np.zeros((n,) + g.shape, np.float32)
    v = np.zeros_like(u)
    u[:, ~mask] = np.nan
    v[:, ~mask] = np.nan
    info = RunInfo("test", "RAFTcorr-s4", "0" * 64, 4, 20, "tile", "cpu", RunConfig().to_dict())
    res = np.where(mask, 1.0, np.nan)[None].repeat(n, 0).astype(np.float32)
    return DisplacementSeries(g, mask, u, v, np.asarray(done, bool), tuple(f"f{i}" for i in range(n)),
                              FrameSchedule.accumulative(n), info, residual=res)


# ================================================================================ 1. warning colour
def test_engine_warning_level_is_coloured_like_warn(qtbot):
    from raftcorr.gui.theme import COLORS
    from raftcorr.gui.widgets.console_log import ConsoleLog

    log = ConsoleLog()
    qtbot.addWidget(log)
    assert ConsoleLog._line("t", "m", "warning") == ConsoleLog._line("t", "m", "warn")
    log.append_log("Stopped.", "warning")
    assert COLORS.WARNING.lower() in log.toHtml().lower()


def test_pipeline_controller_emits_the_gui_warning_spelling():
    """The GUI settles on "warn"; the engine's "warning" is accepted by the LOG."""
    src = Path(__file__).resolve().parents[1] / "src/raftcorr/gui/controllers/pipeline_controller.py"
    assert '"warning")' not in src.read_text(encoding="utf-8")


# ================================================================================== 2. out of memory
def _failing_worker(monkeypatch, exc: Exception):
    import raftcorr.engine.pipeline as pipeline_mod
    from raftcorr.gui.controllers.pipeline_controller import PipelineWorker

    class Boom:
        def __init__(self, *_a, **_k) -> None:
            pass

        def run(self, *_a, **_k):
            raise exc

    monkeypatch.setattr(pipeline_mod, "Pipeline", Boom)
    worker = PipelineWorker(RunConfig(), None, None, {}, {})
    monkeypatch.setattr(worker, "_load_model", lambda cb: None)
    errors: list[tuple[str, str]] = []
    worker.fatal_error.connect(lambda t, m: errors.append((t, m)))
    worker.run()            # synchronously, in this thread
    assert len(errors) == 1
    return errors[0][1]


def test_out_of_memory_advice_names_the_gpu_memory_control(monkeypatch):
    text = _failing_worker(monkeypatch, RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert "memory fraction" not in text
    assert "GPU Memory" in text and "ADVANCED" in text
    assert "lower" in text
    assert text.endswith("CUDA out of memory. Tried to allocate 2 GiB")


def test_memory_budget_error_gets_friendly_advice(monkeypatch):
    raw = "the memory budget allows windows of about 90 x 90 px, below the model's 128 px minimum"
    text = _failing_worker(monkeypatch, MemoryError(raw))
    assert text.endswith(raw) and text != raw
    assert "GPU Memory" in text and "ADVANCED" in text
    assert "raise" in text            # a larger budget helps here, a smaller one cannot


def test_other_errors_get_no_memory_advice(monkeypatch):
    assert _failing_worker(monkeypatch, ValueError("bad thing")) == "bad thing"


# ============================================================================ 3. stale progress line
def test_second_run_does_not_show_the_previous_runs_frame(qtbot):
    from raftcorr.gui.app_state import AppState, RunState
    from raftcorr.gui.panels.right_sidebar import RightSidebar

    st = AppState.instance()
    bar = RightSidebar(types.SimpleNamespace(_start_time=0.0))
    qtbot.addWidget(bar)
    st.set_run_state(RunState.RUNNING)
    st.set_progress(1.0, "Frame 5/5")
    st.set_run_state(RunState.DONE)
    assert "5/5" in bar._progress_label.text()
    # a second run, as PipelineController.start() starts it
    st.set_progress(0.0, "Starting...")
    st.set_run_state(RunState.RUNNING)
    assert "5/5" not in bar._progress_label.text()
    st.set_progress(0.3, "Frame 2/5")
    assert "2/5" in bar._progress_label.text()


def test_pause_and_resume_keep_the_frame(qtbot):
    from raftcorr.gui.app_state import AppState, RunState
    from raftcorr.gui.panels.right_sidebar import RightSidebar

    st = AppState.instance()
    bar = RightSidebar(types.SimpleNamespace(_start_time=0.0))
    qtbot.addWidget(bar)
    st.set_run_state(RunState.RUNNING)
    st.set_progress(0.4, "Frame 2/5")
    st.set_run_state(RunState.PAUSED)
    st.set_run_state(RunState.RUNNING)
    assert "2/5" in bar._progress_label.text()


def test_a_new_run_forgets_a_download_status(qtbot):
    from raftcorr.gui.app_state import AppState, RunState
    from raftcorr.gui.panels.right_sidebar import RightSidebar

    st = AppState.instance()
    bar = RightSidebar(types.SimpleNamespace(_start_time=0.0))
    qtbot.addWidget(bar)
    st.set_run_state(RunState.RUNNING)
    st.set_progress_status("Downloading RAFTcorr-s4")
    st.set_run_state(RunState.DONE)      # e.g. the download failed
    st.set_progress(0.0, "Starting...")
    st.set_run_state(RunState.RUNNING)
    assert "Downloading" not in bar._progress_label.text()


# ================================================================== 4. strain window after a run
@pytest.fixture()
def win(qtbot):
    from raftcorr.gui.app import MainWindow

    w = MainWindow()
    opened: list[int] = []
    w._on_open_strain_window = lambda: opened.append(1)
    w.opened = opened
    qtbot.addWidget(w)
    yield w
    w._state.set_results(None)


def _finish(win, result) -> None:
    """A run that ends with *result*, through the controller's own finish handler."""
    from raftcorr.gui.app_state import RunState

    st, ctrl = win._state, win._pipeline_ctrl
    st.set_run_state(RunState.RUNNING)
    worker = types.SimpleNamespace(wait=lambda _ms: True, sequence=st.sequence)
    ctrl._worker = worker
    ctrl._on_finished(worker, result)


def test_strain_window_opens_after_a_run_with_results(win):
    _finish(win, _series())
    assert win.opened == [1]


def test_strain_window_stays_closed_after_a_failed_run(win):
    from raftcorr.gui.app_state import RunState

    win._state.set_results(_series())        # older results
    win._state.set_run_state(RunState.DONE)
    win.opened.clear()
    _finish(win, None)                       # failed (or stopped by an exception)
    assert win._state.run_state == RunState.DONE
    assert win.opened == []


def test_strain_window_stays_closed_after_a_run_cancelled_before_its_first_frame(win):
    from raftcorr.gui.app_state import RunState

    win._state.set_results(_series())
    win._state.set_run_state(RunState.DONE)
    win.opened.clear()
    _finish(win, _series(done=(True, False, False)))
    assert win._state.run_state == RunState.DONE
    assert win.opened == []


def test_strain_window_opens_again_on_the_next_successful_run(win):
    _finish(win, _series())
    _finish(win, None)
    _finish(win, _series())
    assert win.opened == [1, 1]


# ===================================================================== 5. Hide untracked tooltips
def test_hide_untracked_tooltips_say_where_they_apply(qtbot):
    from raftcorr.gui.panels.right_sidebar import RightSidebar
    from raftcorr.gui.widgets.strain_viz_panel import StrainVizPanel

    bar = RightSidebar(types.SimpleNamespace(_start_time=0.0))
    panel = StrainVizPanel()
    qtbot.addWidget(bar)
    qtbot.addWidget(panel)
    for tip in (bar._hide_untracked_cb.toolTip(), panel._hide_untracked_check.toolTip()):
        assert "Reference frame" in tip and "Deformed frame" in tip


# ================================================================ 6. strain view: untracked fill
def _strain(series: DisplacementSeries) -> StrainSeries:
    """exx = 1 left of the crack, 2 right of it; an untracked patch left of the crack, next to
    it; the two top rows edge-trimmed (NaN and marked)."""
    n = series.n_frames
    exx = np.where(np.arange(W) < CRACK_COL, 1.0, 2.0)[None, None].repeat(H, 1).repeat(n, 0)
    exx = exx.astype(np.float32)
    exx[:, :, CRACK_COL] = np.nan
    exx[:, 2:, 15:CRACK_COL] = np.nan              # untracked, inside the ROI, by the crack
    trimmed = np.zeros_like(exx, dtype=bool)
    trimmed[:, :2, :] = series.mask[:2, :]
    exx[trimmed] = np.nan
    settings = StrainSettings("infinitesimal", 5, "uniform", 1, 1, None, 0.0, True, 0.5)
    return StrainSeries(series.grid, 1, {"exx": exx}, series.done.copy(), settings, trimmed)


def _render(hide_untracked: bool, fill_trimmed: bool):
    from raftcorr.gui.panels.strain_canvas import FieldRenderer
    from raftcorr.viz.fields import FieldProvider

    series = _series()
    prov = FieldProvider(series, _strain(series))
    out = FieldRenderer().render(prov, "exx", 1, deformed=False, cmap="turbo", alpha=1.0,
                                 range_of=lambda _v: (0.0, 2.0), hide_untracked=hide_untracked,
                                 fill_trimmed=fill_trimmed)
    assert out is not None
    return out[1].values, prov


def test_untracked_fill_leaves_the_trimmed_band_empty():
    vals, _ = _render(hide_untracked=False, fill_trimmed=False)
    assert np.isnan(vals[:2, :]).all()                     # the trimmed band
    assert np.isfinite(vals[2:, 15:CRACK_COL]).all()     # the untracked patch is filled
    assert np.isnan(vals[:, CRACK_COL]).all()              # outside the ROI


def test_untracked_fill_does_not_cross_a_crack():
    vals, _ = _render(hide_untracked=False, fill_trimmed=False)
    assert (vals[2:, 15:CRACK_COL] == 1.0).all()         # from its own side only


def test_with_fill_trimmed_edges_the_band_is_filled_as_that_option_fills_it():
    from raftcorr.post.strain import fill_trimmed

    vals, prov = _render(hide_untracked=False, fill_trimmed=True)
    strain = prov.strain
    expected = fill_trimmed(strain.components["exx"][1], strain.trimmed[1])
    assert np.array_equal(vals[:2, :], expected[:2, :], equal_nan=True)
    assert np.isfinite(vals[:2, :CRACK_COL]).all() and (vals[:2, :CRACK_COL] == 1.0).all()


def test_hide_untracked_on_still_hides_everything():
    vals, _ = _render(hide_untracked=True, fill_trimmed=False)
    assert np.isnan(vals[:2, :]).all() and np.isnan(vals[2:, 15:CRACK_COL]).all()


# ===================================================================================== 7. tooltips
def test_output_spacing_tooltip_does_not_promise_speed(qtbot):
    from raftcorr.gui.widgets.strain_param_panel import StrainParamPanel

    panel = StrainParamPanel()
    qtbot.addWidget(panel)
    tip = panel._step_spin.toolTip()
    assert "faster" not in tip
    assert "every pixel" in tip


def test_strain_export_tooltip_lists_what_the_dialog_exports(qtbot):
    from raftcorr.gui.strain_window import StrainWindow

    w = StrainWindow(__import__("raftcorr.gui.app_state", fromlist=["AppState"]).AppState.instance())
    qtbot.addWidget(w)
    tip = w._export_strain_btn.toolTip()
    for word in ("HDF5", "TIFF", "JPEG", "PDF", "SVG", "GIF", "MP4", "report"):
        assert word in tip


# ===================================================================================== 8. dead code
def test_velocity_settings_widget_is_gone():
    root = Path(__file__).resolve().parents[1] / "src/raftcorr/gui/widgets"
    assert not (root / "velocity_settings.py").exists()


# ============================================================================ 9. frame-rate combo
@pytest.mark.parametrize("module, cls", [
    ("raftcorr.gui.widgets.frame_navigator", "FrameNavigator"),
    ("raftcorr.gui.widgets.strain_navigator", "StrainNavigator"),
])
def test_speed_combo_fits_its_longest_item(qtbot, module, cls):
    import importlib

    from PySide6.QtWidgets import QStyle, QStyleOptionComboBox, QVBoxLayout, QWidget

    from raftcorr.gui.theme import build_stylesheet

    host = QWidget()
    host.setStyleSheet(build_stylesheet())       # the application's padding and drop-down
    nav = getattr(importlib.import_module(module), cls)()
    QVBoxLayout(host).addWidget(nav)
    qtbot.addWidget(host)
    host.resize(900, 40)
    host.show()
    qtbot.waitExposed(host)
    combo = nav._speed_combo
    fm = combo.fontMetrics()
    longest = max(fm.horizontalAdvance(combo.itemText(i)) for i in range(combo.count()))
    opt = QStyleOptionComboBox()
    combo.initStyleOption(opt)
    field = combo.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, opt,
                                         QStyle.SubControl.SC_ComboBoxEditField, combo)
    assert field.width() >= longest
