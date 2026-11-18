"""Background tasks without a GPU: ``run_correlation`` with a stand-in pipeline (results, stale
flag, stop, failure), one task at a time, and closing the window while a task runs."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from .gui_helpers import (  # noqa: F401  (fixtures)
    _restore_theme,
    dialogs,
    fake_pipeline,
    loaded,
    log_lines,
    make_window,
    rect_mask,
    seq_folder,
    wait_idle,
    win,
)

pytestmark = pytest.mark.gui

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from raftcorr.gui_basic.workers import TaskWorker  # noqa: E402


def test_run_correlation_delivers_results_to_every_panel(qtbot, loaded, fake_pipeline):
    w = loaded
    st = w.state
    st.set_roi(0, rect_mask())
    crack = rect_mask((100, 100, 120, 120))
    st.set_frame(2)
    w.canvas.shape_finished.emit("rect", [(100.0, 100.0), (119.0, 119.0)])
    st.set_frame(0)
    st.update_view(field="speed")
    w.run_correlation()
    assert w._worker is not None and st.run_state == "running"
    assert not w.left.run.run_btn.isEnabled() and w.left.run.stop_btn.isEnabled()
    assert not w.right.strain.go.isEnabled()
    wait_idle(qtbot, w)
    call = fake_pipeline.calls[0]
    np.testing.assert_array_equal(call["roi"], rect_mask())
    assert list(call["frame_masks"]) == [2]
    np.testing.assert_array_equal(call["frame_masks"][2], crack)
    assert st.series is not None and st.series.done.all()
    assert st.run_state == "idle"
    assert st.current_frame == 3  # jumps to the last computed frame
    assert st.view.field == "u"  # velocity / strain fields fall back to u
    assert not st.is_stale and w.left.run.stale.isHidden()
    assert w.nav._frames == [0, 1, 2, 3]
    assert w.left.run.run_btn.isEnabled() and not w.left.run.stop_btn.isEnabled()
    assert w.left.run.status.text() == "Ready."
    assert w.left.run.bar.value() == 1000
    assert w.right.strain.go.isEnabled()
    assert w.right.strain.status.text() == "Not computed yet."
    assert "correlation finished: 3 frames" in log_lines(w)
    qtbot.waitUntil(lambda: w.view.last_range is not None, timeout=2000)


def test_roi_edit_during_a_run_marks_the_results_stale(qtbot, loaded, fake_pipeline):
    w = loaded
    st = w.state
    st.set_roi(0, rect_mask())
    fake_pipeline.gate = threading.Event()
    w.run_correlation()
    w.canvas.shape_finished.emit("rect", [(10.0, 10.0), (20.0, 20.0)])  # edited while running
    fake_pipeline.gate.set()
    wait_idle(qtbot, w)
    assert st.series is not None
    assert st.is_stale
    assert not w.left.run.stale.isHidden()


def test_run_without_roi_asks_first(qtbot, loaded, fake_pipeline, dialogs):
    w = loaded
    w.run_correlation()  # the stub answers No
    assert dialogs.names() == ["question"] and "No ROI" in dialogs.texts("question")[0]
    assert w._worker is None and fake_pipeline.calls == []
    dialogs.question_answer = QMessageBox.StandardButton.Yes
    w.run_correlation()
    wait_idle(qtbot, w)
    assert fake_pipeline.calls[0]["roi"] is None
    assert w.state.series.grid.shape == w.state.sequence.shape  # the whole image


def test_run_without_images_reports_an_error(win, fake_pipeline, dialogs):
    win.run_correlation()
    assert dialogs.names() == ["critical"]
    assert "Nothing to run" in dialogs.texts("critical")[0]
    assert win._worker is None


def test_stop_keeps_the_computed_frames_and_returns_to_idle(qtbot, loaded, fake_pipeline):
    w = loaded
    st = w.state
    st.set_roi(0, rect_mask())
    fake_pipeline.delay = 0.25
    with qtbot.waitSignal(st.progress, timeout=10000):
        w.run_correlation()
    w.stop_task()
    assert st.run_state == "stopping"
    assert not w.left.run.stop_btn.isEnabled() and not w.left.run.run_btn.isEnabled()
    assert w.left.run.status.text() == "Stopping after the current step…"
    wait_idle(qtbot, w)
    assert st.run_state == "idle"
    assert st.series is not None
    computed = int(st.series.done.sum()) - 1
    assert 1 <= computed < 3
    assert w.nav._frames == st.series.completed_frames
    assert w.left.run.run_btn.isEnabled()
    assert f"correlation finished: {computed} frames" in log_lines(w)


def test_stop_with_nothing_running_is_harmless(win):
    win.stop_task()
    assert win.state.run_state == "idle"


def test_a_failing_run_shows_an_error_and_returns_to_idle(qtbot, loaded, fake_pipeline, dialogs):
    w = loaded
    w.state.set_roi(0, rect_mask())
    fake_pipeline.error = ValueError("synthetic failure")
    w.run_correlation()
    wait_idle(qtbot, w)
    assert dialogs.names() == ["critical"]
    title, msg = dialogs.calls[0][1][1:3]
    assert title == "Correlation failed" and msg == "synthetic failure"
    assert w.state.series is None and w.state.run_state == "idle"
    assert w.left.run.run_btn.isEnabled()


def test_only_one_task_runs_at_a_time(qtbot, loaded, fake_pipeline, dialogs, tmp_path):
    w = loaded
    w.state.set_roi(0, rect_mask())
    fake_pipeline.gate = threading.Event()
    w.run_correlation()
    running = w._worker
    assert running is not None
    w._start(TaskWorker(lambda cb: 1, "second"), lambda _r: None, "Second task")
    assert w._worker is running
    assert "Second task: another task is still running; wait for it or stop it" in log_lines(w)
    dialogs.save_file = str(tmp_path / "p.raftcorr")
    w.save_project(ask=True)
    assert w._worker is running
    assert "Saving the project: another task is still running" in log_lines(w)
    assert not (tmp_path / "p.raftcorr").exists()
    w.run_correlation()  # ignored while the first run is busy
    fake_pipeline.gate.set()
    wait_idle(qtbot, w)
    assert w.state.series is not None
    assert len(fake_pipeline.calls) == 1


def test_suggest_model_while_busy_says_why(qtbot, loaded, fake_pipeline):
    w = loaded
    w.state.set_roi(0, rect_mask())
    fake_pipeline.gate = threading.Event()
    w.run_correlation()
    w.suggest_model()
    last = log_lines(w).strip().splitlines()[-1]
    fake_pipeline.gate.set()
    wait_idle(qtbot, w)
    assert "open an image sequence first" not in last


def test_strain_waits_for_results_and_reports_progress(qtbot, loaded, fake_pipeline):
    w = loaded
    w.compute_strain()  # no results: nothing happens
    assert w._worker is None
    w.state.set_roi(0, rect_mask())
    w.run_correlation()
    wait_idle(qtbot, w)
    seen = []
    w.state.progress.connect(lambda f, m: seen.append(m))
    w.compute_strain()
    assert not w.right.strain.bar.isHidden()
    wait_idle(qtbot, w)
    assert w.state.strain is not None
    assert any(m.startswith("strain: frame") for m in seen)
    assert w.right.strain.bar.isHidden()


def test_close_while_running_asks_and_can_be_cancelled(qtbot, loaded, fake_pipeline, dialogs):
    w = loaded
    w.state.set_roi(0, rect_mask())
    fake_pipeline.gate = threading.Event()
    w.run_correlation()
    assert w.close() is False  # the stub answers No: the window stays
    assert w.isVisible() and "Quit" in dialogs.texts("question")[-1]
    assert w._worker is not None
    dialogs.question_answer = QMessageBox.StandardButton.Yes
    assert w.close() is True  # stops the task (the stand-in honours the stop request) and quits
    assert not w.isVisible()
    qtbot.waitUntil(lambda: w._worker is None, timeout=10000)


def test_close_saves_geometry_in_the_settings(win):
    win.close()
    assert win.settings.value("geometry") is not None


def test_tracking_quality_note_is_shown_after_the_run(qtbot, loaded, fake_pipeline, dialogs):
    w = loaded
    w.state.set_roi(0, rect_mask())
    fake_pipeline.notes = ("tracking quality drops from frame 2 on (intensity residual 9.0 vs 3.0)",)
    w.run_correlation()
    wait_idle(qtbot, w)
    assert dialogs.names() == ["warning"]
    title, text = dialogs.calls[0][1][1:3]
    assert title == "Tracking quality" and text.startswith("Tracking quality drops from frame 2")


def test_tracking_quality_status_survives_the_end_of_the_task(qtbot, loaded, fake_pipeline, dialogs):
    w = loaded
    w.state.set_roi(0, rect_mask())
    fake_pipeline.notes = ("tracking quality drops from frame 2 on (intensity residual 9.0 vs 3.0)",)
    w.run_correlation()
    wait_idle(qtbot, w)
    assert "quality warning" in w.left.run.status.text()


def test_a_run_stopped_before_any_frame_keeps_the_previous_results(qtbot, loaded, fake_pipeline):
    """Run, then Run again and Stop at once: the complete earlier results are not thrown away."""
    w = loaded
    st = w.state
    st.set_roi(0, rect_mask())
    w.run_correlation()
    wait_idle(qtbot, w)
    assert st.series.completed_frames == [0, 1, 2, 3]
    fake_pipeline.gate = threading.Event()
    w.run_correlation()
    w.stop_task()
    wait_idle(qtbot, w)
    assert st.series.completed_frames == [0, 1, 2, 3]
