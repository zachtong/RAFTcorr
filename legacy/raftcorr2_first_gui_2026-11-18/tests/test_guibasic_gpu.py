"""End to end with the real RAFTcorr-s8 model on a synthetic speckle sequence (needs CUDA and the
released weights): run, stale flag, strain, deformed view, readout, project, exports, stop."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from .gui_helpers import (  # noqa: F401  (fixtures)
    SHIFTS,
    _restore_theme,
    accept_with,
    dialogs,
    log_lines,
    make_window,
    rect_mask,
    render,
    seq_folder,
    wait_idle,
    write_sequence,
)

pytestmark = [pytest.mark.gui, pytest.mark.gpu, pytest.mark.weights]

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QMessageBox  # noqa: E402

RUN_TIMEOUT_MS = 120_000
TOL_PX = 0.25  # s8 on a clean synthetic translation


@pytest.fixture
def gpu_window(require_cuda, weights, make_window, seq_folder):
    weights("s8")
    w = make_window()
    assert w.load_folder(seq_folder)
    w.state.set_roi(0, rect_mask())
    w.left.model.buttons["s8"].click()
    assert w.state.run_config.model == "s8"
    return w


@pytest.mark.slow
def test_end_to_end_with_the_s8_model(qtbot, gpu_window, make_window, dialogs, monkeypatch, tmp_path):
    from raftcorr.export.scientific import read_results
    from raftcorr.gui_basic.export_dialogs import PicturesDialog, ResultsDialog
    from raftcorr.gui_basic.project_io import load_project_into

    w = gpu_window
    st = w.state

    # -- run --------------------------------------------------------------------------------
    w.run_correlation()
    assert st.run_state == "running"
    wait_idle(qtbot, w, RUN_TIMEOUT_MS)
    assert "critical" not in dialogs.names(), dialogs.texts("critical")
    s = st.series
    assert s is not None and s.done.all()
    assert s.info.model_name.endswith("s8")
    for k, (du, dv) in enumerate(SHIFTS, 1):
        assert np.nanmedian(s.u[k][s.mask]) == pytest.approx(du, abs=TOL_PX)
        assert np.nanmedian(s.v[k][s.mask]) == pytest.approx(dv, abs=TOL_PX)
    assert st.current_frame == 3 and st.view.field == "u"
    assert "correlation finished: 3 frames" in log_lines(w)

    # -- stale flag -------------------------------------------------------------------------
    assert not st.is_stale and w.left.run.stale.isHidden()
    w.canvas.shape_finished.emit("rect", [(100.0, 100.0), (120.0, 120.0)])
    assert st.is_stale and not w.left.run.stale.isHidden()
    st.undo_roi()
    assert not st.is_stale
    st.set_frame(3)

    # -- readout and deformed configuration ---------------------------------------------------
    st.update_view(field="u", configuration="reference")
    render(w)
    assert w.view.value_at(160.0, 128.0) == pytest.approx(SHIFTS[2][0], abs=TOL_PX)
    w._hover(160.0, 128.0)
    assert w.value_label.text().startswith("u = 5.") or w.value_label.text().startswith("u = 6.")
    st.update_view(configuration="deformed")
    render(w)
    assert w.view.last_range is not None
    du, dv = SHIFTS[2]
    pts = np.array([[160.0, 128.0], [100.0, 80.0]])
    moved = w._probe_mapper()(pts)
    np.testing.assert_allclose(moved - pts, [[du, dv], [du, dv]], atol=TOL_PX)
    assert w.view.value_at(160.0 + du, 128.0 + dv) == pytest.approx(du, abs=TOL_PX)
    st.update_view(configuration="reference")

    # -- strain -------------------------------------------------------------------------------
    w.compute_strain()
    wait_idle(qtbot, w, RUN_TIMEOUT_MS)
    assert st.strain is not None and st.view.field == "exx"
    assert abs(float(np.nanmedian(st.strain.components["exx"][3]))) < 5e-3  # rigid translation
    render(w)
    assert w.colorbar.isVisible()

    # -- probes -------------------------------------------------------------------------------
    w.canvas.shape_finished.emit("probe_point", [(160.0, 128.0)])
    w.canvas.shape_finished.emit("probe_line", [(80.0, 128.0), (240.0, 128.0)])
    w.canvas.shape_finished.emit("probe_rect", [(100.0, 80.0), (200.0, 160.0)])
    assert [p.label for p in st.probes] == ["P1", "L2", "A3"]
    st.update_view(field="u")
    p1 = next(ln for ln in w.chart.ax.lines if ln.get_label() == "P1")
    np.testing.assert_allclose(p1.get_ydata(), [0.0] + [d for d, _ in SHIFTS], atol=TOL_PX)

    # -- project: save, then open in a new window ---------------------------------------------
    proj = tmp_path / "e2e.raftcorr"
    dialogs.save_file = str(proj)
    w.save_project()
    wait_idle(qtbot, w, RUN_TIMEOUT_MS)
    assert proj.is_file()
    w2 = make_window()
    load_project_into(w2, proj)
    wait_idle(qtbot, w2, RUN_TIMEOUT_MS)
    b = w2.state
    np.testing.assert_array_equal(b.roi(0), st.roi(0))
    assert [p.to_dict() for p in b.probes] == [p.to_dict() for p in st.probes]
    np.testing.assert_array_equal(b.series.u, s.u)
    np.testing.assert_array_equal(b.series.v, s.v)
    np.testing.assert_array_equal(b.strain.components["exx"], st.strain.components["exx"])
    assert b.series.info.model_sha256 == s.info.model_sha256

    # -- scientific export ----------------------------------------------------------------------
    h5 = tmp_path / "e2e.h5"
    accept_with(monkeypatch, ResultsDialog, lambda dlg: dlg.path.setText(str(h5)))
    w.export_results()
    wait_idle(qtbot, w, RUN_TIMEOUT_MS)
    data = read_results(h5)
    np.testing.assert_array_equal(data["U"], s.u)
    assert "exx" in data
    assert data["metadata"]["run_info"]["model_name"] == s.info.model_name

    # -- animation ------------------------------------------------------------------------------
    gif = tmp_path / "e2e.gif"

    def pics(dlg):
        dlg.kind.setCurrentIndex(dlg.kind.findData(".gif"))
        dlg.field.setCurrentIndex(dlg.field.findData("u"))
        dlg.path.setText(str(gif))

    accept_with(monkeypatch, PicturesDialog, pics)
    w.export_images()
    wait_idle(qtbot, w, RUN_TIMEOUT_MS)
    from PIL import Image

    with Image.open(gif) as im:
        assert im.n_frames == 4

    # -- probe table ----------------------------------------------------------------------------
    table = tmp_path / "e2e.csv"
    dialogs.save_file = str(table)
    w.export_probes()
    with open(table, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    got = [float(r["P1_u [px]"]) for r in rows]
    np.testing.assert_allclose(got, [0.0] + [d for d, _ in SHIFTS], atol=TOL_PX)
    assert "critical" not in dialogs.names(), dialogs.texts("critical")


@pytest.fixture
def long_window(require_cuda, weights, make_window, tmp_path):
    weights("s8")
    folder = tmp_path / "long"
    write_sequence(folder, shifts=[(0.5 * k, 0.25 * k) for k in range(1, 10)])
    w = make_window()
    assert w.load_folder(folder)
    w.state.set_roi(0, rect_mask())
    w.left.model.buttons["s8"].click()
    return w


def test_stop_right_after_start_returns_to_idle_with_partial_results(qtbot, long_window):
    w = long_window
    st = w.state
    w.run_correlation()
    w.stop_task()  # before the model is even loaded
    assert st.run_state == "stopping"
    assert not w.left.run.stop_btn.isEnabled()
    wait_idle(qtbot, w, RUN_TIMEOUT_MS)
    assert st.run_state == "idle"
    assert w.left.run.run_btn.isEnabled()
    assert st.series is not None
    assert st.series.completed_frames == [0]
    assert any("stopped by the user" in n for n in st.series.info.notes)
    assert "correlation finished: 0 frames" in log_lines(w)


def test_stop_mid_run_keeps_the_frames_already_computed(qtbot, long_window):
    w = long_window
    st = w.state
    w.run_correlation()
    worker = w._worker
    # stop from the worker thread right after the first frame is reported: deterministic
    worker.progress.connect(lambda _f, m: m.startswith("frame ") and worker.request_stop(),
                            Qt.ConnectionType.DirectConnection)
    wait_idle(qtbot, w, RUN_TIMEOUT_MS)
    s = st.series
    assert s is not None and s.completed_frames == [0, 1]
    assert np.nanmedian(s.u[1][s.mask]) == pytest.approx(0.5, abs=TOL_PX)
    assert np.isnan(s.u[2]).all()
    assert w.nav._frames == [0, 1]
    assert st.current_frame == 1
    assert st.run_state == "idle"


def test_suggest_model_with_the_survey(qtbot, gpu_window, dialogs):
    w = gpu_window
    dialogs.question_answer = QMessageBox.StandardButton.Yes
    w.suggest_model()
    wait_idle(qtbot, w, RUN_TIMEOUT_MS)
    assert "critical" not in dialogs.names(), dialogs.texts("critical")
    q = dialogs.texts("question")[-1]
    assert "RAFTcorr-s1 is suggested" in q  # 6.7 px at most: s1 (32 px reach) is enough
    assert w.state.run_config.model == "s1"
    assert w.left.model.buttons["s1"].isChecked()
    assert w.left.model.advice.text().startswith("Suggested: RAFTcorr-s1")


@pytest.mark.slow
def test_self_test_reports_every_model(qtbot, require_cuda, weights, make_window, dialogs):
    for key in ("s1", "s2", "s4", "s8"):
        weights(key)
    w = make_window()
    w.self_test()
    wait_idle(qtbot, w, RUN_TIMEOUT_MS)
    assert dialogs.names() == ["information"], dialogs.calls
    text = dialogs.calls[0][1][2]
    for key in ("s1", "s2", "s4", "s8"):
        assert key in text
