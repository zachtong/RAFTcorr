"""Project files (save, open in a new window, moved images, damaged files) and every export
dialog (scientific data, pictures and animations, probe CSV), with synthetic results (no GPU)."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import pytest

from .gui_helpers import (  # noqa: F401  (fixtures)
    _restore_theme,
    accept_with,
    dialogs,
    linear_field,
    loaded,
    log_lines,
    make_window,
    rect_mask,
    render,
    seq_folder,
    wait_idle,
    win,
    with_results,
    write_sequence,
)

pytestmark = pytest.mark.gui

from PySide6.QtWidgets import QMessageBox  # noqa: E402

from raftcorr.export.scientific import read_results  # noqa: E402
from raftcorr.gui_basic.export_dialogs import PicturesDialog, ResultsDialog  # noqa: E402
from raftcorr.gui_basic.project_io import load_project_into  # noqa: E402


def u_at(k, x, y):
    uu, _vv = linear_field(k, np.array([x]), np.array([y]))
    return float(uu[0])


def _decorate(w, qtbot):
    """Masks on two frames, probes, units, view settings and a strain series."""
    st = w.state
    st.set_frame(2)
    w.canvas.shape_finished.emit("circle", [(150.0, 120.0), (165.0, 120.0)])
    st.set_frame(0)
    w.canvas.shape_finished.emit("probe_point", [(200.0, 100.0)])
    w.canvas.shape_finished.emit("probe_line", [(60.0, 100.0), (260.0, 100.0)])
    w.right.units.unit.setCurrentText("mm")
    w.right.units.pixel_size.setValue(0.02)
    w.left.settings.iters.setValue(9)
    w.right.strain.gauge.setValue(21)
    w.compute_strain()
    wait_idle(qtbot, w)
    st.set_view(st.view.with_display("v", colormap="viridis", auto=False, vmin=-1.0, vmax=4.0))
    st.update_view(field="v", configuration="deformed", opacity=0.5)


def _save(w, qtbot, dialogs, path: Path) -> None:
    dialogs.save_file = str(path)
    w.save_project()
    wait_idle(qtbot, w)


# ======================================================================================= project
def test_project_round_trip_into_a_new_window(qtbot, make_window, with_results, dialogs, tmp_path):
    w = with_results
    _decorate(w, qtbot)
    path = tmp_path / "session.raftcorr"
    _save(w, qtbot, dialogs, tmp_path / "session")  # the extension is added
    assert path.is_file()
    assert w.state.project_path == path
    assert w.windowTitle() == "RAFTcorr - session.raftcorr"
    assert f"saved project {path}" in log_lines(w)

    w2 = make_window()
    load_project_into(w2, path)
    wait_idle(qtbot, w2)
    a, b = w.state, w2.state
    assert b.image_folder == a.image_folder and b.sequence.names == a.sequence.names
    assert sorted(b.roi_masks) == [0, 2]
    for k in a.roi_masks:
        np.testing.assert_array_equal(b.roi_masks[k], a.roi_masks[k])
    assert [p.to_dict() for p in b.probes] == [p.to_dict() for p in a.probes]
    assert b.run_config == a.run_config and b.run_config.iters == 9
    assert b.strain_config == a.strain_config and b.strain_config.gauge_px == 21
    assert b.units == a.units
    assert b.view == a.view
    assert b.series.grid == a.series.grid
    for name in ("u", "v", "mask", "done"):
        np.testing.assert_array_equal(getattr(b.series, name), getattr(a.series, name))
    assert b.strain is not None and b.strain.names == a.strain.names
    for name in a.strain.names:
        np.testing.assert_array_equal(b.strain.components[name], a.strain.components[name])
    assert b.project_path == path
    assert w2.windowTitle() == "RAFTcorr - session.raftcorr"
    assert w2.right.display.field.currentData() == "v"
    assert w2.right.probes.table.rowCount() == 2
    assert w2.right.strain.gauge.value() == 21
    assert not b.roi_history.can_undo
    render(w2)
    assert w2.view.last_range is not None


def test_save_again_reuses_the_project_file(qtbot, with_results, dialogs, tmp_path):
    w = with_results
    path = tmp_path / "p.raftcorr"
    _save(w, qtbot, dialogs, path)
    n_dialogs = dialogs.names().count("getSaveFileName")
    w.left.settings.iters.setValue(5)
    w.save_project()
    wait_idle(qtbot, w)
    assert dialogs.names().count("getSaveFileName") == n_dialogs  # no second dialog
    w.save_project(ask=True)  # "Save as" always asks
    wait_idle(qtbot, w)
    assert dialogs.names().count("getSaveFileName") == n_dialogs + 1


def test_save_cancelled_writes_nothing(qtbot, with_results, dialogs, tmp_path):
    dialogs.save_file = ""
    with_results.save_project()
    assert with_results._worker is None
    assert with_results.state.project_path is None
    assert list(tmp_path.glob("*.raftcorr")) == []


def test_open_project_dialog(qtbot, make_window, with_results, dialogs, tmp_path):
    path = tmp_path / "p.raftcorr"
    _save(with_results, qtbot, dialogs, path)
    w2 = make_window()
    dialogs.open_file = str(path)
    w2.open_project()
    wait_idle(qtbot, w2)
    assert w2.state.series is not None
    assert w2.settings.value("last_project_dir") == str(tmp_path)


def test_out_of_date_results_stay_out_of_date_after_reopening(qtbot, make_window, with_results, dialogs, tmp_path):
    w = with_results
    w.canvas.shape_finished.emit("rect", [(10.0, 10.0), (30.0, 30.0)])
    assert w.state.is_stale
    path = tmp_path / "stale.raftcorr"
    _save(w, qtbot, dialogs, path)
    w2 = make_window()
    load_project_into(w2, path)
    wait_idle(qtbot, w2)
    assert w2.state.series is not None
    assert w2.state.is_stale


def test_project_with_moved_images_asks_for_the_folder(qtbot, make_window, dialogs, tmp_path):
    images = tmp_path / "a" / "imgs"
    write_sequence(images, shifts=((1.0, 0.0),), h=64, w=80)
    w = make_window()
    assert w.load_folder(images)
    w.state.set_roi(0, rect_mask((8, 8, 70, 56), shape=(64, 80)))
    proj_dir = tmp_path / "b"
    proj_dir.mkdir()
    path = proj_dir / "p.raftcorr"
    _save(w, qtbot, dialogs, path)
    shutil.move(str(tmp_path / "a"), str(tmp_path / "c"))
    w2 = make_window()
    dialogs.directory = str(tmp_path / "c" / "imgs")
    load_project_into(w2, path)
    wait_idle(qtbot, w2)
    assert "warning" in dialogs.names() and "getExistingDirectory" in dialogs.names()
    assert w2.state.image_folder == tmp_path / "c" / "imgs"
    assert w2.state.n_frames == 2
    np.testing.assert_array_equal(w2.state.roi(0), w.state.roi(0))
    assert "image folder not found" in log_lines(w2)


def test_project_whose_images_cannot_be_found_is_not_opened(qtbot, make_window, dialogs, tmp_path):
    images = tmp_path / "imgs"
    write_sequence(images, shifts=((1.0, 0.0),), h=64, w=80)
    w = make_window()
    w.load_folder(images)
    path = tmp_path / "p.raftcorr"
    _save(w, qtbot, dialogs, path)
    shutil.rmtree(images)
    w2 = make_window()
    dialogs.directory = ""  # the user cancels the folder dialog
    load_project_into(w2, path)
    wait_idle(qtbot, w2)
    assert dialogs.texts("critical")[-1].startswith("Project not opened")
    assert w2.state.sequence is None


def test_damaged_project_reports_an_error(qtbot, win, dialogs, tmp_path):
    bad = tmp_path / "bad.raftcorr"
    bad.write_bytes(b"this is not a zip archive")
    load_project_into(win, bad)
    wait_idle(qtbot, win)
    assert dialogs.names() == ["critical"]
    title, msg = dialogs.calls[0][1][1:3]
    assert title == "Opening the project failed" and "not a RAFTcorr project" in msg
    assert win.state.sequence is None and win.state.run_state == "idle"


# ============================================================================== scientific export
def _export_results(monkeypatch, qtbot, w, path: Path, setup=None):
    def _setup(dlg):
        if setup is not None:
            setup(dlg)
        dlg.path.setText(str(path))

    shown = accept_with(monkeypatch, ResultsDialog, _setup)
    w.export_results()
    wait_idle(qtbot, w)
    return shown


def test_export_results_to_hdf5_and_read_back(monkeypatch, qtbot, with_results, tmp_path):
    pytest.importorskip("h5py")
    w = with_results
    w.right.units.unit.setCurrentText("mm")
    w.right.units.pixel_size.setValue(0.02)
    path = tmp_path / "out.h5"
    shown = _export_results(monkeypatch, qtbot, w, path)
    assert shown and set(shown[0].checks) == {"U", "V", "Magnitude", "X", "Y", "ROI", "Done"}
    data = read_results(path)
    s = w.state.series
    np.testing.assert_array_equal(data["U"], s.u)
    np.testing.assert_array_equal(data["V"], s.v)
    np.testing.assert_array_equal(data["Done"], s.done.astype(np.uint8))
    meta = data["metadata"]
    assert meta["length_unit"] == "mm" and meta["pixel_size"] == pytest.approx(0.02)
    assert meta["displacement_unit"] == "px"  # data stay in pixels
    assert meta["frame_names"] == w.state.sequence.names
    assert f"exported 7 variables to {path}" in log_lines(w)


def test_export_selected_variables_as_npz(monkeypatch, qtbot, with_results, tmp_path):
    w = with_results
    w.compute_strain()
    wait_idle(qtbot, w)

    def setup(dlg):
        dlg.fmt.button(2).click()  # NumPy
        assert dlg.path.text().endswith(".npz")
        for name, cb in dlg.checks.items():
            cb.setChecked(name in ("U", "exx"))

    path = tmp_path / "subset.npz"
    shown = _export_results(monkeypatch, qtbot, w, path, setup)
    assert "exx" in shown[0].checks  # strain variables are offered once computed
    data = read_results(path)
    assert {"U", "exx"} <= set(data) and "V" not in data
    np.testing.assert_array_equal(data["exx"], w.state.strain.components["exx"])


def test_export_overwrite_needs_confirmation(monkeypatch, qtbot, with_results, dialogs, tmp_path):
    w = with_results
    path = tmp_path / "out.npz"
    _export_results(monkeypatch, qtbot, w, path)
    first = path.stat().st_mtime_ns
    _export_results(monkeypatch, qtbot, w, path)  # the stub answers No
    assert "Overwrite?" in dialogs.texts("question")[-1]
    assert path.stat().st_mtime_ns == first
    dialogs.question_answer = QMessageBox.StandardButton.Yes
    _export_results(monkeypatch, qtbot, w, path)
    assert path.stat().st_mtime_ns != first


def test_export_without_variables_or_results_is_refused(monkeypatch, qtbot, with_results, win, dialogs, tmp_path):
    def none(dlg):
        for cb in dlg.checks.values():
            cb.setChecked(False)

    _export_results(monkeypatch, qtbot, with_results, tmp_path / "x.h5", none)
    assert "Choose at least one variable" in dialogs.texts("critical")[-1]
    assert not (tmp_path / "x.h5").exists()


def test_export_menus_without_results_say_so(make_window, dialogs):
    w = make_window()
    w.export_results()
    w.export_images()
    w.export_probes()
    assert dialogs.names() == ["critical"] * 3
    assert all(t.startswith("Nothing to export") for t in dialogs.texts("critical"))


# ===================================================================================== pictures
def _export_pictures(monkeypatch, qtbot, w, kind: str, out: Path, setup=None):
    def _setup(dlg):
        dlg.kind.setCurrentIndex(dlg.kind.findData(kind))
        if setup is not None:
            setup(dlg)
        dlg.path.setText(str(out))

    shown = accept_with(monkeypatch, PicturesDialog, _setup)
    w.export_images()
    wait_idle(qtbot, w)
    return shown


def test_export_gif_animation(monkeypatch, qtbot, with_results, tmp_path):
    from PIL import Image

    out = tmp_path / "anim.gif"
    _export_pictures(monkeypatch, qtbot, with_results, ".gif", out)
    assert out.is_file()
    with Image.open(out) as im:
        assert im.n_frames == 4
    assert f"saved animation {out}" in log_lines(with_results)


def test_export_png_frames_every_second_frame(monkeypatch, qtbot, with_results, tmp_path):
    out = tmp_path / "figs"

    def setup(dlg):
        assert (dlg.first.value(), dlg.last.value()) == (0, 3)
        dlg.every.setValue(2)
        dlg.field.setCurrentIndex(dlg.field.findData("v"))

    _export_pictures(monkeypatch, qtbot, with_results, ".png", out, setup)
    assert sorted(p.name for p in out.glob("*.png")) == ["v_0000.png", "v_0002.png"]


def test_picture_export_uses_the_display_style(monkeypatch, qtbot, with_results, tmp_path):
    import raftcorr.export.figures as figures

    w = with_results
    st = w.state
    st.set_view(st.view.with_display("u", colormap="viridis", auto=False, vmin=-2.0, vmax=8.0, levels=5))
    st.update_view(configuration="deformed", opacity=0.4, background="frame")
    seen = {}

    def fake_frames(out, prov, seq, frames, style, units, ext, callbacks=None):
        seen.update(style=style, frames=list(frames), ext=ext)
        return []

    monkeypatch.setattr(figures, "export_frames", fake_frames)
    shown = _export_pictures(monkeypatch, qtbot, w, ".png", tmp_path / "f")
    assert shown[0].range.currentData() == "manual"  # the display is manual: "as on screen" preselected
    s = seen["style"]
    assert (s.field, s.configuration, s.background, s.colormap, s.levels) == ("u", "deformed", "frame", "viridis", 5)
    assert s.opacity == pytest.approx(0.4)
    assert (s.vmin, s.vmax) == (-2.0, 8.0)
    assert seen["frames"] == [0, 1, 2, 3] and seen["ext"] == ".png"


def test_as_on_screen_range_matches_the_screen_with_auto_range(monkeypatch, qtbot, with_results, tmp_path):
    import raftcorr.export.figures as figures

    w = with_results
    w.state.set_frame(2)
    render(w)
    screen = w.view.last_range
    seen = {}
    monkeypatch.setattr(figures, "export_frames",
                        lambda out, prov, seq, frames, style, units, ext, callbacks=None: seen.update(style=style) or [])
    _export_pictures(monkeypatch, qtbot, w, ".png", tmp_path / "f",
                     lambda dlg: dlg.range.setCurrentIndex(dlg.range.findData("manual")))
    assert (seen["style"].vmin, seen["style"].vmax) == pytest.approx((screen.vmin, screen.vmax))


def test_switching_picture_kinds_back_and_forth_keeps_the_folder(with_results):
    dlg = PicturesDialog(with_results)
    start = dlg.path.text()
    dlg.kind.setCurrentIndex(dlg.kind.findData(".gif"))
    assert dlg.path.text().endswith(".gif") and dlg.fps.isEnabled()
    dlg.kind.setCurrentIndex(dlg.kind.findData(".png"))
    assert not dlg.fps.isEnabled()
    assert dlg.path.text() == start


# ========================================================================================== CSV
def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_export_probe_table(qtbot, with_results, dialogs, tmp_path):
    w = with_results
    w.canvas.shape_finished.emit("probe_point", [(200.0, 100.0)])
    w.canvas.shape_finished.emit("probe_line", [(60.0, 100.0), (260.0, 100.0)])
    out = tmp_path / "probes.csv"
    dialogs.save_file = str(out)
    w.right.probes.metric_combo.setCurrentText("max")
    w.export_probes()
    rows = _rows(out)
    assert len(rows) == 4
    hdr = list(rows[0])
    assert hdr[0] == "frame" and "time_s" not in hdr
    for col in ("P1_u [px]", "P1_vel_u [px/frame]", "L2_u_max [px]", "L2_length [px]", "L2_extensometer_strain [-]"):
        assert col in hdr
    for k, r in enumerate(rows):
        assert int(r["frame"]) == k
        assert float(r["P1_u [px]"]) == pytest.approx(u_at(k, 200.0, 100.0), abs=1e-5)
        assert float(r["L2_length [px]"]) == pytest.approx(200.0 * (1 + 0.01 * k), abs=1e-4)
    assert "saved the probe table" in log_lines(w)


def test_probe_table_with_a_frame_rate_and_units(qtbot, with_results, dialogs, tmp_path):
    w = with_results
    w.canvas.shape_finished.emit("probe_point", [(200.0, 100.0)])
    w.right.units.unit.setCurrentText("mm")
    w.right.units.pixel_size.setValue(0.5)
    w.right.units.rate_on.setChecked(True)
    w.right.units.rate.setValue(4.0)
    out = tmp_path / "p.csv"
    dialogs.save_file = str(out)
    w.export_probes()
    rows = _rows(out)
    assert float(rows[2]["time_s"]) == pytest.approx(0.5)
    assert float(rows[2]["P1_u [mm]"]) == pytest.approx(0.5 * u_at(2, 200.0, 100.0), abs=1e-6)
    assert "P1_vel_u [mm/s]" in rows[0]


def test_probe_table_needs_probes(with_results, dialogs):
    with_results.export_probes()
    assert dialogs.names() == ["critical"]
