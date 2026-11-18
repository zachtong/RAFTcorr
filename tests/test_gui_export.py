"""The pyALDIC-derived export dialog on RAFTcorr results (offscreen, no GPU).

Synthetic displacement / strain series and a tiny image sequence in tmp_path; every export is
written into tmp_path and read back (data files with raftcorr.export.read_results).
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.gui

from raftcorr.callbacks import StopRequested, check_stop  # noqa: E402
from raftcorr.config import RunConfig  # noqa: E402
from raftcorr.engine.mapping import Grid  # noqa: E402
from raftcorr.export.scientific import read_results  # noqa: E402
from raftcorr.post.strain import COMPONENTS  # noqa: E402
from raftcorr.results import DisplacementSeries, RunInfo, StrainSeries, StrainSettings  # noqa: E402
from raftcorr.schedule import FrameSchedule  # noqa: E402

H, W, T = 50, 64, 4
GRID = Grid(8, 6, 40, 30, H, W)
TS = "20261118120000"
WAIT_MS = 60_000


# ------------------------------------------------------------------------------ synthetic data
def make_series() -> DisplacementSeries:
    """Linear u, uniform v; a hole in the ROI, one untracked point, frame 3 not computed."""
    X, _ = GRID.coords()
    u = np.stack([0.05 * k * (X - 28.0) for k in range(T)]).astype(np.float32)
    v = np.stack([np.full(GRID.shape, -0.5 * k) for k in range(T)]).astype(np.float32)
    mask = np.ones(GRID.shape, bool)
    mask[10:16, 15:25] = False
    u[:, ~mask] = np.nan
    v[:, ~mask] = np.nan
    u[1, 3, 4] = np.nan
    residual = np.where(mask, 2.0, np.nan)[None].repeat(T, 0).astype(np.float32)
    info = RunInfo("test", "RAFTcorr-s4", "ab" * 32, 4, 20, "tile", "cpu", RunConfig().to_dict())
    return DisplacementSeries(GRID, mask, u, v, np.array([True, True, True, False]),
                              tuple(f"img_{i}.png" for i in range(T)), FrameSchedule.accumulative(T), info,
                              residual=residual)


def make_strain(step: int = 1) -> StrainSeries:
    """Strain of frames 0 and 1 only."""
    hs, ws = -(-GRID.height // step), -(-GRID.width // step)
    rng = np.random.default_rng(3)
    comps = {c: (rng.normal(size=(T, hs, ws)) * 1e-3).astype(np.float32) for c in COMPONENTS}
    settings = StrainSettings("green_lagrange", 41, "uniform", 1, step, None, 0.0, True)
    return StrainSeries(GRID, step, comps, np.array([True, True, False, False]), settings)


def write_images(folder: Path) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    files = []
    for i in range(T):
        p = folder / f"img_{i}.png"
        Image.fromarray(rng.integers(0, 255, (H, W), dtype=np.uint8)).save(p)
        files.append(str(p))
    return files


# ------------------------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def fresh_state(qapp):
    from PySide6.QtCore import QCoreApplication

    from raftcorr.gui.app_state import AppState

    QCoreApplication.setOrganizationName("RAFTcorr-tests")
    QCoreApplication.setApplicationName("RAFTcorr-tests")
    AppState._instance = None
    yield
    AppState._instance = None


@pytest.fixture
def loaded(tmp_path: Path):
    """AppState holding a run: images, displacement, strain."""
    from raftcorr.gui.app_state import AppState
    from raftcorr.io.images import ImageSequence

    files = write_images(tmp_path / "imgs")
    st = AppState.instance()
    st.set_image_files(files, ImageSequence(files), tmp_path / "imgs")
    series, strain = make_series(), make_strain()
    st.set_results(series, strain)
    return st


@pytest.fixture
def dialog(qtbot, loaded, tmp_path: Path, monkeypatch):
    from raftcorr.gui.dialogs import export_jobs as jobs
    from raftcorr.gui.dialogs.export_dialog import ExportDialog, VizExportHint

    monkeypatch.setattr(jobs, "make_timestamp", lambda: TS)
    dlg = ExportDialog(loaded.results, loaded.image_folder, VizExportHint(show_deformed=True),
                       image_files=loaded.image_files, strain=loaded.strain_results, sequence=loaded.sequence)
    qtbot.addWidget(dlg)
    dlg._folder_edit.setText(str(tmp_path / "out"))
    yield dlg
    dlg._shutdown_workers()


def run(qtbot, dlg, tab: str) -> None:
    """Click a tab's export button and wait for its worker."""
    ctl = dlg._controls[tab]
    assert ctl.run_btn.isEnabled()
    ctl.run_btn.click()
    qtbot.waitUntil(lambda: not ctl.running, timeout=WAIT_MS)


def only_fields(rows, names: set[str]) -> None:
    for row in rows:
        row.set_enabled_field(row.field_name in names)


def out_dir(tmp_path: Path) -> Path:
    return tmp_path / "out"


def expected_uv(series: DisplacementSeries) -> tuple[np.ndarray, np.ndarray]:
    keep = series.mask[None] & series.done[:, None, None]
    return np.where(keep, series.u, np.nan), np.where(keep, series.v, np.nan)


# --------------------------------------------------------------------------------------- build
def test_dialog_has_pyaldic_tabs_and_follows_the_results(dialog):
    from raftcorr.gui.dialogs.export_dialog import mp4_available

    tabs = [dialog._tabs.tabText(i) for i in range(dialog._tabs.count())]
    assert tabs == ["Data", "Images", "Animation", "Report", "Preview && Colorbar"]   # "&&" shows one "&"
    assert dialog._data_strain_checks["strain_exx"].isEnabled()
    assert dialog._data_disp_checks["residual"].isEnabled()
    assert [r.field_name for r in dialog._img_field_rows if r.get_config().enabled] == \
        ["disp_u", "disp_v", "strain_exx", "strain_eyy", "strain_exy"]
    mp4 = dialog._anim_fmt_combo.model().item(0)
    assert mp4.isEnabled() == mp4_available()
    if not mp4_available():
        assert dialog._anim_fmt_combo.currentData() == ".gif" and "imageio" in mp4.toolTip()
    assert all(c.run_btn.isEnabled() for c in dialog._controls.values())
    dialog._folder_edit.setText("")
    assert not any(c.run_btn.isEnabled() for c in dialog._controls.values())


def test_without_strain_the_strain_fields_are_disabled(qtbot, loaded):
    from raftcorr.gui.dialogs.export_dialog import ExportDialog, VizExportHint

    dlg = ExportDialog(loaded.results, None, VizExportHint(), strain=None, sequence=None)
    qtbot.addWidget(dlg)
    assert not any(c.isEnabled() for c in dlg._data_strain_checks.values())
    assert not any(r.get_config().enabled for r in dlg._img_field_rows if r.field_name.startswith("strain_"))
    assert not dlg._img_bg_check.isEnabled()                      # no images: pictures on white
    assert dlg.images_job().fields[0].style.background == "none"


def test_strain_and_images_default_to_the_state_of_these_results(qtbot, loaded):
    """The strain window's call passes neither: they come from AppState."""
    from raftcorr.gui.dialogs.export_dialog import ExportDialog, VizExportHint

    dlg = ExportDialog(loaded.results, loaded.image_folder, VizExportHint(), image_files=loaded.image_files)
    qtbot.addWidget(dlg)
    assert dlg._strain is loaded.strain_results and dlg._images is loaded.sequence


# ---------------------------------------------------------------------------------------- data
def test_data_export_writes_every_format_and_reads_back(qtbot, dialog, loaded, tmp_path):
    dialog._h5_check.setChecked(True)
    dialog._csv_check.setChecked(True)
    dialog._data_disp_checks["residual"].setChecked(True)
    for chk in dialog._data_strain_checks.values():
        chk.setChecked(True)
    box = dialog._units_box
    box.enable_check.setChecked(True)
    box.pixel_size_spin.setValue(0.5)
    box.unit_combo.setCurrentText("\u03bcm")
    box.frame_rate_spin.setValue(10.0)
    run(qtbot, dialog, "data")

    out = out_dir(tmp_path)
    assert "Exported" in dialog._controls["data"].status_lbl.text()
    series = loaded.results
    u, v = expected_uv(series)
    for ext in (".npz", ".mat", ".h5"):
        data = read_results(out / f"imgs_results_{TS}{ext}")
        assert set(data) - {"metadata"} == {"U", "V", "Magnitude", "Residual", "X", "Y", "ROI", "Done",
                                            *COMPONENTS, "X_strain", "Y_strain"}
        np.testing.assert_array_equal(data["U"], u.astype(np.float32))
        np.testing.assert_array_equal(data["V"], v.astype(np.float32))
        assert np.isnan(data["exx"][2]).all() and np.isfinite(data["exx"][1]).all()
        meta = data["metadata"]
        assert meta["pixel_size"] == 0.5 and meta["length_unit"] == "\u03bcm" and meta["frame_rate"] == 10.0

    params = json.loads((out / f"imgs_parameters_{TS}.json").read_text(encoding="utf-8"))
    assert params["run_info"]["model_name"] == "RAFTcorr-s4"
    assert params["physical_units"] == {"scale": 0.5, "unit": "\u03bcm", "frame_rate": 10.0}
    assert params["strain"]["settings"]["gauge_px"] == 41

    tables = sorted((out / f"imgs_csv_{TS}").glob("*.csv"))
    assert [p.name for p in tables] == ["frame_0001.csv", "frame_0002.csv", "frame_0003.csv"]  # frame 4: not computed
    with open(tables[1], newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][:7] == ["x [px]", "y [px]", "U [px]", "V [px]", "Magnitude [px]", "Residual [grey levels]",
                           "exx [-]"]
    assert len(rows) - 1 == int(series.mask.sum())
    first = rows[1]
    assert float(first[0]) == GRID.x0 and float(first[1]) == GRID.y0
    assert float(first[2]) == pytest.approx(float(series.u[1, 0, 0]))
    untracked = next(r for r in rows[1:] if float(r[0]) == GRID.x0 + 4 and float(r[1]) == GRID.y0 + 3)
    assert untracked[2] == "" and untracked[3] != ""                 # NaN is an empty field
    with open(tables[2], newline="", encoding="utf-8") as fh:
        frame2 = list(csv.reader(fh))
    assert frame2[1][6] == ""                                        # strain of frame 2 not computed


def test_data_without_strain_variables_holds_no_strain(qtbot, dialog, tmp_path):
    dialog._mat_check.setChecked(False)
    for chk in dialog._data_strain_checks.values():
        chk.setChecked(False)
    run(qtbot, dialog, "data")
    data = read_results(out_dir(tmp_path) / f"imgs_results_{TS}.npz")
    assert set(data) - {"metadata"} == {"U", "V", "Magnitude", "X", "Y", "ROI", "Done"}
    assert data["metadata"]["strain_settings"] is None
    assert not (out_dir(tmp_path) / f"imgs_results_{TS}.mat").exists()


def test_data_export_needs_a_format_and_a_variable(dialog):
    for chk in (dialog._npz_check, dialog._mat_check):
        chk.setChecked(False)
    assert dialog.data_job() == "Select at least one format."
    dialog._npz_check.setChecked(True)
    for chk in [*dialog._data_disp_checks.values(), *dialog._data_strain_checks.values()]:
        chk.setChecked(False)
    dialog._export_data_btn.click()
    assert "variable" in dialog._controls["data"].status_lbl.text()
    assert dialog._workers["data"] is None


def test_cancelled_data_export_leaves_no_partial_file(loaded, tmp_path):
    from raftcorr.gui.dialogs import export_jobs as jobs

    names = jobs.OutputNames(tmp_path / "out", "run", TS)
    job = jobs.DataJob(names, loaded.results, loaded.strain_results, formats=(".npz", ".h5"))
    reports: list = []
    written: list = []
    with pytest.raises(StopRequested):
        job.run(lambda *a: reports.append(a), lambda: bool(reports), written)
    assert written == [names.parameters()]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [names.parameters().name]


# -------------------------------------------------------------------------------------- images
@pytest.mark.parametrize("fmt", [".png", ".tif", ".jpg", ".pdf", ".svg"])
def test_images_one_file_per_computed_frame(qtbot, dialog, tmp_path, fmt):
    only_fields(dialog._img_field_rows, {"disp_u", "strain_exx"})
    dialog._img_fmt_combo.setCurrentIndex(dialog._img_fmt_combo.findData(fmt))
    box = dialog._img_range_box
    box.all_check.setChecked(False)
    box.start_spin.setValue(2)          # 1-based: frames 2..4 = indices 1..3
    box.end_spin.setValue(4)
    run(qtbot, dialog, "img")

    root = out_dir(tmp_path) / f"imgs_images_{TS}"
    assert sorted(p.name for p in (root / "disp_u").iterdir()) == [f"frame_0002{fmt}", f"frame_0003{fmt}"]
    assert sorted(p.name for p in (root / "strain_exx").iterdir()) == [f"frame_0002{fmt}"]  # strain: frames 1, 2
    assert "Exported 3 image(s)" in dialog._controls["img"].status_lbl.text()
    if fmt in (".png", ".tif", ".jpg"):
        with Image.open(root / "disp_u" / f"frame_0002{fmt}") as im:
            assert min(im.size) > 100


def test_images_job_follows_the_rows_and_settings(dialog):
    only_fields(dialog._img_field_rows, {"disp_v"})
    row = next(r for r in dialog._img_field_rows if r.field_name == "disp_v")
    row.set_appearance(colormap="viridis", auto=False, vmin=-2.0, vmax=1.0, opacity=0.4)
    dialog._img_geom_combo.setCurrentIndex(1)       # reference frame
    dialog._img_bg_check.setChecked(False)
    dialog._img_range_combo.setCurrentIndex(1)      # whole sequence
    dialog._img_dpi_spin.setValue(90)
    dialog._img_colorbar_check.setChecked(False)
    job = dialog.images_job()
    (field,) = job.fields
    s = field.style
    assert (field.name, s.field, s.colormap, s.vmin, s.vmax, s.opacity) == ("disp_v", "v", "viridis", -2.0, 1.0, 0.4)
    assert (s.configuration, s.background, s.range_mode, s.dpi, s.colorbar, s.title) == \
        ("reference", "none", "sequence", 90, False, False)
    assert field.frames == (0, 1, 2)                 # frame 3 not computed
    row.set_appearance(vmin=1.0, vmax=1.0)
    assert "Min must be below Max" in dialog.images_job()


def test_rows_start_from_the_main_window_colour_state(qtbot, loaded):
    from raftcorr.gui.dialogs.export_dialog import ExportDialog, VizExportHint

    loaded.display_field = "disp_mag"               # an older name of |U|
    loaded.set_colormap("plasma")
    loaded.set_color_range(False, 0.25, 3.5)
    dlg = ExportDialog(loaded.results, None, VizExportHint())
    qtbot.addWidget(dlg)
    row = next(r for r in dlg._img_field_rows if r.field_name == "disp_magnitude")
    assert row.get_appearance() == dict(colormap="plasma", auto=False, vmin=0.25, vmax=3.5, opacity=0.7)


# ----------------------------------------------------------------------------------- animation
def test_animation_gif_one_frame_per_computed_frame(qtbot, dialog, tmp_path):
    only_fields(dialog._anim_field_rows, {"disp_u"})
    dialog._anim_fmt_combo.setCurrentIndex(dialog._anim_fmt_combo.findData(".gif"))
    dialog._anim_fps_spin.setValue(10)
    run(qtbot, dialog, "anim")
    gif = out_dir(tmp_path) / f"imgs_animation_{TS}" / "disp_u.gif"
    with Image.open(gif) as im:
        assert im.n_frames == 3 and im.info["duration"] == 100
    assert "Exported 1 animation" in dialog._controls["anim"].status_lbl.text()


def test_animation_frame_step_keeps_the_playback_duration(dialog):
    only_fields(dialog._anim_field_rows, {"disp_u"})
    dialog._anim_fps_spin.setValue(10)
    dialog._anim_step_spin.setValue(2)
    job = dialog.animation_job()
    assert job.fields[0].frames == (0, 2) and job.fps == pytest.approx(5.0)
    assert job.fields[0].style.range_mode == "sequence"


# -------------------------------------------------------------------------------------- report
def test_report_is_self_contained_html(qtbot, dialog, tmp_path):
    for chk in [*dialog._report_disp_checks.values(), *dialog._report_strain_checks.values()]:
        chk.setChecked(False)
    dialog._report_disp_checks["disp_u"].setChecked(True)
    dialog._report_strain_checks["strain_exx"].setChecked(True)
    dialog._report_sample_spin.setValue(2)
    run(qtbot, dialog, "report")
    page = (out_dir(tmp_path) / f"imgs_report_{TS}.html").read_text(encoding="utf-8")
    assert "RAFTcorr-s4" in page and "Run parameters" in page
    assert page.count("data:image/png;base64,") == 2 + 2     # U: frames 0, 2; exx: frames 0, 1
    assert "http://" not in page and "https://" not in page
    assert "<td>4</td><td colspan='4'>N/A</td>" not in page  # only computed frames are listed
    assert "Report saved" in dialog._controls["report"].status_lbl.text()


# --------------------------------------------------------------------------- cancel / overwrite
def _blocking_export_frames(started: threading.Event):
    def fake(folder, provider, images, frames, style, units, fmt=".png", prefix=None, callbacks=None, first_number=0,
             **_options):
        started.set()
        deadline = time.monotonic() + 30
        while not callbacks.should_stop() and time.monotonic() < deadline:
            callbacks.progress(0.5, "")
            time.sleep(0.01)
        check_stop(callbacks)
        return []
    return fake


def test_cancel_stops_a_running_export(qtbot, dialog, monkeypatch):
    from raftcorr.gui.dialogs import export_jobs as jobs

    started = threading.Event()
    monkeypatch.setattr(jobs, "export_frames", _blocking_export_frames(started))
    only_fields(dialog._img_field_rows, {"disp_u", "disp_v"})   # 3 computed frames each
    ctl = dialog._controls["img"]
    ctl.run_btn.click()
    qtbot.waitUntil(started.is_set, timeout=WAIT_MS)
    assert ctl.running and not ctl.cancel_btn.isHidden() and not ctl.run_btn.isEnabled()
    qtbot.waitUntil(lambda: ctl.progress_lbl.text().startswith("Rendering U ("), timeout=WAIT_MS)
    assert ctl.progress.value() == 25                   # half of U = a quarter of U + V
    ctl.cancel_btn.click()
    qtbot.waitUntil(lambda: not ctl.running, timeout=WAIT_MS)
    assert "cancelled" in ctl.status_lbl.text()
    assert ctl.run_btn.isEnabled() and ctl.cancel_btn.isHidden()


def test_a_failed_export_is_reported(qtbot, dialog, loaded, monkeypatch):
    from raftcorr.gui.dialogs import export_jobs as jobs

    def broken(*_a, **_k):
        raise RuntimeError("disk full")

    logged: list = []
    loaded.log_message.connect(lambda msg, level: logged.append((msg, level)))
    monkeypatch.setattr(jobs, "export_animation", broken)
    run(qtbot, dialog, "anim")
    assert dialog._controls["anim"].status_lbl.text() == "Error: RuntimeError: disk full"
    assert logged == [("Export failed: RuntimeError: disk full", "error")]
    assert dialog._export_anim_btn.isEnabled()


def test_closing_the_dialog_stops_its_exports(qtbot, dialog, monkeypatch):
    from raftcorr.gui.dialogs import export_jobs as jobs

    started = threading.Event()
    monkeypatch.setattr(jobs, "export_frames", _blocking_export_frames(started))
    dialog._export_img_btn.click()
    qtbot.waitUntil(started.is_set, timeout=WAIT_MS)
    worker = dialog._workers["img"]
    dialog.reject()
    assert not worker.isRunning() and not dialog.is_busy()


def test_existing_files_are_replaced_only_after_asking(qtbot, dialog, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    asked: list[str] = []
    answer = [QMessageBox.StandardButton.No]

    def fake_question(parent, title, text, *args, **kwargs):
        asked.append(text)
        return answer[0]

    monkeypatch.setattr(QMessageBox, "question", fake_question)
    run(qtbot, dialog, "data")
    assert asked == []                                  # nothing there yet
    npz = out_dir(tmp_path) / f"imgs_results_{TS}.npz"
    npz.write_bytes(b"not an export")

    dialog._export_data_btn.click()                     # same timestamp: the files exist
    assert len(asked) == 1 and npz.name in asked[0]
    assert dialog._workers["data"] is not None and not dialog._controls["data"].running
    assert "kept" in dialog._controls["data"].status_lbl.text()
    assert npz.read_bytes() == b"not an export"

    answer[0] = QMessageBox.StandardButton.Yes
    run(qtbot, dialog, "data")
    assert len(asked) == 2
    assert "U" in read_results(npz)


# ------------------------------------------------------------------------- pyALDIC's extra options
def test_npz_one_file_per_frame(qtbot, dialog, loaded, tmp_path):
    per_frame = dialog._npz_per_frame_check
    assert per_frame.isEnabled() and not per_frame.isChecked()
    dialog._npz_check.setChecked(False)
    assert not per_frame.isEnabled()                    # only meaningful with NPZ
    dialog._npz_check.setChecked(True)
    per_frame.setChecked(True)
    assert dialog.data_job().npz_per_frame
    run(qtbot, dialog, "data")
    out = out_dir(tmp_path)
    assert not (out / f"imgs_results_{TS}.npz").exists() and (out / f"imgs_results_{TS}.mat").exists()
    files = sorted((out / f"imgs_npz_{TS}").iterdir())
    assert [p.name for p in files] == ["frame_0001.npz", "frame_0002.npz", "frame_0003.npz"]   # computed frames
    assert "Exported 5 file(s)" in dialog._controls["data"].status_lbl.text()   # JSON + MAT + 3 frames
    u, _ = expected_uv(loaded.results)
    data = read_results(files[1])
    assert data["metadata"]["frame"] == 1 and {"X", "Y", "ROI", "Done", "exx"} <= set(data)
    np.testing.assert_array_equal(data["U"], u[1].astype(np.float32))


def test_resolution_presets_and_jpeg_quality(qtbot, dialog, tmp_path):
    texts = [dialog._img_res_combo.itemText(i) for i in range(dialog._img_res_combo.count())]
    assert texts == ["1024 px", "768 px", "512 px", "1536 px", "2048 px", "Full resolution"]
    assert dialog._img_res_combo.currentData() == 0                 # pictures: the size the DPI gives
    assert dialog._anim_res_combo.currentData() == 1024
    quality = dialog._img_quality_spin
    assert quality.value() == 92 and not quality.isEnabled()        # PNG
    only_fields(dialog._img_field_rows, {"disp_u"})
    dialog._img_fmt_combo.setCurrentIndex(dialog._img_fmt_combo.findData(".jpg"))
    assert quality.isEnabled()
    quality.setValue(70)
    dialog._img_res_combo.setCurrentIndex(dialog._img_res_combo.findData(512))
    job = dialog.images_job()
    assert (job.fmt, job.max_side, job.jpeg_quality) == (".jpg", 512, 70)
    run(qtbot, dialog, "img")
    for path in (out_dir(tmp_path) / f"imgs_images_{TS}" / "disp_u").iterdir():
        with Image.open(path) as im:
            assert max(im.size) == 512 and im.mode == "RGB"


def test_animation_resolution_cap(qtbot, dialog, tmp_path):
    only_fields(dialog._anim_field_rows, {"disp_u"})
    dialog._anim_fmt_combo.setCurrentIndex(dialog._anim_fmt_combo.findData(".gif"))
    dialog._anim_res_combo.setCurrentIndex(dialog._anim_res_combo.findData(512))
    assert dialog.animation_job().max_side == 512
    run(qtbot, dialog, "anim")
    with Image.open(out_dir(tmp_path) / f"imgs_animation_{TS}" / "disp_u.gif") as im:
        assert max(im.size) == 512


def test_colorbar_style_reaches_every_export_and_the_preview(qtbot, dialog):
    from raftcorr.export.figure_layout import ColorbarStyle

    dialog.show()
    dialog._tabs.setCurrentIndex(dialog._preview_tab_index)
    preview, look = dialog._preview, dialog._preview.look
    assert dialog._tabs.tabText(dialog._preview_tab_index) == "Preview && Colorbar"
    assert look.colorbar_style() == ColorbarStyle()                  # the look of the pictures so far
    before = preview.preview_image()
    look.position_combo.setCurrentIndex(look.position_combo.findData("top"))
    look.font_spin.setValue(14)
    look.font_combo.setCurrentIndex(look.font_combo.findData("serif"))
    look.thickness_spin.setValue(0.1)
    look.background_combo.setCurrentIndex(look.background_combo.findData("black"))
    look.margin_spin.setValue(0.05)
    look.margin_color_combo.setCurrentIndex(look.margin_color_combo.findData("black"))
    want = ColorbarStyle(position="top", font_size=14, font_family="serif", thickness=0.1, background="black")
    for job in (dialog.images_job(), dialog.animation_job(), dialog.report_job()):
        s = job.fields[0].style
        assert (s.colorbar_style, s.margin, s.margin_color) == (want, 0.05, "black")
    after = preview.preview_image()
    assert after.shape[0] > before.shape[0]                          # the bar now sits on top
    assert tuple(after[0, 0]) == (0, 0, 0, 255)                      # black margin


def test_hint_starts_hidden_background_and_hide_untracked(qtbot, loaded):
    from raftcorr.gui.dialogs.export_dialog import ExportDialog, VizExportHint

    dlg = ExportDialog(loaded.results, None, VizExportHint(hidden_bg_color="black", hide_untracked=False))
    qtbot.addWidget(dlg)
    assert dlg._preview.look.hidden_background() == "black"
    assert not dlg._img_hide_check.isChecked() and not dlg._anim_hide_check.isChecked()
    for job in (dlg.images_job(), dlg.animation_job()):
        s = job.fields[0].style
        assert (s.hidden_background, s.hide_untracked) == ("black", False)
    default = ExportDialog(loaded.results, None, VizExportHint())
    qtbot.addWidget(default)
    assert default._preview.look.hidden_background() == "white" and default._img_hide_check.isChecked()


def test_transparent_hidden_background_in_files_and_preview(qtbot, dialog, tmp_path):
    dialog.show()
    dialog._tabs.setCurrentIndex(dialog._preview_tab_index)
    only_fields(dialog._img_field_rows, {"disp_u"})
    dialog._preview.activate()
    dialog._img_bg_check.setChecked(False)
    look = dialog._preview.look
    look.hidden_bg_combo.setCurrentIndex(look.hidden_bg_combo.findData("transparent"))
    shown = dialog._preview.preview_image()
    assert shown[..., 3].min() == 0                                  # PNG keeps the transparency
    run(qtbot, dialog, "img")
    png = sorted((out_dir(tmp_path) / f"imgs_images_{TS}" / "disp_u").glob("*.png"))[1]   # frame 2, as shown
    with Image.open(png) as im:
        written = np.asarray(im)
    assert im.mode == "RGBA" and written[..., 3].min() == 0
    # The same picture at another DPI: same layout, so the same transparent share.
    assert written.shape[1] / written.shape[0] == pytest.approx(shown.shape[1] / shown.shape[0], rel=0.01)
    assert (written[..., 3] == 0).mean() == pytest.approx((shown[..., 3] == 0).mean(), abs=0.02)
    dialog._img_fmt_combo.setCurrentIndex(dialog._img_fmt_combo.findData(".jpg"))
    assert dialog._preview.preview_image()[..., 3].min() == 255      # JPEG: white instead


def test_hide_untracked_reaches_the_pictures(qtbot, dialog):
    """make_series leaves u[1, 3, 4] untracked inside the ROI: filled only with the box off."""
    dialog.show()
    dialog._tabs.setCurrentIndex(dialog._preview_tab_index)
    dialog._img_geom_combo.setCurrentIndex(1)                          # reference frame
    dialog._img_bg_check.setChecked(False)
    preview = dialog._preview
    preview.frame_slider.setValue(1)
    hidden = preview.preview_image()
    dialog._img_hide_check.setChecked(False)
    assert not dialog.images_job().fields[0].style.hide_untracked
    filled = preview.preview_image()
    assert not np.array_equal(hidden, filled)
    dialog._anim_hide_check.setChecked(False)
    assert not dialog.animation_job().fields[0].style.hide_untracked


# ------------------------------------------------------------------------------------- preview
def test_preview_draws_the_exported_picture_and_edits_the_row(qtbot, dialog):
    dialog.show()
    dialog._tabs.setCurrentIndex(dialog._preview_tab_index)
    preview = dialog._preview
    assert preview.field_combo.currentData() == "disp_u"
    img = preview.preview_image()
    assert img is not None and img.ndim == 3 and img.shape[2] == 4
    preview._cmap_combo.setCurrentText("viridis")
    row = next(r for r in dialog._img_field_rows if r.field_name == "disp_u")
    assert row.get_config().colormap == "viridis"
    preview.frame_slider.setValue(3)                    # not computed
    assert preview.preview_image() is None
    preview.render()
    assert "No data" in preview._label.text()


# -------------------------------------------------------------------------------- right sidebar
def test_right_sidebar_export_passes_the_strain_and_the_images(qtbot, loaded, monkeypatch):
    from raftcorr.gui.dialogs.export_dialog import ExportDialog
    from raftcorr.gui.panels.right_sidebar import RightSidebar

    seen = []
    monkeypatch.setattr(ExportDialog, "exec", lambda self: seen.append(self) or 0)
    bar = RightSidebar(pipeline_ctrl=None)
    qtbot.addWidget(bar)
    bar._on_export()
    (dlg,) = seen
    assert dlg._results is loaded.results and dlg._strain is loaded.strain_results
    assert dlg._images is loaded.sequence


def test_results_replaced_after_opening_are_not_exported_silently(qtbot, dialog, loaded, tmp_path):
    """A new run (or project) finishing while the dialog is open: refuse, do not export the old one."""
    loaded.set_results(make_series())                     # new results, same shape
    ctl = dialog._controls["data"]
    ctl.run_btn.click()
    assert not ctl.running
    assert "changed" in ctl.status_lbl.text()
    assert not (out_dir(tmp_path)).exists() or not any(out_dir(tmp_path).iterdir())
