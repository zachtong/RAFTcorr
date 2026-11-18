"""The strain window's Analysis tab on RAFTcorr results (offscreen, no GPU).

A synthetic :class:`~raftcorr.results.DisplacementSeries` and
:class:`~raftcorr.results.StrainSeries` with known linear fields stand in for a run, so every
number a probe reports has an analytic value.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QLocale, Qt  # noqa: E402
from PySide6.QtWidgets import QFileDialog, QMessageBox  # noqa: E402

from raftcorr.analysis.load_data import LoadData, LoadSync, LoadTable  # noqa: E402
from raftcorr.analysis.probes import AreaGeom, LineGeom, PointGeom  # noqa: E402
from raftcorr.analysis.series import FrameStatus  # noqa: E402
from raftcorr.gui.app_state import AppState  # noqa: E402
from raftcorr.gui.panels.analysis.quantities import Quantity  # noqa: E402

from .test_analysis_engine import build_series, build_strain  # noqa: E402

H, W = 60, 80
N_DEFORMED = 3
SLOPE = 0.01
X_C = 40.0
ROI_BOX = (5, 5, 75, 55)          # x0, y0, x1, y1 (half-open)
STRAIN_STEP = 2


def u_of(x: float, f: int) -> float:
    return f * (SLOPE * (x - X_C) + 0.5)


def v_of(f: int) -> float:
    return 0.25 * f


def linear(X, Y, f):
    return f * (SLOPE * (X - X_C) + 0.5), np.full_like(Y, 0.25 * f)


def exx_of(x: float, f: int) -> float:
    return 0.001 * f * x


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def test_numbers_format_without_pyaldic():
    """``locale_format`` once imported the pyALDIC package and every line label crashed."""
    from raftcorr.gui.support.locale_format import format_number

    assert format_number(400.0, 1) in ("400.0", "400,0")


def roi_mask() -> np.ndarray:
    x0, y0, x1, y1 = ROI_BOX
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def write_images(folder: Path, n: int = N_DEFORMED + 1) -> list[Path]:
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    paths = []
    for k in range(n):
        p = folder / f"img_{k:02d}.png"
        Image.fromarray(rng.integers(0, 255, (H, W), dtype=np.uint8)).save(p)
        paths.append(p)
    return paths


def load_images(state: AppState, folder: Path) -> list[Path]:
    from raftcorr.io.images import ImageSequence

    files = write_images(folder)
    state.set_image_files([str(f) for f in files], sequence=ImageSequence(files), folder=folder)
    return files


def run_results(lost: dict | None = None):
    series = build_series(linear, N_DEFORMED, shape=(H, W), roi=roi_mask(), lost=lost)
    strain = build_strain(series, lambda X, Y, f: 0.001 * f * X, step=STRAIN_STEP)
    return series, strain


@pytest.fixture
def state(qapp, tmp_path):
    AppState._instance = None
    st = AppState()
    AppState._instance = st
    load_images(st, tmp_path / "images")
    st.set_frame_roi(0, roi_mask())
    st.set_results(*run_results())
    yield st
    AppState._instance = None


@pytest.fixture
def tab(qtbot, state):
    from raftcorr.gui.panels.analysis import AnalysisTab

    t = AnalysisTab(state)
    qtbot.addWidget(t)
    t.resize(1200, 720)
    t.show()
    qtbot.waitExposed(t)
    return t


def place(tab, kind: str, geometry):
    """Place a probe as the canvas reports a finished placement."""
    tab._canvas.probe_requested.emit(kind, geometry)
    return list(tab._state.probes)[-1]


def series_of(tab, probe, quantity: Quantity, statistic: str = "mean"):
    return tab._series(probe, quantity, statistic)


DISP_U = Quantity("field", "disp_u")


def click(qtbot, tab, x: float, y: float) -> None:
    canvas = tab._canvas
    qtbot.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, pos=canvas.image_to_view(x, y))


def save_dialog(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(path), ""))


def read_csv(path: Path):
    with open(path, encoding="utf-8-sig") as fh:
        lines = fh.read().splitlines()
    comments = [ln[1:].strip() for ln in lines if ln.startswith("#")]
    rows = list(csv.reader(ln for ln in lines if not ln.startswith("#")))
    return comments, rows[0], rows[1:]


# ---------------------------------------------------------------------------
# building and placing
# ---------------------------------------------------------------------------

def test_the_tab_builds_on_the_reference_frame(tab, state):
    assert state.probes is not None and len(state.probes) == 0
    pixmap = tab._canvas._bg_item.pixmap()
    assert (pixmap.width(), pixmap.height()) == (W, H)
    assert tab._nav._n_frames == N_DEFORMED + 1
    assert not tab._chart.has_data, "no probe yet: a placeholder"
    assert "probe" in tab._chart.figure.axes[0].texts[0].get_text().lower()


def test_a_clicked_point_lands_on_the_image_pixel_it_shows(qtbot, tab):
    """Probe coordinates are image pixels (pixel centres on integers), not scene pixels."""
    qtbot.mouseClick(tab._tool_buttons["point"], Qt.MouseButton.LeftButton)
    assert tab._canvas.tool == "point"
    click(qtbot, tab, 30.0, 20.0)
    (probe,) = list(tab._state.probes)
    tolerance = 1.0 / tab._canvas.transform().m11()      # one screen pixel
    assert probe.geometry.x == pytest.approx(30.0, abs=tolerance)
    assert probe.geometry.y == pytest.approx(20.0, abs=tolerance)
    assert tab._canvas.tool == "none"
    ts = series_of(tab, probe, DISP_U, "value")
    np.testing.assert_allclose(ts.values, [u_of(probe.geometry.x, f) for f in range(4)], atol=1e-5)
    assert tab._chart.has_data
    assert [p.label for p, _, _ in tab._chart_panel.plotted] == ["P1"]


def test_an_extensometer_reads_the_applied_stretch(qtbot, tab):
    qtbot.mouseClick(tab._tool_buttons["extensometer"], Qt.MouseButton.LeftButton)
    click(qtbot, tab, 15.0, 30.0)
    click(qtbot, tab, 65.0, 30.0)
    (probe,) = list(tab._state.probes)
    assert probe.kind == "line"
    assert tab._chart_panel.quantity() == Quantity("gauge", "strain")
    ts = series_of(tab, probe, Quantity("gauge", "strain"))
    # u = f (SLOPE (x - X_C) + 0.5), v uniform: engineering strain f * SLOPE along x
    g = probe.geometry
    expected = [np.hypot(g.x1 - g.x0 + f * SLOPE * (g.x1 - g.x0), g.y1 - g.y0) / g.length() - 1
                for f in range(4)]
    np.testing.assert_allclose(ts.values, expected, atol=1e-6)
    np.testing.assert_allclose(ts.values, [0, 0.01, 0.02, 0.03], atol=2e-4)
    assert set(ts.status) == {FrameStatus.OK}


def test_regions_give_the_mean_of_the_field_over_them(tab):
    rect = place(tab, "area", AreaGeom.rect(20.0, 15.0, 50.0, 45.0))
    circle = place(tab, "area", AreaGeom.circle(35.0, 30.0, 10.0))
    triangle = place(tab, "area", AreaGeom.polygon([(20.0, 20.0), (50.0, 20.0), (35.0, 45.0)]))
    for probe in (rect, circle):
        np.testing.assert_allclose(series_of(tab, probe, DISP_U).values,
                                   [u_of(35.0, f) for f in range(4)], atol=1e-5)
    np.testing.assert_allclose(series_of(tab, rect, Quantity("field", "disp_v")).values,
                               [v_of(f) for f in range(4)], atol=1e-6)
    tri = series_of(tab, triangle, DISP_U).values
    np.testing.assert_allclose(tri, [u_of(35.0, f) for f in range(4)], atol=0.02)
    assert {p.label for p, _, _ in tab._chart_panel.plotted} == {"P1", "P2", "P3"}


def test_a_strain_probe_reads_the_strain_grid(tab):
    probe = place(tab, "point", PointGeom(31.0, 22.0))
    assert tab._chart_panel.set_quantity(Quantity("field", "strain_exx"))
    tab._refresh()
    ts = series_of(tab, probe, Quantity("field", "strain_exx"), "value")
    np.testing.assert_allclose(ts.values, [exx_of(31.0, f) for f in range(4)], atol=1e-6)
    assert ts.unit == ""
    assert tab._unit_box.isVisibleTo(tab), "strain may be shown as %, µε"
    assert tab._chart.has_data


def test_strain_not_computed_says_so(tab, state):
    place(tab, "point", PointGeom(31.0, 22.0))
    state.set_strain_results(None)
    assert tab._chart_panel.set_quantity(Quantity("field", "strain_exx"))
    tab._refresh()
    assert not tab._chart.has_data
    assert "not been computed" in tab._chart.figure.axes[0].texts[0].get_text()


def test_the_chart_draws_one_curve_per_probe(tab):
    place(tab, "point", PointGeom(30.0, 20.0))
    place(tab, "line", LineGeom(10.0, 40.0, 60.0, 40.0))
    ax = tab._chart.figure.axes[0]
    labels = [line.get_label() for line in ax.get_lines() if not line.get_label().startswith("_")]
    assert labels == ["P1", "P2"]
    p1 = next(line for line in ax.get_lines() if line.get_label() == "P1")
    np.testing.assert_allclose(p1.get_xdata(), [1, 2, 3, 4])           # frames are 1-based
    np.testing.assert_allclose(p1.get_ydata(), [u_of(30.0, f) for f in range(4)], atol=1e-5)


def test_hiding_renaming_and_deleting_probes(tab):
    a = place(tab, "point", PointGeom(30.0, 20.0))
    place(tab, "point", PointGeom(40.0, 20.0))
    tab._probe_table.visibility_changed.emit(a.id, False)
    assert [p.label for p, _, _ in tab._chart_panel.plotted] == ["P2"]
    tab._probe_table.renamed.emit(a.id, "tip")
    assert tab._state.probes.get(a.id).label == "tip"
    tab._on_table_selected(a.id)
    tab._on_delete()
    assert [p.label for p in tab._state.probes] == ["P2"]


def test_the_line_views_show_the_field_along_the_line(tab):
    line = place(tab, "line", LineGeom(10.0, 30.0, 70.0, 30.0))
    tab._view_bar.setCurrentIndex(tab._view_bar_index("profile"))
    tab.set_frame(2)
    assert tab._chart_panel.line_shown == (line, "disp_u")
    current = next(ln for ln in tab._chart.figure.axes[0].get_lines() if ln.get_gid() == "current")
    distance = np.asarray(current.get_xdata())
    np.testing.assert_allclose(current.get_ydata(), [u_of(10.0 + d, 2) for d in distance], atol=1e-5)

    tab._view_bar.setCurrentIndex(tab._view_bar_index("kymograph"))
    image = tab._chart.figure.axes[0].get_images()[0]
    assert image.get_array().shape == (len(distance), N_DEFORMED + 1)
    assert tab._chart.has_data


def test_untracked_material_is_flagged_not_blanked(tab, state):
    gone = np.zeros((H, W), dtype=bool)
    gone[:, 40] = True                      # a crack along x = 40 from frame 2 on
    state.set_results(*run_results(lost={2: gone}))
    line = place(tab, "line", LineGeom(10.0, 30.0, 70.0, 30.0))
    ts = series_of(tab, line, DISP_U)
    assert ts.status[1] is FrameStatus.OK and ts.status[2] is FrameStatus.UNTRACKED
    assert np.isfinite(ts.values).all()
    assert tab._chart_panel.notes[line.id] == "partly untracked from frame 3"


def test_the_frame_follows_the_strain_window(tab):
    place(tab, "point", PointGeom(30.0, 20.0))
    tab.set_frame(2)
    assert tab._nav._current == 2
    assert tab._chart._cursor_x == pytest.approx(3.0)
    seen = []
    tab.frame_requested.connect(seen.append)
    tab._on_chart_clicked(4.2)
    assert seen == [3]


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def test_csv_export_writes_the_plotted_curves(tab, monkeypatch, tmp_path):
    point = place(tab, "point", PointGeom(30.0, 20.0))
    place(tab, "area", AreaGeom.rect(20.0, 15.0, 50.0, 45.0))
    out = tmp_path / "probes.csv"
    save_dialog(monkeypatch, out)
    tab._chart_panel.export_csv_requested.emit()
    comments, header, rows = read_csv(out)
    assert header[:2] == ["frame", "image"] and "time_s" not in header   # no frame rate set
    assert "P1_disp_u_value" in header and "P2_disp_u_mean_flag" in header
    assert [r[0] for r in rows] == ["1", "2", "3", "4"]
    assert [r[1] for r in rows] == [f"img_{k:02d}.png" for k in range(4)]
    col = header.index("P1_disp_u_value")
    np.testing.assert_allclose([float(r[col]) for r in rows],
                               [u_of(point.geometry.x, f) for f in range(4)], atol=1e-5)
    assert {r[header.index("P2_disp_u_mean_flag")] for r in rows} == {"ok"}
    blob = "\n".join(comments)
    assert "RAFTcorr" in blob and "image axes" in blob and "model: synthetic" in blob
    assert "strain: green-lagrange, virtual strain gauge 41 px" in blob


def test_line_csv_export(tab, monkeypatch, tmp_path):
    place(tab, "line", LineGeom(10.0, 30.0, 70.0, 30.0))
    tab._view_bar.setCurrentIndex(tab._view_bar_index("kymograph"))
    out = tmp_path / "line.csv"
    save_dialog(monkeypatch, out)
    tab._chart_panel.export_line_requested.emit()
    _, header, rows = read_csv(out)
    assert header == ["distance_px", "x_px", "y_px", "frame_1", "frame_2", "frame_3", "frame_4"]
    assert len(rows) == 61
    for r in rows:
        x = float(r[1])
        np.testing.assert_allclose([float(c) for c in r[3:]], [u_of(x, f) for f in range(4)], atol=1e-5)


def test_physical_units_scale_lengths_but_not_strain(tab, state):
    point = place(tab, "point", PointGeom(30.0, 20.0))
    line = place(tab, "line", LineGeom(10.0, 30.0, 70.0, 30.0))
    state.set_physical_units(True, 0.5, 10.0, "mm")
    disp = series_of(tab, point, DISP_U, "value")
    assert disp.unit == "mm"
    np.testing.assert_allclose(disp.values, [0.5 * u_of(30.0, f) for f in range(4)], atol=1e-5)
    np.testing.assert_allclose(series_of(tab, line, Quantity("gauge", "strain")).values,
                               [0, 0.01, 0.02, 0.03], atol=1e-6)


# ---------------------------------------------------------------------------
# the testing machine's record
# ---------------------------------------------------------------------------

def test_load_data_gives_load_and_stress_axes(tab, state, monkeypatch, tmp_path):
    line = place(tab, "line", LineGeom(10.0, 30.0, 70.0, 30.0))
    table = LoadTable(("frame", "load"), (np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.0, 2.0, 3.0])),
                      "machine.csv")
    state.set_load_data(LoadData(table, LoadSync(mode="frame", load_column="load", frame_column="frame",
                                                 load_unit="kN"), area_mm2=10.0))
    keys = [tab._x_box.itemData(i) for i in range(tab._x_box.count())]
    assert {"load", "stress"} <= set(keys)
    assert tab._chart_panel.set_quantity(Quantity("gauge", "strain"))
    tab._view_bar.setCurrentIndex(tab._view_bar_index("stress_strain"))
    assert tab._chart.has_data
    curve = next(ln for ln in tab._chart.figure.axes[0].get_lines() if ln.get_label() == line.label)
    np.testing.assert_allclose(curve.get_xdata(), [0, 0.01, 0.02, 0.03], atol=1e-6)
    np.testing.assert_allclose(curve.get_ydata(), [0, 100, 200, 300])      # kN / 10 mm² in MPa

    out = tmp_path / "with_load.csv"
    save_dialog(monkeypatch, out)
    tab._chart_panel.export_csv_requested.emit()
    _, header, rows = read_csv(out)
    assert header[2:4] == ["load_N", "stress_MPa"]
    assert [r[3] for r in rows] == ["0", "100", "200", "300"]


def test_csv_has_a_time_column_with_physical_units(tab, state, monkeypatch, tmp_path):
    place(tab, "point", PointGeom(30.0, 20.0))
    state.set_physical_units(True, 0.5, 10.0, "mm")
    out = tmp_path / "timed.csv"
    save_dialog(monkeypatch, out)
    tab._chart_panel.export_csv_requested.emit()
    _, header, rows = read_csv(out)
    assert header[:3] == ["frame", "image", "time_s"]
    np.testing.assert_allclose([float(r[2]) for r in rows], [0.0, 0.1, 0.2, 0.3])


def test_the_load_data_button_imports_a_machine_record(tab, state, monkeypatch, tmp_path):
    from raftcorr.gui.dialogs.load_data_dialog import LoadDataDialog

    record = tmp_path / "machine.csv"
    record.write_text("frame,load (kN)\n1,0.0\n2,1.5\n3,3.0\n4,4.5\n", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(record), ""))

    def choose_and_accept(dialog) -> int:
        dialog._on_choose()
        dialog.accept()
        return dialog.result()

    monkeypatch.setattr(LoadDataDialog, "exec", choose_and_accept)
    tab._chart_panel.load_data_requested.emit()
    assert state.load_data is not None
    assert state.load_data.sync.mode == "frame" and state.load_data.sync.load_unit == "kN"
    np.testing.assert_allclose(state.load_data.load_n(4, 1.0), [0.0, 1500.0, 3000.0, 4500.0])
    assert "load" in [tab._x_box.itemData(i) for i in range(tab._x_box.count())]


def test_the_chart_exports_as_an_image_and_as_text(tab, monkeypatch, tmp_path, qapp):
    place(tab, "point", PointGeom(30.0, 20.0))
    out = tmp_path / "chart.svg"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    tab._chart_panel.export_chart_requested.emit()
    assert out.is_file() and "<svg" in out.read_text(encoding="utf-8")[:500]
    tab._chart_panel.copy_data_requested.emit()
    rows = [r.split("\t") for r in qapp.clipboard().text().splitlines()]
    assert rows[0][0] == "Frame" and len(rows) == 5
    np.testing.assert_allclose([float(r[1]) for r in rows[1:]], [u_of(30.0, f) for f in range(4)], atol=1e-5)


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def test_new_images_clear_the_probes(tab, state, tmp_path):
    place(tab, "point", PointGeom(30.0, 20.0))
    load_images(state, tmp_path / "other")
    assert len(state.probes) == 0


def test_probes_survive_a_project_round_trip(tab, state, tmp_path):
    from raftcorr.gui.controllers.image_controller import ImageController
    from raftcorr.gui.project_bridge import apply_project, project_from_state
    from raftcorr.io.project import load_project, save_project
    from raftcorr.post.probes import Probe as BasicProbe

    place(tab, "point", PointGeom(30.25, 20.5))
    line = place(tab, "line", LineGeom(10.0, 30.0, 70.0, 30.0))
    place(tab, "area", AreaGeom.rect(20.0, 15.0, 50.0, 45.0))
    place(tab, "area", AreaGeom.circle(35.0, 30.0, 10.0))
    place(tab, "area", AreaGeom.polygon([(20.0, 20.0), (50.0, 20.0), (35.0, 45.0)]))
    dropped = place(tab, "point", PointGeom(60.0, 40.0))
    tab._probe_table.visibility_changed.emit(line.id, False)
    tab._on_table_selected(dropped.id)
    tab._on_delete()
    before = list(state.probes)

    path = save_project(tmp_path / "run.raftcorr", project_from_state(state))
    prj = load_project(path)
    # The other interface reads the same entries.
    assert [BasicProbe.from_dict(d).kind for d in prj.probes] == ["point", "line", "rect", "circle", "polygon"]

    AppState._instance = None
    other = AppState()
    apply_project(other, ImageController(other), prj, tmp_path / "images", path)
    assert list(other.probes) == before
    assert other.results is not None
    assert other.probes.add("point", PointGeom(1.0, 1.0)).id == dropped.id + 1, "ids are not reused"


def test_a_malformed_probe_in_a_project_is_reported_not_fatal(state, tmp_path):
    from raftcorr.gui.controllers.image_controller import ImageController
    from raftcorr.gui.project_bridge import apply_project, project_from_state

    prj = project_from_state(state)
    bad = type(prj)(**{**prj.__dict__, "probes": [{"id": 1, "kind": "wedge", "points": [[0, 0]],
                                                   "label": "x", "color": "#000000"}]})
    logged = []
    other = AppState()
    other.log_message.connect(lambda msg, level: logged.append((level, msg)))
    apply_project(other, ImageController(other), bad, tmp_path / "images", tmp_path / "x.raftcorr")
    assert len(other.probes) == 0
    assert any(level == "warn" and "probes" in msg for level, msg in logged)


def test_clear_all_asks_first(tab, monkeypatch):
    place(tab, "point", PointGeom(30.0, 20.0))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
    tab._on_clear()
    assert len(tab._state.probes) == 1
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    tab._on_clear()
    assert len(tab._state.probes) == 0
    assert not tab._chart.has_data


def test_time_axis_needs_physical_units(tab, state):
    """The fps box keeps its value while disabled: no time axis from a rate the user never set."""
    state.frame_rate = 25.0
    state.use_physical_units = False
    assert tab._frame_rate() == 0.0
    state.use_physical_units = True
    assert tab._frame_rate() == 25.0


def test_time_synced_load_without_a_frame_rate_says_so():
    data = LoadData(LoadTable(("t", "load"), (np.array([0.0, 1.0]), np.array([1.0, 2.0]))),
                    LoadSync(mode="time", load_column="load", time_column="t"))
    text = " ".join(data.describe(0.0))
    assert "no camera frame rate" in text and "/ 0" not in text


def test_probe_label_box_fits_the_painted_text(qtbot):
    """The label's box is measured at construction and its text painted later: same face, same width."""
    from PySide6.QtCore import QPointF
    from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

    from raftcorr.gui.panels import probe_canvas as pc

    painted: dict = {}
    original = pc.ProbeLabel.paint

    def spy(self, painter, option, widget=None):
        original(self, painter, option, widget)
        painted["advance"] = painter.fontMetrics().horizontalAdvance(self._text)
        painted["box"] = self._rect.width()

    scene = QGraphicsScene()
    view = QGraphicsView(scene)
    qtbot.addWidget(view)
    view.resize(400, 200)
    view.show()
    scene.addItem(pc.ProbeLabel("P1 · 400.0 px", "#ff0000", QPointF(40, 60)))
    pc.ProbeLabel.paint = spy
    try:
        view.grab()
    finally:
        pc.ProbeLabel.paint = original
    assert painted and painted["advance"] <= painted["box"]
