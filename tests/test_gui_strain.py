"""Offscreen tests of the strain post-processing window (pyALDIC's, on RAFTcorr results).

A synthetic :class:`~raftcorr.results.DisplacementSeries` with a known affine field is built by
hand, so no GPU, model or image is needed. The Analysis tab is replaced by a recording stub (it is
ported separately); one test checks the placeholder shown when it cannot be built.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import Signal  # noqa: E402
from PySide6.QtWidgets import QLabel, QMessageBox, QWidget  # noqa: E402

pytestmark = pytest.mark.gui

#: Image size and the grid (bounding box of the frame-0 ROI) inside it.
IMG_H, IMG_W = 120, 160
GX0, GY0, GW, GH = 20, 10, 120, 100
N_FRAMES = 4
#: Displacement gradient of frame k is k * GRAD (about the centre XC, YC): du/dx, du/dy, dv/dx, dv/dy.
GRAD = (0.010, 0.004, -0.002, -0.006)
XC, YC = 80.0, 60.0
SHIFT = (1.5, -0.5)  # translation per frame (px)


# ===================================================================================== synthetic data
def gradients(k: int) -> tuple[float, float, float, float]:
    return tuple(k * g for g in GRAD)  # type: ignore[return-value]


def affine_uv(k: int, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b, c, d = gradients(k)
    u = a * (X - XC) + b * (Y - YC) + k * SHIFT[0]
    v = c * (X - XC) + d * (Y - YC) + k * SHIFT[1]
    return u, v


def make_series(done=None, hole: bool = True, nan_frame: int | None = None):
    """``N_FRAMES`` frames of the affine field on a 100 x 120 grid at (20, 10) of a 120 x 160 image."""
    from raftcorr.config import RunConfig
    from raftcorr.engine.mapping import Grid
    from raftcorr.results import DisplacementSeries, RunInfo
    from raftcorr.schedule import FrameSchedule

    grid = Grid(GX0, GY0, GW, GH, IMG_H, IMG_W)
    mask = np.ones(grid.shape, bool)
    if hole:
        mask[40:55, 90:105] = False  # a hole: strain must never be fitted across it
    X, Y = grid.coords()
    done_arr = np.ones(N_FRAMES, bool) if done is None else np.asarray(done, bool)
    done_arr[0] = True
    u = np.full((N_FRAMES,) + grid.shape, np.nan, np.float32)
    v = np.full_like(u, np.nan)
    for k in range(N_FRAMES):
        if done_arr[k]:
            uk, vk = affine_uv(k, X, Y)
            u[k] = np.where(mask, uk, np.nan)
            v[k] = np.where(mask, vk, np.nan)
    if nan_frame is not None:
        u[nan_frame, 20:30, 20:30] = np.nan  # untracked patch inside the ROI
        v[nan_frame, 20:30, 20:30] = np.nan
    info = RunInfo(software_version="test", model_name="synthetic", model_sha256="", downsample_factor=8,
                   iters=20, corr_mode="volume", device="cpu", config=RunConfig().to_dict())
    names = tuple(f"img_{k:02d}.png" for k in range(N_FRAMES))
    return DisplacementSeries(grid, mask, u, v, done_arr, names, FrameSchedule.accumulative(N_FRAMES), info)


def expected_strain(k: int, measure: str) -> dict[str, float]:
    ux, uy, vx, vy = gradients(k)
    if measure == "green_lagrange":
        return {"exx": ux + 0.5 * (ux * ux + vx * vx), "eyy": vy + 0.5 * (uy * uy + vy * vy),
                "exy": 0.5 * (uy + vx) + 0.5 * (ux * uy + vx * vy)}
    if measure == "infinitesimal":
        return {"exx": ux, "eyy": vy, "exy": 0.5 * (uy + vx)}
    raise ValueError(measure)


def interior(series, margin: int) -> np.ndarray:
    """ROI points at least *margin* px from the ROI border and the hole."""
    from scipy import ndimage

    return ndimage.binary_erosion(series.mask, iterations=margin, border_value=0)


# ========================================================================================== fixtures
class StubAnalysisTab(QWidget):
    """Records what the strain window tells the Analysis tab."""

    frame_requested = Signal(int)
    instances: list["StubAnalysisTab"] = []

    def __init__(self, state, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.calls: list[tuple] = []
        self.renderer = None
        self.parameters = None
        StubAnalysisTab.instances.append(self)

    def set_frame(self, frame: int) -> None:
        self.calls.append(("set_frame", frame))

    def set_default_field(self, name: str) -> None:
        self.calls.append(("set_default_field", name))

    def set_parameters_provider(self, provider) -> None:
        self.parameters = provider

    def set_field_renderer(self, renderer) -> None:
        self.renderer = renderer

    def last(self, name: str):
        return next((c[1] for c in reversed(self.calls) if c[0] == name), None)


@pytest.fixture
def stub_analysis(monkeypatch):
    StubAnalysisTab.instances = []
    mod = types.ModuleType("raftcorr.gui.panels.analysis")
    mod.AnalysisTab = StubAnalysisTab
    monkeypatch.setitem(sys.modules, "raftcorr.gui.panels.analysis", mod)
    return StubAnalysisTab


@pytest.fixture
def warnings_shown(monkeypatch):
    """Replace the modal warning box; records (title, text)."""
    shown: list[tuple[str, str]] = []

    def fake(_parent, title, text, *_a, **_k):
        shown.append((title, text))
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "warning", fake)
    return shown


@pytest.fixture
def state(qapp):
    from raftcorr.gui.app_state import AppState

    AppState._instance = None
    st = AppState.instance()
    yield st
    AppState._instance = None


@pytest.fixture
def series():
    return make_series()


@pytest.fixture
def make_window(qtbot, state, stub_analysis, warnings_shown):
    from raftcorr.gui.strain_window import StrainWindow

    def _stop(w) -> None:
        worker = w._strain_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(30_000)

    def _make(results=True):
        if results is True:
            state.set_results(make_series())
        elif results is not None and results is not False:
            state.set_results(results)
        w = StrainWindow(state)
        qtbot.addWidget(w, before_close_func=_stop)
        w.show()
        qtbot.waitExposed(w)
        return w

    return _make


@pytest.fixture
def window(make_window):
    return make_window()


def overlay(window):
    return window.canvas()._overlay_item


def has_overlay(window) -> bool:
    return not overlay(window).pixmap().isNull()


def wait_idle(qtbot, window, timeout: int = 30_000) -> None:
    qtbot.waitUntil(lambda: not window.is_computing() and window._compute_btn.isEnabled(), timeout=timeout)


# ============================================================================================ opening
def test_window_opens_with_tabs_and_displacement(window, state, stub_analysis):
    assert window.windowTitle() == "Strain Post-Processing"
    assert [window._tabs.tabText(i) for i in range(window._tabs.count())] == ["Strain Field", "Analysis"]
    assert window.current_field() == "disp_u"
    assert window.strain_current_frame() == 0
    assert window._frame_nav._n_frames == N_FRAMES
    assert window._export_strain_btn.isEnabled()
    assert not window._cancel_btn.isVisibleTo(window)
    # frame 0: zero displacement drawn at the grid origin, colorbar shown
    assert has_overlay(window) and window._colorbar.isVisible()
    pos = overlay(window).pos()
    assert (pos.x(), pos.y()) == (GX0, GY0)
    assert window._colorbar._label == "U (px)"
    # the Analysis tab is wired
    tab = stub_analysis.instances[-1]
    assert tab.last("set_default_field") == "disp_u"
    assert tab.renderer is not None and tab.parameters is not None


def test_window_without_results(make_window, state):
    w = make_window(results=None)
    assert not w._export_strain_btn.isEnabled()
    assert not has_overlay(w)
    w.trigger_compute()  # no-op
    assert state.strain_results is None
    w._on_compute_clicked()
    assert not w.is_computing()


def test_analysis_placeholder_when_unavailable(qtbot, state, monkeypatch, warnings_shown):
    from raftcorr.gui.strain_window import StrainWindow

    broken = types.ModuleType("raftcorr.gui.panels.analysis")

    class Broken(QWidget):
        def __init__(self, *_a, **_k):
            raise RuntimeError("not ported yet")

    broken.AnalysisTab = Broken
    monkeypatch.setitem(sys.modules, "raftcorr.gui.panels.analysis", broken)
    state.set_results(make_series())
    w = StrainWindow(state)
    qtbot.addWidget(w)
    w.show()
    page = w._tabs.widget(1)
    assert isinstance(page, QLabel) and page.text() == "Analysis is not available"
    assert w._tabs.tabText(1) == "Analysis"
    # the field view still works, and switching tabs is harmless
    w._tabs.setCurrentIndex(1)
    w._tabs.setCurrentIndex(0)
    w.set_strain_frame(2)
    assert has_overlay(w)


def test_frame_navigation_is_independent(window, state, stub_analysis):
    state.current_frame = 0
    window.set_strain_frame(2)
    assert window.strain_current_frame() == 2
    assert state.current_frame == 0
    assert window._frame_nav._label.text() == "FRAME 3/4"
    assert stub_analysis.instances[-1].last("set_frame") == 2
    window.set_strain_frame(99)  # clamped
    assert window.strain_current_frame() == N_FRAMES - 1
    # the Analysis tab can move the field view
    stub_analysis.instances[-1].frame_requested.emit(1)
    assert window.strain_current_frame() == 1
    # the navigator's slider too
    window._frame_nav._slider.setValue(3)
    assert window.strain_current_frame() == 3


# ============================================================================================ compute
@pytest.mark.parametrize("measure", ["green_lagrange", "infinitesimal"])
def test_compute_affine_field_gives_exact_strain(window, state, measure):
    from dataclasses import replace

    window.param_panel().set_config(replace(window.param_panel().get_config(), measure=measure, gauge_px=15))
    window.trigger_compute()
    strain = state.strain_results
    assert strain is not None and strain.done.all()
    assert strain.settings.measure == measure and strain.settings.gauge_px == 15
    assert state.strain_config.measure == measure
    inner = interior(state.results, 8)
    for k in range(N_FRAMES):
        exp = expected_strain(k, measure)
        for name in ("exx", "eyy", "exy"):
            got = strain.components[name][k][inner]
            assert np.all(np.isfinite(got))
            np.testing.assert_allclose(got, exp[name], atol=2e-6, err_msg=f"{name} frame {k}")
    assert not window.is_stale()


def test_compute_button_runs_worker(window, state, qtbot):
    assert state.strain_results is None
    window._compute_btn.click()
    assert window._strain_progress.isVisibleTo(window)
    wait_idle(qtbot, window)
    qtbot.waitUntil(lambda: state.strain_results is not None, timeout=30_000)
    assert window._strain_progress_label.text() == "Complete"
    assert not window._cancel_btn.isVisibleTo(window)
    assert state.strain_config == window.param_panel().get_config()
    exp = expected_strain(3, "green_lagrange")["exx"]
    np.testing.assert_allclose(np.nanmedian(state.strain_results.components["exx"][3]), exp, atol=1e-6)
    # the strain window offers the settings for an export header
    params = window._strain_parameters()
    assert params["strain measure"] == "Green-Lagrangian" and params["strain window"].startswith("41 px")


def test_cancel_keeps_previous_strain(window, state, qtbot, monkeypatch):
    from raftcorr.callbacks import check_stop
    import raftcorr.gui.controllers.strain_controller as sc

    window.trigger_compute()
    before = state.strain_results
    assert before is not None
    started = threading.Event()

    def endless(series, cfg, frames=None, callbacks=None):
        started.set()
        deadline = time.perf_counter() + 20.0
        while time.perf_counter() < deadline:
            callbacks.progress(0.25, "")
            check_stop(callbacks)  # raises StopRequested once Cancel is clicked
            time.sleep(0.005)
        raise AssertionError("never cancelled")

    monkeypatch.setattr(sc, "compute_strain", endless)
    window._compute_btn.click()
    assert started.wait(10)
    qtbot.waitUntil(lambda: window._cancel_btn.isVisibleTo(window), timeout=5000)
    assert not window._compute_btn.isEnabled()
    window._cancel_btn.click()
    wait_idle(qtbot, window)
    assert state.strain_results is before
    assert not window._cancel_btn.isVisibleTo(window)
    assert not window._strain_progress.isVisibleTo(window)
    assert "canceled" in window._console.toPlainText()


def test_cancel_clicked_without_worker_is_noop(window):
    window._strain_worker = None
    window._on_cancel_clicked()


def test_compute_error_is_reported(window, state, qtbot, monkeypatch, warnings_shown):
    import raftcorr.gui.controllers.strain_controller as sc

    def fail(*_a, **_k):
        raise ValueError("bad gauge")

    monkeypatch.setattr(sc, "compute_strain", fail)
    window._compute_btn.click()
    wait_idle(qtbot, window)
    qtbot.waitUntil(lambda: bool(warnings_shown), timeout=5000)
    assert warnings_shown[-1][0] == "Strain Computation Failed"
    assert "bad gauge" in warnings_shown[-1][1]
    assert state.strain_results is None


def test_strain_of_replaced_results_is_discarded(state):
    from raftcorr.gui.controllers.strain_controller import StrainController

    old = make_series()
    state.set_results(old)
    ctrl = StrainController(state)
    strain = ctrl.compute_all_frames(state.strain_config, series=old)
    state.set_results(make_series())  # a new run finished meanwhile
    assert not ctrl.store(old, state.strain_config, strain)
    assert state.strain_results is None


def test_controller_reports_progress_and_requires_results(state):
    from raftcorr.gui.controllers.strain_controller import StrainComputationCancelled, StrainController

    ctrl = StrainController(state)
    with pytest.raises(RuntimeError):
        ctrl.compute_all_frames(state.strain_config)
    state.set_results(make_series(done=[True, True, False, True]))
    messages = []
    strain = ctrl.compute_all_frames(state.strain_config, progress_cb=lambda f, m: messages.append((f, m)))
    assert [m for _f, m in messages] == ["Strain frame 1/3", "Strain frame 2/3", "Strain frame 3/3"]
    assert list(strain.done) == [True, True, False, True]
    with pytest.raises(StrainComputationCancelled):
        ctrl.compute_all_frames(state.strain_config, should_stop=lambda: True)


# ======================================================================================= parameters
def test_param_panel_round_trip_and_odd_gauge(qapp):
    from raftcorr.config import StrainConfig
    from raftcorr.gui.widgets.strain_param_panel import StrainParamPanel

    cfg = StrainConfig(measure="euler_almansi", gauge_px=25, weighting="gaussian", poly_order=2, step=3,
                       gaussian_sigma=5.0, smoothing_sigma=1.5, connectivity_aware=False, min_coverage=0.4)
    panel = StrainParamPanel(cfg)
    assert panel.get_config() == cfg
    assert not panel.is_dirty()
    panel._vsg_spin.setValue(30)  # even -> next odd
    assert panel.get_config().gauge_px == 31
    assert panel.is_dirty()
    panel._sigma_spin.setValue(0.0)  # Auto
    assert panel.get_config().gaussian_sigma is None
    panel._weight_combo.setCurrentIndex(panel._weight_combo.findData("uniform"))
    assert not panel._sigma_spin.isEnabled()


def test_paper_settings_button(window, state):
    from dataclasses import replace

    from raftcorr.config import PAPER_STRAIN

    panel = window.param_panel()
    panel._vsg_spin.setValue(21)
    panel._type_combo.setCurrentIndex(panel._type_combo.findData("infinitesimal"))
    panel._method_combo.setCurrentIndex(panel._method_combo.findData(2))
    assert state.strain_config.gauge_px == 21 and state.strain_config.measure == "infinitesimal"
    assert not panel.is_paper_settings()
    window.trigger_compute()
    assert not window.is_stale()
    panel._paper_btn.click()
    assert panel.get_config() == replace(PAPER_STRAIN)
    assert panel.is_paper_settings()
    assert state.strain_config == PAPER_STRAIN
    assert window.is_stale()
    assert "Params changed" in window._stale_label.text()
    window.trigger_compute()
    assert not window.is_stale() and window._stale_label.text() == ""
    assert state.strain_results.settings.gauge_px == PAPER_STRAIN.gauge_px


def test_project_settings_reach_the_panel(window, state):
    from raftcorr.config import StrainConfig

    state.strain_config = StrainConfig(gauge_px=31, step=2)
    state.params_changed.emit()  # what opening a project does
    assert window.param_panel().get_config() == state.strain_config
    assert not window.is_stale()


# ========================================================================================== display
def test_every_component_renders(window, state):
    from raftcorr.gui.widgets.strain_field_selector import FIELD_NAMES, field_colorbar_label

    window.set_strain_frame(2)
    window.set_current_field("strain_exx")
    assert not has_overlay(window)
    assert window._hint_label.text() == "Click Compute Strain to show this field."
    window.trigger_compute()
    for name in FIELD_NAMES:
        window.set_current_field(name)
        assert has_overlay(window), name
        assert window._colorbar.isVisible()
        assert window._colorbar._label == field_colorbar_label(name)
        assert window._hint_label.text() == ""
    window.set_current_field("strain_rotation")
    assert window._colorbar._label.endswith("(deg)")
    window.set_current_field("strain_eyy")
    exp = expected_strain(2, "green_lagrange")["eyy"]
    np.testing.assert_allclose(window.shown_field().value_at(GX0 + 10.0, GY0 + 10.0), exp, atol=1e-6)


def test_displacement_and_velocity_before_compute(window, state):
    window.set_strain_frame(2)
    shown = window.shown_field()
    X, Y = GX0 + 30.0, GY0 + 20.0
    u, _v = affine_uv(2, np.array(X), np.array(Y))
    window.viz_panel().set_show_deformed(False)
    np.testing.assert_allclose(window.shown_field().value_at(X, Y), float(u), atol=1e-4)
    window.set_current_field("velocity")
    assert has_overlay(window) and window._colorbar._label == "Velocity (px/frame)"
    du1, dv1 = affine_uv(1, np.array(X), np.array(Y))
    du3, dv3 = affine_uv(3, np.array(X), np.array(Y))
    speed = np.hypot((du3 - du1) / 2, (dv3 - dv1) / 2)  # central difference over frames 1 and 3
    np.testing.assert_allclose(window.shown_field().value_at(X, Y), float(speed), atol=1e-4)
    window.set_current_field("disp_magnitude")
    assert has_overlay(window)
    assert state.strain_results is None and shown is not None


def test_deformed_configuration(window, state):
    window.set_strain_frame(3)
    window.viz_panel().set_show_deformed(False)
    ref_pos = overlay(window).pos()
    assert (ref_pos.x(), ref_pos.y()) == (GX0, GY0)
    window.viz_panel().set_show_deformed(True)
    assert has_overlay(window)
    shown = window.shown_field()
    # the window of the deformed frame, not the reference grid
    assert (shown.x0, shown.y0) != (GX0, GY0)
    assert (overlay(window).pos().x(), overlay(window).pos().y()) == (shown.x0, shown.y0)
    # a material point X is drawn at x = X + u(X) with its own value u(X)
    X, Y = GX0 + 25.0, GY0 + 30.0
    u, v = affine_uv(3, np.array(X), np.array(Y))
    np.testing.assert_allclose(shown.value_at(X + float(u), Y + float(v)), float(u), atol=0.02)
    # strain in the deformed configuration
    window.trigger_compute()
    window.set_current_field("strain_exx")
    assert has_overlay(window)
    exp = expected_strain(3, "green_lagrange")["exx"]
    np.testing.assert_allclose(window.shown_field().value_at(X + float(u), Y + float(v)), exp, atol=1e-5)
    # frame 0 is its own deformed configuration
    window.set_strain_frame(0)
    assert (window.shown_field().x0, window.shown_field().y0) == (GX0, GY0)


def test_reference_field_image_for_analysis(window, state, stub_analysis):
    from raftcorr.gui.panels.strain_canvas import FieldImage

    renderer = stub_analysis.instances[-1].renderer
    window.set_strain_frame(2)  # the view is deformed by default; the probes need the reference
    img = renderer("disp_u", 2)
    assert isinstance(img, FieldImage) and (img.x, img.y, img.scale) == (GX0, GY0, 1.0)
    assert renderer("strain_exx", 2) is None
    window.trigger_compute()
    img = renderer("strain_exx", 2)
    assert img is not None and not img.pixmap.isNull()
    assert img.label == "εxx"
    assert renderer("no_such_field", 2) is None


def test_untracked_points_and_frames(make_window, state):
    w = make_window(make_series(done=[True, True, False, True], nan_frame=1))
    w.viz_panel().set_show_deformed(False)
    w.set_strain_frame(2)
    assert not has_overlay(w)
    assert w._hint_label.text() == "Frame 3 was not computed."
    w.set_strain_frame(1)
    vals = w.shown_field().values
    assert np.isnan(vals[25, 25])            # hidden (transparent) by default
    w.viz_panel()._hide_untracked_check.setChecked(False)
    vals = w.shown_field().values
    assert np.isfinite(vals[25, 25])         # filled from the nearest tracked point
    assert np.isnan(vals[45, 95])            # the hole stays outside the ROI


def test_velocity_needs_two_frames(make_window, state):
    w = make_window(make_series(done=[True, False, False, False]))
    w.set_current_field("velocity")
    assert not has_overlay(w) and not w._colorbar.isVisible()
    assert w._hint_label.text() == "Velocity needs at least two computed frames."


def test_colour_settings_and_auto_to_fixed(window, state):
    window.set_strain_frame(3)
    window.viz_panel()._cmap_combo.setCurrentText("viridis")
    assert overlay(window).pixmap() is not None and window._colorbar._cmap_name == "viridis"
    assert state.get_field_state("disp_u").colormap == "viridis"
    vmin, vmax = window._last_rendered_vmin, window._last_rendered_vmax
    window.viz_panel()._auto_check.setChecked(False)
    st = window.viz_panel().get_state()
    assert (st["vmin"], st["vmax"]) == pytest.approx((vmin, vmax))
    window.viz_panel()._vmax_spin.setValue(vmax + 1.0)
    assert window._colorbar._vmax == pytest.approx(vmax + 1.0)
    window.viz_panel()._opacity_slider.setValue(40)
    assert overlay(window).opacity() == pytest.approx(0.4)
    # the main window's own display settings are untouched
    assert state.display_field == "disp_u" and state.current_frame == 0
    # switching fields restores each field's own settings
    window.set_current_field("disp_v")
    assert window.viz_panel().get_state()["colormap"] == state.get_field_state("disp_v").colormap
    window.set_current_field("disp_u")
    assert window.viz_panel().get_state()["colormap"] == "viridis"


def test_hidden_background_and_physical_units(window, state):
    window.set_strain_frame(2)
    window.viz_panel()._background_check.setChecked(False)
    bg = window.canvas()._bg_item.pixmap()
    assert (bg.width(), bg.height()) == (IMG_W, IMG_H)
    X, Y = GX0 + 30.0, GY0 + 20.0
    window.viz_panel().set_show_deformed(False)
    raw = window.shown_field().value_at(X, Y)
    state.set_physical_units(True, 0.5, 10.0, "mm")
    assert window._colorbar._label == "U (mm)"
    np.testing.assert_allclose(window.shown_field().value_at(X, Y), raw * 0.5, rtol=1e-5)
    window.set_current_field("velocity")
    assert window._colorbar._label == "Velocity (mm/s)"
    # strain is never scaled by the physical units
    window.trigger_compute()
    window.set_current_field("strain_exx")
    assert window._colorbar._label == "εxx"
    exp = expected_strain(2, "green_lagrange")["exx"]
    np.testing.assert_allclose(window.shown_field().value_at(X, Y), exp, atol=1e-6)


def test_cursor_readout(window, state):
    window.set_strain_frame(1)
    window.viz_panel().set_show_deformed(False)
    X, Y = GX0 + 40.0, GY0 + 30.0
    window._on_scene_hover(X + 0.5, Y + 0.5)  # scene pixel centre of image pixel (X, Y)
    u, _v = affine_uv(1, np.array(X), np.array(Y))
    text = window._readout.text()
    assert text.startswith(f"x {X:.1f}  y {Y:.1f} px")
    assert f"U = {float(u):.5g} px" in text
    window.canvas().hover_left.emit()
    assert window._readout.text() == ""


def test_new_results_reset_strain_view(window, state):
    window.trigger_compute()
    window.set_strain_frame(3)
    window.set_current_field("strain_exx")
    assert has_overlay(window)
    state.set_results(make_series(hole=False))  # a new run: strain is dropped
    assert state.strain_results is None
    assert not has_overlay(window)
    assert window._hint_label.text() == "Click Compute Strain to show this field."


def test_export_opens_dialog_with_this_windows_settings(window, state, monkeypatch):
    from dataclasses import dataclass

    opened = []

    @dataclass
    class Hint:
        colormap: str = "jet"
        auto_range: bool = True
        vmin: float = 0.0
        vmax: float = 1.0
        show_deformed: bool = False
        overlay_alpha: float = 0.7
        use_physical_units: bool = False

    class Dialog:
        def __init__(self, results, folder, hint, **kwargs):
            opened.append((results, hint, kwargs))

        def exec(self):
            return 0

        def deleteLater(self):
            pass

    mod = types.ModuleType("raftcorr.gui.dialogs.export_dialog")
    mod.ExportDialog, mod.VizExportHint = Dialog, Hint
    monkeypatch.setitem(sys.modules, "raftcorr.gui.dialogs.export_dialog", mod)
    window.viz_panel()._cmap_combo.setCurrentText("plasma")
    window.viz_panel()._opacity_slider.setValue(55)
    window._export_strain_btn.click()
    results, hint, kwargs = opened[-1]
    assert results is state.results
    assert hint.colormap == "plasma" and hint.show_deformed is True
    assert hint.overlay_alpha == pytest.approx(0.55)
    assert kwargs["parent"] is window


def test_export_unavailable_is_reported(window, monkeypatch, warnings_shown):
    mod = types.ModuleType("raftcorr.gui.dialogs.export_dialog")

    class Dialog:
        def __init__(self, *_a, **_k):
            raise ImportError("not ported yet")

    mod.ExportDialog, mod.VizExportHint = Dialog, lambda **_k: None
    monkeypatch.setitem(sys.modules, "raftcorr.gui.dialogs.export_dialog", mod)
    window._export_strain_btn.click()
    assert warnings_shown and "not ported yet" in warnings_shown[-1][1]
    assert "Export is not available" in window._console.toPlainText()


def test_close_disconnects_from_state(window, state):
    window.close()
    state.set_results(make_series())  # must not reach the closed window
    assert not window._state_connected
    window.show()
    assert window._state_connected
