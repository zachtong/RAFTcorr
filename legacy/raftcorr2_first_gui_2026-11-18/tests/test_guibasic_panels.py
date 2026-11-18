"""Sidebar panels (settings, model, display, units, strain) and the frame navigator."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from .gui_helpers import (  # noqa: F401  (fixtures)
    _restore_theme,
    dialogs,
    loaded,
    make_window,
    render,
    seq_folder,
    win,
    with_results,
)

pytestmark = pytest.mark.gui

from PySide6.QtWidgets import QPushButton  # noqa: E402

from raftcorr.config import PAPER_STRAIN, RunConfig, StrainConfig, TilingConfig, TrackingConfig  # noqa: E402
from raftcorr.gui_basic.state import FieldDisplay, ViewSettings  # noqa: E402


# ===================================================================================== settings
def test_settings_widgets_update_the_run_config(win):
    p = win.left.settings
    st = win.state
    p.iters.setValue(12)
    assert st.run_config.iters == 12
    p.smooth.setValue(1.5)
    assert st.run_config.smoothing_sigma == 1.5
    p.mode.setCurrentIndex(p.mode.findData("incremental"))
    assert st.run_config.tracking.mode == "incremental"
    p.mode.setCurrentIndex(p.mode.findData("every_n"))
    p.every.setValue(3)
    assert st.run_config.tracking == TrackingConfig("every_n", 3, ())
    assert not p.every_row.isHidden() and p.keys.isHidden()
    p.mode.setCurrentIndex(p.mode.findData("custom"))
    assert p.every_row.isHidden() and not p.keys.isHidden()
    p.keys.setText("0; 2")
    p.keys.editingFinished.emit()
    assert st.run_config.tracking.key_frames == (0, 2)
    p.keys.setText("zero, two")  # unreadable: the previous key frames are kept
    p.keys.editingFinished.emit()
    assert st.run_config.tracking.key_frames == (0, 2)
    p.memfrac.setValue(0.5)
    assert st.run_config.tiling.memory_fraction == 0.5
    p.blend.setValue(16)
    assert st.run_config.tiling.blend_px == 16
    p.corr.setCurrentIndex(p.corr.findData("tile"))
    assert st.run_config.corr_mode == "tile"


def test_margin_auto_and_manual(win):
    p = win.left.settings
    st = win.state
    assert p.margin_auto.isChecked() and st.run_config.tiling.margin_px is None
    assert not p.margin.isEnabled()
    assert p.margin.value() == 128  # s4: 32 * 4
    p.margin_auto.setChecked(False)
    assert p.margin.isEnabled()
    assert st.run_config.tiling.margin_px == 128
    p.margin.setValue(64)
    assert st.run_config.tiling.margin_px == 64
    p.margin_auto.setChecked(True)
    assert st.run_config.tiling.margin_px is None
    assert not p.margin.isEnabled()


def test_settings_sync_reflects_state_changes(win):
    p = win.left.settings
    st = win.state
    cfg = replace(st.run_config, iters=7, smoothing_sigma=2.0, corr_mode="volume",
                  tracking=TrackingConfig("custom", 10, (0, 5)),
                  tiling=TilingConfig(margin_px=40, memory_fraction=0.3, blend_px=8))
    st.set_run_config(cfg)
    assert p.iters.value() == 7
    assert p.smooth.value() == 2.0
    assert p.mode.currentData() == "custom"
    assert p.keys.text() == "0, 5"
    assert not p.margin_auto.isChecked() and p.margin.value() == 40
    assert p.memfrac.value() == pytest.approx(0.3)
    assert p.blend.value() == 8
    assert p.corr.currentData() == "volume"
    assert st.run_config == cfg  # sync must not write back a different config


def test_model_buttons_and_device_update_the_config(win):
    m = win.left.model
    st = win.state
    assert m.buttons["s4"].isChecked()  # the default model
    for key in ("s8", "s1", "s2"):
        if not m.buttons[key].isEnabled():
            pytest.skip(f"weights for {key} not installed")
        m.buttons[key].click()
        assert st.run_config.model == key
        assert sum(b.isChecked() for b in m.buttons.values()) == 1
    st.set_run_config(replace(st.run_config, model="s8"))
    assert m.buttons["s8"].isChecked()
    assert win.left.settings.margin.value() == 256  # auto margin follows the model's reach
    m.device.setCurrentText("cpu")
    assert st.run_config.device == "cpu"
    st.set_run_config(replace(st.run_config, device="cuda"))
    assert m.device.currentText() == "cuda"


def test_model_change_marks_results_stale(with_results):
    st = with_results.state
    assert not st.is_stale
    with_results.left.model.buttons["s8"].click()
    assert st.is_stale
    assert not with_results.left.run.stale.isHidden()
    with_results.left.model.buttons["s4"].click()
    assert not st.is_stale
    assert with_results.left.run.stale.isHidden()


def test_run_panel_status_follows_the_workflow(win, seq_folder):
    from .gui_helpers import rect_mask

    run = win.left.run
    assert not run.stop_btn.isEnabled()
    win.load_folder(seq_folder)
    assert "Draw an ROI on frame 0" in run.status.text()
    win.state.set_roi(0, rect_mask())
    assert run.status.text() == "Ready."


# ======================================================================================= display
def test_display_widgets_update_the_view(with_results):
    d = with_results.right.display
    st = with_results.state
    d.field.setCurrentIndex(d.field.findData("v"))
    assert st.view.field == "v"
    d.cmap.setCurrentText("viridis")
    assert st.view.display("v").colormap == "viridis"
    assert st.view.display("u").colormap == "turbo"  # colours are per field
    d.levels.setCurrentIndex(d.levels.findData(8))
    assert st.view.display("v").levels == 8
    d.opacity.setValue(40)
    assert st.view.opacity == pytest.approx(0.40)
    d.def_btn.click()
    assert st.view.configuration == "deformed"
    d.ref_btn.click()
    assert st.view.configuration == "reference"
    d.background.setCurrentIndex(d.background.findData("none"))
    assert st.view.background == "none"
    d.scope.setCurrentIndex(d.scope.findData("sequence"))
    assert st.view.range_scope == "sequence"
    d.show_roi.setChecked(False)
    d.show_overlay.setChecked(False)
    assert st.view.show_roi is False and st.view.show_overlay is False


def test_manual_range_and_auto_range_readout(with_results):
    d = with_results.right.display
    st = with_results.state
    render(with_results)
    rng = with_results.view.last_range
    assert rng is not None
    assert d.vmin.value() == pytest.approx(rng.vmin, abs=1e-6)
    assert d.vmax.value() == pytest.approx(rng.vmax, abs=1e-6)
    assert not d.vmin.isEnabled()
    d.auto.setChecked(False)
    assert st.view.display("u").auto is False
    assert d.vmin.isEnabled() and not d.scope.isEnabled()
    d.vmin.setValue(-1.0)
    d.vmax.setValue(3.0)
    assert (st.view.display("u").vmin, st.view.display("u").vmax) == (-1.0, 3.0)
    render(with_results)
    assert (with_results.view.last_range.vmin, with_results.view.last_range.vmax) == (-1.0, 3.0)
    assert with_results.colorbar._vmin == -1.0 and with_results.colorbar._vmax == 3.0
    d.vmax.setValue(-5.0)  # max below min: nudged above min
    disp = st.view.display("u")
    assert disp.vmax > disp.vmin == -1.0
    d.auto.setChecked(True)
    assert st.view.display("u").auto is True


def test_display_sync_reflects_the_view(with_results):
    d = with_results.right.display
    st = with_results.state
    v = st.view.with_display("v", colormap="RdBu_r", auto=False, vmin=-2.0, vmax=2.0, levels=10)
    st.set_view(replace(v, field="v", configuration="deformed", opacity=0.2, background="frame"))
    assert d.field.currentData() == "v"
    assert d.cmap.currentText() == "RdBu_r"
    assert d.levels.currentData() == 10
    assert not d.auto.isChecked()
    assert (d.vmin.value(), d.vmax.value()) == (-2.0, 2.0)
    assert d.def_btn.isChecked() and not d.ref_btn.isChecked()
    assert d.opacity.value() == 20
    assert d.background.currentData() == "frame"
    assert d.readout.text() == "Units: px"


def test_field_list_grows_with_strain(with_results, qtbot):
    d = with_results.right.display
    keys = [d.field.itemData(i) for i in range(d.field.count())]
    assert keys == ["u", "v", "magnitude", "vel_u", "vel_v", "speed"]
    with_results.compute_strain()
    qtbot.waitUntil(lambda: with_results._worker is None, timeout=30000)
    keys = [d.field.itemData(i) for i in range(d.field.count())]
    assert "exx" in keys and "von_mises" in keys
    assert with_results.state.view.field == "exx"
    assert d.field.currentData() == "exx"


def test_view_settings_round_trip():
    v = ViewSettings(field="v", configuration="deformed", background="none", opacity=0.3, show_roi=False,
                     show_overlay=False, range_scope="sequence",
                     fields={"v": FieldDisplay("viridis", False, -1.5, 2.5, 8), "u": FieldDisplay()})
    assert ViewSettings.from_dict(v.to_dict()) == v
    assert ViewSettings.from_dict(json.loads(json.dumps(v.to_dict()))) == v
    assert ViewSettings.from_dict({}) == ViewSettings()
    partial = ViewSettings.from_dict({"field": "magnitude", "bogus": 1, "fields": {"u": {"auto": False, "x": 0}}})
    assert partial.field == "magnitude" and partial.display("u").auto is False
    assert partial.display("u").colormap == FieldDisplay().colormap


# ========================================================================================== units
def test_units_panel(with_results):
    u = with_results.right.units
    st = with_results.state
    assert not u.pixel_size.isEnabled()
    u.unit.setCurrentText("mm")
    u.pixel_size.setValue(0.01)
    assert st.units.length_unit == "mm" and st.units.pixel_size == 0.01
    assert u.pixel_size.isEnabled()
    u.rate_on.setChecked(True)
    u.rate.setValue(50.0)
    assert st.units.frame_rate == 50.0
    render(with_results)
    assert with_results.colorbar._label == "u (mm)"
    assert with_results.right.display.readout.text() == "Units: mm"
    u.unit.setCurrentText("px")
    # the calibration is kept while px is selected; px means no scaling
    assert st.units.length_unit == "px" and st.units.scale == 1.0
    u.rate_on.setChecked(False)
    assert st.units.frame_rate is None


# ========================================================================================= strain
def _paper_button(panel) -> QPushButton:
    return next(b for b in panel.findChildren(QPushButton) if b.text() == "Paper settings")


def test_strain_widgets_update_the_strain_config(win):
    s = win.right.strain
    st = win.state
    s.measure.setCurrentIndex(s.measure.findData("infinitesimal"))
    s.weight.setCurrentIndex(s.weight.findData("gaussian"))
    s.order.setCurrentIndex(s.order.findData(2))
    s.step.setValue(4)
    s.presmooth.setValue(1.0)
    s.crack.setChecked(False)
    c = st.strain_config
    assert (c.measure, c.weighting, c.poly_order, c.step, c.smoothing_sigma, c.connectivity_aware) == \
        ("infinitesimal", "gaussian", 2, 4, 1.0, False)
    assert not s.go.isEnabled()  # no results yet
    assert s.status.text() == "Run the correlation first."


def test_paper_settings_button_restores_the_paper_strain(win):
    s = win.right.strain
    st = win.state
    s.measure.setCurrentIndex(s.measure.findData("euler_almansi"))
    s.gauge.setValue(21)
    s.step.setValue(3)
    assert st.strain_config != PAPER_STRAIN
    _paper_button(s).click()
    assert st.strain_config == PAPER_STRAIN
    assert s.measure.currentData() == "green_lagrange"
    assert s.gauge.value() == 41 and s.step.value() == 1


def test_even_gauge_becomes_the_next_odd_size(win):
    s = win.right.strain
    s.gauge.setValue(42)
    assert win.state.strain_config.gauge_px == 43
    assert s.gauge.value() == 43


def test_even_gauge_below_the_current_size_is_shown_as_used(win):
    s = win.right.strain
    assert win.state.strain_config.gauge_px == 41
    s.gauge.setValue(40)
    assert s.gauge.value() == win.state.strain_config.gauge_px


def test_strain_status_after_compute_and_after_a_settings_change(with_results, qtbot):
    s = with_results.right.strain
    assert s.go.isEnabled()
    assert s.status.text() == "Not computed yet."
    s.go.click()
    qtbot.waitUntil(lambda: with_results._worker is None, timeout=30000)
    assert with_results.state.strain is not None
    assert "Computed: Green-Lagrange, Lg 41 px" in s.status.text()
    assert "(paper settings)" in s.status.text()
    assert "Settings changed" not in s.status.text()
    s.gauge.setValue(21)
    assert "Settings changed since: compute again." in s.status.text()
    assert s.bar.isHidden()


# ================================================================================ frame navigator
@pytest.fixture
def nav(qtbot):
    from raftcorr.gui_basic.widgets import FrameNavigator

    n = FrameNavigator()
    qtbot.addWidget(n)
    n.frame_requested.connect(n.set_frame)  # what the window does through the state
    n.set_count(5, [f"f{i}.png" for i in range(5)])
    return n


def test_navigator_step_with_and_without_loop(nav):
    assert nav.isEnabled() and nav.slider.maximum() == 4
    nav.step(1)
    assert nav.slider.value() == 1
    assert nav.label.text() == "frame 1 / 4  f1.png"
    nav.request(4)
    nav.step(1)
    assert nav.slider.value() == 0  # loop on by default
    nav.step(-1)
    assert nav.slider.value() == 4
    nav.loop.setChecked(False)
    nav.step(1)
    assert nav.slider.value() == 4
    nav.request(0)
    nav.step(-1)
    assert nav.slider.value() == 0
    nav.request(99)  # clamped
    assert nav.slider.value() == 4


def test_navigator_playable_frames(nav):
    nav.set_playable([4, 0, 2])
    nav.step(1)
    assert nav.slider.value() == 2
    nav.step(1)
    assert nav.slider.value() == 4
    nav.step(1)
    assert nav.slider.value() == 0  # loop
    nav.step(-1)
    assert nav.slider.value() == 4
    nav.loop.setChecked(False)
    nav.step(1)
    assert nav.slider.value() == 4
    nav.request(0)
    nav.step(-1)
    assert nav.slider.value() == 0
    nav.set_playable(None)
    nav.step(1)
    assert nav.slider.value() == 1


def test_navigator_play_and_stop(nav, qtbot):
    nav.speed.setCurrentText("30")
    nav.loop.setChecked(False)
    nav.toggle_play()
    assert nav.playing
    qtbot.waitUntil(lambda: not nav.playing, timeout=3000)  # stops at the last frame without loop
    assert nav.slider.value() == 4
    nav.loop.setChecked(True)
    nav.toggle_play()
    qtbot.waitUntil(lambda: nav.slider.value() == 1, timeout=3000)  # wrapped around
    nav.toggle_play()
    assert not nav.playing


def test_navigator_cannot_play_a_single_frame(qtbot):
    from raftcorr.gui_basic.widgets import FrameNavigator

    n = FrameNavigator()
    qtbot.addWidget(n)
    assert not n.isEnabled()
    n.set_count(1)
    n.toggle_play()
    assert not n.playing


def test_window_navigator_is_restricted_to_computed_frames(loaded):
    from .gui_helpers import make_series, rect_mask

    st = loaded.state
    st.set_results(make_series(rect_mask(), done=[True, False, True, False], names=st.sequence.names))
    assert loaded.nav._frames == [0, 2]
    loaded.nav.step(1)
    assert st.current_frame == 2
    loaded.nav.step(1)
    assert st.current_frame == 0
    st.set_results(None)
    assert loaded.nav._frames is None


def test_run_config_json_round_trip_of_gui_settings(win):
    p = win.left.settings
    p.mode.setCurrentIndex(p.mode.findData("custom"))
    p.keys.setText("0, 2")
    p.keys.editingFinished.emit()
    p.margin_auto.setChecked(False)
    cfg = win.state.run_config
    assert RunConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    assert StrainConfig.from_dict(json.loads(json.dumps(win.state.strain_config.to_dict()))) == win.state.strain_config
