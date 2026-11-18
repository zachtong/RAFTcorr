"""Results on the canvas (reference / deformed configuration, readout, colour bar), probes and the
probe chart, using a synthetic displacement series with a known linear field (no GPU needed)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from .gui_helpers import (  # noqa: F401  (fixtures)
    SLOPE,
    _restore_theme,
    dialogs,
    linear_field,
    loaded,
    make_series,
    make_window,
    rect_mask,
    render,
    seq_folder,
    win,
    with_results,
)

pytestmark = pytest.mark.gui

from PySide6.QtWidgets import QPushButton  # noqa: E402

from raftcorr.viz.render import grey_rgba  # noqa: E402


def u_at(k: int, x: float, y: float) -> tuple[float, float]:
    uu, vv = linear_field(k, np.array([x]), np.array([y]))
    return float(uu[0]), float(vv[0])


# ======================================================================================= display
def test_results_are_drawn_in_the_reference_configuration(with_results):
    w = with_results
    st = w.state
    st.set_frame(2)
    render(w)
    assert w.view.last_range is not None
    assert w.colorbar.isVisible()
    assert w.colorbar._label == "u (px)"
    pix = w.canvas._overlay.pixmap()
    g = st.series.grid
    assert (pix.width(), pix.height()) == (g.width, g.height)
    assert (w.canvas._overlay.pos().x(), w.canvas._overlay.pos().y()) == (g.x0, g.y0)
    np.testing.assert_array_equal(w.view.background(), grey_rgba(st.sequence[0]))  # frame 0 under the reference
    lo, hi = u_at(2, 40, 0)[0], u_at(2, 279, 0)[0]  # auto range: trimmed by 0.5 % at each end
    assert lo <= w.view.last_range.vmin < lo + 0.02 * (hi - lo)
    assert hi - 0.02 * (hi - lo) < w.view.last_range.vmax <= hi
    assert "✓" in w.left.images.list.item(2).text()


def test_cursor_readout_in_the_reference_configuration(with_results):
    w = with_results
    w.state.set_frame(2)
    render(w)
    expect = u_at(2, 200.0, 100.0)[0]
    assert w.view.value_at(200.0, 100.0) == pytest.approx(expect, abs=1e-5)
    w._hover(200.0, 100.0)
    assert w.pos_label.text() == "x 200.0  y 100.0 px"
    assert w.value_label.text() == f"u = {expect:.5g} px"
    assert w.view.value_at(10.0, 10.0) is None  # outside the ROI
    w._hover(float("nan"), float("nan"))
    assert w.pos_label.text() == "" and w.value_label.text() == ""


def test_readout_uses_the_display_units(with_results):
    w = with_results
    w.state.set_frame(2)
    w.right.units.unit.setCurrentText("mm")
    w.right.units.pixel_size.setValue(0.5)
    render(w)
    expect = 0.5 * u_at(2, 200.0, 100.0)[0]
    assert w.view.value_at(200.0, 100.0) == pytest.approx(expect, abs=1e-5)
    w._hover(200.0, 100.0)
    assert w.value_label.text().endswith(" mm")


def test_deformed_configuration_overlay_and_readout(with_results):
    w = with_results
    st = w.state
    st.set_frame(2)
    st.update_view(configuration="deformed")
    render(w)
    assert w.view.last_range is not None
    np.testing.assert_array_equal(w.view.background(), grey_rgba(st.sequence[2]))  # the current frame
    du, dv = u_at(2, 200.0, 100.0)
    # the material point X = (200, 100) is drawn at x = X + u(X) and shows u(X)
    assert w.view.value_at(200.0 + du, 100.0 + dv) == pytest.approx(du, abs=0.02)


def test_probe_mapper_moves_points_with_the_material(with_results):
    w = with_results
    st = w.state
    st.set_frame(3)
    assert w._probe_mapper() is None  # reference configuration: probes stay put
    st.update_view(configuration="deformed")
    mapper = w._probe_mapper()
    pts = np.array([[200.0, 100.0], [60.0, 150.0], [5.0, 5.0]])
    out = mapper(pts)
    for (x, y), (xo, yo) in zip(pts[:2], out[:2]):
        du, dv = u_at(3, x, y)
        assert (xo, yo) == pytest.approx((x + du, y + dv), abs=1e-4)
    assert tuple(out[2]) == (5.0, 5.0)  # outside the ROI: not moved
    st.set_results(make_series(rect_mask(), done=[True, True, True, False], names=st.sequence.names))
    st.set_frame(3)
    assert w._probe_mapper() is None  # frame not computed


def test_background_modes(with_results):
    w = with_results
    st = w.state
    st.set_frame(1)
    st.update_view(background="frame")
    np.testing.assert_array_equal(w.view.background(), grey_rgba(st.sequence[1]))
    st.update_view(background="reference", configuration="deformed")
    np.testing.assert_array_equal(w.view.background(), grey_rgba(st.sequence[0]))
    st.update_view(background="none")
    bg = w.view.background()
    assert (bg[..., :3] == 255).all()  # white, as in exported pictures
    st.update_view(show_overlay=False)  # no field: "none" falls back to the frame itself
    np.testing.assert_array_equal(w.view.background(), grey_rgba(st.sequence[1]))
    render(w)
    assert w.view.last_range is None and not w.colorbar.isVisible()


def test_sequence_range_scope_covers_every_frame(with_results):
    w = with_results
    st = w.state
    st.update_view(range_scope="sequence")
    st.set_frame(1)
    render(w)
    r1 = w.view.last_range
    st.set_frame(3)
    render(w)
    assert w.view.last_range == r1  # fixed while playing
    assert u_at(3, 200, 0)[0] < r1.vmax <= u_at(3, 279, 0)[0]  # frame 1 alone never exceeds 3.2 px
    assert r1.vmin < u_at(1, 279, 0)[0]


def test_field_keys_choose_the_field(with_results):
    w = with_results
    w._field_key(1)
    assert w.state.view.field == "v"
    w._field_key(2)
    assert w.state.view.field == "magnitude"
    w._field_key(8)  # no ninth field without strain: ignored
    assert w.state.view.field == "magnitude"


def test_velocity_field_uses_the_frame_rate(with_results):
    w = with_results
    st = w.state
    st.set_frame(2)
    st.update_view(field="vel_u")
    render(w)
    # central difference of u over frames 1 and 3 at X = 200
    expect = (u_at(3, 200.0, 100.0)[0] - u_at(1, 200.0, 100.0)[0]) / 2.0
    assert w.view.value_at(200.0, 100.0) == pytest.approx(expect, abs=1e-5)
    assert w.colorbar._label == "vel_u (px/frame)"
    w.right.units.rate_on.setChecked(True)
    w.right.units.rate.setValue(10.0)
    render(w)
    assert w.view.value_at(200.0, 100.0) == pytest.approx(10.0 * expect, abs=1e-4)
    assert w.colorbar._label == "vel_u (px/s)"


def test_new_results_reset_a_strain_field(with_results, qtbot):
    w = with_results
    w.compute_strain()
    qtbot.waitUntil(lambda: w._worker is None, timeout=30000)
    assert w.state.view.field == "exx"
    w.state.set_results(None)
    assert w.state.view.field == "u"


def test_strain_of_the_synthetic_field(with_results, qtbot):
    w = with_results
    w.right.strain.go.click()
    qtbot.waitUntil(lambda: w._worker is None, timeout=30000)
    st = w.state
    assert st.strain is not None and st.view.field == "exx"
    assert "strain computed (green_lagrange, Lg 41 px, uniform)" in w.logview.toPlainText()
    st.set_frame(3)
    render(w)
    a = 3 * SLOPE
    assert w.view.value_at(160.0, 128.0) == pytest.approx(a + 0.5 * a * a, abs=1e-5)  # Green-Lagrange
    assert w.colorbar._label == "exx (-)"


# ============================================================================== probes and chart
def _add_all_probe_kinds(w):
    c = w.canvas
    c.shape_finished.emit("probe_point", [(200.0, 100.0)])
    c.shape_finished.emit("probe_line", [(60.0, 80.0), (120.0, 90.0), (260.0, 80.0)])  # drag path: ends used
    c.shape_finished.emit("probe_rect", [(100.0, 60.0), (150.0, 100.0)])
    c.shape_finished.emit("probe_circle", [(200.0, 150.0), (220.0, 150.0)])
    c.shape_finished.emit("probe_polygon", [(60.0, 120.0), (120.0, 120.0), (120.0, 200.0), (60.0, 200.0)])


def test_probes_get_labels_colours_and_table_rows(loaded):
    from raftcorr.post.probes import COLORS

    w = loaded
    _add_all_probe_kinds(w)
    ps = w.state.probes
    assert [p.label for p in ps] == ["P1", "L2", "A3", "A4", "A5"]
    assert [p.kind for p in ps] == ["point", "line", "rect", "circle", "polygon"]
    assert [p.color for p in ps] == list(COLORS[:5])
    assert ps[1].points == ((60.0, 80.0), (260.0, 80.0))
    t = w.right.probes.table
    assert t.rowCount() == 5
    assert [t.item(i, 0).text() for i in range(5)] == ["P1", "L2", "A3", "A4", "A5"]
    assert t.item(1, 2).text() == "(60, 80) - (260, 80), L0 = 200.0 px"
    assert w.tabs.currentWidget() is w.chart
    assert "probe A4: centre (200, 150), r = 20.0 px" in w.logview.toPlainText()
    render(w)
    assert len(w.canvas._probe_items) >= 5


def test_invalid_probe_is_rejected_with_a_warning(loaded):
    loaded.canvas.shape_finished.emit("probe_polygon", [(1.0, 1.0), (5.0, 5.0)])
    assert loaded.state.probes == []
    assert "at least 3 points" in loaded.logview.toPlainText()


def test_remove_and_clear_probes(loaded):
    w = loaded
    _add_all_probe_kinds(w)
    p = w.right.probes
    p.table.selectRow(1)
    p._remove()
    assert [q.label for q in w.state.probes] == ["P1", "A3", "A4", "A5"]
    p.table.clearSelection()
    p._remove()  # nothing selected: the last probe goes
    assert [q.label for q in w.state.probes] == ["P1", "A3", "A4"]
    w.canvas.shape_finished.emit("probe_point", [(10.0, 10.0)])
    assert w.state.probes[-1].label == "P5"  # ids keep increasing
    next(b for b in p.findChildren(QPushButton) if b.text() == "Clear").click()
    assert w.state.probes == [] and p.table.rowCount() == 0


def test_chart_without_and_with_results(loaded):
    w = loaded
    _add_all_probe_kinds(w)
    assert w.chart.title.text() == "Run the correlation to plot probe time series."
    st = w.state
    st.set_roi(0, rect_mask())
    st.set_results(make_series(rect_mask(), names=st.sequence.names))
    labels = [ln.get_label() for ln in w.chart.ax.lines]
    for lab in ("P1", "L2", "A3", "A4", "A5"):
        assert lab in labels
    p1 = next(ln for ln in w.chart.ax.lines if ln.get_label() == "P1")
    expect = [u_at(k, 200.0, 100.0)[0] for k in range(4)]
    np.testing.assert_allclose(p1.get_ydata(), expect, atol=1e-5)
    assert "areas and lines: mean" in w.chart.title.text()
    w.right.probes.metric_combo.setCurrentText("max")
    assert "areas and lines: max" in w.chart.title.text()
    a3 = next(ln for ln in w.chart.ax.lines if ln.get_label() == "A3")
    np.testing.assert_allclose(a3.get_ydata()[1:], [u_at(k, 150.0, 0)[0] for k in (1, 2, 3)], atol=1e-4)
    st.clear_probes()
    assert w.chart.title.text() == "Place probes on the canvas to plot their time series."


def test_chart_extensometer_and_frame_click(with_results):
    w = with_results
    w.canvas.shape_finished.emit("probe_line", [(60.0, 100.0), (260.0, 100.0)])
    w.chart.ext.setChecked(True)
    line = next(ln for ln in w.chart.ax.lines if ln.get_label() == "L1 strain")
    np.testing.assert_allclose(line.get_ydata(), [k * SLOPE for k in range(4)], atol=1e-6)
    assert w.chart.ax.get_ylabel() == "strain (-)"
    w.chart._clicked(SimpleNamespace(inaxes=w.chart.ax, xdata=2.2))
    assert w.state.current_frame == 2
    w.chart._clicked(SimpleNamespace(inaxes=w.chart.ax, xdata=99.0))
    assert w.state.current_frame == 3


def test_chart_refreshes_on_theme_and_unit_changes(with_results):
    w = with_results
    w.canvas.shape_finished.emit("probe_point", [(200.0, 100.0)])
    w.set_theme("dark")
    w.right.units.unit.setCurrentText("mm")
    w.right.units.pixel_size.setValue(0.1)
    p1 = next(ln for ln in w.chart.ax.lines if ln.get_label() == "P1")
    np.testing.assert_allclose(p1.get_ydata(), [0.1 * u_at(k, 200.0, 100.0)[0] for k in range(4)], atol=1e-6)
    assert w.chart.ax.get_ylabel().endswith("(mm)")


def test_probes_are_drawn_at_the_deformed_position(with_results):
    w = with_results
    st = w.state
    w.canvas.shape_finished.emit("probe_point", [(200.0, 100.0)])
    st.set_frame(2)
    st.update_view(configuration="deformed")
    render(w)
    marker = next(it for it in w.canvas._probe_items if type(it).__name__ == "QGraphicsEllipseItem")
    du, dv = u_at(2, 200.0, 100.0)
    assert (marker.pos().x(), marker.pos().y()) == pytest.approx((200.0 + du, 100.0 + dv), abs=1e-3)


def test_probe_placed_in_the_deformed_configuration_lands_where_clicked(with_results):
    w = with_results
    st = w.state
    st.set_frame(2)
    st.update_view(configuration="deformed")
    du, dv = u_at(2, 200.0, 100.0)
    clicked = (200.0 + du, 100.0 + dv)  # where material point (200, 100) is on the deformed frame
    w.set_tool("probe_point")
    w.canvas.shape_finished.emit("probe_point", [clicked])
    drawn = w._probe_mapper()(np.array(st.probes[0].points))[0]
    assert tuple(drawn) == pytest.approx(clicked, abs=0.05)


def test_editing_the_mask_of_frame_k_shows_frame_k(with_results):
    w = with_results
    w.state.set_frame(2)
    w.set_tool("rect")
    np.testing.assert_array_equal(w.view.background(), grey_rgba(w.state.sequence[2]))


def test_roi_edit_after_a_run_marks_results_stale(with_results):
    w = with_results
    st = w.state
    assert not st.is_stale and w.left.run.stale.isHidden()
    w.canvas.shape_finished.emit("rect", [(10.0, 10.0), (30.0, 30.0)])
    assert st.is_stale
    assert not w.left.run.stale.isHidden()
    st.undo_roi()
    assert not st.is_stale and w.left.run.stale.isHidden()
    st.set_frame(2)
    w.canvas.shape_finished.emit("rect", [(10.0, 10.0), (30.0, 30.0)])  # a crack mask on frame 2 matters too
    assert st.is_stale
