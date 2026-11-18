"""Region of interest: shapes, add / cut, undo / redo, brush, per-frame masks, mask files, and the
canvas mouse tools (drag, click, polygon, wheel zoom, pan)."""

from __future__ import annotations

import numpy as np
import pytest

from .gui_helpers import (  # noqa: F401  (fixtures)
    H,
    W,
    _restore_theme,
    dialogs,
    loaded,
    make_window,
    scene_xy,
    seq_folder,
    view_pos,
    win,
)

pytestmark = pytest.mark.gui

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton  # noqa: E402

from raftcorr.roi.shapes import Circle, Polygon, Rect, brush_stroke, rasterize  # noqa: E402


def _button(panel, text: str) -> QPushButton:
    return next(b for b in panel.findChildren(QPushButton) if b.text() == text)


def _rect(x0, y0, x1, y1):
    return rasterize(Rect(x0, y0, x1, y1), H, W)


# ======================================================================================== shapes
def test_rect_shape_adds_a_mask_on_the_current_frame(loaded):
    st = loaded.state
    loaded.canvas.shape_finished.emit("rect", [(50.0, 60.0), (100.0, 120.0)])
    m = st.roi(0)
    assert m is not None and m.dtype == bool and m.shape == (H, W)
    np.testing.assert_array_equal(m, _rect(50, 60, 100, 120))
    assert m.sum() == 51 * 61
    assert not m.flags.writeable  # stored masks are frozen
    assert "Frame 0 ROI: 3,111 px, box 51 x 61 at (50, 60)" in loaded.left.roi.status.text()
    assert "ROI" in loaded.left.images.list.item(0).text()
    assert loaded.left.run.status.text() == "Ready."


def test_reversed_drag_gives_the_same_rect(loaded):
    loaded.canvas.shape_finished.emit("rect", [(100.0, 120.0), (50.0, 60.0)])
    np.testing.assert_array_equal(loaded.state.roi(0), _rect(50, 60, 100, 120))


def test_cut_mode_removes_the_shape(loaded):
    st = loaded.state
    loaded.canvas.shape_finished.emit("rect", [(20.0, 20.0), (200.0, 200.0)])
    loaded.left.roi.mode_cut.setChecked(True)
    assert loaded.left.roi.mode == "cut"
    loaded.canvas.shape_finished.emit("rect", [(50.0, 50.0), (100.0, 100.0)])
    np.testing.assert_array_equal(st.roi(0), _rect(20, 20, 200, 200) & ~_rect(50, 50, 100, 100))
    loaded.left.roi.mode_add.setChecked(True)
    assert loaded.left.roi.mode == "add"


def test_cutting_everything_removes_the_mask(loaded):
    loaded.canvas.shape_finished.emit("rect", [(50.0, 50.0), (60.0, 60.0)])
    loaded.left.roi.mode_cut.setChecked(True)
    loaded.canvas.shape_finished.emit("rect", [(0.0, 0.0), (300.0, 250.0)])
    assert loaded.state.roi(0) is None
    assert "no ROI" in loaded.left.roi.status.text()


def test_circle_and_polygon_shapes(loaded):
    st = loaded.state
    loaded.canvas.shape_finished.emit("circle", [(160.0, 128.0), (190.0, 168.0)])
    circle = rasterize(Circle(160.0, 128.0, 50.0), H, W)
    np.testing.assert_array_equal(st.roi(0), circle)
    tri = [(10.0, 10.0), (60.0, 10.0), (10.0, 60.0)]
    loaded.canvas.shape_finished.emit("polygon", tri)
    np.testing.assert_array_equal(st.roi(0), circle | rasterize(Polygon(tuple(tri)), H, W))


def test_degenerate_polygon_is_ignored(loaded):
    loaded.canvas.shape_finished.emit("polygon", [(10.0, 10.0), (60.0, 10.0)])
    assert loaded.state.roi(0) is None
    assert not loaded.state.roi_history.can_undo


def test_shapes_are_ignored_without_images(win):
    win.canvas.shape_finished.emit("rect", [(1.0, 1.0), (10.0, 10.0)])
    win.canvas.brush_stroke.emit([(5.0, 5.0)], 3.0, True)
    assert win.state.roi_masks == {}


def test_mask_on_another_frame_is_a_per_frame_mask(loaded):
    st = loaded.state
    st.set_frame(2)
    loaded.canvas.shape_finished.emit("rect", [(30.0, 30.0), (90.0, 90.0)])
    assert st.roi(0) is None
    np.testing.assert_array_equal(st.roi(2), _rect(30, 30, 90, 90))
    assert "[mask]" in loaded.left.images.list.item(2).text()
    txt = loaded.left.roi.status.text()
    assert "Frame 2 mask:" in txt and "Masks on frames: 2" in txt
    st.set_frame(0)
    assert "Frame 0: no ROI. Draw one to start." in loaded.left.roi.status.text()


# ===================================================================================== undo / redo
def test_undo_redo_restore_exact_masks_across_frames(loaded):
    st = loaded.state
    c = loaded.canvas
    c.shape_finished.emit("rect", [(10.0, 10.0), (100.0, 100.0)])
    m1 = st.roi(0).copy()
    c.shape_finished.emit("circle", [(150.0, 150.0), (170.0, 150.0)])
    m2 = st.roi(0).copy()
    st.set_frame(3)
    c.shape_finished.emit("rect", [(5.0, 5.0), (40.0, 40.0)])
    f3 = st.roi(3).copy()

    st.undo_roi()  # frame 3 edit
    assert st.roi(3) is None and st.current_frame == 3
    st.undo_roi()  # the circle on frame 0; jumps to frame 0
    assert st.current_frame == 0
    np.testing.assert_array_equal(st.roi(0), m1)
    st.undo_roi()
    assert st.roi(0) is None
    st.undo_roi()  # nothing left: no-op
    assert st.roi_masks == {}

    st.redo_roi()
    np.testing.assert_array_equal(st.roi(0), m1)
    st.redo_roi()
    np.testing.assert_array_equal(st.roi(0), m2)
    st.redo_roi()
    np.testing.assert_array_equal(st.roi(3), f3)
    assert st.current_frame == 3
    assert not st.roi_history.can_redo


def test_new_edit_clears_redo(loaded):
    st = loaded.state
    loaded.canvas.shape_finished.emit("rect", [(10.0, 10.0), (50.0, 50.0)])
    st.undo_roi()
    assert st.roi_history.can_redo
    loaded.canvas.shape_finished.emit("rect", [(60.0, 60.0), (80.0, 80.0)])
    assert not st.roi_history.can_redo


def test_undo_redo_buttons_and_menu_actions(loaded):
    st = loaded.state
    loaded.canvas.shape_finished.emit("rect", [(10.0, 10.0), (50.0, 50.0)])
    _button(loaded.left.roi, "Undo").click()
    assert st.roi(0) is None
    _button(loaded.left.roi, "Redo").click()
    np.testing.assert_array_equal(st.roi(0), _rect(10, 10, 50, 50))
    loaded.undo_act.trigger()
    assert st.roi(0) is None
    loaded.redo_act.trigger()
    assert st.roi(0) is not None


def test_invert_and_clear_buttons(loaded):
    st = loaded.state
    roi = loaded.left.roi
    _button(roi, "Invert").click()  # no mask yet: everything
    assert st.roi(0).all()
    _button(roi, "Clear").click()
    assert st.roi(0) is None
    loaded.canvas.shape_finished.emit("rect", [(10.0, 10.0), (50.0, 50.0)])
    _button(roi, "Invert").click()
    np.testing.assert_array_equal(st.roi(0), ~_rect(10, 10, 50, 50))
    st.undo_roi()
    np.testing.assert_array_equal(st.roi(0), _rect(10, 10, 50, 50))
    n_undo = len(st.roi_history._undo)
    st.set_frame(1)
    _button(roi, "Clear").click()  # nothing to clear on frame 1: no undo step
    assert len(st.roi_history._undo) == n_undo


# ========================================================================================= brush
def test_brush_stroke_commits_one_undo_step(loaded):
    st = loaded.state
    c = loaded.canvas
    pts = [(100.0, 100.0), (120.0, 100.0), (140.0, 110.0)]
    n0 = len(st.roi_history._undo)
    c.brush_stroke.emit(pts[:1], 8.0, False)
    c.brush_stroke.emit(pts[:2], 8.0, False)
    assert st.roi(0) is None  # preview only while dragging
    c.brush_stroke.emit(pts, 8.0, True)
    np.testing.assert_array_equal(st.roi(0), brush_stroke(H, W, pts, 8.0))
    assert len(st.roi_history._undo) == n0 + 1
    assert loaded._brush_base is None
    st.undo_roi()
    assert st.roi(0) is None


def test_brush_in_cut_mode_erases(loaded):
    st = loaded.state
    loaded.canvas.shape_finished.emit("rect", [(50.0, 50.0), (200.0, 200.0)])
    base = st.roi(0).copy()
    loaded.left.roi.mode_cut.setChecked(True)
    loaded.canvas.brush_stroke.emit([(100.0, 100.0), (150.0, 100.0)], 10.0, True)
    expect = base & ~brush_stroke(H, W, [(100.0, 100.0), (150.0, 100.0)], 10.0)
    np.testing.assert_array_equal(st.roi(0), expect)
    st.undo_roi()
    np.testing.assert_array_equal(st.roi(0), base)


def test_brush_radius_field_sets_the_canvas_radius(loaded):
    loaded.left.roi.radius.setValue(33)
    assert loaded.canvas.brush_radius == 33.0


def test_selecting_a_roi_tool_shows_the_roi_and_fills_it(loaded):
    st = loaded.state
    st.update_view(show_roi=False)
    loaded.canvas.shape_finished.emit("rect", [(50.0, 50.0), (100.0, 100.0)])
    assert loaded.view.roi_layer(editing=False) is None  # hidden
    loaded.set_tool("rect")
    assert st.view.show_roi is True
    layer = loaded.view.roi_layer(editing=True)
    assert layer.shape == (H, W, 4)
    assert layer[75, 75, 3] > 0 and layer[10, 10, 3] == 0


def test_roi_tool_buttons_are_exclusive_and_sync_with_the_canvas(loaded):
    roi = loaded.left.roi
    btn = {b.property("tool"): b for b in roi.tools.buttons()}
    btn["circle"].click()
    assert loaded.canvas.tool == "circle" and btn["circle"].isChecked()
    btn["brush"].click()
    assert loaded.canvas.tool == "brush"
    assert btn["brush"].isChecked() and not btn["circle"].isChecked()
    btn["brush"].click()  # unchecking returns to pan
    assert loaded.canvas.tool == "pan"
    probe_btn = next(b for b in loaded.right.probes.tools.buttons() if b.property("tool") == "probe_line")
    probe_btn.click()
    assert loaded.canvas.tool == "probe_line"
    assert not any(b.isChecked() for b in roi.tools.buttons())


def test_unknown_tool_is_rejected(loaded):
    with pytest.raises(ValueError):
        loaded.set_tool("lasso")


# ===================================================================================== mask files
def test_save_and_import_a_mask(loaded, dialogs, tmp_path):
    st = loaded.state
    loaded.canvas.shape_finished.emit("circle", [(160.0, 128.0), (200.0, 128.0)])
    m = st.roi(0).copy()
    out = tmp_path / "mask0.png"
    dialogs.save_file = str(out)
    _button(loaded.left.roi, "Save…").click()
    assert out.is_file()
    st.set_frame(1)
    dialogs.open_file = str(out)
    _button(loaded.left.roi, "Import…").click()
    np.testing.assert_array_equal(st.roi(1), m)
    assert "mask of frame 1 from mask0.png" in loaded.logview.toPlainText()


def test_saving_without_a_mask_warns(loaded, dialogs):
    loaded.save_current_mask()
    assert "getSaveFileName" not in dialogs.names()
    assert "this frame has no mask to save" in loaded.logview.toPlainText()


def test_importing_an_unreadable_mask_reports_an_error(loaded, dialogs, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    dialogs.open_file = str(bad)
    loaded.import_mask()
    assert dialogs.names()[-1] == "critical"
    assert loaded.state.roi(0) is None


def test_import_mask_set_by_frame_number(loaded, dialogs, tmp_path):
    from raftcorr.roi.masks import save_mask

    folder = tmp_path / "masks"
    folder.mkdir()
    a, b = _rect(10, 10, 60, 60), _rect(100, 100, 200, 180)
    save_mask(folder / "mask_0000.png", a)
    save_mask(folder / "mask_0002.png", b)
    dialogs.directory = str(folder)
    dialogs.question_answer = QMessageBox.StandardButton.Yes
    _button(loaded.left.roi, "Import set…").click()
    assert "by the number in the name" in dialogs.texts("question")[-1]
    st = loaded.state
    np.testing.assert_array_equal(st.roi(0), a)
    np.testing.assert_array_equal(st.roi(2), b)
    assert st.roi(1) is None and st.roi(3) is None


def test_import_mask_set_declined_changes_nothing(loaded, dialogs, tmp_path):
    from raftcorr.roi.masks import save_mask

    folder = tmp_path / "masks"
    folder.mkdir()
    save_mask(folder / "a.png", _rect(10, 10, 60, 60))
    dialogs.directory = str(folder)
    loaded.import_frame_masks()  # the stub answers No
    assert loaded.state.roi_masks == {}


# ================================================================================= mouse on canvas
def _vp(win):
    return win.canvas.viewport()


def test_rect_drag_with_the_mouse(qtbot, loaded):
    c = loaded.canvas
    loaded.set_tool("rect")
    p1, p2 = view_pos(c, 60.0, 50.0), view_pos(c, 180.0, 140.0)
    with qtbot.waitSignal(c.shape_finished, timeout=2000) as blocker:
        qtbot.mousePress(_vp(loaded), Qt.MouseButton.LeftButton, pos=p1)
        qtbot.mouseMove(_vp(loaded), p2)
        qtbot.mouseRelease(_vp(loaded), Qt.MouseButton.LeftButton, pos=p2)
    tool, pts = blocker.args
    assert tool == "rect"
    assert pts[0] == pytest.approx(scene_xy(c, p1))
    assert pts[-1] == pytest.approx(scene_xy(c, p2))
    # within one screen pixel of the intended image coordinates
    px = 1.0 / c.zoom
    assert pts[0] == pytest.approx((60.0, 50.0), abs=px)
    assert pts[-1] == pytest.approx((180.0, 140.0), abs=px)
    (x0, y0), (x1, y1) = pts
    np.testing.assert_array_equal(loaded.state.roi(0), _rect(x0, y0, x1, y1))
    assert not c._preview  # the dashed preview is removed


def test_tiny_drag_emits_nothing(qtbot, loaded):
    c = loaded.canvas
    loaded.set_tool("rect")
    p = view_pos(c, 100.0, 100.0)
    with qtbot.assertNotEmitted(c.shape_finished):
        qtbot.mousePress(_vp(loaded), Qt.MouseButton.LeftButton, pos=p)
        qtbot.mouseRelease(_vp(loaded), Qt.MouseButton.LeftButton, pos=p)
    assert loaded.state.roi(0) is None


def test_polygon_clicks_closed_with_enter(qtbot, loaded):
    c = loaded.canvas
    loaded.set_tool("polygon")
    vertices = [(40.0, 40.0), (200.0, 50.0), (120.0, 200.0)]
    for x, y in vertices:
        qtbot.mouseClick(_vp(loaded), Qt.MouseButton.LeftButton, pos=view_pos(c, x, y))
    assert len(c._points) == 3 and c._preview
    with qtbot.waitSignal(c.shape_finished, timeout=2000) as blocker:
        qtbot.keyClick(c, Qt.Key.Key_Return)
    tool, pts = blocker.args
    assert tool == "polygon" and len(pts) == 3
    assert np.allclose(pts, vertices, atol=1.0 / c.zoom)
    np.testing.assert_array_equal(loaded.state.roi(0), rasterize(Polygon(tuple(pts)), H, W))


def test_polygon_closed_by_right_click_and_by_clicking_the_first_vertex(qtbot, loaded):
    c = loaded.canvas
    loaded.set_tool("polygon")
    for x, y in [(40.0, 40.0), (100.0, 40.0), (70.0, 90.0)]:
        qtbot.mouseClick(_vp(loaded), Qt.MouseButton.LeftButton, pos=view_pos(c, x, y))
    with qtbot.waitSignal(c.shape_finished, timeout=2000):
        qtbot.mouseClick(_vp(loaded), Qt.MouseButton.RightButton, pos=view_pos(c, 10.0, 10.0))
    for x, y in [(150.0, 40.0), (220.0, 40.0), (180.0, 100.0)]:
        qtbot.mouseClick(_vp(loaded), Qt.MouseButton.LeftButton, pos=view_pos(c, x, y))
    with qtbot.waitSignal(c.shape_finished, timeout=2000) as blocker:
        qtbot.mouseClick(_vp(loaded), Qt.MouseButton.LeftButton, pos=view_pos(c, 150.5, 40.5))
    assert len(blocker.args[1]) == 3


def test_right_click_cancels_a_rect_in_progress(qtbot, loaded):
    c = loaded.canvas
    loaded.set_tool("polygon")
    qtbot.mouseClick(_vp(loaded), Qt.MouseButton.LeftButton, pos=view_pos(c, 40.0, 40.0))
    qtbot.mouseClick(_vp(loaded), Qt.MouseButton.LeftButton, pos=view_pos(c, 80.0, 40.0))
    with qtbot.assertNotEmitted(c.shape_finished):
        qtbot.mouseClick(_vp(loaded), Qt.MouseButton.RightButton, pos=view_pos(c, 60.0, 60.0))
    assert c._points == [] and not c._preview


def test_brush_with_the_mouse(qtbot, loaded):
    c = loaded.canvas
    loaded.set_tool("brush")
    loaded.left.roi.radius.setValue(6)
    a, b = view_pos(c, 100.0, 100.0), view_pos(c, 160.0, 100.0)
    qtbot.mousePress(_vp(loaded), Qt.MouseButton.LeftButton, pos=a)
    qtbot.mouseMove(_vp(loaded), b)
    assert loaded.state.roi(0) is None
    qtbot.mouseRelease(_vp(loaded), Qt.MouseButton.LeftButton, pos=b)
    m = loaded.state.roi(0)
    assert m is not None
    assert m[100, 100] and m[100, 158] and not m[120, 130]
    assert len(loaded.state.roi_history._undo) == 1


def test_probe_point_click(qtbot, loaded):
    c = loaded.canvas
    loaded.set_tool("probe_point")
    p = view_pos(c, 123.0, 77.0)
    qtbot.mouseClick(_vp(loaded), Qt.MouseButton.LeftButton, pos=p)
    assert len(loaded.state.probes) == 1
    assert loaded.state.probes[0].points[0] == pytest.approx(scene_xy(c, p))


def test_hover_reports_image_coordinates(qtbot, loaded):
    c = loaded.canvas
    p = view_pos(c, 150.0, 90.0)
    with qtbot.waitSignal(c.hovered, timeout=2000) as blocker:
        qtbot.mouseMove(_vp(loaded), p)
    x, y = blocker.args
    assert (x, y) == pytest.approx(scene_xy(c, p))
    assert loaded.pos_label.text().startswith("x 1")
    assert loaded.value_label.text() == ""  # no field yet


def _wheel(canvas, pos: QPoint, dy: int) -> None:
    vp = canvas.viewport()
    ev = QWheelEvent(QPointF(pos), QPointF(vp.mapToGlobal(pos)), QPoint(0, 0), QPoint(0, dy),
                     Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(vp, ev)


def test_wheel_zoom_fit_and_actual_size(loaded):
    c = loaded.canvas
    loaded.canvas.fit()
    z_fit = c.zoom
    assert z_fit > 0
    centre = c.viewport().rect().center()
    _wheel(c, centre, 120)
    assert c.zoom == pytest.approx(z_fit * 1.15)
    assert loaded.zoom_label.text() == f"{c.zoom * 100:.0f} %"
    _wheel(c, centre, -120)
    _wheel(c, centre, -120)
    assert c.zoom == pytest.approx(z_fit / 1.15)
    c.set_zoom(1.0)
    assert c.zoom == pytest.approx(1.0)
    c.set_zoom(1000.0)  # clamped
    assert c.zoom == pytest.approx(40.0)
    c.fit()
    assert c.zoom == pytest.approx(z_fit)


def test_wheel_zoom_keeps_the_point_under_the_cursor(qtbot, loaded):
    from PySide6.QtGui import QCursor

    c = loaded.canvas
    c.fit()
    pos = view_pos(c, 200.0, 150.0)
    QCursor.setPos(c.viewport().mapToGlobal(pos))
    qtbot.mouseMove(_vp(loaded), pos)
    if not c.underMouse():
        pytest.skip("the offscreen platform does not track the cursor (AnchorUnderMouse needs it)")
    before = scene_xy(c, pos)
    _wheel(c, pos, 120)
    _wheel(c, pos, 120)
    after = scene_xy(c, pos)
    assert after == pytest.approx(before, abs=2.0 / c.zoom)


def test_pan_drag_scrolls_when_zoomed_in(qtbot, loaded):
    c = loaded.canvas
    c.set_zoom(4.0)
    loaded.set_tool("pan")
    h0 = c.horizontalScrollBar().value()
    centre = c.viewport().rect().center()
    qtbot.mousePress(_vp(loaded), Qt.MouseButton.LeftButton, pos=centre)
    qtbot.mouseMove(_vp(loaded), centre - QPoint(60, 0))
    qtbot.mouseRelease(_vp(loaded), Qt.MouseButton.LeftButton, pos=centre - QPoint(60, 0))
    assert c.horizontalScrollBar().value() == h0 + 60
    assert loaded.state.roi(0) is None
