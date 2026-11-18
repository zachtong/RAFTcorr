"""Main window: construction, themes, menus, shortcuts, help, opening image folders."""

from __future__ import annotations

import numpy as np
import pytest

from .gui_helpers import (  # noqa: F401  (fixtures)
    H,
    W,
    _restore_theme,
    dialogs,
    loaded,
    log_lines,
    make_window,
    rect_mask,
    render,
    seq_folder,
    win,
    write_plain_images,
    write_sequence,
)

pytestmark = pytest.mark.gui

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def _menu_actions(win) -> dict[str, dict[str, object]]:
    out = {}
    for top in win.menuBar().actions():
        menu = top.menu()
        if menu is None:
            continue
        out[top.text().replace("&", "")] = {a.text(): a for a in menu.actions() if a.text()}
    return out


# ================================================================================== construction
@pytest.mark.parametrize("name", ["light", "dark"])
def test_window_builds_with_each_stylesheet(qapp, make_window, name):
    from raftcorr.gui_basic import theme

    pal = theme.set_current(name)
    css = theme.stylesheet()
    assert pal.BG_WINDOW in css and pal.ACCENT in css
    qapp.setStyleSheet(css)
    w = make_window()
    assert w.isVisible()
    assert w.windowTitle().startswith("RAFTcorr")
    assert w.state.sequence is None
    assert not w.nav.isEnabled()  # no frames yet
    assert w.left.images.info.text() == "No images loaded."
    w.colorbar.repaint()  # paints without error in either palette
    w.chart.refresh()


def test_build_app_returns_the_running_application(qapp):
    from raftcorr.gui_basic.app import build_app

    assert build_app([]) is qapp
    assert qapp.applicationName() == "RAFTcorr"


def test_unknown_theme_is_rejected():
    from raftcorr.gui_basic import theme

    with pytest.raises(ValueError):
        theme.set_current("sepia")


def test_set_theme_switches_palette_and_persists(win, qapp):
    from raftcorr.gui_basic import theme

    win.set_theme("dark")
    assert theme.current().is_dark
    assert theme.DARK.BG_WINDOW in qapp.styleSheet()
    assert win.settings.value("theme") == "dark"
    win.set_theme("light")
    assert not theme.current().is_dark
    assert theme.LIGHT.BG_WINDOW in qapp.styleSheet()


def test_menus_and_their_shortcuts(win):
    m = _menu_actions(win)
    assert {"File", "Edit", "View", "Analysis", "Help"} <= set(m)
    expect = {
        ("Analysis", "Run correlation"): "F5",
        ("Analysis", "Stop"): "Shift+F5",
        ("Analysis", "Compute strain"): "F6",
        ("Help", "Keyboard shortcuts"): "F1",
        ("File", "Export results…"): "Ctrl+E",
        ("Edit", "Redo ROI edit"): "Ctrl+Y",
        ("View", "Fit image"): "Ctrl+0",
    }
    for (menu, text), key in expect.items():
        assert text in m[menu], f"{menu} > {text} missing"
        assert m[menu][text].shortcut().toString() == key
    assert "Ctrl+Z" in m["Edit"]["Undo ROI edit"].shortcut().toString()


def test_f1_shows_a_non_empty_shortcut_list(win, dialogs):
    _menu_actions(win)["Help"]["Keyboard shortcuts"].trigger()
    assert dialogs.names() == ["information"]
    text = dialogs.calls[0][1][2]
    assert len(text) > 100
    for key in ("F5", "Ctrl+Z", "Space", "Esc"):
        assert key in text


def test_about_box(win, dialogs):
    win.show_about()
    assert dialogs.names() == ["about"]
    assert "RAFTcorr" in dialogs.calls[0][1][2]


def test_debug_messages_hidden_until_enabled(win):
    win.log("visible line", "info")
    win.log("hidden detail", "debug")
    assert "visible line" in log_lines(win)
    assert "hidden detail" not in log_lines(win)
    dbg = _menu_actions(win)["View"]["Show debug messages"]
    dbg.setChecked(True)
    win.log("shown detail", "debug")
    assert "shown detail" in log_lines(win)


def test_log_escapes_html(win):
    win.log("<b>not bold</b> & co", "warning")
    assert "<b>not bold</b> & co" in log_lines(win)


# =================================================================================== load folder
def test_load_folder_sets_state_navigator_and_list(win, seq_folder):
    assert win.load_folder(seq_folder)
    st = win.state
    assert st.n_frames == 4
    assert st.sequence.shape == (H, W)
    assert st.image_folder == seq_folder
    assert st.current_frame == 0
    assert win.nav.isEnabled()
    assert win.nav.slider.maximum() == 3
    assert win.nav.label.text().startswith("frame 0 / 3")
    lst = win.left.images.list
    assert lst.count() == 4
    assert lst.item(0).text().startswith("  0  img_00.png")
    assert "[reference]" in lst.item(0).text()
    info = win.left.images.info.text()
    assert "4 frames" in info and f"{W} x {H} px" in info and "8-bit" in info
    assert "opened 4 frames" in log_lines(win)
    assert win.settings.value("last_folder") == str(seq_folder)
    render(win)
    assert win.canvas._image_size == (H, W)
    assert not win.colorbar.isVisible()  # no results yet


def test_open_folder_uses_the_directory_dialog(win, dialogs, seq_folder):
    dialogs.directory = str(seq_folder)
    win.open_folder()
    assert "getExistingDirectory" in dialogs.names()
    assert win.state.n_frames == 4


def test_open_folder_cancelled_changes_nothing(win, dialogs):
    dialogs.directory = ""
    win.open_folder()
    assert win.state.sequence is None
    assert "critical" not in dialogs.names()


def test_image_list_selection_and_navigator_change_the_frame(loaded):
    st = loaded.state
    loaded.left.images.list.setCurrentRow(2)
    assert st.current_frame == 2
    assert loaded.nav.slider.value() == 2
    assert loaded.nav.label.text().startswith("frame 2 / 3")
    loaded.nav.slider.setValue(3)
    assert st.current_frame == 3
    assert loaded.left.images.list.currentRow() == 3


def test_natural_sort_toggle_reloads_in_the_other_order(win, tmp_path):
    folder = tmp_path / "numbered"
    write_plain_images(folder, ["f1.png", "f2.png", "f10.png"])
    assert win.left.images.natural.isChecked()
    assert win.load_folder(folder)
    assert win.state.sequence.names == ["f1.png", "f2.png", "f10.png"]
    win.left.images.natural.setChecked(False)  # emits sort_changed -> reload
    assert win.state.sequence.names == ["f1.png", "f10.png", "f2.png"]
    assert win.state.natural_sort is False
    assert "natural sort would order them differently" in log_lines(win)
    assert win.left.images.list.item(1).text().startswith("  1  f10.png")
    win.left.images.natural.setChecked(True)
    assert win.state.sequence.names == ["f1.png", "f2.png", "f10.png"]
    assert win.state.natural_sort is True


def test_bad_folder_shows_an_error_and_keeps_the_state(loaded, dialogs, tmp_path):
    st = loaded.state
    st.set_roi(0, rect_mask())
    seq_before, roi_before = st.sequence, st.roi(0)
    assert loaded.load_folder(tmp_path / "does_not_exist") is False
    assert dialogs.names() == ["critical"]
    assert st.sequence is seq_before
    assert st.roi(0) is roi_before
    empty = tmp_path / "empty"
    empty.mkdir()
    assert loaded.load_folder(empty) is False
    assert "no images" in dialogs.texts("critical")[-1]
    assert st.sequence is seq_before and st.n_frames == 4


def test_loading_a_folder_resets_rois_probes_and_results(loaded, seq_folder):
    from raftcorr.post.probes import Probe

    from .gui_helpers import make_series

    st = loaded.state
    st.set_roi(0, rect_mask())
    st.add_probe(Probe(1, "point", ((100.0, 100.0),)))
    st.set_results(make_series(rect_mask(), names=st.sequence.names))
    st.set_frame(2)
    assert loaded.load_folder(seq_folder)
    assert st.roi_masks == {} and st.probes == [] and st.series is None and st.current_frame == 0
    assert not st.roi_history.can_undo


def test_mixed_image_sizes_do_not_crash_the_canvas(win, qtbot, tmp_path):
    """A folder whose frames differ in size must be refused, or the bad frame reported."""
    from PIL import Image

    folder = tmp_path / "mixed"
    write_sequence(folder, shifts=((1.0, 0.0),), h=64, w=80)
    Image.fromarray(np.zeros((32, 40), np.uint8)).save(folder / "img_02.png")
    ok = win.load_folder(folder)
    if ok:
        with qtbot.captureExceptions() as exceptions:
            win.state.set_frame(2)
            qtbot.wait(50)
            win.state.set_frame(0)  # leave a renderable frame so teardown stays clean
            qtbot.wait(50)
        assert not exceptions, f"unhandled exception while showing the bad frame: {exceptions[0][1]!r}"


def test_opening_a_folder_with_fewer_frames(win, qtbot, seq_folder, tmp_path):
    small = tmp_path / "two_frames"
    write_sequence(small, shifts=((1.0, 0.0),), h=64, w=80)
    assert win.load_folder(seq_folder)
    with qtbot.captureExceptions() as exceptions:
        assert win.load_folder(small)
        qtbot.wait(20)
    assert win.state.n_frames == 2 and win.left.images.list.count() == 2
    assert not exceptions, f"{exceptions[0][1]!r}"


def test_save_after_opening_a_new_folder_asks_for_a_file_name(win, dialogs, seq_folder, tmp_path):
    old = tmp_path / "old_project.raftcorr"
    win.state.project_path = old  # as after "Open project" or a first "Save project"
    other = tmp_path / "other_images"
    write_sequence(other, shifts=((1.0, 1.0),), h=64, w=80)
    assert win.load_folder(other)
    dialogs.save_file = ""  # the user cancels the dialog if it is shown
    win.save_project()
    if win._worker is not None:
        win._worker.wait(10000)
    assert "getSaveFileName" in dialogs.names(), "no file dialog: the session went into the old project file"


# ===================================================================================== shortcuts
def _activate(qtbot, win):
    win.activateWindow()
    win.raise_()
    qtbot.waitUntil(lambda: QApplication.activeWindow() is win, timeout=3000)


def test_frame_keyboard_shortcuts(qtbot, loaded):
    _activate(qtbot, loaded)
    loaded.canvas.setFocus()
    st = loaded.state
    qtbot.keyClick(loaded.canvas, Qt.Key.Key_Right)
    assert st.current_frame == 1
    qtbot.keyClick(loaded.canvas, Qt.Key.Key_End)
    assert st.current_frame == 3
    qtbot.keyClick(loaded.canvas, Qt.Key.Key_Right)  # loop is on by default
    assert st.current_frame == 0
    qtbot.keyClick(loaded.canvas, Qt.Key.Key_Left)
    assert st.current_frame == 3
    qtbot.keyClick(loaded.canvas, Qt.Key.Key_Home)
    assert st.current_frame == 0


def test_tool_keys_toggle_roi_tools(qtbot, loaded):
    _activate(qtbot, loaded)
    loaded.canvas.setFocus()
    qtbot.keyClick(loaded.canvas, Qt.Key.Key_R)
    assert loaded.canvas.tool == "rect"
    rect_btn = next(b for b in loaded.left.roi.tools.buttons() if b.property("tool") == "rect")
    assert rect_btn.isChecked()
    qtbot.keyClick(loaded.canvas, Qt.Key.Key_R)
    assert loaded.canvas.tool == "pan" and not rect_btn.isChecked()
    qtbot.keyClick(loaded.canvas, Qt.Key.Key_P)
    assert loaded.canvas.tool == "polygon"
    qtbot.keyClick(loaded.canvas, Qt.Key.Key_Escape)
    assert loaded.canvas.tool == "pan"


def test_tool_keys_do_nothing_without_images(qtbot, win):
    _activate(qtbot, win)
    win.canvas.setFocus()
    qtbot.keyClick(win.canvas, Qt.Key.Key_B)
    assert win.canvas.tool == "pan"


def test_typing_digits_in_a_spin_box_does_not_switch_the_field(qtbot, loaded):
    """1-9 are window shortcuts for the field; they must not steal digits from number fields."""
    _activate(qtbot, loaded)
    it = loaded.left.settings.iters
    it.setFocus()
    it.selectAll()
    qtbot.keyClicks(it, "7")
    qtbot.keyClick(it, Qt.Key.Key_Return)
    assert loaded.state.run_config.iters == 7
    assert loaded.state.view.field == "u"


def test_overlay_and_configuration_keys(qtbot, loaded):
    _activate(qtbot, loaded)
    loaded.canvas.setFocus()
    st = loaded.state
    qtbot.keyClick(loaded.canvas, Qt.Key.Key_F)
    assert st.view.show_overlay is False
    qtbot.keyClick(loaded.canvas, Qt.Key.Key_D)
    assert st.view.configuration == "deformed"
    assert loaded.right.display.def_btn.isChecked()
    qtbot.keyClick(loaded.canvas, Qt.Key.Key_D)
    assert st.view.configuration == "reference"


# ====================================================================== paths, recent, drop, plan
def test_open_path_dispatches_on_the_kind_of_path(win, seq_folder, tmp_path):
    win.open_path(seq_folder / "img_02.png")  # an image: its folder
    assert win.state.image_folder == seq_folder and win.state.n_frames == 4
    other = tmp_path / "notes.txt"
    other.write_text("x")
    win.open_path(other)
    assert f"cannot open {other}" in log_lines(win)
    assert win.state.image_folder == seq_folder


def test_dropping_a_folder_opens_it(win, seq_folder):
    from PySide6.QtCore import QMimeData, QPointF, QUrl
    from PySide6.QtGui import QDropEvent

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(seq_folder))])
    ev = QDropEvent(QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton,
                    Qt.KeyboardModifier.NoModifier)
    win.dropEvent(ev)
    assert win.state.n_frames == 4


def test_recent_menu_lists_and_reopens_folders(win, seq_folder, tmp_path):
    win._fill_recent()
    acts = win.recent_menu.actions()
    assert [a.text() for a in acts] == ["(nothing yet)"] and not acts[0].isEnabled()
    other = tmp_path / "second"
    write_sequence(other, shifts=((1.0, 0.0), (2.0, 0.0), (3.0, 0.0)), h=64, w=80)  # same frame count
    win.load_folder(seq_folder)
    win.load_folder(other)
    assert win.recent()[:2] == [str(other), str(seq_folder)]
    win.load_folder(seq_folder)
    assert win.recent()[:2] == [str(seq_folder), str(other)]  # most recent first, no duplicates
    win._fill_recent()
    acts = win.recent_menu.actions()
    assert [a.text() for a in acts][:2] == [str(seq_folder), str(other)]
    acts[1].trigger()
    assert win.state.image_folder == other


def test_plan_preview_describes_the_next_run(qtbot, loaded):
    label = loaded.left.run.plan
    qtbot.waitUntil(lambda: label.text().startswith("Plan:"), timeout=3000)
    loaded.state.set_roi(0, rect_mask())
    loaded.left.model.buttons["s8"].click()
    qtbot.waitUntil(lambda: label.text().startswith("Plan:"), timeout=3000)
    assert "per window" in label.text() or "Plan:" in label.text()


def test_space_toggles_a_focused_check_box(qtbot, loaded):
    _activate(qtbot, loaded)
    cb = loaded.right.display.show_roi
    cb.setFocus()
    before = cb.isChecked()
    qtbot.keyClick(cb, Qt.Key.Key_Space)
    playing = loaded.nav.playing
    loaded.nav.stop()
    assert cb.isChecked() != before and not playing


def test_arrow_keys_move_a_focused_slider(qtbot, loaded):
    _activate(qtbot, loaded)
    s = loaded.right.display.opacity
    s.setFocus()
    before = s.value()
    qtbot.keyClick(s, Qt.Key.Key_Right)
    assert s.value() != before and loaded.state.current_frame == 0
