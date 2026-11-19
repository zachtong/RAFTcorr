"""Render the one-minute RAFTcorr quick-start video.

The real application is driven step by step (off-screen, device-pixel ratio 2) on the aluminium plate
with two holes in tension: images, region of interest, Suggest a Model, run, displacement, strain,
probes and export. Nothing is mocked: every frame is a picture of what RAFTcorr itself drew; the
pointer, click ripples, captions and cards are painted on top (``compositor.py``).

    python scripts/media/quickstart_video.py --images <folder of Al_with_holes_1> --lang en \
        --out docs/media/RAFTcorr_quickstart_en.mp4
    python scripts/media/quickstart_video.py --images ... --lang zh_CN --out docs/media/RAFTcorr_quickstart_zh.mp4

Needs RAFTcorr with its model files, OpenCV (``pip install opencv-python-headless``) and an ffmpeg
with libx264 (``imageio-ffmpeg``, a RAFTcorr dependency, bundles one; or set ``FFMPEG``). The run and
the strain computation take their real time on this computer (GPU recommended); the video shows them
as a time-lapse. Nothing appears on screen and the user's own settings are untouched.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from app_setup import (  # noqa: E402  (configures Qt before PySide6 is imported)
    DESK_H, DESK_W, ROOT, SPECIMEN, find_ffmpeg, grab, grab_with_popups, make_app, qimage_to_rgb,
    settle, wait_until,
)

import cv2  # noqa: E402
from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

from compositor import (  # noqa: E402
    ACCENT, FONT_CJK, FONT_LATIN, FULL_CAM, TEXT, TEXT_MUTED, Caption, Overlay, Video, cam_around,
    render_card,
)
from director import Director  # noqa: E402
from storyboard import CREDIT, GROUP, REPO_URL, SCRIPTS, Script  # noqa: E402

ICON_SVG = ROOT / "src" / "raftcorr" / "gui" / "assets" / "icon" / "RAFTcorr.svg"
CAM_LEFT = cam_around(560, 560, 1.4)       # settings column and canvas (ROI, model)
RUN_SHOW_S = 3.0                            # playback length of the correlation run
SUGGEST_SHOW_S = 1.6                        # ... of the model survey
STRAIN_SHOW_S = 1.2                         # ... of the strain computation
DLG_MARGIN = 14                             # px kept free above and below the export dialog
MENU_RECTANGLE, MENU_CIRCLE = 1, 2          # ROI shape menus: Polygon, Rectangle, Circle, Circle (3-point)


def icon_image(px: int = 512) -> QImage:
    img = QImage(px, px, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    QSvgRenderer(str(ICON_SVG)).render(p, QRectF(0, 0, px, px))
    p.end()
    return img


def cards(s: Script) -> tuple:
    fam = FONT_CJK if s.cjk else FONT_LATIN
    icon = icon_image()
    title = render_card([
        (s.title, 76, QFont.Weight.DemiBold, TEXT, FONT_LATIN),
        (s.subtitle, 30, QFont.Weight.Normal, TEXT_MUTED, fam),
    ], icon)
    end = render_card([
        (s.title, 60, QFont.Weight.DemiBold, TEXT, FONT_LATIN),
        (s.end_tagline, 25, QFont.Weight.Normal, TEXT_MUTED, fam),
        (REPO_URL, 30, QFont.Weight.DemiBold, ACCENT, FONT_LATIN),
        (s.end_install, 20, QFont.Weight.Normal, QColor("#9ca3af"), fam),
        (CREDIT, 21, QFont.Weight.Normal, TEXT, FONT_LATIN),
        (GROUP, 19, QFont.Weight.Normal, TEXT_MUTED, FONT_LATIN),
    ], icon, gap_out=14)
    return title, end


class Tutorial:
    def __init__(self, script: Script, images: Path, out: str) -> None:
        self.s = script
        self.images = images
        self.app, self.win = make_app(script.lang)
        self.v = Video(out, find_ffmpeg(), cjk=script.cjk)
        self.d = Director(self.app, self.v, self.win)
        self.st = self.win._state
        self.ls = self.win._left_sidebar
        self.rs = self.win._right_sidebar
        self.canvas = self.win._canvas_area.canvas
        self.sw = None
        self._patch_export_dialog()

    def cap(self, i: int, tag: str = "") -> None:
        c = self.s.steps[i - 1]
        self.v.caption_to(Caption(c.step, c.title, c.sub, tag) if tag else c)

    # ------------------------------------------------------------------ scenes
    def intro(self) -> None:
        title, self.end_card = cards(self.s)
        self.d.snap()
        self.v.cursor_visible = False
        self.v.show_card(title, 0.0)
        self.v.hold(1.9)
        self.v.hide_card(0.7)
        self.v.hold(0.5)
        self.v.cursor_visible = True

    def import_images(self) -> None:
        self.cap(1)
        self.v.hold(0.5)

        def load() -> None:
            self.win._image_ctrl.load_folder(str(self.images))
            wait_until(self.app, lambda: self.st.sequence is not None, timeout=120)
            settle(self.app, 0.2)
            self.canvas.fit_to_view()

        self.d.click_widget(self.ls._drop_zone, load, fy=0.4, fade=0.35, wait=0.2)
        self.v.hold(0.5)
        self.d.go(self.ls._image_list, 0.35, 0.3)
        self.v.hold(0.8)

    def draw_roi(self) -> None:
        self.cap(2)
        c, tb, sp = self.canvas, self.ls.roi_toolbar, SPECIMEN
        self.v.zoom(CAM_LEFT, dur=0.9)
        self.v.move(*self.d.scene_pos(c, 276, 1900))
        self.d.canvas_zoom(c, c.sceneRect(), QRectF(*sp.gauge_view), dur=0.8)
        self.v.hold(0.1)
        self.d.pick_from_menu(tb._btn_add, tb._add_menu, MENU_RECTANGLE,
                              lambda: tb._on_shape_selected("rect", "add"))
        x0, y0, x1, y1 = sp.gauge
        self.d.mouse_drag(c, (x0, y0), (x1, y1), dur=0.8)
        self.d.snap(fade=0.15, wait=0.3)
        self.v.hold(0.2)
        for cx, cy in sp.holes:
            self.d.pick_from_menu(tb._btn_cut, tb._cut_menu, MENU_CIRCLE,
                                  lambda: tb._on_shape_selected("circle", "cut"))
            self.d.mouse_drag(c, (cx, cy), (cx + sp.hole_cut_radius, cy), dur=0.45)
            self.d.snap(fade=0.12, wait=0.3)
            self.v.hold(0.05)
        if self.st.per_frame_rois.get(0) is None:
            raise RuntimeError("the region of interest was not drawn")
        self.v.hold(0.35)

    def suggest(self) -> None:
        self.cap(3)
        btn = self.ls._workflow_panel._suggest_btn
        self.d.go(btn)
        self.v.click(settle=0.1)
        btn.click()
        wall = self.d.record_live(lambda: self.win._task is None, show_for=SUGGEST_SHOW_S,
                                  capture=lambda: qimage_to_rgb(grab_with_popups(self.win)))
        print(f"  model survey filmed over {wall:.1f} s: {self.st.model_advice}")
        self.d.go(self.ls._workflow_panel._model, 0.5, 0.5)
        self.v.hold(1.6)

    def run(self) -> None:
        from raftcorr.gui.app_state import RunState
        from raftcorr.gui.strain_window import StrainWindow

        # The strain window opens by itself when the run is done: make it now, the size of the screen.
        self.sw = StrainWindow(self.st, parent=None)
        self.sw.move(0, 0)
        self.sw.resize(DESK_W, DESK_H)
        self.win._strain_window = self.sw
        self.cap(4, self.s.timelapse)
        self.v.zoom(FULL_CAM, dur=0.8)
        self.d.go(self.rs._run_btn, 0.45, 0.5, dur=0.9)
        self.v.hold(0.2)
        self.v.click(settle=0.12)
        self.rs._run_btn.click()
        wall = self.d.record_live(
            lambda: self.st.run_state not in (RunState.RUNNING, RunState.PAUSED)
            and self.win._pipeline_ctrl._worker is None, show_for=RUN_SHOW_S)
        if self.st.run_state != RunState.DONE:
            raise RuntimeError(f"the run did not finish: {self.st.run_state}")
        print(f"  run filmed over {wall:.1f} s")
        self.v.hold(0.5)

    def displacement(self) -> None:
        self.cap(5)
        fs = self.rs._field_selector
        self.d.click_widget(fs._buttons["disp_v"], fs._buttons["disp_v"].click, fade=0.25, wait=0.15)
        self.v.hold(0.9)
        nav = self.win._canvas_area._frame_nav
        nav._slider.setValue(0)
        self.d.snap(fade=0.2, wait=0.15)
        self.v.hold(0.3)
        self.d.drag_slider(nav._slider, nav._slider.maximum(), dur=2.0)
        self.v.hold(1.2)

    def strain(self) -> None:
        self.cap(6, self.s.timelapse)
        sw, sc = self.sw, self.sw._canvas
        self.d.click_widget(self.rs._strain_btn, self.rs._strain_btn.click, fade=0.0)
        sw.set_strain_frame(len(self.st.image_files) - 1)
        sc.fit_to_view()
        self.d.switch_view(sw, fade=0.45)
        self.v.hold(0.3)
        self.v.move(*self.d.scene_pos(sc, 262, 2080))
        self.d.canvas_zoom(sc, sc.sceneRect(), QRectF(*SPECIMEN.gauge_view), dur=0.7)
        self.d.go(sw._compute_btn, 0.5, 0.5)
        self.v.click(settle=0.1)
        sw._compute_btn.click()
        self.d.record_live(lambda: not sw.is_computing() and sw._compute_btn.isEnabled(), show_for=STRAIN_SHOW_S)
        b = sw._field_selector.button("strain_eyy")
        self.d.click_widget(b, b.click, fade=0.25, wait=0.2)
        self.v.hold(2.8)

    def probes(self) -> None:
        self.cap(7)
        sw = self.sw
        tab = sw._analysis_tab
        bar = sw._tabs.tabBar()
        self.v.move(*self._pos_in(sw, bar, bar.tabRect(1).center()))
        self.v.click(settle=0.1)
        sw._tabs.setCurrentWidget(tab)
        self.d.snap(fade=0.3, wait=0.4)
        pc = tab._canvas
        self.d.canvas_zoom(pc, pc.sceneRect(), QRectF(*SPECIMEN.gauge_view), dur=0.6)
        buttons = tab._tool_buttons
        x0, y0, x1, y1 = SPECIMEN.extensometer
        self.d.click_widget(buttons["extensometer"], buttons["extensometer"].click, fade=0.1)
        self.d.mouse_clicks(pc, [(x0, y0), (x1, y1)], dur=0.7)
        self.d.snap(fade=0.25, wait=0.6)
        if len(self.st.probes) < 1:
            raise RuntimeError("the extensometer was not placed")
        self.v.hold(2.4)

    def export(self) -> None:
        self.cap(8)
        b = self.sw._export_strain_btn
        self.d.go(b)
        self.v.click(settle=0.1)
        b.click()                          # runs _export_scene inside the patched exec()

    def _patch_export_dialog(self) -> None:
        from raftcorr.gui.dialogs.export_dialog import ExportDialog

        tutorial = self

        def scripted_exec(dlg) -> int:
            tutorial._export_scene(dlg)
            return 0

        ExportDialog.exec = scripted_exec

    def _export_scene(self, dlg) -> None:
        dlg.show()                         # showEvent fits it to the (virtual) screen
        settle(self.app, 0.3)
        k = min(1.0, (self.d.view.height() - 2 * DLG_MARGIN) / dlg.height())
        x = (self.d.view.width() - dlg.width() * k) / 2
        y = (self.d.view.height() - dlg.height() * k) / 2
        print(f"  export dialog {dlg.width()}x{dlg.height()} shown at scale {k:.3f}")
        self._dlg_overlay(dlg, x, y, k)
        self.v.hold(1.3)
        tabs = dlg._tabs
        bar = tabs.tabBar()
        for index, hold in ((1, 1.1), (2, 1.1), (3, 1.1)):
            tab_pos = bar.mapTo(dlg, bar.tabRect(index).center())
            self.v.move(x + tab_pos.x() * k, y + tab_pos.y() * k)
            self.v.click(settle=0.1)
            tabs.setCurrentIndex(index)
            settle(self.app, 0.2)
            self._dlg_overlay(dlg, x, y, k)
            self.v.hold(hold)
        dlg.hide()
        self.v.overlays = []

    def _dlg_overlay(self, dlg, x: float, y: float, k: float) -> None:
        img = grab(dlg)
        if k < 1.0:
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        self.v.overlays = [Overlay(img, x, y)]

    def _pos_in(self, root, widget, point) -> tuple[float, float]:
        p = widget.mapTo(root, point)
        return float(p.x()), float(p.y())

    def outro(self) -> None:
        self.v.hold(0.4)
        self.v.caption_to(None)
        self.v.cursor_visible = False
        self.v.show_card(self.end_card, 0.8)
        self.v.hold(3.6)

    def render(self) -> None:
        t0 = time.perf_counter()
        for scene in (self.intro, self.import_images, self.draw_roi, self.suggest, self.run,
                      self.displacement, self.strain, self.probes, self.export, self.outro):
            scene()
            print(f"{scene.__name__:<14} ends at {self.v.t:6.2f} s", flush=True)
        self.v.close()
        print(f"video {self.v.t:.1f} s, rendered in {time.perf_counter() - t0:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, type=Path, help="folder of the Al_with_holes_1 images")
    ap.add_argument("--lang", choices=sorted(SCRIPTS), default="en")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Tutorial(SCRIPTS[a.lang], a.images, a.out).render()


if __name__ == "__main__":
    main()
