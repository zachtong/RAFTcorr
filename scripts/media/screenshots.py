"""Pictures of the real application for the user guide (``docs/user-guide/figures``) and the README.

The whole analysis is run once on the aluminium plate with two holes in tension (light theme, device
pixel ratio 2), and each picture is taken at the moment it shows: a panel, a dialog, a menu, or the
whole window. ``fig-main-window.png`` gets numbered callouts for the overview of the guide.

    python scripts/media/screenshots.py --images <folder of Al_with_holes_1>

Needs RAFTcorr with its model files; a GPU makes the run take about two minutes instead of many.
Nothing appears on screen and the user's own settings are untouched.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from app_setup import (  # noqa: E402  (configures Qt before PySide6 is imported)
    DESK_H, DESK_W, DPR, ROOT, SPECIMEN, grab_with_popups, make_app, settle, specimen_roi, wait_until,
    zoom_canvas,
)

from PySide6.QtCore import QPoint, QPointF, QRect, QRectF, Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen  # noqa: E402
from PySide6.QtSvg import QSvgRenderer  # noqa: E402

FIGURES = ROOT / "docs" / "user-guide" / "figures"
ICON_SVG = ROOT / "src" / "raftcorr" / "gui" / "assets" / "icon" / "RAFTcorr.svg"
ACCENT = QColor("#4f46e5")
MAX_WIDTH = 2400                       # px of a saved figure
COLORBAR_W = 175                       # logical px at the right of the canvas that hold its colour bar


class Shots:
    def __init__(self, images: Path, out: Path) -> None:
        self.images = images
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.app, self.win = make_app("en", "light")
        self.st = self.win._state
        self.ls = self.win._left_sidebar
        self.rs = self.win._right_sidebar
        self.canvas = self.win._canvas_area.canvas

    # ------------------------------------------------------------------ helpers
    def settle(self, s: float = 0.3) -> None:
        settle(self.app, s)

    def rect_of(self, root, widgets, pad: int = 6) -> QRect:
        """Union of the widgets' rectangles in ``root`` (logical px), padded and kept inside root."""
        r = QRect()
        for w in widgets:
            tl = w.mapTo(root, QPoint(0, 0)) if not w.isWindow() else (w.mapToGlobal(QPoint(0, 0))
                                                                       - root.mapToGlobal(QPoint(0, 0)))
            r = r.united(QRect(tl, w.size()))
        return r.adjusted(-pad, -pad, pad, pad).intersected(root.rect())

    def save(self, name: str, root, crop: QRect | None = None) -> QImage:
        self.settle(0.25)
        img = grab_with_popups(root)
        if crop is not None:
            img = img.copy(QRect(crop.x() * DPR, crop.y() * DPR, crop.width() * DPR, crop.height() * DPR))
        if img.width() > MAX_WIDTH:          # whole windows: still about 370 dpi across the text width
            img = img.scaledToWidth(MAX_WIDTH, Qt.TransformationMode.SmoothTransformation)
        path = self.out / name
        img.save(str(path))
        print(f"saved {path.relative_to(ROOT)}  {img.width()}x{img.height()}", flush=True)
        return img

    def save_canvas(self, name: str, colorbar: bool, half_width: int = 230) -> None:
        """The canvas around the specimen (it is narrow and tall), with the canvas's own colour bar
        moved up beside it: both are parts of one picture of the window, nothing is redrawn."""
        self.settle(0.3)
        win, area = self.win, self.win._canvas_area
        img = grab_with_popups(win)
        a = self.rect_of(win, [area], 0)
        vp = self.canvas.viewport()
        centre = vp.mapTo(win, self.canvas.mapFromScene(QPointF(276.0, 2000.0)))
        x0 = max(a.left(), centre.x() - half_width)
        x1 = min(a.right(), centre.x() + half_width)
        parts = [img.copy(QRect(x0 * DPR, a.top() * DPR, (x1 - x0) * DPR, a.height() * DPR))]
        if colorbar:                          # the overlay covers the viewport; the bar is at its right edge
            vr = self.rect_of(win, [vp], 0)
            left = vr.right() - COLORBAR_W
            parts.append(img.copy(QRect(left * DPR, a.top() * DPR, COLORBAR_W * DPR, a.height() * DPR)))
        for part in parts:
            part.setDevicePixelRatio(1.0)    # copies keep the grab's ratio 2; paste them pixel for pixel
        out = QImage(sum(p.width() for p in parts), parts[0].height(), QImage.Format.Format_RGB32)
        out.fill(parts[0].pixelColor(4, parts[0].height() // 2))
        p = QPainter(out)
        x = 0
        for part in parts:
            p.drawImage(x, 0, part)
            x += part.width()
        p.end()
        out.save(str(self.out / name))
        print(f"saved {(self.out / name).relative_to(ROOT)}  {out.width()}x{out.height()}", flush=True)

    # ------------------------------------------------------------------ the session
    def run(self) -> None:
        self.icon()
        self.models_window()
        self.win._image_ctrl.load_folder(str(self.images))
        wait_until(self.app, lambda: self.st.sequence is not None, timeout=120)
        self.canvas.fit_to_view()
        self.settle(0.5)
        self.images_panel()
        self.workflow_panel()
        self.roi()
        self.parameters()
        self.batch_import()
        self.suggest_and_run()
        self.results()
        self.strain()
        self.analysis()
        self.export_dialog()
        self.readme_pictures()

    def readme_pictures(self) -> None:
        """Lighter copies (1600 px wide) of three whole-window figures for the README."""
        media = ROOT / "docs" / "media"
        media.mkdir(parents=True, exist_ok=True)
        for src, dst in (("fig-main-window-plain.png", "readme-main-window.png"),
                         ("fig-strain-window.png", "readme-strain-window.png"),
                         ("fig-analysis.png", "readme-analysis.png")):
            img = QImage(str(self.out / src))
            img = img.scaledToWidth(1600, Qt.TransformationMode.SmoothTransformation)
            img.save(str(media / dst))
            print(f"saved docs/media/{dst}", flush=True)
        (self.out / "fig-main-window-plain.png").unlink()      # the guide uses the numbered one

    def icon(self) -> None:
        img = QImage(1024, 1024, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        QSvgRenderer(str(ICON_SVG)).render(p, QRectF(0, 0, 1024, 1024))
        p.end()
        img.save(str(self.out / "raftcorr-icon.png"))

    def models_window(self) -> None:
        import os

        from raftcorr.gui.dialogs.models_dialog import ModelsDialog

        # The window shows the per-user model folder; a public picture should not carry the name of
        # this computer's account. The models themselves are found in the checkout's models/.
        saved = os.environ.get("LOCALAPPDATA")
        if sys.platform == "win32":
            os.environ["LOCALAPPDATA"] = "C:\\Users\\you\\AppData\\Local"
        try:
            dlg = ModelsDialog(self.win)
        finally:
            if saved is not None:
                os.environ["LOCALAPPDATA"] = saved
        dlg.show()
        self.settle(0.4)
        self.save("fig-models-window.png", dlg)
        dlg.close()

    def images_panel(self) -> None:
        ls = self.ls
        self.save("fig-images.png", self.win, self.rect_of(self.win, [ls._images_header, ls._image_list], 4))

    def workflow_panel(self) -> None:
        self.save("fig-workflow.png", self.win, self.rect_of(self.win, [self.ls._workflow_section], 4))

    def roi(self) -> None:
        ls, tb = self.ls, self.ls.roi_toolbar
        roi = specimen_roi(self.win)
        self.st.set_frame_roi(0, roi)
        self.win._on_roi_edit_for_frame(0)
        zoom_canvas(self.canvas, SPECIMEN.gauge_view)
        self.settle(0.4)
        self.save_canvas("fig-roi-canvas.png", colorbar=False, half_width=330)
        tb._show_add_menu()
        self.settle(0.3)
        menu = tb._add_menu
        above = ls._workflow_panel._model           # the menu opens upwards over this row
        self.save("fig-roi-toolbar.png", self.win, self.rect_of(self.win, [ls._roi_section, menu, above], 4))
        menu.hide()
        self.settle(0.2)

    def parameters(self) -> None:
        ls = self.ls
        ls._advanced_section.set_expanded(True)
        self.settle(0.3)
        ls._settings_scroll.ensureWidgetVisible(ls._advanced_section)
        self.settle(0.3)
        self.save("fig-parameters.png", self.win,
                  self.rect_of(self.win, [ls._params_section, ls._advanced_section], 4))
        ls._advanced_section.set_expanded(False)
        ls._settings_scroll.verticalScrollBar().setValue(0)
        self.settle(0.2)

    def batch_import(self) -> None:
        from raftcorr.gui.dialogs.batch_import_dialog import BatchImportDialog

        shots = self

        def show_instead(dlg) -> int:
            dlg.show()
            shots.settle(0.4)
            shots.save("fig-batch-import.png", dlg)
            dlg.close()
            return 0

        original = BatchImportDialog.exec
        BatchImportDialog.exec = show_instead
        try:
            self.win._on_batch_import()
        finally:
            BatchImportDialog.exec = original
        self.settle(0.2)

    def suggest_and_run(self) -> None:
        from raftcorr.gui.app_state import RunState
        from raftcorr.gui.strain_window import StrainWindow

        self.st.roi_editing = False
        self.st.display_changed.emit()
        self.win._on_suggest_model()
        wait_until(self.app, lambda: self.win._task is None)
        print("advice:", self.st.model_advice, flush=True)
        sw = StrainWindow(self.st, parent=None)
        sw.move(0, 0)
        sw.resize(DESK_W, DESK_H)
        self.win._strain_window = sw
        self.sw = sw
        rs = self.rs
        self.win._pipeline_ctrl.start()
        taken = False
        t0 = time.perf_counter()
        while self.st.run_state in (RunState.RUNNING, RunState.PAUSED) or self.win._pipeline_ctrl._worker:
            self.settle(0.2)
            if not taken and rs._progress_bar.value() >= 35 and time.perf_counter() - t0 > 3:
                self.save("fig-progress.png", self.win,
                          self.rect_of(self.win, [rs._run_btn, rs._remaining_label, rs._elapsed_label], 6))
                taken = True
        print(f"run {time.perf_counter() - t0:.0f} s, notes: {self.st.results.info.notes}", flush=True)
        self.settle(0.5)

    def results(self) -> None:
        rs, st = self.rs, self.st
        last = st.results.completed_frames[-1]
        st.set_current_frame(last)
        st.show_deformed = True
        st.set_display_field("disp_v")
        self.settle(0.8)
        self.canvas.fit_to_view()
        self.settle(0.5)
        img = self.save("fig-main-window-plain.png", self.win)
        self.callouts(img)
        self.save("fig-display.png", self.win,
                  self.rect_of(self.win, [rs._field_selector, rs._physical_units], 6))
        zoom_canvas(self.canvas, SPECIMEN.gauge_view)
        self.settle(0.5)
        self.save_canvas("fig-displacement.png", colorbar=True)
        st.set_display_field("residual")
        self.settle(0.8)
        self.save_canvas("fig-residual.png", colorbar=True)
        st.set_display_field("disp_v")
        self.canvas.fit_to_view()
        self.settle(0.3)

    def callouts(self, img: QImage) -> None:
        """fig-main-window.png: the plain picture with five numbered regions."""
        ls, rs, win = self.ls, self.rs, self.win
        regions = [
            self.rect_of(win, [ls._images_header, ls._image_list], 2),
            self.rect_of(win, [ls._settings_scroll], 2),
            self.rect_of(win, [win._canvas_area], 2),
            self.rect_of(win, [rs._run_btn, rs._remaining_label, rs._elapsed_label], 2),
            self.rect_of(win, [rs._field_selector, rs._console], 2),
        ]
        out = QImage(img)
        out.setDevicePixelRatio(out.width() / win.width())      # paint in the window's logical px
        p = QPainter(out)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)    # the grab carries DPR 2: logical px here
        font = QFont("Segoe UI")
        font.setPixelSize(15)
        font.setBold(True)
        p.setFont(font)
        for k, r in enumerate(regions, start=1):
            edge = QColor(ACCENT)
            edge.setAlphaF(0.85)
            p.setPen(QPen(edge, 2.0))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(r).adjusted(2, 2, -2, -2), 6, 6)
            c = QPointF(r.right() - 22, r.top() + 1)     # on the top border: section titles are left
            p.setPen(QPen(QColor("white"), 2.0))
            p.setBrush(ACCENT)
            p.drawEllipse(c, 13, 13)
            p.setPen(QColor("white"))
            p.drawText(QRectF(c.x() - 13, c.y() - 13, 26, 26), Qt.AlignmentFlag.AlignCenter, str(k))
        p.end()
        path = self.out / "fig-main-window.png"
        out.save(str(path))
        print(f"saved {path.relative_to(ROOT)}", flush=True)

    def strain(self) -> None:
        sw, st = self.sw, self.st
        if not sw.isVisible():
            self.win._on_open_strain_window()
        self.settle(0.5)
        t0 = time.perf_counter()
        sw._on_compute_clicked()
        wait_until(self.app, lambda: not sw.is_computing())
        print(f"strain {time.perf_counter() - t0:.0f} s", flush=True)
        sw.set_current_field("strain_eyy")
        sw.set_strain_frame(st.results.completed_frames[-1])
        self.settle(0.6)
        zoom_canvas(sw._canvas, SPECIMEN.gauge_view)
        self.settle(0.6)
        self.save("fig-strain-window.png", sw)
        sw._advanced_section.set_expanded(True)
        self.settle(0.4)
        self.save("fig-strain-params.png", sw,
                  self.rect_of(sw, [sw._params_section, sw._advanced_section], 4))
        sw._advanced_section.set_expanded(False)
        self.settle(0.2)
        self.banner()

    def banner(self) -> None:
        """docs/media/banner.png from the arrays of this run (last frame)."""
        import numpy as np

        from banner import _on_image, render_banner

        st = self.st
        res, strain = st.results, st.strain_results
        last = res.completed_frames[-1]
        shape = st.sequence.shape
        ref = np.asarray(st.sequence[0], np.float32)
        ref = ref.mean(axis=2) if ref.ndim == 3 else ref
        g = res.grid
        v = _on_image(np.where(res.mask, res.v[last], np.nan), g.x0, g.y0, 1, shape)
        sg = strain.grid
        eyy = _on_image(strain.get("eyy", last), sg.x0, sg.y0, strain.step, shape)
        out = ROOT / "docs" / "media" / "banner.png"
        render_banner(ref, v, eyy, self.out / "raftcorr-icon.png", out)
        print(f"saved {out.relative_to(ROOT)}", flush=True)

    def analysis(self) -> None:
        from raftcorr.analysis.probes import AreaGeom, LineGeom, PointGeom

        sw = self.sw
        tab = sw._analysis_tab
        sw._tabs.setCurrentWidget(tab)
        self.settle(0.5)
        tab._armed_tool = "extensometer"
        tab._on_probe_placed("line", LineGeom(*SPECIMEN.extensometer))
        tab._armed_tool = "point"
        tab._on_probe_placed("point", PointGeom(*SPECIMEN.point))
        tab._armed_tool = "area"
        tab._on_probe_placed("area", AreaGeom.rect(*SPECIMEN.area))
        tab._on_canvas_selected(self.st.probes.of_kind("line")[0].id)
        self.settle(1.0)
        zoom_canvas(tab._canvas, SPECIMEN.gauge_view)
        self.settle(0.8)
        self.save("fig-analysis.png", sw)
        sw._tabs.setCurrentIndex(0)
        self.settle(0.3)

    def export_dialog(self) -> None:
        from raftcorr.gui.dialogs.export_dialog import ExportDialog

        shots = self
        names = ("fig-export-data.png", "fig-export-images.png")       # the tabs the guide shows

        def show_instead(dlg) -> int:
            dlg.show()
            shots.settle(0.6)
            for index, name in enumerate(names):
                dlg._tabs.setCurrentIndex(index)
                shots.settle(0.5)
                shots.save(name, dlg)
            dlg.close()
            return 0

        ExportDialog.exec = show_instead
        self.rs._on_export()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, type=Path, help="folder of the Al_with_holes_1 images")
    ap.add_argument("--out", type=Path, default=FIGURES)
    a = ap.parse_args()
    Shots(a.images, a.out).run()


if __name__ == "__main__":
    main()
