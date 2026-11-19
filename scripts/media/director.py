"""The 'camera operator' that turns real GUI actions into video time.

The Director links the live (off-screen) RAFTcorr windows with the Video timeline: it moves the
drawn pointer to real widget positions, performs the real action at the moment of the click, and
captures the window again, so the video shows exactly what the application drew. Clicks and drags on
the canvases are delivered as real mouse events, so the tools draw their own previews.

Adapted from the pyALDIC tutorial-video director (same author).
"""
from __future__ import annotations

import time
from typing import Callable

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QAbstractSlider, QApplication, QGraphicsView, QMenu, QStyle,
    QStyleOptionSlider, QWidget,
)

from app_setup import grab, offscreen, settle
from compositor import FPS, Overlay, Video, ease_in_out_cubic, still


class Director:
    def __init__(self, app: QApplication, video: Video, view: QWidget) -> None:
        self.app = app
        self.v = video
        self.view = view            # the window currently filling the screen

    # -- capture --------------------------------------------------------
    def snap(self, fade: float = 0.0, wait: float = 0.05) -> None:
        """Re-capture the current window after letting Qt settle."""
        settle(self.app, wait)
        self.v.set_desktop(grab(self.view), fade)

    def switch_view(self, window: QWidget, fade: float = 0.45) -> None:
        self.view = window
        self.snap(fade=fade, wait=0.2)

    # -- geometry -------------------------------------------------------
    def pos(self, w: QWidget, fx: float = 0.5, fy: float = 0.5) -> tuple[float, float]:
        r = w.rect()
        p = w.mapTo(self.view, QPoint(0, 0))
        return p.x() + r.width() * fx, p.y() + r.height() * fy

    def scene_pos(self, canvas: QGraphicsView, x: float, y: float) -> tuple[float, float]:
        vp = canvas.viewportTransform().map(QPointF(x, y))
        o = canvas.viewport().mapTo(self.view, QPoint(0, 0))
        return o.x() + vp.x(), o.y() + vp.y()

    # -- cursor ---------------------------------------------------------
    def go(self, w: QWidget, fx: float = 0.5, fy: float = 0.5, dur: float = 0.0) -> None:
        self.v.move(*self.pos(w, fx, fy), dur=dur)

    def click(self, action: Callable[[], None] | None = None,
              fade: float = 0.16, wait: float = 0.08) -> None:
        """Ripple at the cursor, run the real action, show its result."""
        self.v.click(settle=0.14)
        if action is not None:
            action()
            self.snap(fade=fade, wait=wait)

    def click_widget(self, w: QWidget, action: Callable[[], None],
                     fx: float = 0.5, fy: float = 0.5, dur: float = 0.0,
                     fade: float = 0.16, wait: float = 0.08) -> None:
        self.go(w, fx, fy, dur)
        self.click(action, fade, wait)

    # -- popups ---------------------------------------------------------
    def pick_from_menu(self, button: QWidget, menu: QMenu, index: int,
                       on_pick: Callable[[], None]) -> None:
        """Click a button that opens ``menu`` above it and choose item ``index``."""
        self.go(button)
        self.v.click(settle=0.1)
        offscreen(menu)
        menu.popup(QPoint(0, 0))
        settle(self.app, 0.05)
        bx, by = self.pos(button, 0.0, 0.0)
        mx, my = bx, by - menu.height()
        self.v.overlays = [Overlay(grab(menu), mx, my)]
        self.v.hold(0.22)
        act = menu.actions()[index]
        g = menu.actionGeometry(act)
        self.v.move(mx + g.center().x() - 10, my + g.center().y(), dur=0.42)
        menu.setActiveAction(act)
        settle(self.app, 0.03)
        self.v.overlays = [Overlay(grab(menu), mx, my)]
        self.v.hold(0.12)
        self.v.click(settle=0.12)
        menu.hide()
        self.v.overlays = []
        on_pick()
        self.snap(fade=0.1)

    def pick_from_combo(self, combo, text: str) -> None:
        """Open a QComboBox list, hover ``text`` and choose it."""
        self.go(combo)
        self.v.click(settle=0.1)
        container = combo.view().window()
        offscreen(container)
        combo.showPopup()
        settle(self.app, 0.08)
        origin = self.view.mapToGlobal(QPoint(0, 0))
        tl = container.mapToGlobal(QPoint(0, 0)) - origin
        self.v.overlays = [Overlay(grab(container), tl.x(), tl.y())]
        self.v.hold(0.22)
        row = combo.findText(text)
        view = combo.view()
        idx = view.model().index(row, 0)
        r = view.visualRect(idx)
        vo = view.viewport().mapTo(container, QPoint(0, 0))
        self.v.move(tl.x() + vo.x() + r.center().x(), tl.y() + vo.y() + r.center().y(), dur=0.4)
        view.setCurrentIndex(idx)
        settle(self.app, 0.03)
        self.v.overlays = [Overlay(grab(container), tl.x(), tl.y())]
        self.v.hold(0.12)
        self.v.click(settle=0.12)
        combo.hidePopup()
        self.v.overlays = []
        combo.setCurrentIndex(row)
        self.snap(fade=0.1, wait=0.1)

    # -- continuous interactions ---------------------------------------
    def canvas_drag(self, canvas: QGraphicsView, p0: tuple[float, float],
                    p1: tuple[float, float], dur: float,
                    on_press, on_move, on_release) -> None:
        """Press at scene point p0, drag to p1 with live rubber-band frames."""
        self.v.move(*self.scene_pos(canvas, *p0), dur=0.0)
        self.v.click(settle=0.1)
        on_press(QPointF(*p0))
        n = max(2, round(dur * FPS))
        for i in range(1, n + 1):
            u = ease_in_out_cubic(i / n)
            q = QPointF(p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u)
            on_move(q)
            self.app.processEvents()
            self.v.cursor = still(self.scene_pos(canvas, q.x(), q.y()))
            self.v.set_desktop(grab(self.view))
            self.v.hold(1 / FPS)
        on_release(QPointF(*p1))

    def _mouse(self, view: QGraphicsView, kind: QEvent.Type, scene: QPointF,
               button: Qt.MouseButton, buttons: Qt.MouseButton) -> None:
        """Send a mouse event to ``view``'s viewport at scene point ``scene``."""
        vp = view.viewport()
        local = QPointF(view.mapFromScene(scene))
        glob = QPointF(vp.mapToGlobal(local.toPoint()))
        QApplication.sendEvent(vp, QMouseEvent(kind, local, glob, button, buttons,
                                               Qt.KeyboardModifier.NoModifier))

    def mouse_drag(self, view: QGraphicsView, p0: tuple[float, float],
                   p1: tuple[float, float], dur: float) -> None:
        """Press at scene point p0, drag to p1 and release, with live frames, as a user would."""
        left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
        self.v.move(*self.scene_pos(view, *p0))
        self.v.click(settle=0.08)
        self._mouse(view, QEvent.Type.MouseButtonPress, QPointF(*p0), left, left)
        n = max(2, round(dur * FPS))
        q = QPointF(*p0)
        for i in range(1, n + 1):
            u = ease_in_out_cubic(i / n)
            q = QPointF(p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u)
            self._mouse(view, QEvent.Type.MouseMove, q, none, left)
            self.app.processEvents()
            self.v.cursor = still(self.scene_pos(view, q.x(), q.y()))
            self.v.set_desktop(grab(self.view))
            self.v.hold(1 / FPS)
        self._mouse(view, QEvent.Type.MouseButtonRelease, q, left, none)

    def mouse_clicks(self, view: QGraphicsView, points: list[tuple[float, float]],
                     dur: float = 0.6) -> None:
        """Click scene points one after the other; the moves between them show the tool's preview."""
        left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
        self.v.move(*self.scene_pos(view, *points[0]))
        for k, (x, y) in enumerate(points):
            if k > 0:
                (x0, y0) = points[k - 1]
                n = max(2, round(dur * FPS))
                for i in range(1, n + 1):
                    u = ease_in_out_cubic(i / n)
                    q = QPointF(x0 + (x - x0) * u, y0 + (y - y0) * u)
                    self._mouse(view, QEvent.Type.MouseMove, q, none, none)
                    self.app.processEvents()
                    self.v.cursor = still(self.scene_pos(view, q.x(), q.y()))
                    self.v.set_desktop(grab(self.view))
                    self.v.hold(1 / FPS)
            self.v.click(settle=0.08)
            self._mouse(view, QEvent.Type.MouseButtonPress, QPointF(x, y), left, left)
            self._mouse(view, QEvent.Type.MouseButtonRelease, QPointF(x, y), left, none)
            self.snap(wait=0.05)

    def canvas_zoom(self, canvas: QGraphicsView, r0: QRectF, r1: QRectF, dur: float) -> None:
        """Smoothly zoom a canvas (as a mouse-wheel zoom would) from r0 to r1."""
        n = max(2, round(dur * FPS))
        for i in range(1, n + 1):
            u = ease_in_out_cubic(i / n)
            h = r0.height() * (r1.height() / r0.height()) ** u      # geometric zoom
            w = r0.width() * (r1.width() / r0.width()) ** u
            cx = r0.center().x() + (r1.center().x() - r0.center().x()) * u
            cy = r0.center().y() + (r1.center().y() - r0.center().y()) * u
            canvas.fitInView(QRectF(cx - w / 2, cy - h / 2, w, h),
                             Qt.AspectRatioMode.KeepAspectRatio)
            canvas._zoom_level = canvas.transform().m11()
            canvas.view_changed.emit()
            self.app.processEvents()
            self.v.set_desktop(grab(self.view))
            self.v.hold(1 / FPS)

    def drag_slider(self, slider: QAbstractSlider, to_value: int, dur: float) -> None:
        """Grab a slider handle and drag it to ``to_value``, frame by frame."""
        def handle_center(value: int) -> tuple[float, float]:
            old = slider.value()
            slider.blockSignals(True)
            slider.setValue(value)
            opt = QStyleOptionSlider()
            slider.initStyleOption(opt)
            hr = slider.style().subControlRect(
                QStyle.ComplexControl.CC_Slider, opt,
                QStyle.SubControl.SC_SliderHandle, slider)
            slider.setValue(old)
            slider.blockSignals(False)
            p = slider.mapTo(self.view, hr.center())
            return float(p.x()), float(p.y())

        start = slider.value()
        x0, y0 = handle_center(start)
        x1, _ = handle_center(to_value)
        self.v.move(x0, y0, dur=0.0)
        self.v.click(settle=0.08)
        n = max(2, round(dur * FPS))
        for i in range(1, n + 1):
            u = ease_in_out_cubic(i / n)
            x = x0 + (x1 - x0) * u
            value = round(start + (to_value - start) * u)
            if value != slider.value():
                slider.setValue(value)
                self.snap(wait=0.02)
            self.v.cursor = still((x, y0))
            self.v.hold(1 / FPS)

    def type_into(self, line_edit, text: str, per_char: float = 0.13) -> None:
        """Select the field's text and type ``text`` key by key."""
        from PySide6.QtTest import QTest
        self.v.click(settle=0.06)
        self.v.click(settle=0.1)                  # double-click selects all
        line_edit.selectAll()
        self.snap(wait=0.03)
        self.v.hold(0.15)
        for ch in text:
            QTest.keyClicks(line_edit, ch)
            self.snap(wait=0.02)
            self.v.hold(per_char)
        QTest.keyClick(line_edit, Qt.Key.Key_Return)

    def record_live(self, done: Callable[[], bool], show_for: float,
                    sample_every: float = 0.2, timeout: float = 900.0,
                    max_samples: int = 64, capture: Callable | None = None) -> float:
        """Film a background job and play it back over ``show_for`` seconds.

        Captures are spaced ``sample_every`` apart so the worker thread is not starved of the GIL
        (the application's own log then reports a realistic duration); playback is time-scaled to
        ``show_for``. A long job keeps at most ``max_samples`` captures (every other one is dropped
        and the spacing doubled), which bounds the memory. Returns the wall time.
        """
        shoot = capture or (lambda: grab(self.view))
        samples = []
        t0 = time.perf_counter()
        next_sample = 0.0
        while not done():
            self.app.processEvents()
            now = time.perf_counter() - t0
            if now >= next_sample:
                samples.append((now, shoot()))
                next_sample = now + sample_every
                if len(samples) > max_samples:
                    samples = samples[::2]
                    sample_every *= 2
            else:
                time.sleep(0.01)          # leave the GIL to the worker thread
            if now > timeout:
                raise TimeoutError("background job did not finish")
        wall = time.perf_counter() - t0
        settle(self.app, 0.3)
        samples.append((wall, shoot()))
        scale = show_for / max(wall, 1e-3)
        start = self.v.t
        for ts, img in samples:
            self.v.set_desktop(img)
            due = start + ts * scale
            if due > self.v.t:
                self.v.hold(due - self.v.t)
        return wall
