"""Zoomable, pannable image canvas with layers and drawing tools.

Scene coordinates are image pixels with pixel ``(col, row)`` covering ``[col - 0.5, col + 0.5)``,
so a point's scene position equals its image coordinate (the convention of :mod:`raftcorr.roi`).

Layers (z order): background image, field overlay, ROI mask, probes, drawing preview.
Navigation: wheel = zoom about the cursor; middle drag (or any drag with the pan tool) = pan;
double-click with the pan tool = fit.
Tools emit :attr:`shape_finished` ``(tool, points)`` in image coordinates.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QWidget,
)

SHAPE_TOOLS = ("rect", "circle", "polygon")
PROBE_TOOLS = ("probe_point", "probe_line", "probe_rect", "probe_circle", "probe_polygon")
TOOLS = ("pan", "brush") + SHAPE_TOOLS + PROBE_TOOLS

ZOOM_STEP = 1.15
ZOOM_MIN, ZOOM_MAX = 0.02, 40.0


def rgba_to_qimage(rgba: np.ndarray) -> QImage:
    """QImage that owns a copy of an ``(h, w, 4)`` uint8 RGBA array."""
    a = np.ascontiguousarray(rgba, dtype=np.uint8)
    h, w = a.shape[:2]
    img = QImage(a.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return img.copy()


class ImageCanvas(QGraphicsView):
    shape_finished = Signal(str, list)        # tool, [(x, y), ...]
    brush_stroke = Signal(list, float, bool)  # points, radius, finished
    hovered = Signal(float, float)            # image x, y (NaN when outside)
    zoom_changed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._bg = QGraphicsPixmapItem()
        self._overlay = QGraphicsPixmapItem()
        self._roi = QGraphicsPixmapItem()
        for z, item in enumerate((self._bg, self._overlay, self._roi)):
            item.setZValue(z)
            item.setOffset(-0.5, -0.5)  # pixel centres at integer coordinates
            item.setTransformationMode(Qt.TransformationMode.FastTransformation)
            self._scene.addItem(item)
        self._probe_items: list[QGraphicsItem] = []
        self._preview: list[QGraphicsItem] = []
        self._image_size: tuple[int, int] | None = None

        self.tool = "pan"
        self.brush_radius = 20.0
        self.draw_color = QColor(74, 108, 247)
        self._points: list[tuple[float, float]] = []
        self._dragging = False
        self._panning = False
        self._pan_origin = None
        self._space = False
        self._brush_points: list[tuple[float, float]] = []
        self.placeholder = "\n".join((
            "Open an image folder to start  (File > Open image folder, Ctrl+O)",
            "or drop a folder or a .raftcorr project here",
        ))

    # -- content ----------------------------------------------------------------------------------
    def set_background(self, rgba: np.ndarray | None) -> None:
        if rgba is None:
            self._bg.setPixmap(QPixmap())
            return
        h, w = rgba.shape[:2]
        first = self._image_size is None
        resized = self._image_size != (h, w)
        self._image_size = (h, w)
        self._bg.setPixmap(QPixmap.fromImage(rgba_to_qimage(rgba)))
        if resized:
            self._scene.setSceneRect(QRectF(-0.5, -0.5, w, h).adjusted(-w * 0.1, -h * 0.1, w * 0.1, h * 0.1))
        if first or resized:
            self.fit()

    def set_overlay(self, rgba: np.ndarray | None, x0: float = 0.0, y0: float = 0.0, step: float = 1.0) -> None:
        if rgba is None or rgba.size == 0:
            self._overlay.setPixmap(QPixmap())
            return
        self._overlay.setPixmap(QPixmap.fromImage(rgba_to_qimage(rgba)))
        self._overlay.setScale(step)
        # the -0.5 offset centres cell j on local coordinate j; after scaling, cell j sits at
        # pos + step * j, so pos = (x0, y0) puts cell (0, 0) on image pixel (x0, y0)
        self._overlay.setPos(x0, y0)

    def set_roi_overlay(self, rgba: np.ndarray | None) -> None:
        if rgba is None:
            self._roi.setPixmap(QPixmap())
        else:
            self._roi.setPixmap(QPixmap.fromImage(rgba_to_qimage(rgba)))

    def set_overlay_visible(self, visible: bool) -> None:
        self._overlay.setVisible(visible)

    def set_probes(self, probes: list, show_labels: bool = True, mapper=None) -> None:
        """Draw probes; ``mapper`` maps ``(N, 2)`` reference points to where they are drawn
        (the deformed positions in the deformed configuration)."""
        from ..post.probes import outline

        for it in self._probe_items:
            self._scene.removeItem(it)
        self._probe_items = []
        for p in probes:
            color = QColor(p.color)
            pen = QPen(color, 2.0)
            pen.setCosmetic(True)
            pts = outline(p)
            if mapper is not None:
                pts = mapper(pts)
            items: list[QGraphicsItem] = []
            if p.kind == "point":
                items.append(self._marker(pts[0, 0], pts[0, 1], color))
            elif p.kind == "line":
                path = QPainterPath(QPointF(*pts[0]))
                for x, y in pts[1:]:
                    path.lineTo(QPointF(x, y))
                it = QGraphicsPathItem(path)
                it.setPen(pen)
                items += [it, self._marker(*pts[0], color, 3), self._marker(*pts[-1], color, 3)]
            else:
                poly = QGraphicsPolygonItem(QPolygonF([QPointF(x, y) for x, y in pts]))
                poly.setPen(pen)
                poly.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
                items.append(poly)
            if show_labels:
                t = QGraphicsSimpleTextItem(p.label)
                t.setBrush(QBrush(color))
                t.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                t.setPos(float(pts[0, 0]) + 2, float(pts[0, 1]) + 2)
                items.append(t)
            for it in items:
                it.setZValue(10)
                self._scene.addItem(it)
            self._probe_items += items

    def _marker(self, x: float, y: float, color: QColor, r: float = 5.0) -> QGraphicsItem:
        it = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        it.setPos(x, y)
        it.setBrush(QBrush(color))
        it.setPen(QPen(Qt.GlobalColor.white, 1.0))
        it.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        return it

    def _shape_item(self, kind: str, pts: list | tuple, pen: QPen, brush: QColor | None = None) -> QGraphicsItem:
        if kind in ("rect", "probe_rect"):
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            it: QGraphicsItem = QGraphicsRectItem(QRectF(QPointF(x0, y0), QPointF(x1, y1)).normalized())
        elif kind in ("circle", "probe_circle"):
            (cx, cy), (ex, ey) = pts[0], pts[-1]
            r = math.hypot(ex - cx, ey - cy)
            it = QGraphicsEllipseItem(cx - r, cy - r, 2 * r, 2 * r)
        else:
            it = QGraphicsPolygonItem(QPolygonF([QPointF(x, y) for x, y in pts]))
        it.setPen(pen)
        if brush is not None:
            it.setBrush(QBrush(brush))
        return it

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        """Guidance text while there is no image."""
        super().drawForeground(painter, rect)
        if self._image_size is not None or not self.placeholder:
            return
        painter.save()
        painter.resetTransform()
        painter.setPen(QColor(120, 128, 140))
        f = painter.font()
        f.setPointSizeF(11.0)
        painter.setFont(f)
        painter.drawText(self.viewport().rect(), int(Qt.AlignmentFlag.AlignCenter), self.placeholder)
        painter.restore()

    # -- view -------------------------------------------------------------------------------------
    @property
    def zoom(self) -> float:
        return float(self.transform().m11())

    def fit(self) -> None:
        if self._image_size is None:
            return
        h, w = self._image_size
        self.fitInView(QRectF(-0.5, -0.5, w, h), Qt.AspectRatioMode.KeepAspectRatio)
        self.zoom_changed.emit(self.zoom)

    def set_zoom(self, z: float) -> None:
        z = float(np.clip(z, ZOOM_MIN, ZOOM_MAX))
        f = z / max(self.zoom, 1e-9)
        self.scale(f, f)
        self.zoom_changed.emit(self.zoom)

    def wheelEvent(self, event) -> None:  # noqa: N802
        f = ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP
        if ZOOM_MIN <= self.zoom * f <= ZOOM_MAX:
            self.scale(f, f)
            self.zoom_changed.emit(self.zoom)
        event.accept()

    # -- tools ------------------------------------------------------------------------------------
    def set_tool(self, tool: str) -> None:
        if tool not in TOOLS:
            raise ValueError(f"unknown tool {tool!r}")
        self.cancel_drawing()
        self.tool = tool
        self.viewport().setCursor(Qt.CursorShape.OpenHandCursor if tool == "pan" else Qt.CursorShape.CrossCursor)

    def cancel_drawing(self) -> None:
        self._points = []
        self._brush_points = []
        self._dragging = False
        self._clear_preview()

    def _clear_preview(self) -> None:
        for it in self._preview:
            self._scene.removeItem(it)
        self._preview = []

    def _scene_xy(self, event) -> tuple[float, float]:
        p = self.mapToScene(event.position().toPoint())
        return float(p.x()), float(p.y())

    def _base_tool(self) -> str:
        return self.tool.removeprefix("probe_")

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and (self.tool == "pan" or self._space)
        ):
            self._panning = True
            self._pan_origin = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            if self._points and self._base_tool() == "polygon":
                self._finish_polygon()
            else:
                self.cancel_drawing()
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        x, y = self._scene_xy(event)
        base = self._base_tool()
        if self.tool == "brush":
            self._dragging = True
            self._brush_points = [(x, y)]
            self._preview_brush()
        elif base == "point":
            self.shape_finished.emit(self.tool, [(x, y)])
        elif base in ("rect", "circle", "line"):
            self._dragging = True
            self._points = [(x, y), (x, y)]
            self._preview_shape()
        elif base == "polygon":
            if len(self._points) >= 3 and self._near_first(x, y):
                self._finish_polygon()
            else:
                self._points.append((x, y))
                self._preview_shape(hover=(x, y))
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._panning and self._pan_origin is not None:
            d = event.position() - self._pan_origin
            self._pan_origin = event.position()
            self.horizontalScrollBar().setValue(int(self.horizontalScrollBar().value() - d.x()))
            self.verticalScrollBar().setValue(int(self.verticalScrollBar().value() - d.y()))
            event.accept()
            return
        x, y = self._scene_xy(event)
        if self._image_size is not None:
            h, w = self._image_size
            inside = -0.5 <= x < w - 0.5 and -0.5 <= y < h - 0.5
            self.hovered.emit(x if inside else float("nan"), y if inside else float("nan"))
        if self.tool == "brush":
            if self._dragging:
                self._brush_points.append((x, y))
                self.brush_stroke.emit(list(self._brush_points), self.brush_radius, False)
            self._preview_brush(cursor=(x, y))
        elif self._dragging and self._points:
            self._points[-1] = (x, y)
            self._preview_shape()
        elif self._points and self._base_tool() == "polygon":
            self._preview_shape(hover=(x, y))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning and event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton):
            self._panning = False
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor if self.tool == "pan" else Qt.CursorShape.CrossCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            if self.tool == "brush":
                self.brush_stroke.emit(list(self._brush_points), self.brush_radius, True)
                self._brush_points = []
                self._clear_preview()
            else:
                pts = list(self._points)
                self._points = []
                self._clear_preview()
                (x0, y0), (x1, y1) = pts
                if math.hypot(x1 - x0, y1 - y0) >= 1.0:
                    self.shape_finished.emit(self.tool, pts)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if self._base_tool() == "polygon" and len(self._points) >= 3:
            self._finish_polygon()
            event.accept()
            return
        if self.tool == "pan":
            self.fit()
        super().mouseDoubleClickEvent(event)

    #: Keys the canvas leaves to the main window (frame navigation, tools, fields).
    PASS_KEYS = (Qt.Key.Key_Space, Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down,
                 Qt.Key.Key_Home, Qt.Key.Key_End)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape and (self._points or self._dragging):
            self.cancel_drawing()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self._base_tool() == "polygon":
            self._finish_polygon()
        else:
            # not QGraphicsView's arrow-key scrolling: let the window handle navigation keys
            event.ignore()
            return
        event.accept()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.hovered.emit(float("nan"), float("nan"))
        if self.tool == "brush" and not self._dragging:
            self._clear_preview()
        super().leaveEvent(event)

    def _near_first(self, x: float, y: float) -> bool:
        fx, fy = self._points[0]
        return math.hypot(x - fx, y - fy) * self.zoom < 8.0

    def _finish_polygon(self) -> None:
        pts = list(self._points)
        self._points = []
        self._clear_preview()
        if len(pts) >= 3:
            self.shape_finished.emit(self.tool, pts)

    def _pen(self) -> QPen:
        pen = QPen(self.draw_color, 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        return pen

    def _preview_shape(self, hover: tuple[float, float] | None = None) -> None:
        self._clear_preview()
        base = self._base_tool()
        pen = self._pen()
        if base == "line":
            (x0, y0), (x1, y1) = self._points
            it: QGraphicsItem = QGraphicsLineItem(x0, y0, x1, y1)
            it.setPen(pen)
            self._add_preview(it)
        elif base in ("rect", "circle"):
            self._add_preview(self._shape_item(base, self._points, pen))
        elif base == "polygon":
            pts = list(self._points) + ([hover] if hover else [])
            path = QPainterPath(QPointF(*pts[0]))
            for p in pts[1:]:
                path.lineTo(QPointF(*p))
            it = QGraphicsPathItem(path)
            it.setPen(pen)
            self._add_preview(it)
            for x, y in self._points:
                self._add_preview(self._marker(x, y, self.draw_color, 3))

    def _preview_brush(self, cursor: tuple[float, float] | None = None) -> None:
        self._clear_preview()
        r = self.brush_radius
        pts = self._brush_points[-1:] if self._dragging else ([cursor] if cursor else [])
        for x, y in pts:
            it = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
            it.setPen(self._pen())
            self._add_preview(it)

    def _add_preview(self, it: QGraphicsItem) -> None:
        it.setZValue(20)
        self._scene.addItem(it)
        self._preview.append(it)

    def cursor_image_pos(self) -> tuple[float, float]:
        p = self.mapToScene(self.mapFromGlobal(QCursor.pos()))
        return float(p.x()), float(p.y())
