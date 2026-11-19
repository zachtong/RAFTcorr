"""Read-only canvas for the strain post-processing window, and its renderer.

:class:`StrainCanvas` is a stripped-down :class:`QGraphicsView` with two
layers:

    z=0  Background image  (``_bg_item``)
    z=5  Field overlay     (``_overlay_item``)

It supports wheel zoom and middle-button pan ONLY, plus a hover read-out
signal. All ROI editing, brush refinement, shape drawing, and tool switching
from :class:`canvas_area.ImageCanvas` are intentionally absent -- the strain
window never edits its inputs, it only views them.

:class:`FieldRenderer` turns one field of a
:class:`raftcorr.viz.fields.FieldProvider` into a :class:`FieldImage`: values
in the reference configuration (frame-0 positions) or the deformed one
(where the material points are in the frame), coloured with
:func:`raftcorr.viz.render.colorize`. Scene coordinates: scene pixel
``(c, r)`` covers ``[c, c + 1)``; an array whose cell ``(0, 0)`` is image pixel
``(x0, y0)`` with spacing ``step`` is placed at ``(x0, y0)`` with scale
``step``.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QMouseEvent,
    QPainter,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from raftcorr.gui.support import colormaps as _gui_colormaps  # noqa: F401  (registers black_rainbow)
from raftcorr.gui.theme import COLORS
from raftcorr.gui.theme_manager import connect_theme

# Match canvas_area zoom defaults so the two views feel identical.
_ZOOM_FACTOR = 1.15
_ZOOM_MIN = 0.10
_ZOOM_MAX = 20.0

_PIXMAP_CACHE = 48

RangeFn = Callable[[NDArray[np.float32]], tuple[float, float]]


@dataclass(frozen=True)
class FieldImage:
    """A rendered field, ready for a canvas and its colorbar."""

    pixmap: QPixmap
    x: float | None             # scene position of the top-left; None: keep
    y: float | None
    scale: float                # scene pixels per pixmap pixel
    alpha: float
    cmap: str
    vmin: float
    vmax: float
    label: str                  # colorbar title


@dataclass(frozen=True)
class ShownField:
    """The numbers behind a :class:`FieldImage`, for the cursor read-out."""

    name: str
    values: NDArray[np.float32]  # raw (unscaled) values
    x0: float
    y0: float
    step: int
    factor: float                # display unit per raw unit
    unit: str                    # "" for dimensionless strain

    def value_at(self, x: float, y: float) -> float | None:
        """Displayed value at image point ``(x, y)`` (bilinear), or None."""
        from raftcorr.engine.mapping import sample_bilinear

        if not (np.isfinite(x) and np.isfinite(y)) or self.values.size == 0:
            return None
        v = sample_bilinear(np.asarray(self.values, dtype=np.float64),
                            np.array([(x - self.x0) / self.step]),
                            np.array([(y - self.y0) / self.step]))[0]
        return float(v) * self.factor if np.isfinite(v) else None


def blank_pixmap(width: int, height: int, color: str = "white") -> QPixmap:
    """A solid pixmap to stand in for a hidden background image.

    "transparent" paints nothing, so the view's own background shows through
    and the field appears to float -- which is what transparency means when
    there is no file behind it to reveal.  Names match
    ``export_png.HIDDEN_BG_COLORS`` so the canvas and the exporter cannot
    disagree about what the user picked.
    """
    pm = QPixmap(width, height)
    if color == "transparent":
        pm.fill(Qt.GlobalColor.transparent)
    elif color == "black":
        pm.fill(Qt.GlobalColor.black)
    else:
        pm.fill(Qt.GlobalColor.white)
    return pm


def percentile_range(values: NDArray, colormap: str = "turbo") -> tuple[float, float]:
    """Auto colour range: :func:`raftcorr.viz.render.auto_range` (2-98 %, symmetric for diverging
    colormaps), the rule of the main window and the exports, so all colour a field alike."""
    from raftcorr.viz.render import auto_range

    rng = auto_range(values, colormap)
    return rng.vmin, rng.vmax


def _safe_colormap(name: str) -> str:
    """*name* when matplotlib knows it, else jet (pyALDIC's silent fallback)."""
    from raftcorr.viz.colormaps import lut

    try:
        lut(name)
        return name
    except KeyError:
        return "jet"


class FieldRenderer:
    """Cached rendering of FieldProvider fields into :class:`FieldImage`."""

    def __init__(self, cache_size: int = _PIXMAP_CACHE) -> None:
        self._cache: OrderedDict[tuple, QPixmap] = OrderedDict()
        self._cache_size = cache_size

    def clear(self, strain_only: bool = False) -> None:
        """Forget cached pixmaps (only the strain fields' with *strain_only*)."""
        if not strain_only:
            self._cache.clear()
            return
        for ck in [k for k in self._cache if k[1] is not None]:
            del self._cache[ck]

    def render(
        self,
        provider,
        key: str,
        frame: int,
        *,
        deformed: bool,
        cmap: str,
        alpha: float,
        range_of: RangeFn,
        factor: float = 1.0,
        unit: str = "",
        label: str = "",
        hide_untracked: bool = True,
        fill_trimmed: bool = False,
    ) -> tuple[FieldImage, ShownField] | None:
        """FieldProvider field *key* of *frame*, or None when there is none.

        *deformed* asks for the deformed configuration; the reference is used
        for frame 0 (it is its own deformed configuration). *factor* converts
        the raw values to the displayed unit (physical units). *range_of* maps
        the displayed values to ``(vmin, vmax)``. *fill_trimmed* fills the
        edge-trimmed band of a strain field from its reliable points.
        """
        if provider is None:
            return None
        series = provider.series
        if not 0 <= frame < series.n_frames:
            return None
        if key not in {f.key for f in provider.available()} or not provider.has_frame(key, frame):
            return None
        use_deformed = bool(deformed) and frame > 0
        if use_deformed:
            vals, pl = provider.deformed(key, frame, fill_trimmed=fill_trimmed)
            vals = np.asarray(vals, dtype=np.float32)
        else:
            vals, pl = provider.values(key, frame, fill_trimmed=fill_trimmed)
            step = int(pl.step)
            inside = series.mask[::step, ::step] if step > 1 else series.mask
            vals = np.where(inside, np.asarray(vals, dtype=np.float32), np.float32(np.nan))
            if not hide_untracked:
                from raftcorr.viz.fields import fill_untracked

                vals = fill_untracked(vals, inside, provider.trimmed_band(key, frame))
        if vals.size == 0 or not np.isfinite(vals).any():
            return None  # nothing to draw: a 0..1 colorbar would mislead
        shown_vals = vals * np.float32(factor) if factor != 1.0 else vals
        vmin, vmax = range_of(shown_vals)
        cmap = _safe_colormap(cmap)
        from raftcorr.viz.fields import ALL_FIELDS

        # Displacement pixmaps survive a new strain result; strain ones do not.
        strain_id = id(provider.strain) if ALL_FIELDS[key].is_strain else None
        ck = (id(series), strain_id, key, frame, use_deformed, cmap, float(vmin), float(vmax),
              float(factor), bool(hide_untracked), bool(fill_trimmed))
        pixmap = self._cache.get(ck)
        if pixmap is None:
            pixmap = _to_pixmap(shown_vals, vmin, vmax, cmap)
            self._cache[ck] = pixmap
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(ck)
        image = FieldImage(
            pixmap=pixmap, x=float(pl.x0), y=float(pl.y0), scale=float(pl.step), alpha=float(alpha),
            cmap=cmap, vmin=float(vmin), vmax=float(vmax), label=label,
        )
        shown = ShownField(label.split(" (")[0], vals, float(pl.x0), float(pl.y0), int(pl.step),
                           float(factor), unit)
        return image, shown


def _to_pixmap(values: NDArray[np.float32], vmin: float, vmax: float, cmap: str) -> QPixmap:
    from raftcorr.viz.render import ColorRange, colorize

    rgba = np.ascontiguousarray(colorize(values, ColorRange(float(vmin), float(vmax)), cmap))
    h, w = rgba.shape[:2]
    qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimg)


class StrainCanvas(QGraphicsView):
    """Read-only zoomable image canvas with a single overlay layer."""

    view_changed = Signal()
    #: Scene position under the mouse (no button needed).
    scene_hover = Signal(float, float)
    #: The mouse left the canvas.
    hover_left = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # --- Layer 0: background image ---
        self._bg_item = QGraphicsPixmapItem()
        self._scene.addItem(self._bg_item)
        self._bg_item.setZValue(0)

        # --- Layer 5: field overlay (semi-transparent colormap) ---
        self._overlay_item = QGraphicsPixmapItem()
        self._scene.addItem(self._overlay_item)
        self._overlay_item.setZValue(5)
        self._overlay_item.setOpacity(0.7)

        # --- Render settings (mirror ImageCanvas) ---
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate
        )
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.setBackgroundBrush(QBrush(QColor(COLORS.BG_CANVAS)))
        # Hover read-out needs move events without a button pressed.
        self.setMouseTracking(True)
        connect_theme(self._on_theme_changed)

        # --- Pan state (middle button only) ---
        self._panning = False
        self._pan_start = QPointF()
        self._zoom_level: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _on_theme_changed(self, _name: str) -> None:
        """The canvas around the image in the new theme's colour."""
        self.setBackgroundBrush(QBrush(QColor(COLORS.BG_CANVAS)))

    def set_image(self, rgb: NDArray[np.uint8]) -> None:
        """Display *rgb* (H, W, 3) uint8 array as the background image."""
        rgb = np.ascontiguousarray(rgb)
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        # .copy() so QImage owns the data after numpy array may be freed
        self._bg_item.setPixmap(QPixmap.fromImage(qimg.copy()))
        self._scene.setSceneRect(QRectF(0, 0, w, h))

    def set_blank(self, height: int, width: int,
                  color: str = "white") -> None:
        """Show a solid *color* instead of an image, at the given size."""
        if height <= 0 or width <= 0:
            return
        self._bg_item.setPixmap(blank_pixmap(width, height, color))
        self._scene.setSceneRect(QRectF(0, 0, width, height))

    def set_overlay_pixmap(self, pixmap: QPixmap) -> None:
        """Install *pixmap* on the overlay layer."""
        self._overlay_item.setPixmap(pixmap)

    def set_overlay_alpha(self, alpha: float) -> None:
        """Set the overlay opacity. *alpha* is clamped to ``[0, 1]``."""
        a = max(0.0, min(1.0, float(alpha)))
        self._overlay_item.setOpacity(a)

    def set_overlay_pos(self, x: float, y: float) -> None:
        """Offset the overlay item from the scene origin."""
        self._overlay_item.setPos(float(x), float(y))

    def clear_overlay(self) -> None:
        """Drop the overlay pixmap (back to a null pixmap)."""
        self._overlay_item.setPixmap(QPixmap())

    def show_field(self, image: FieldImage) -> None:
        """Install a rendered field on the overlay layer."""
        self.set_overlay_pixmap(image.pixmap)
        self.set_overlay_alpha(image.alpha)
        self._overlay_item.setScale(float(image.scale))
        if image.x is not None and image.y is not None:
            self.set_overlay_pos(image.x, image.y)

    def zoom_in(self) -> None:
        self._apply_zoom(_ZOOM_FACTOR)

    def zoom_out(self) -> None:
        self._apply_zoom(1.0 / _ZOOM_FACTOR)

    def zoom_to_100(self) -> None:
        """Reset zoom to 100% (1:1 pixels)."""
        self.resetTransform()
        self._zoom_level = 1.0
        self.view_changed.emit()

    def fit_to_view(self) -> None:
        """Reset the view to fit the current scene rect."""
        rect = self._scene.sceneRect()
        if rect.isEmpty():
            return
        self.resetTransform()
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_level = self.transform().m11()
        self.view_changed.emit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_zoom(self, factor: float) -> None:
        new_level = self._zoom_level * factor
        if new_level < _ZOOM_MIN or new_level > _ZOOM_MAX:
            return
        self.scale(factor, factor)
        self._zoom_level = new_level
        self.view_changed.emit()

    # ------------------------------------------------------------------
    # Qt event overrides
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta > 0:
            self._apply_zoom(_ZOOM_FACTOR)
        elif delta < 0:
            self._apply_zoom(1.0 / _ZOOM_FACTOR)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
            return
        sp = self.mapToScene(event.position().toPoint())
        self.scene_hover.emit(sp.x(), sp.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            self._panning
            and event.button() == Qt.MouseButton.MiddleButton
        ):
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.hover_left.emit()
        super().leaveEvent(event)

    def scrollContentsBy(self, dx: int, dy: int) -> None:  # noqa: N802
        super().scrollContentsBy(dx, dy)
        self.view_changed.emit()


__all__ = ["FieldImage", "FieldRenderer", "ShownField", "StrainCanvas", "blank_pixmap", "percentile_range"]
