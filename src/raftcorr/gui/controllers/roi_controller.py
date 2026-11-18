"""ROI controller -- boolean mask with add / cut shape operations (pyALDIC API, no OpenCV).

Shapes are rasterised by :mod:`raftcorr.roi.shapes` (a pixel belongs to a shape when its
centre does), the same rules the engine uses.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from raftcorr.roi.masks import save_mask as _save_mask
from raftcorr.roi.shapes import Circle, Polygon, Rect, brush_stroke, rasterize


class ROIController:
    """Boolean ROI mask with add/cut shape operations."""

    def __init__(self, img_shape: tuple[int, int]) -> None:
        if len(img_shape) != 2 or img_shape[0] <= 0 or img_shape[1] <= 0:
            raise ValueError(f"img_shape must be (H, W) with positive dimensions, got {img_shape}")
        self._shape = (int(img_shape[0]), int(img_shape[1]))
        self.mask: NDArray[np.bool_] = np.zeros(self._shape, dtype=bool)

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    def add_rectangle(self, x1: int, y1: int, x2: int, y2: int, mode: str = "add") -> None:
        """Add or cut a filled rectangle (both corners included)."""
        self._apply(rasterize(Rect(x1, y1, x2, y2), *self._shape), mode)

    def add_polygon(self, points: list[tuple[int, int]], mode: str = "add") -> None:
        """Add or cut a filled polygon."""
        if len(points) < 3:
            return
        self._apply(rasterize(Polygon(tuple(points)), *self._shape), mode)

    def add_circle(self, cx: int, cy: int, radius: int, mode: str = "add") -> None:
        """Add or cut a filled circle."""
        if radius <= 0:
            return
        self._apply(rasterize(Circle(cx, cy, radius), *self._shape), mode)

    def stroke_segment(self, x1: int, y1: int, x2: int, y2: int, radius: int, mode: str = "add") -> None:
        """Paint or erase a thick segment (freehand brush)."""
        if radius <= 0:
            return
        self._apply(brush_stroke(*self._shape, [(x1, y1), (x2, y2)], radius), mode)

    def import_mask(self, path: str) -> None:
        """Import a mask image (bright = inside), resized to the image size if needed."""
        from raftcorr.gui.support.io_compat import read_mask_as_bool

        self.mask = read_mask_as_bool(path, target_shape=self._shape)

    def clear(self) -> None:
        self.mask = np.zeros(self._shape, dtype=bool)

    def invert(self) -> None:
        self.mask = ~self.mask

    def save_mask(self, path: str) -> None:
        """Save the mask as a grey PNG (255 = inside)."""
        _save_mask(path, self.mask)

    def _apply(self, region: NDArray[np.bool_], mode: str) -> None:
        if mode == "add":
            self.mask = self.mask | region
        elif mode == "cut":
            self.mask = self.mask & ~region
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Expected 'add' or 'cut'.")
