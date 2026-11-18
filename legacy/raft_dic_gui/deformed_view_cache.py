"""Deformed View Cache Manager.

Manages image loading and forward-warp caches for deformed frame visualization.
The primary inverse-mapping algorithm lives in server/deformed_warp.py; this
module retains the forward-scatter warp (used by the legacy export pipeline)
and the LRU image cache (used by Flask render endpoints).
"""

import numpy as np
import cv2
from collections import OrderedDict
from typing import Dict, List, Optional

# Disable verbose debug output in production
DEBUG_DEFORMED_VIEW = False

_MAX_WARPED_DISP = 10  # Bounded LRU for warped displacement cache


def _debug(msg: str):
    if DEBUG_DEFORMED_VIEW:
        print(f"[DeformedView] {msg}")


class DeformedViewCache:
    """Image LRU cache + forward-scatter warp for deformed-mode visualization."""

    def __init__(self, max_images: int = 5):
        # LRU cache for loaded deformed frame images
        self.image_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.max_images = max_images

        # Bounded LRU for warped displacement (used by legacy export)
        self.warped_disp_cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()

        # Image file paths — set by images.py / processing.py on load
        self._image_paths: List[str] = []
        self._load_image_func = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_image_paths(self, paths: List[str]):
        self._image_paths = paths
        self.image_cache.clear()

    def set_load_function(self, load_func):
        self._load_image_func = load_func

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------

    def invalidate_all(self, reason: str = "unknown"):
        _debug(f"invalidate_all: reason='{reason}'")
        self.warped_disp_cache.clear()
        self.image_cache.clear()

    # ------------------------------------------------------------------
    # Image loading (LRU)
    # ------------------------------------------------------------------

    def get_deformed_image(self, frame_idx: int) -> Optional[np.ndarray]:
        """Load deformed frame image with LRU caching."""
        if frame_idx in self.image_cache:
            self.image_cache.move_to_end(frame_idx)
            return self.image_cache[frame_idx]

        if not self._image_paths:
            return None
        if frame_idx < 0 or frame_idx >= len(self._image_paths):
            return None
        if self._load_image_func is None:
            return None

        try:
            image = self._load_image_func(self._image_paths[frame_idx])
        except Exception:
            return None

        if len(self.image_cache) >= self.max_images:
            self.image_cache.popitem(last=False)
        self.image_cache[frame_idx] = image
        return image

    # ------------------------------------------------------------------
    # Warped displacement cache (bounded LRU, used by legacy export)
    # ------------------------------------------------------------------

    def get_warped_displacement(self, frame_idx: int) -> Optional[Dict[str, np.ndarray]]:
        if frame_idx in self.warped_disp_cache:
            self.warped_disp_cache.move_to_end(frame_idx)
            return self.warped_disp_cache[frame_idx]
        return None

    def set_warped_displacement(self, frame_idx: int, u_warped: np.ndarray, v_warped: np.ndarray):
        if frame_idx in self.warped_disp_cache:
            self.warped_disp_cache.move_to_end(frame_idx)
        else:
            while len(self.warped_disp_cache) >= _MAX_WARPED_DISP:
                self.warped_disp_cache.popitem(last=False)
        self.warped_disp_cache[frame_idx] = {
            'u_warped': u_warped,
            'v_warped': v_warped,
        }

    # ------------------------------------------------------------------
    # Forward-scatter warp (used by legacy export_images.py)
    # ------------------------------------------------------------------

    def warp_data_forward(
        self,
        data: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        roi_rect: tuple,
        output_shape: tuple,
    ) -> np.ndarray:
        """Warp data from reference to deformed coordinates using forward scatter.

        Note: For real-time visualization the inverse-mapping approach in
        ``server/deformed_warp.py`` is preferred (no holes, sub-pixel accuracy).
        This method is retained for backward compatibility with the batch export
        pipeline in ``raft_dic_gui/export_images.py``.
        """
        xmin, ymin, xmax, ymax = roi_rect
        h, w = output_shape

        y_ref, x_ref = np.mgrid[ymin:ymax, xmin:xmax]

        x_def = (x_ref + U).astype(np.float32)
        y_def = (y_ref + V).astype(np.float32)

        warped_data = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        valid_mask = ~np.isnan(U) & ~np.isnan(V) & ~np.isnan(data)

        x_idx = np.round(x_def[valid_mask]).astype(np.int32)
        y_idx = np.round(y_def[valid_mask]).astype(np.int32)
        data_valid = data[valid_mask].astype(np.float32)

        in_bounds = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
        x_idx = x_idx[in_bounds]
        y_idx = y_idx[in_bounds]
        data_valid = data_valid[in_bounds]

        np.add.at(warped_data, (y_idx, x_idx), data_valid)
        np.add.at(count_map, (y_idx, x_idx), 1.0)

        nonzero = count_map > 0
        warped_data[nonzero] /= count_map[nonzero]
        warped_data[~nonzero] = np.nan

        return warped_data
