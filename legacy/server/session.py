"""Server-side session state — replaces RAFTDICGUI class for state management."""

import os
import shutil
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from raft_dic_gui.config import DICConfig
from raft_dic_gui.controller import DICProcessor
from raft_dic_gui.probe_manager import ProbeManager
from raft_dic_gui.deformed_view_cache import DeformedViewCache
from server.deformed_warp import InverseMapCache

# ---------------------------------------------------------------------------
# Session cache limits
# ---------------------------------------------------------------------------

# Maximum frames held in the LRU image cache.  Each frame is a
# full-resolution numpy array, so this caps resident memory for raw images.
IMAGE_CACHE_MAX_FRAMES: int = 50

# Maximum total bytes for the image cache (500 MB default).
# Prevents OOM when caching large (e.g. 4096×4096) images.
IMAGE_CACHE_MAX_BYTES: int = 500 * 1024 * 1024


def _make_set_event() -> threading.Event:
    """Create an Event that starts in the 'set' (unblocked) state."""
    e = threading.Event()
    e.set()
    return e


@dataclass
class AppSession:
    """Singleton holding all application state between requests."""

    # Configuration
    config: DICConfig = field(default_factory=DICConfig)

    # Image state
    image_dir: str = ""
    image_files: List[str] = field(default_factory=list)
    reference_image: Optional[np.ndarray] = None
    image_width: int = 0
    image_height: int = 0

    # ROI state
    roi_mask: Optional[np.ndarray] = None
    roi_rect: Optional[Tuple[int, int, int, int]] = None
    roi_confirmed: bool = False

    # Per-frame ROI masks: 0-based frame index -> bool mask (H, W)
    # per_frame_rois[0] is kept in sync with roi_mask (Frame 1)
    per_frame_rois: Dict[int, np.ndarray] = field(default_factory=dict)

    # Envelope rect — dynamic bounding box of displacement data (may grow beyond roi_rect)
    envelope_rect: Optional[Tuple[int, int, int, int]] = None

    # Processing state
    processor: DICProcessor = field(default_factory=DICProcessor)
    processing_active: bool = False
    stop_requested: bool = False
    pause_requested: bool = False
    pause_event: threading.Event = field(default_factory=lambda: _make_set_event(), repr=False)
    displacement_results: List[Any] = field(default_factory=list)
    result_version: int = 0

    # Strain state
    strain_results: List[Any] = field(default_factory=list)
    strain_computing: bool = False
    strain_components: List[str] = field(default_factory=list)

    # Probes
    probe_manager: ProbeManager = field(default_factory=ProbeManager)

    # Cache
    deformed_view_cache: DeformedViewCache = field(
        default_factory=DeformedViewCache
    )
    inverse_map_cache: InverseMapCache = field(
        default_factory=InverseMapCache,
        repr=False,
    )

    # Visualization settings (used by render endpoints)
    vis_settings: Dict[str, Any] = field(default_factory=lambda: {
        "colormap": "turbo",
        "alpha": 0.7,
        "background": "reference",
        "log_scale": False,
        "physical_ratio": 1.0,
        "physical_unit": "px",
        "dt": 1.0,
        "fps": 1.0,
    })

    # Export state
    export_active: bool = False
    export_progress: int = 0
    export_total: int = 0
    export_cancel: threading.Event = field(default_factory=threading.Event)

    # Image frame cache (LRU, dual-limited by count and bytes)
    _image_cache: OrderedDict = field(default_factory=OrderedDict, repr=False)
    _image_cache_max: int = IMAGE_CACHE_MAX_FRAMES
    _image_cache_max_bytes: int = IMAGE_CACHE_MAX_BYTES
    _image_cache_bytes: int = 0

    # Temporary output directory (set by processing, cleaned on reset)
    _temp_output_dir: Optional[str] = None

    # Lock for thread-safe access
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_cached_image(self, idx: int):
        """Get cached image by index, or None."""
        if idx in self._image_cache:
            self._image_cache.move_to_end(idx)
            return self._image_cache[idx]
        return None

    def cache_image(self, idx: int, img):
        """Cache an image array with byte-budget enforcement."""
        # If replacing an existing entry, subtract its size first
        if idx in self._image_cache:
            old = self._image_cache[idx]
            self._image_cache_bytes -= getattr(old, "nbytes", 0)

        self._image_cache[idx] = img
        self._image_cache.move_to_end(idx)
        self._image_cache_bytes += getattr(img, "nbytes", 0)

        # Evict oldest entries until under both limits
        while (
            len(self._image_cache) > self._image_cache_max
            or self._image_cache_bytes > self._image_cache_max_bytes
        ):
            if not self._image_cache:
                break
            _, evicted = self._image_cache.popitem(last=False)
            self._image_cache_bytes -= getattr(evicted, "nbytes", 0)

        # Guard against negative drift from non-array values
        if self._image_cache_bytes < 0:
            self._image_cache_bytes = 0

    def reset(self):
        """Reset all state to defaults."""
        self.image_dir = ""
        self.image_files = []
        self.reference_image = None
        self.image_width = 0
        self.image_height = 0
        self.roi_mask = None
        self.roi_rect = None
        self.roi_confirmed = False
        self.per_frame_rois = {}
        self.envelope_rect = None
        self.processing_active = False
        self.stop_requested = False
        self.pause_requested = False
        self.pause_event.set()
        self.displacement_results = []
        self.strain_results = []
        self.strain_computing = False
        self.strain_components = []
        self.probe_manager = ProbeManager()
        self.deformed_view_cache = DeformedViewCache()
        self.inverse_map_cache.clear()
        self._image_cache.clear()
        self._image_cache_bytes = 0
        self.result_version += 1
        self.export_active = False
        self.export_progress = 0
        self.export_total = 0
        self.export_cancel.clear()
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Remove temporary output directory from previous processing."""
        if self._temp_output_dir and os.path.isdir(self._temp_output_dir):
            shutil.rmtree(self._temp_output_dir, ignore_errors=True)
            self._temp_output_dir = None


# Module-level singleton
session = AppSession()
