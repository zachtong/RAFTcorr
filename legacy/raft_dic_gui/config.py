import dataclasses
from dataclasses import dataclass, field
from typing import Tuple, Optional, Any

# ---------------------------------------------------------------------------
# Named constants — extracted from hardcoded values across the codebase.
# Each constant documents its purpose so the "why" is never lost.
# ---------------------------------------------------------------------------

# ROI context: extra pixels beyond the ROI fed to RAFT so the network
# has spatial context at the boundary.  Too small -> edge artifacts.
DEFAULT_CONTEXT_PADDING: int = 64

# Overlap between adjacent tiles prevents visible seam artifacts where
# tiles meet.  Must be >= the RAFT correlation radius (typically 4 px at
# each pyramid level).
DEFAULT_TILE_OVERLAP: int = 64

# Maximum number of pixels per tile (~1100 x 1100).  Chosen to fit
# RAFT-Large + context within 8 GB VRAM with ~55 % safety margin.
MAX_TILE_PIXELS: int = 1_210_000

# Default RAFT iterations.  12 is a good speed / accuracy trade-off;
# diminishing returns beyond ~20.
DEFAULT_ITERATIONS: int = 12


@dataclass
class DICConfig:
    """
    Centralized configuration for RAFT-DIC processing.
    Holds all parameters required for the processing pipeline, decoupling them from UI state.
    """
    # Path Settings
    img_dir: str = ""
    project_root: str = ""
    model_path: str = ""
    model_label: str = ""

    # Processing Mode
    mode: str = "accumulative"  # 'accumulative' or 'incremental'

    # Smoothing Settings
    use_smooth: bool = True
    sigma: float = 2.0

    # Tiling / ROI Settings
    context_padding: int = DEFAULT_CONTEXT_PADDING
    tile_overlap: int = DEFAULT_TILE_OVERLAP
    p_max_pixels: int = MAX_TILE_PIXELS

    # Runtime / Hardware
    device: str = "cuda"

    # Incremental mode settings
    key_frames: Optional[list] = None        # User-specified key frames (1-indexed)
    key_frame_interval: Optional[int] = None  # Every N frames shortcut
    mask_dir: Optional[str] = None            # Per-frame mask folder path
    use_median_filter: bool = False           # Median filter accumulated displacement (reduces error accumulation)

    # Internal / Metadata
    model_metadata: Optional[Any] = None

    def validate(self) -> Tuple[bool, str]:
        """Validate the configuration. Returns (is_valid, error_message)."""
        if not self.img_dir:
            return False, "Input directory is not selected."
        # if not self.project_root:
        #     return False, "Output directory is not selected."
        if not self.model_path:
            return False, "Model checkpoint is not selected."

        return True, ""
