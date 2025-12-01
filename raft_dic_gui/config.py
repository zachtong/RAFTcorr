import dataclasses
from dataclasses import dataclass, field
from typing import Tuple, Optional, Any

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
    
    # Crop Settings
    use_crop: bool = False
    crop_size: Tuple[int, int] = (0, 0)  # (H, W)
    shift: int = 1800
    
    # Smoothing Settings
    use_smooth: bool = True
    sigma: float = 2.0
    
    # Tiling / ROI Settings
    D_global: int = 100      # Context padding
    g_tile: int = 100        # Guard band
    overlap_ratio: float = 0.10
    p_max_pixels: int = 1100 * 1100
    prefer_square: bool = False
    
    # Runtime / Hardware
    device: str = "cuda"
    
    # Internal / Metadata
    model_metadata: Optional[Any] = None

    def validate(self) -> Tuple[bool, str]:
        """Validate the configuration. Returns (is_valid, error_message)."""
        if not self.img_dir:
            return False, "Input directory is not selected."
        if not self.project_root:
            return False, "Output directory is not selected."
        if not self.model_path:
            return False, "Model checkpoint is not selected."
        
        if self.use_crop:
            h, w = self.crop_size
            if h <= 0 or w <= 0:
                return False, "Invalid crop size."
                
        return True, ""
