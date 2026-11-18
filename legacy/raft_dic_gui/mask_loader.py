"""Load and match per-frame masks from a user-provided folder."""

import os
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg"}


def _natural_sort_key(s: str):
    """Sort key that treats numeric substrings as integers."""
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", s)]


def _load_mask(path: Path, image_shape: Tuple[int, int]) -> np.ndarray | None:
    """Load a mask image, validate dimensions, and return a boolean array.

    Returns None if the mask cannot be loaded or has wrong dimensions.
    """
    from PIL import Image

    try:
        img = Image.open(path)
        arr = np.array(img)
    except Exception as e:
        print(f"{path.name}: failed to load ({e}), skipped")
        return None

    # Convert RGB/RGBA to grayscale by taking first channel
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    # Validate dimensions
    h, w = image_shape
    if arr.shape[0] != h or arr.shape[1] != w:
        print(
            f"{path.name}: size {arr.shape[1]}x{arr.shape[0]} doesn't match "
            f"image {w}x{h}, skipped"
        )
        return None

    # Binarize
    return arr > 0


def discover_masks(
    mask_dir: str,
    image_files: List[str],
    image_shape: Tuple[int, int],
) -> Dict[int, np.ndarray]:
    """Discover and load per-frame masks from a directory.

    Matching strategy: natural-sort mask files, then pair with image files
    by positional index (0-th mask → 0-th image, 1st mask → 1st image, …).
    Extra masks beyond the image count are ignored.

    Parameters
    ----------
    mask_dir : str
        Path to folder containing mask images.
    image_files : list of str
        Ordered list of image filenames (basenames).
    image_shape : (int, int)
        Expected (H, W) dimensions for masks.

    Returns
    -------
    dict
        Mapping of 0-based frame index to boolean numpy array (H, W).
    """
    mask_path = Path(mask_dir)
    if not mask_path.is_dir():
        print(f"Mask directory not found: {mask_dir}")
        return {}

    num_images = len(image_files)
    result: Dict[int, np.ndarray] = {}

    # Collect mask files, natural-sorted
    mask_files = sorted(
        (f for f in mask_path.iterdir()
         if f.is_file() and f.suffix.lower() in MASK_EXTENSIONS),
        key=lambda f: _natural_sort_key(f.name),
    )

    for idx, mask_file in enumerate(mask_files):
        if idx >= num_images:
            print(f"{mask_file.name}: index {idx} exceeds image count "
                  f"({num_images}), skipped")
            break

        mask = _load_mask(mask_file, image_shape)
        if mask is None:
            continue

        result[idx] = mask
        print(f"  mask[{idx}] = {mask_file.name} -> {image_files[idx]}")

    # Summary
    if not result:
        print("No valid masks found, using auto warp for all frames")
    else:
        print(
            f"Loaded {len(result)} masks for {num_images} images. "
            f"Remaining frames will use auto warp."
        )

    return result
