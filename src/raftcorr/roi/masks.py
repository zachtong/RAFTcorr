"""Mask utilities: import, clean-up, bounding boxes, compact undo history, per-frame ROI sets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..io.images import IMAGE_EXTENSIONS, natural_key, read_raw, to_grey

BBox = tuple[int, int, int, int]  # (x0, y0, width, height)


def bbox_of(mask: np.ndarray) -> BBox | None:
    """Tight ``(x0, y0, width, height)`` of the True pixels, or None for an empty mask."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1)


def resize_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resize of a boolean mask (pixel-centre sampling)."""
    h, w = mask.shape
    if (h, w) == (height, width):
        return mask.copy()
    ri = np.minimum(((np.arange(height) + 0.5) * h / height).astype(int), h - 1)
    ci = np.minimum(((np.arange(width) + 0.5) * w / width).astype(int), w - 1)
    return mask[ri][:, ci]


def remove_small_regions(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Drop 8-connected True regions smaller than ``min_area_px``."""
    if min_area_px <= 0 or not mask.any():
        return mask.copy()
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return mask.copy()
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_area_px
    keep[0] = False
    return keep[lab]


def smooth_edges(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Round off jagged edges: Gaussian blur of the indicator, threshold at one half."""
    if radius_px <= 0:
        return mask.copy()
    return ndimage.gaussian_filter(mask.astype(np.float32), radius_px / 2.0) >= 0.5


def load_mask(path: str | Path, shape: tuple[int, int] | None = None, threshold: float = 127.0) -> np.ndarray:
    """Read a mask image (white = inside). Resized by nearest neighbour to ``shape`` if given."""
    grey = to_grey(read_raw(path)).astype(np.float64)
    if grey.max() <= 1.0:
        grey = grey * 255.0
    elif grey.max() > 255.0:
        grey = grey * (255.0 / grey.max())
    mask = grey > threshold
    if shape is not None and mask.shape != tuple(shape):
        mask = resize_nearest(mask, int(shape[0]), int(shape[1]))
    return mask


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(str(path))


_LAST_NUMBER = re.compile(r"(\d+)(?!.*\d)")


def match_mask_files(files: list[Path], n_frames: int, strategy: str = "sequential") -> dict[int, Path]:
    """Frame index (0-based) -> mask file.

    * ``sequential``: the k-th file in natural order belongs to frame k;
    * ``by_number``: the last number in the file name is the 0-based frame index.

    Files that map past the last frame are ignored.
    """
    ordered = sorted(files, key=lambda p: natural_key(p.name))
    out: dict[int, Path] = {}
    if strategy == "sequential":
        for k, p in enumerate(ordered):
            if k < n_frames:
                out[k] = p
    elif strategy == "by_number":
        for p in ordered:
            m = _LAST_NUMBER.search(p.stem)
            if m:
                k = int(m.group(1))
                if 0 <= k < n_frames:
                    out[k] = p
    else:
        raise ValueError(f"strategy must be 'sequential' or 'by_number', got {strategy!r}")
    return out


def list_mask_files(folder: str | Path) -> list[Path]:
    d = Path(folder)
    return sorted((p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                  key=lambda p: natural_key(p.name))


@dataclass(frozen=True)
class PackedMask:
    """A boolean mask stored as bits (8x smaller), for undo history and project files."""

    shape: tuple[int, int]
    bits: bytes

    @classmethod
    def pack(cls, mask: np.ndarray) -> "PackedMask":
        m = np.asarray(mask, dtype=bool)
        return cls((int(m.shape[0]), int(m.shape[1])), np.packbits(m, axis=None).tobytes())

    def unpack(self) -> np.ndarray:
        n = self.shape[0] * self.shape[1]
        return np.unpackbits(np.frombuffer(self.bits, dtype=np.uint8), count=n).astype(bool).reshape(self.shape)


class MaskHistory:
    """Undo / redo of edits to a set of per-frame masks (one shared stack across frames)."""

    def __init__(self, depth: int = 30) -> None:
        self.depth = max(1, int(depth))
        self._undo: list[tuple[int, PackedMask | None]] = []
        self._redo: list[tuple[int, PackedMask | None]] = []

    def record(self, frame: int, before: np.ndarray | None) -> None:
        """Call before changing ``frame``'s mask, with its current mask (None = no mask)."""
        self._undo.append((frame, None if before is None else PackedMask.pack(before)))
        if len(self._undo) > self.depth:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self, current: dict[int, np.ndarray]) -> tuple[int, np.ndarray | None] | None:
        """Return ``(frame, mask to restore)`` and push the current state for redo."""
        if not self._undo:
            return None
        frame, packed = self._undo.pop()
        cur = current.get(frame)
        self._redo.append((frame, None if cur is None else PackedMask.pack(cur)))
        return frame, None if packed is None else packed.unpack()

    def redo(self, current: dict[int, np.ndarray]) -> tuple[int, np.ndarray | None] | None:
        if not self._redo:
            return None
        frame, packed = self._redo.pop()
        cur = current.get(frame)
        self._undo.append((frame, None if cur is None else PackedMask.pack(cur)))
        return frame, None if packed is None else packed.unpack()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()
