"""Image folder loading and frame access (pyALDIC's ImageController API, RAFTcorr readers).

Frames come from :class:`raftcorr.io.images.ImageSequence`: grey levels 0-255 with one intensity
scale for the whole sequence, which is exactly what the network sees. ``read_image`` returns them
in ``[0, 1]`` (pyALDIC's convention), ``read_image_rgb`` as uint8 RGB for display.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from PySide6.QtCore import QObject

from raftcorr.gui.app_state import AppState
from raftcorr.i18n import tr_args
from raftcorr.io.images import IMAGE_EXTENSIONS, ImageSequence, natural_key

SUPPORTED_EXTENSIONS = set(IMAGE_EXTENSIONS)
_RGB_CACHE_SIZE = 12


def _natural_sort_key(name: str) -> list:
    return natural_key(name)


def _lexicographic_sort_key(name: str) -> str:
    return name.lower()


class ImageMessages(QObject):
    """The image controller's sentences (context ``ImageMessages``).

    ``ImageController`` is not a QObject, and lupdate extracts only ``self.tr`` calls on a QObject
    (with the count passed as a plain name for plurals), as in ``pipeline_messages``.
    """

    def cannot_open_title(self) -> str:
        return self.tr("Cannot open the images")

    def loaded(self, n: int, width: int, height: int, scale: str) -> str:
        """Console line after loading: ``n`` images, their size and grey-level scale."""
        return tr_args(self.tr("%n image(s), %1 x %2 px, %3", "", n), width, height, scale)


class ImageController:
    """Loads a folder, sorts it, and serves frames (grey and RGB) with small caches."""

    def __init__(self, state: AppState) -> None:
        self._state = state
        self._cache_rgb: "OrderedDict[int, NDArray[np.uint8]]" = OrderedDict()
        # Numbered DIC sequences (img1, img2, ..., img10) need natural order: on by default.
        self._natural_sort: bool = True
        self._raw_files: list[str] = []
        state.images_changed.connect(self._on_images_changed)

    def _on_images_changed(self) -> None:
        self._cache_rgb.clear()

    @property
    def natural_sort(self) -> bool:
        return self._natural_sort

    def set_natural_sort(self, enabled: bool) -> None:
        """Change sort mode and re-sort the current file list.

        Re-sorting publishes a new set of images, which drops the ROIs, results and probes;
        when the order does not change (zero-padded names) only the mode is recorded.
        """
        if enabled == self._natural_sort:
            return
        reorders = self.reorders(enabled)
        self._natural_sort = enabled
        if reorders:
            self._sort_and_update()

    def reorders(self, enabled: bool) -> bool:
        """Whether sorting with natural order ``enabled`` would change the current frame order."""
        return bool(self._raw_files) and self._sorted(enabled) != list(self._state.image_files)

    def _sorted(self, natural: bool) -> list[str]:
        if natural:
            return sorted(self._raw_files, key=lambda p: _natural_sort_key(Path(p).name))
        return sorted(self._raw_files, key=lambda p: _lexicographic_sort_key(Path(p).name))

    def load_folder(self, folder: str) -> None:
        """Scan a folder for images, sort them and publish them in the state."""
        self._cache_rgb.clear()
        folder_path = Path(folder)
        if not folder_path.is_dir():
            self._raw_files = []
            self._state.set_image_files([])
            return
        self._raw_files = [str(c) for c in folder_path.iterdir()
                           if c.is_file() and c.suffix.lower() in SUPPORTED_EXTENSIONS and not c.name.startswith(".")]
        self._sort_and_update(folder_path)

    def set_files(self, files: list[str]) -> None:
        """Replace the file list without rescanning the folder: a new set of images."""
        self._raw_files = list(files)
        self._cache_rgb.clear()
        self._publish(list(files), self._state.image_folder)

    def remove_frames(self, frames: Iterable[int]) -> None:
        """Remove frames from the current set (the image list's Delete).

        Not a new set of images: the ROIs of the remaining frames are re-keyed to their new
        numbers and probes, project and load data stay; only the results go, since frame numbers
        change.
        """
        state = self._state
        drop = {int(k) for k in frames}
        keep = [k for k in range(len(state.image_files)) if k not in drop]
        if len(keep) == len(state.image_files):
            return
        files = [state.image_files[k] for k in keep]
        self._raw_files = list(files)
        self._cache_rgb.clear()
        if not files:
            state.set_image_files([])
            return
        try:
            seq = ImageSequence([Path(f) for f in files], cache_size=12)
        except Exception as exc:  # a file vanished since it was listed
            state.log_message.emit(str(exc), "error")
            state.fatal_error.emit(ImageMessages().cannot_open_title(), str(exc))
            return
        new_index = {old: new for new, old in enumerate(keep)}
        rois = {new_index[k]: m for k, m in state.per_frame_rois.items() if k in new_index}
        state.set_remaining_frames(files, seq, rois)

    def _sort_and_update(self, folder: Path | None = None) -> None:
        self._publish(self._sorted(self._natural_sort), folder or self._state.image_folder)

    def _publish(self, files: list[str], folder: Path | None) -> None:
        self._cache_rgb.clear()
        if not files:
            self._state.set_image_files([])
            return
        try:
            seq = ImageSequence([Path(f) for f in files], cache_size=12)
        except Exception as exc:  # unreadable file or frames of different sizes
            self._state.log_message.emit(str(exc), "error")
            self._state.fatal_error.emit(ImageMessages().cannot_open_title(), str(exc))
            self._state.set_image_files([])
            return
        self._state.set_image_files(files, sequence=seq, folder=folder)
        h, w = seq.shape
        self._state.log_message.emit(ImageMessages().loaded(len(files), w, h, seq.scale.description), "info")

    def _sequence(self) -> ImageSequence:
        seq = self._state.sequence
        if seq is None:
            raise IndexError("no images loaded")
        return seq

    def read_image(self, idx: int) -> NDArray[np.float64]:
        """Frame ``idx`` as float64 grey in [0, 1]."""
        if idx < 0 or idx >= len(self._state.image_files):
            raise IndexError(f"Image index {idx} out of range")
        return self._sequence()[idx].astype(np.float64) / 255.0

    def read_image_rgb(self, idx: int) -> NDArray[np.uint8]:
        """Frame ``idx`` as (H, W, 3) uint8 RGB for display."""
        if idx < 0 or idx >= len(self._state.image_files):
            raise IndexError(f"Image index {idx} out of range")
        hit = self._cache_rgb.get(idx)
        if hit is not None:
            self._cache_rgb.move_to_end(idx)
            return hit
        g = np.clip(self._sequence()[idx] + 0.5, 0, 255).astype(np.uint8)
        rgb = np.repeat(g[:, :, None], 3, axis=2)
        self._cache_rgb[idx] = rgb
        if len(self._cache_rgb) > _RGB_CACHE_SIZE:
            self._cache_rgb.popitem(last=False)
        return rgb

    def image_dimensions(self, idx: int) -> tuple[int, int]:
        seq = self._sequence()
        return seq.shape

    def clear_cache(self) -> None:
        self._cache_rgb.clear()
