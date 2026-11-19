"""Image discovery and reading.

Every frame becomes a 2-D float32 array of grey levels in ``[0, 255]`` (the network's input
range). Rules, applied identically to every frame of a sequence so that brightness constancy
between frames is never broken by the reader:

* colour -> luminance (ITU-R BT.601 weights); alpha is dropped;
* 8-bit stays as is;
* integer images with more bits are scaled by their *effective* bit depth, fixed once per
  sequence from the reference frame (a 12-bit camera stored in 16-bit files gives 0-4095, so the
  scale is ``255 / 4095``, not ``255 / 65535``);
* float images: ``[0, 1]`` data is scaled by 255, anything else by the reference frame's maximum;
* values are clipped to ``[0, 255]`` and kept as float (no re-quantisation).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)
_NUM_RE = re.compile(r"(\d+)")


def natural_key(name: str) -> list:
    """Sort key that orders ``img2`` before ``img10``."""
    return [int(t) if t.isdigit() else t.lower() for t in _NUM_RE.split(name)]


def list_images(folder: str | Path, natural: bool = True, extensions: tuple[str, ...] = IMAGE_EXTENSIONS) -> list[Path]:
    """Image files directly inside ``folder``, naturally sorted by default."""
    d = Path(folder)
    if not d.is_dir():
        raise NotADirectoryError(f"not a folder: {d}")
    exts = {e.lower() for e in extensions}
    files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")]
    key = (lambda p: natural_key(p.name)) if natural else (lambda p: p.name.lower())
    return sorted(files, key=key)


def lexicographic_differs(files: list[Path]) -> bool:
    """True when plain sorting would order numbered names differently from natural sorting."""
    names = [p.name for p in files]
    return sorted(names, key=str.lower) != sorted(names, key=natural_key)


def read_raw(path: str | Path) -> np.ndarray:
    """Pixel array as stored (2-D grey or 3-D colour), first page of multi-page files."""
    p = Path(path)
    if p.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile

            arr = tifffile.imread(str(p), key=0)
            return np.asarray(arr)
        except Exception:
            pass  # not a real TIFF or an exotic variant: fall back to Pillow
    from PIL import Image

    with Image.open(p) as im:
        if im.mode == "P":
            im = im.convert("RGBA" if "transparency" in im.info else "RGB")
        elif im.mode in ("1", "CMYK", "YCbCr", "LAB", "HSV"):
            im = im.convert("RGB" if im.mode != "1" else "L")
        elif im.mode == "I;16B":
            im = im.convert("I;16")
        arr = np.asarray(im)
    return arr


def to_grey(arr: np.ndarray) -> np.ndarray:
    """2-D array in the source dtype's units (float64 for colour)."""
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.moveaxis(a, 0, -1)  # channel-first TIFF
    if a.ndim != 3:
        raise ValueError(f"unsupported image shape {a.shape}")
    if a.shape[-1] == 1:
        return a[..., 0]
    if a.shape[-1] == 2:  # grey + alpha
        return a[..., 0]
    return a[..., :3].astype(np.float64) @ _LUMA


@dataclass(frozen=True)
class IntensityScale:
    """Linear map of stored values to ``[0, 255]``: ``out = clip(value * gain, 0, 255)``."""

    gain: float
    description: str

    def apply(self, grey: np.ndarray) -> np.ndarray:
        """New float32 array; ``grey`` is never modified."""
        out = np.asarray(grey, dtype=np.float32) * np.float32(self.gain)  # always a new array
        return np.clip(out, 0.0, 255.0, out=out)


def effective_bit_depth(max_value: float) -> int:
    """Smallest even bit depth >= 8 that holds ``max_value`` (8, 10, 12, 14 or 16)."""
    for bits in (8, 10, 12, 14, 16):
        if max_value <= 2**bits - 1:
            return bits
    return 16


def scale_for(raw: np.ndarray) -> IntensityScale:
    """The intensity scale a sequence uses, decided from its reference frame."""
    a = np.asarray(raw)
    if a.dtype == np.uint8:
        return IntensityScale(1.0, "8-bit")
    if np.issubdtype(a.dtype, np.integer) or a.dtype == bool:
        bits = effective_bit_depth(float(a.max()) if a.size else 0.0)
        return IntensityScale(255.0 / (2**bits - 1), f"{bits}-bit (stored as {a.dtype})")
    finite = a[np.isfinite(a)]
    mx = float(finite.max()) if finite.size else 1.0
    if mx <= 1.0:
        return IntensityScale(255.0, "float in [0, 1]")
    return IntensityScale(255.0 / mx, f"float, scaled by the reference maximum {mx:g}")


def read_grey(path: str | Path, scale: IntensityScale | None = None) -> np.ndarray:
    """One frame as float32 grey levels in ``[0, 255]``.

    Pass the sequence's ``scale`` (from :func:`scale_for` on the reference frame) for every frame
    of a sequence; without it the frame's own scale is used.
    """
    raw = read_raw(path)
    grey = to_grey(raw)
    s = scale or scale_for(grey if np.issubdtype(np.asarray(raw).dtype, np.floating) else raw)
    out = s.apply(grey)
    out = np.nan_to_num(out, nan=0.0, posinf=255.0, neginf=0.0)
    return np.ascontiguousarray(out, dtype=np.float32)


def image_size(path: str | Path) -> tuple[int, int]:
    """``(height, width)`` without decoding the whole image when possible."""
    p = Path(path)
    if p.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile

            with tifffile.TiffFile(str(p)) as tf:
                shape = tf.pages[0].shape
            if len(shape) == 3 and shape[0] in (3, 4) and shape[-1] not in (3, 4):
                return int(shape[1]), int(shape[2])
            return int(shape[0]), int(shape[1])
        except Exception:
            pass
    from PIL import Image

    with Image.open(p) as im:
        w, h = im.size
    return int(h), int(w)


class ImageSequence:
    """An ordered list of frames that share one intensity scale (from frame 0), with a small cache."""

    def __init__(self, files: list[Path] | list[str], cache_size: int = 8) -> None:
        if not files:
            raise ValueError("an image sequence needs at least one file")
        self.files = [Path(f) for f in files]
        self._cache: dict[int, np.ndarray] = {}
        self._order: list[int] = []
        self._lock = threading.Lock()  # the GUI thread and a worker may read frames concurrently
        self.cache_size = max(1, int(cache_size))
        raw0 = read_raw(self.files[0])
        grey0 = to_grey(raw0)
        self.scale = scale_for(grey0 if np.issubdtype(np.asarray(raw0).dtype, np.floating) else raw0)
        first = np.nan_to_num(self.scale.apply(grey0), nan=0.0).astype(np.float32)
        self.shape: tuple[int, int] = (int(first.shape[0]), int(first.shape[1]))
        self._put(0, first)
        self._check_sizes()

    def _check_sizes(self) -> None:
        """Refuse a sequence whose frames differ in size (read from the headers, fast)."""
        bad = []
        for k, p in enumerate(self.files[1:], start=2):  # frame numbers count from 1, as on screen
            try:
                h, w = image_size(p)
                if (h, w) != self.shape:
                    bad.append(f"frame {k}, {p.name} ({w} x {h} px)")
            except Exception as exc:  # unreadable header
                bad.append(f"frame {k}, {p.name} ({exc})")
        if bad:
            more = f" and {len(bad) - 5} more" if len(bad) > 5 else ""
            raise ValueError(f"every frame must have the size of frame 1, {self.files[0].name} "
                             f"({self.shape[1]} x {self.shape[0]} px); different: {'; '.join(bad[:5])}{more}")

    @classmethod
    def from_folder(cls, folder: str | Path, natural: bool = True, cache_size: int = 8) -> "ImageSequence":
        files = list_images(folder, natural=natural)
        if not files:
            raise FileNotFoundError(f"no images ({', '.join(IMAGE_EXTENSIONS)}) in {folder}")
        return cls(files, cache_size=cache_size)

    def __len__(self) -> int:
        return len(self.files)

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.files]

    def _put(self, i: int, img: np.ndarray) -> None:
        img.setflags(write=False)
        self._cache[i] = img
        if i in self._order:
            self._order.remove(i)
        self._order.append(i)
        while len(self._order) > self.cache_size:
            self._cache.pop(self._order.pop(0), None)

    def __getitem__(self, i: int) -> np.ndarray:
        if i < 0:
            i += len(self.files)
        if not 0 <= i < len(self.files):
            raise IndexError(f"frame {i} out of range [0, {len(self.files) - 1}]")
        with self._lock:
            hit = self._cache.get(i)
            if hit is not None:
                self._order.remove(i)
                self._order.append(i)
                return hit
        img = read_grey(self.files[i], self.scale)
        if img.shape != self.shape:
            raise ValueError(f"frame {i + 1}, {self.files[i].name}: {img.shape[1]} x {img.shape[0]} px differs "
                             f"from frame 1, {self.files[0].name} ({self.shape[1]} x {self.shape[0]} px)")
        with self._lock:
            self._put(i, img)
        return img
