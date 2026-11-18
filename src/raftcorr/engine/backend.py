"""One image pair in, one displacement field out.

:class:`Predictor` wraps a :class:`~raftcorr.models.LoadedModel` with the canonical inference of
the paper: grey levels 0-255 (float) tiled to three channels, replicate padding on the right and
bottom to a multiple of 8, ``iters`` refinement iterations, crop back. Nothing else: no smoothing,
no sign change, no masking. Tiling and sequences live in :mod:`raftcorr.engine.tiling` and
:mod:`raftcorr.engine.pipeline`.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..conventions import PAD_MULTIPLE

DEFAULT_ITERS = 20


def _as_tensor(img: Any, device: str):
    import torch

    if isinstance(img, torch.Tensor):
        t = img.to(device=device, dtype=torch.float32)
    else:
        a = np.asarray(img)
        if a.ndim != 2:
            raise ValueError(f"expected a 2-D grey image, got shape {a.shape}")
        t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
    if t.ndim != 2:
        raise ValueError(f"expected a 2-D grey image, got shape {tuple(t.shape)}")
    return t


def pad_amounts(h: int, w: int, multiple: int = PAD_MULTIPLE) -> tuple[int, int]:
    """Rows and columns to append so that both sides are multiples of ``multiple``."""
    return (-h) % multiple, (-w) % multiple


def min_input_side(downsample_factor: int, corr_levels: int = 4) -> int:
    """Smallest image side a model accepts: its coarsest correlation level must be >= 2 px wide.

    The feature grid is ``side / s`` and each of the ``corr_levels - 1`` poolings halves it, so the
    side must be at least ``2 * 2^(levels - 1) * s`` = ``16 s`` px for four levels (128 px for s8).
    Below that the stored-volume lookup divides by zero.
    """
    return int(2 * 2 ** (corr_levels - 1) * downsample_factor)


class Predictor:
    """Runs a loaded network on single (reference, deformed) pairs."""

    def __init__(self, loaded: Any, iters: int = DEFAULT_ITERS) -> None:
        if iters < 1:
            raise ValueError(f"iters must be >= 1, got {iters}")
        self.loaded = loaded
        self.iters = int(iters)

    @property
    def device(self) -> str:
        return str(self.loaded.device)

    @property
    def downsample_factor(self) -> int:
        return int(self.loaded.downsample_factor)

    @property
    def min_side(self) -> int:
        return min_input_side(self.downsample_factor, int(self.loaded.network.cfg.corr_levels))

    def predict_tensor(self, reference: Any, deformed: Any):
        """(2, H, W) float32 torch tensor on the model's device: channel 0 = u, 1 = v."""
        import torch
        import torch.nn.functional as F

        ref = _as_tensor(reference, self.device)
        dfm = _as_tensor(deformed, self.device)
        if ref.shape != dfm.shape:
            raise ValueError(f"reference {tuple(ref.shape)} and deformed {tuple(dfm.shape)} differ in size")
        h, w = ref.shape
        if min(h, w) < self.min_side:
            raise ValueError(
                f"image {h} x {w} px is smaller than the {self.min_side} px minimum of this model "
                f"(s={self.downsample_factor})"
            )
        ph, pw = pad_amounts(h, w)
        i1 = ref[None, None].expand(1, 3, h, w)
        i2 = dfm[None, None].expand(1, 3, h, w)
        if ph or pw:
            i1 = F.pad(i1, (0, pw, 0, ph), mode="replicate")
            i2 = F.pad(i2, (0, pw, 0, ph), mode="replicate")
        with torch.inference_mode():
            _, flow_up = self.loaded.network(i1.contiguous(), i2.contiguous(), iters=self.iters, test_mode=True)
        return flow_up[0, :, :h, :w].float()

    def predict(self, reference: Any, deformed: Any) -> np.ndarray:
        """(2, H, W) float32 numpy array: channel 0 = u (+right), channel 1 = v (+down)."""
        return self.predict_tensor(reference, deformed).cpu().numpy()

    def warmup(self, size: int = 64) -> float:
        """One forward pass on a small random pair (allocator / cuDNN autotune). Returns seconds."""
        rng = np.random.default_rng(0)
        size = max(size, self.min_side)
        a = rng.uniform(0, 255, (size, size)).astype(np.float32)
        t0 = time.perf_counter()
        self.predict_tensor(a, a)
        _synchronize(self.device)
        return time.perf_counter() - t0


def _synchronize(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()


def empty_cache(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
