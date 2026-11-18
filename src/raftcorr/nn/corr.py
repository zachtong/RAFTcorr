# Derived from RAFT (Teed and Deng, ECCV 2020), BSD 3-Clause License: see LICENSE-RAFT in this folder.
"""Correlation lookups: the stored all-pairs volume and the factory that picks an implementation.

``CorrBlock`` is the research ``raftcorr/core/corr.py`` without the optional compiled kernel.
``CorrBlockOTF`` (``corr_otf.py``) computes the same lookup on demand with memory linear in
the image size. Both are mathematically identical; they differ in speed and memory only.

The network receives its correlation implementation as a constructor argument
(:data:`CORR_MODES`), never through a patched module global.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .corr_otf import CorrBlockOTF
from .utils import bilinear_sampler

#: "volume": stored all-pairs pyramid; "tile" / "rows": on demand (corr_otf.py implementations).
CORR_MODES: tuple[str, ...] = ("volume", "tile", "rows")


class CorrBlock:
    """All-pairs correlation volume with a ``num_levels`` average-pooled pyramid."""

    def __init__(self, fmap1: torch.Tensor, fmap2: torch.Tensor, num_levels: int = 4, radius: int = 4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for _ in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


CorrFactory = Callable[[torch.Tensor, torch.Tensor, int, int], object]


def default_corr_mode(downsample_factor: int) -> str:
    """Stored volume for s8 (small on the s8 grid and faster); on demand ('tile') for s <= 4."""
    return "volume" if downsample_factor >= 8 else "tile"


def corr_factory(mode: str) -> CorrFactory:
    """Return ``f(fmap1, fmap2, num_levels, radius) -> lookup`` for a mode in :data:`CORR_MODES`."""
    if mode == "volume":
        return lambda f1, f2, levels, radius: CorrBlock(f1, f2, num_levels=levels, radius=radius)
    if mode in ("tile", "rows"):
        return lambda f1, f2, levels, radius: CorrBlockOTF(f1, f2, num_levels=levels, radius=radius, impl=mode)
    raise ValueError(f"unknown correlation mode {mode!r}; expected one of {CORR_MODES}")
