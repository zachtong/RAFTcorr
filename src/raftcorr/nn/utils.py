# Derived from RAFT (Teed and Deng, ECCV 2020), BSD 3-Clause License: see LICENSE-RAFT in this folder.
"""Small tensor helpers shared by the network modules.

Copied from the research repository (``raftcorr/core/utils/utils.py``) without functional
changes; ``InputPadder`` lives in :mod:`raftcorr.engine.backend` instead.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

CUDNN_GRID_SAMPLE_MAX = 2**31  # cuDNN's grid sampler fails above this many input elements


def bilinear_sampler(img: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """``grid_sample`` with pixel coordinates (x, y in the last dimension), zero padding."""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)
    if img.numel() > CUDNN_GRID_SAMPLE_MAX:
        # a large all-pairs volume: PyTorch's own kernel supports 64-bit indexing
        with torch.backends.cudnn.flags(enabled=False):
            return F.grid_sample(img, grid, align_corners=True)
    return F.grid_sample(img, grid, align_corners=True)


def coords_grid(batch: int, ht: int, wd: int, device: torch.device | str) -> torch.Tensor:
    """(batch, 2, ht, wd) pixel coordinates, channel 0 = x, channel 1 = y."""
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij"
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow(flow: torch.Tensor, factor: int, mode: str = "bilinear") -> torch.Tensor:
    """Bilinearly upsample a flow field by ``factor`` and scale its magnitude accordingly."""
    new_size = (factor * flow.shape[2], factor * flow.shape[3])
    return factor * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)
