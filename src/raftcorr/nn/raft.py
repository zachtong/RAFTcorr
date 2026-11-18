# Derived from RAFT (Teed and Deng, ECCV 2020), BSD 3-Clause License: see LICENSE-RAFT in this folder.
"""The unified RAFTcorr network (one architecture for s = 1, 2, 4, 8).

Copied from the research repository (``raftcorr/core/raft.py``). Differences, none of which
changes a result:

* the correlation implementation is a constructor argument (``corr_mode``) instead of the
  module global the research scripts patch;
* in ``test_mode`` the convex upsampling runs once, after the last iteration, instead of after
  every iteration (the returned field is identical: it depends only on the final state);
* mixed precision is requested only for CUDA tensors.

Convention (``raftcorr.conventions``): ``forward(reference, deformed)`` returns the
reference -> deformed displacement ``coords1 - coords0 = (u, v)``, +u right, +v down. There is
no sign flip anywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .corr import CORR_MODES, corr_factory, default_corr_mode
from .extractor import BasicEncoder, SmallEncoder
from .update import BasicUpdateBlock, SmallUpdateBlock
from .utils import coords_grid, upflow

SUPPORTED_FACTORS: tuple[int, ...] = (1, 2, 4, 8)


def _autocast(device_type: str, enabled: bool):
    return torch.amp.autocast(device_type, enabled=enabled and device_type == "cuda")


@dataclass(frozen=True)
class RAFTConfig:
    """Architecture of one model; stored in every checkpoint as ``model_cfg``."""

    downsample_factor: int = 8
    corr_levels: int = 4
    corr_radius: int = 4
    small: bool = False
    dropout: float = 0.0
    mixed_precision: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "RAFTConfig":
        fields = set(cls.__dataclass_fields__)
        cfg = cls(**{k: v for k, v in d.items() if k in fields})
        if cfg.downsample_factor not in SUPPORTED_FACTORS:
            raise ValueError(
                f"unsupported downsample_factor={cfg.downsample_factor}; supported: {SUPPORTED_FACTORS}"
            )
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def lookup_radius_px(self) -> int:
        """Nominal reach of the coarsest correlation level: ``r * 2^(levels-1) * s`` image px."""
        return int(self.corr_radius * 2 ** (self.corr_levels - 1) * self.downsample_factor)


class RAFT(nn.Module):
    def __init__(self, cfg: RAFTConfig, corr_mode: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.factor = cfg.downsample_factor
        self.corr_mode = corr_mode or default_corr_mode(self.factor)

        if cfg.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128

        fac = cfg.downsample_factor
        # Convex upsampling only makes sense when features are downsampled.
        mask_channels = (fac * fac * 9) if fac > 1 else None

        if cfg.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn="instance", dropout=cfg.dropout, downsample_factor=fac)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn="none", dropout=cfg.dropout, downsample_factor=fac)
            self.update_block = SmallUpdateBlock(cfg, hidden_dim=hdim)
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn="instance", dropout=cfg.dropout, downsample_factor=fac)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn="batch", dropout=cfg.dropout, downsample_factor=fac)
            self.update_block = BasicUpdateBlock(cfg, hidden_dim=hdim, mask_channels=mask_channels)

    # -- correlation ------------------------------------------------------------------------------
    @property
    def corr_mode(self) -> str:
        return self._corr_mode

    @corr_mode.setter
    def corr_mode(self, mode: str) -> None:
        if mode not in CORR_MODES:
            raise ValueError(f"unknown correlation mode {mode!r}; expected one of {CORR_MODES}")
        self._corr_mode = mode
        self._corr_factory = corr_factory(mode)

    # -- helpers ----------------------------------------------------------------------------------
    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """flow = coords1 - coords0 at feature resolution (input / downsample_factor)."""
        N, _, H, W = img.shape
        h, w = H // self.factor, W // self.factor
        coords0 = coords_grid(N, h, w, device=img.device)
        coords1 = coords_grid(N, h, w, device=img.device)
        return coords0, coords1

    def upsample_flow(self, flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Convex-combination upsampling from feature resolution to full resolution."""
        fac = self.factor
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, fac, fac, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(fac * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, fac * H, fac * W)

    def _full_resolution(self, flow: torch.Tensor, up_mask: torch.Tensor | None) -> torch.Tensor:
        if self.factor == 1:
            return flow
        if up_mask is None:
            return upflow(flow, self.factor)
        return self.upsample_flow(flow, up_mask)

    # -- forward ----------------------------------------------------------------------------------
    def forward(self, image1, image2, iters: int = 12, flow_init=None, test_mode: bool = False):
        """Estimate the reference -> deformed flow. ``image1`` = reference, ``image2`` = deformed.

        Images are (N, 3, H, W) grey levels in [0, 255]; H and W must be multiples of the
        downsample factor. In ``test_mode`` returns ``(flow_low, flow_up)``, otherwise the list
        of full-resolution predictions of every iteration (training).
        """
        dev = image1.device.type
        amp = self.cfg.mixed_precision
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        with _autocast(dev, amp):
            fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = self._corr_factory(fmap1, fmap2, self.cfg.corr_levels, self.cfg.corr_radius)

        with _autocast(dev, amp):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        up_mask = None
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with _autocast(dev, amp):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow
            if not test_mode:
                flow_predictions.append(self._full_resolution(coords1 - coords0, up_mask))

        if test_mode:
            flow_low = coords1 - coords0
            return flow_low, self._full_resolution(flow_low, up_mask)
        return flow_predictions
