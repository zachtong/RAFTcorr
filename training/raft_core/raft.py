# Derived from RAFT (Teed and Deng, ECCV 2020), BSD 3-Clause License: see LICENSE-RAFT in raft_core/.
"""Unified RAFTcorr model.

A single model spanning the legacy L (1/8 features, 4-level pyramid, convex upsampling) and F
(full-resolution, 2-level pyramid, no upsampling) variants. The only thing that changes is the
config: ``downsample_factor`` and ``corr_levels``. Intermediate factors (2, 4) are supported
for study #1.

Convention (CONVENTION.md): ``forward(image1=reference, image2=deformed)`` returns the
reference->deformed flow ``coords1 - coords0 = (u, v)`` with ``+u`` right / ``+v`` down. No
sign flip is applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .corr import CorrBlock
from .extractor import BasicEncoder, SmallEncoder
from .update import BasicUpdateBlock, SmallUpdateBlock
from .utils.utils import coords_grid, upflow

try:
    autocast = torch.amp.autocast
    _AUTOCAST_NEW = True
except AttributeError:  # very old torch
    autocast = torch.cuda.amp.autocast
    _AUTOCAST_NEW = False


def _autocast(enabled):
    if _AUTOCAST_NEW:
        return autocast("cuda", enabled=enabled)
    return autocast(enabled=enabled)


@dataclass
class RAFTConfig:
    """Model configuration. L and F are just different instances of this."""

    downsample_factor: int = 8     # 8 = RAFTcorr-L, 1 = RAFTcorr-F; 2/4 for study #1
    corr_levels: int = 4
    corr_radius: int = 4
    small: bool = False
    dropout: float = 0.0
    mixed_precision: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "RAFTConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


class RAFT(nn.Module):
    def __init__(self, cfg: RAFTConfig):
        super().__init__()
        self.cfg = cfg
        self.factor = cfg.downsample_factor

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
            self.fnet = SmallEncoder(output_dim=128, norm_fn="instance",
                                     dropout=cfg.dropout, downsample_factor=fac)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn="none",
                                     dropout=cfg.dropout, downsample_factor=fac)
            self.update_block = SmallUpdateBlock(cfg, hidden_dim=hdim)
            self._small_update = True
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn="instance",
                                     dropout=cfg.dropout, downsample_factor=fac)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn="batch",
                                     dropout=cfg.dropout, downsample_factor=fac)
            self.update_block = BasicUpdateBlock(cfg, hidden_dim=hdim, mask_channels=mask_channels)
            self._small_update = False

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """flow = coords1 - coords0 at feature resolution (input / downsample_factor)."""
        N, C, H, W = img.shape
        h, w = H // self.factor, W // self.factor
        coords0 = coords_grid(N, h, w, device=img.device)
        coords1 = coords_grid(N, h, w, device=img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
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

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """Estimate the reference->deformed flow. image1=reference, image2=deformed."""
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        with _autocast(self.cfg.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.cfg.corr_levels,
                            radius=self.cfg.corr_radius)

        with _autocast(self.cfg.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        flow_up = None
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with _autocast(self.cfg.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow

            if self.factor == 1:
                flow_up = coords1 - coords0
            elif up_mask is None:
                flow_up = upflow(coords1 - coords0, self.factor)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
        return flow_predictions
