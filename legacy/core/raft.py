import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8

# use new torch.amp.autocast API; context chosen per device
def _amp_autocast(enabled: bool, is_cuda: bool):
    try:
        return torch.amp.autocast(device_type='cuda', enabled=(enabled and is_cuda))
    except Exception:
        # Fallback: no autocast
        class _NoOp:
            def __enter__(self):
                return None
            def __exit__(self, *args):
                return False
        return _NoOp()


VARIANT_PYRAMID = "pyramid"
VARIANT_FULL_RES = "full_res"
DEFAULT_VARIANT = VARIANT_PYRAMID
_VARIANT_SPECS = {
    VARIANT_PYRAMID: {
        "full_resolution": False,
        "default_corr_levels": 4,
        "default_radius_small": 3,
        "default_radius_large": 4,
        "needs_upsample": True,
    },
    VARIANT_FULL_RES: {
        "full_resolution": True,
        "default_corr_levels": 2,
        "default_radius_small": 3,
        "default_radius_large": 4,
        "needs_upsample": False,
    },
}


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        variant_raw = getattr(args, 'variant', DEFAULT_VARIANT)
        variant = variant_raw.lower().replace('-', '_') if isinstance(variant_raw, str) else DEFAULT_VARIANT
        if variant not in _VARIANT_SPECS:
            variant = DEFAULT_VARIANT
        self.variant = variant
        spec = _VARIANT_SPECS[variant]
        self.full_resolution = spec["full_resolution"]
        self._needs_upsample = spec["needs_upsample"]
        setattr(self.args, 'variant', self.variant)
        setattr(self.args, 'full_resolution', self.full_resolution)

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            default_levels = spec["default_corr_levels"]
            default_radius = spec["default_radius_small"]
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            default_levels = spec["default_corr_levels"]
            default_radius = spec["default_radius_large"]

        if not hasattr(self.args, 'corr_levels') or self.args.corr_levels is None:
            self.args.corr_levels = default_levels
        else:
            self.args.corr_levels = int(self.args.corr_levels)

        if not hasattr(self.args, 'corr_radius') or self.args.corr_radius is None:
            self.args.corr_radius = default_radius
        else:
            self.args.corr_radius = int(self.args.corr_radius)

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        encoder_kwargs = {'full_resolution': self.full_resolution}

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout, **encoder_kwargs)
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout, **encoder_kwargs)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout, **encoder_kwargs)
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout, **encoder_kwargs)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        if self.full_resolution:
            coords0 = coords_grid(N, H, W, device=img.device)
            coords1 = coords_grid(N, H, W, device=img.device)
        else:
            coords0 = coords_grid(N, H//8, W//8, device=img.device)
            coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with _amp_autocast(self.args.mixed_precision, image1.is_cuda):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.args.corr_levels, radius=self.args.corr_radius)

        # run the context network
        with _amp_autocast(self.args.mixed_precision, image1.is_cuda):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with _amp_autocast(self.args.mixed_precision, image1.is_cuda):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if not self._needs_upsample:
                flow_up = coords1 - coords0
            else:
                if up_mask is None:
                    flow_up = upflow8(coords1 - coords0)
                else:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
