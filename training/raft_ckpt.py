"""RAFT with optional gradient checkpointing of every GRU iteration (same math, less memory).

Each refinement iteration (correlation lookup + update block + upsampling) is recomputed during the
backward pass instead of storing its activations, so memory no longer grows with the activations of
all 12 iterations at feature resolution - the dominant cost for s1/s2 at large crops.
The correlation class is read from raft_core.raft at call time, so the on-the-fly CorrBlockOTF
patch (train_v2.py) applies here too.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

import raft_core.raft as raft_mod
from raft_core.raft import RAFT, RAFTConfig, _autocast
from raft_core.utils.utils import upflow


class RAFTCkpt(RAFT):
    def __init__(self, cfg: RAFTConfig, grad_ckpt: bool = False):
        super().__init__(cfg)
        self.grad_ckpt = grad_ckpt

    def _iteration(self, net, inp, coords0, coords1, corr_fn):
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
        return net, coords1, flow_up

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        hdim, cdim = self.hidden_dim, self.context_dim
        with _autocast(self.cfg.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])
        corr_fn = raft_mod.CorrBlock(fmap1.float(), fmap2.float(), num_levels=self.cfg.corr_levels,
                                     radius=self.cfg.corr_radius)
        with _autocast(self.cfg.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net, inp = torch.tanh(net), torch.relu(inp)
        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init
        preds, flow_up = [], None
        use_ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
        for _ in range(iters):
            coords1 = coords1.detach()
            if use_ckpt:
                net, coords1, flow_up = checkpoint(self._iteration, net, inp, coords0, coords1, corr_fn,
                                                   use_reentrant=False)
            else:
                net, coords1, flow_up = self._iteration(net, inp, coords0, coords1, corr_fn)
            preds.append(flow_up)
        if test_mode:
            return coords1 - coords0, flow_up
        return preds


def build_raft_ckpt(model_cfg: dict, grad_ckpt: bool = False) -> RAFTCkpt:
    return RAFTCkpt(RAFTConfig.from_dict(model_cfg), grad_ckpt=grad_ckpt)
