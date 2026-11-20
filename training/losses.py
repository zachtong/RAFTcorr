"""RAFT sequence loss: L1 over the GRU predictions with exponential weights, as in RAFT."""
from __future__ import annotations

import torch

MAX_FLOW = 400


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """L1 over the GRU prediction sequence with exponential weights; EPE of the last prediction."""
    n_predictions = len(flow_preds)
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    flow_loss = 0.0
    for i in range(n_predictions):
        w = gamma ** (n_predictions - i - 1)
        flow_loss += w * (valid[:, None] * (flow_preds[i] - flow_gt).abs()).mean()
    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt().view(-1)[valid.view(-1)]
    return flow_loss, {"epe": epe.mean().item()}
