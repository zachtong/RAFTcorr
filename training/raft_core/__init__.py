"""The RAFTcorr network (one RAFT for every downsampling factor s, set by RAFTConfig)."""

import torch

from .raft import RAFT, RAFTConfig


def build_raft(model_cfg: dict) -> RAFT:
    """Construct a RAFT model from a config dict (e.g. the ``model:`` block of a train YAML)."""
    return RAFT(RAFTConfig.from_dict(model_cfg))


def load_model(ckpt_path, device=None) -> RAFT:
    """Load a trained RAFTcorr model from a checkpoint saved by ``train_v2.py``.

    The checkpoint stores ``model_cfg`` so the architecture (L/F/factor) is reconstructed
    exactly; weights are then loaded and the model is set to eval mode.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_raft(ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


__all__ = ["RAFT", "RAFTConfig", "build_raft", "load_model"]
