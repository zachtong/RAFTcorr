"""Shared helpers of the training scripts: import path, default folders and model loading."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from raft_core import build_raft  # noqa: E402

RUNS = HERE / "runs"          # default --runs-root; the validation cache goes next to it


def load_any(path, device="cuda", eval_mode=True):
    """Load a RAFTcorr checkpoint ({"model", "model_cfg", ...}) and check that every key is present."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "model" in ck and "model_cfg" in ck):
        raise ValueError(f"{path}: not a RAFTcorr checkpoint (needs 'model' and 'model_cfg')")
    model = build_raft(dict(ck["model_cfg"]))
    sd = {k.replace("module.", ""): v for k, v in ck["model"].items()}
    missing, _unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"{path}: {len(missing)} missing keys, e.g. {missing[:3]}")
    model = model.to(device)
    return model.eval() if eval_mode else model
