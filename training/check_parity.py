"""Check that the training code's network is the network of the RAFTcorr application.

Loads each released model twice, with this folder's ``raft_core`` (as ``train_v2.py`` builds it) and
with the installed ``raftcorr`` package (the application's inference), runs both on the same synthetic
speckle pairs with 20 iterations, and prints the largest pixelwise difference of (u, v). Needs a CUDA
GPU, ``pip install raftcorr`` and the model files (``raftcorr models --download``).

    python training/check_parity.py
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raft_core.raft as raft_mod  # noqa: E402
from common import load_any  # noqa: E402
from corr_otf import CorrBlockOTF  # noqa: E402
from raft_core.corr import CorrBlock  # noqa: E402

from raftcorr import synthetic as syn  # noqa: E402  (the application)
from raftcorr.engine.backend import Predictor  # noqa: E402
from raftcorr.models import available_models, load_model  # noqa: E402

PAIRS = {
    "translation (+2.30, -1.20) px": (256, 256, syn.translation(2.30, -1.20)),
    "affine with a 2 degree rotation": (256, 320, syn.affine(0.01, -0.035, 0.035, -0.005, tx=1.5, ty=0.7,
                                                             cx=160, cy=128)),
    "sinusoid, 250 x 333 px": (250, 333, syn.sinusoid(1.5, 60.0, "x")),
}
TOLERANCE_PX = 1e-3


def predict(model, ref: np.ndarray, dfm: np.ndarray, iters: int = 20) -> np.ndarray:
    """Inference as in the paper: grey levels tiled to 3 channels, replicate-padded to a multiple of 8."""
    t = lambda a: torch.tensor(a, dtype=torch.float32, device="cuda")  # noqa: E731
    a, b = t(ref), t(dfm)
    h, w = a.shape
    ph, pw = (-h) % 8, (-w) % 8
    if ph or pw:
        a = F.pad(a[None, None], (0, pw, 0, ph), mode="replicate")[0, 0]
        b = F.pad(b[None, None], (0, pw, 0, ph), mode="replicate")[0, 0]
    with torch.no_grad():
        _, up = model(a[None, None].expand(1, 3, -1, -1).contiguous(),
                      b[None, None].expand(1, 3, -1, -1).contiguous(), iters=iters, test_mode=True)
    return up[0, :, :h, :w].float().cpu().numpy()


def main() -> int:
    pairs = {name: syn.render_pair(h, w, f, seed=7) for name, (h, w, f) in PAIRS.items()}
    worst = 0.0
    for key, path in available_models().items():
        if path is None:
            print(f"{key}: model file not found, skipped (raftcorr models --download)")
            continue
        # stored all-pairs volume for s8, the on-the-fly correlation for s <= 4 (same values)
        mode = "volume" if key == "s8" else "rows"
        raft_mod.CorrBlock = CorrBlock if mode == "volume" else functools.partial(CorrBlockOTF, impl=mode)
        training_net = load_any(path)
        app = Predictor(load_model(path, device="cuda", corr_mode=mode), iters=20)
        for name, (a, b) in pairs.items():
            d = float(np.abs(predict(training_net, a, b) - app.predict(a, b)).max())
            worst = max(worst, d)
            print(f"{key}  {name:34s} max |training code - application| = {d:.2e} px")
        del training_net, app
        torch.cuda.empty_cache()
    ok = worst <= TOLERANCE_PX
    print(("OK" if ok else "FAILED") + f": largest difference {worst:.2e} px (tolerance {TOLERANCE_PX:g} px)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
