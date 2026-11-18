"""Measure the peak device memory of one forward pass and fit the memory model.

For each model and correlation mode, runs single pairs of increasing size and records
``torch.cuda.max_memory_allocated`` above the baseline (weights already resident). Fits the
per-pixel coefficient of the on-demand modes, ``bytes/px = A / s^2 + B``, and checks the stored
volume's quadratic term. Prints the constants for ``raftcorr/engine/memory.py``.

    python scripts/calibrate_memory.py [--out outputs/memory]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from raftcorr.engine.backend import Predictor  # noqa: E402
from raftcorr.engine.memory import estimate_peak_bytes  # noqa: E402
from raftcorr.models import available_models, load_model  # noqa: E402

SIDES = {"tile": [256, 384, 512, 768], "volume": [256, 384, 512, 768, 1024]}
VOLUME_LIMIT_GB = 6.0


def measure(pred: Predictor, side: int) -> int:
    import torch

    rng = np.random.default_rng(0)
    a = rng.uniform(0, 255, (side, side)).astype(np.float32)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    pred.predict_tensor(a, a)
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated() - base)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "memory"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, path in available_models().items():
        if path is None:
            continue
        s = int(key[1:])
        for mode in ("tile", "volume"):
            if mode == "tile" and s == 8:
                continue
            loaded = load_model(path, device="cuda", corr_mode=mode)
            pred = Predictor(loaded, iters=20)
            pred.warmup(128)
            for side in SIDES[mode]:
                n = side * side
                if mode == "volume" and 8.0 * (n / s**2) ** 2 > VOLUME_LIMIT_GB * 1024**3:
                    continue
                peak = measure(pred, side)
                est = estimate_peak_bytes(n, s, mode)
                rows.append({"model": key, "s": s, "mode": mode, "side": side, "pixels": n,
                             "peak_bytes": peak, "estimate_bytes": est})
                print(f"{key} {mode:6s} {side:5d}^2  peak {peak / 1024**2:8.1f} MB   estimate {est / 1024**2:8.1f} MB")
            del pred, loaded
    (out / "memory.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # fit bytes/px of the on-demand mode per s (slope of peak vs pixels), then A / s^2 + B
    slopes = {}
    for s in (1, 2, 4):
        r = [x for x in rows if x["s"] == s and x["mode"] == "tile"]
        if len(r) >= 2:
            n = np.array([x["pixels"] for x in r], float)
            p = np.array([x["peak_bytes"] for x in r], float)
            slopes[s] = np.polyfit(n, p, 1)
            print(f"s{s} tile: {slopes[s][0]:.1f} bytes/px + {slopes[s][1] / 1024**2:.0f} MB")
    if len(slopes) >= 2:
        inv = np.array([1.0 / s**2 for s in slopes])
        k = np.array([slopes[s][0] for s in slopes])
        A, B = np.polyfit(inv, k, 1)
        print(f"fit: bytes/px = {A:.0f} / s^2 + {B:.0f}")
    # volume: residual after the linear part -> coefficient of n^2 (n = pixels / s^2)
    for s in (4, 8):
        r = [x for x in rows if x["s"] == s and x["mode"] == "volume"]
        if len(r) >= 2 and s in slopes:
            pass
        if r:
            q = [(x["peak_bytes"] - x["pixels"] * (slopes.get(s, [0])[0] if s in slopes else 0)) / (x["pixels"] / s**2) ** 2
                 for x in r]
            print(f"s{s} volume: peak / n^2 = {[round(v, 2) for v in q]} (n = pixels / s^2)")


if __name__ == "__main__":
    main()
