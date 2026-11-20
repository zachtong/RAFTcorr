"""Write a few training pairs of the data generator, with their exact displacement, without training.

Each sample is the reference image, the deformed image and the exact Lagrangian displacement at every
reference pixel, drawn exactly as ``train_v2.py`` draws them for the variant ``--s`` (same recipe as
the released models: min_edge 0.125 px, p_detail 1.0). Saves, in ``--out``:

* ``sample_<k>.npz`` with ``ref``, ``dfm`` (uint8, H x W), ``gt`` (float32, 2 x H x W: channel 0 = u,
  +x right; channel 1 = v, +y down; px), ``valid`` (uint8, H x W: 1 where the loss is evaluated)
  and ``type`` (translation, rotation, deformation or composite);
* ``sample_<k>_ref.png`` and ``sample_<k>_def.png``;
* with ``--figure``, ``samples.png``: reference, deformed, u and v of every sample.

    python training/generate_samples.py --field-dir path/to/DIV2K_train_HR --s 4 --n 4 --out samples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fields_v2 import TYPES, FieldCfgV2, NaturalFieldBank, SpeckleSynthV2  # noqa: E402

CROPS = {1: 128, 2: 256, 4: 512, 8: 512}          # training crop of each released model


def generate(field_dir: str, s: int, n: int, seed: int, crop: int | None, device: str) -> list[dict]:
    """``n`` samples of the variant ``s`` at a fixed seed, as numpy arrays."""
    cfg = FieldCfgV2(crop=crop or CROPS[s], s=s, min_edge=0.125, p_detail=1.0)
    bank = NaturalFieldBank(field_dir, device, side=768)
    syn = SpeckleSynthV2(cfg, bank, device, seed=seed)
    img1, img2, d, valid, info = syn.sample(n)
    return [{"ref": img1[k, 0].to(torch.uint8).cpu().numpy(),
             "dfm": img2[k, 0].to(torch.uint8).cpu().numpy(),
             "gt": d[k].float().cpu().numpy(),
             "valid": (valid[k] > 0.5).to(torch.uint8).cpu().numpy(),
             "type": TYPES[int(info["type"][k])]} for k in range(n)]


def save(samples: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for k, smp in enumerate(samples):
        np.savez_compressed(out / f"sample_{k}.npz", **smp)
        Image.fromarray(smp["ref"]).save(out / f"sample_{k}_ref.png")
        Image.fromarray(smp["dfm"]).save(out / f"sample_{k}_def.png")


def figure(samples: list[dict], path: Path, s: int, seed: int) -> None:
    """One row per sample: reference, deformed, u, v (each displacement with its own colour bar)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 11})
    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(12.5, 3.1 * n), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for k, smp in enumerate(samples):
        peak = float(np.hypot(*smp["gt"]).max())
        panels = (("Reference", smp["ref"], "gray"), ("Deformed", smp["dfm"], "gray"),
                  ("u (px)", smp["gt"][0], "turbo"), ("v (px)", smp["gt"][1], "turbo"))
        for j, (title, img, cmap) in enumerate(panels):
            ax = axes[k, j]
            im = ax.imshow(img, cmap=cmap, vmin=0 if cmap == "gray" else None,
                           vmax=255 if cmap == "gray" else None)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title if k == 0 else "", fontsize=12)
            if cmap != "gray":
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        axes[k, 0].set_ylabel(f"#{k}  {smp['type']}\npeak |d| {peak:.1f} px", fontsize=11)
    fig.suptitle(f"Generator samples, s = {s}, seed {seed}", fontsize=13)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--field-dir", required=True, metavar="path/to/DIV2K_train_HR",
                    help="folder of natural images for the smooth deformation fields (DIV2K_train_HR)")
    ap.add_argument("--s", type=int, default=4, choices=[1, 2, 4, 8], help="variant whose samples to draw")
    ap.add_argument("--n", type=int, default=4, help="number of samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", type=int, default=None, help="sample side in px (default: the training crop)")
    ap.add_argument("--out", type=Path, default=Path("samples"))
    ap.add_argument("--figure", action="store_true", help="also save samples.png")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    samples = generate(a.field_dir, a.s, a.n, a.seed, a.crop, a.device)
    save(samples, a.out)
    if a.figure:
        figure(samples, a.out / "samples.png", a.s, a.seed)
    print(f"{a.n} samples of s{a.s} (seed {a.seed}) in {a.out}")


if __name__ == "__main__":
    main()
