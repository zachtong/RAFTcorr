"""Train one variant (s1/s2/s4/s8) of the RAFTcorr study from scratch with generator v2.

Standard strategy: random init, exact labels, infinite on-the-fly data (diverse procedural speckle,
translation / rotation / natural-image deformation / composite, octave-balanced peak displacement up to
D_max = 4*s*8, 25% occlusion), uniform crop, convex upsampling (s > 1). --otf auto switches to the
on-the-fly correlation (identical to all-pairs, memory O(C*H*W)) only when the all-pairs volume of a
sample would exceed ALLPAIRS_MAX_GB. Validation: a fixed seeded v2 set of the SAME variant, EPE by
octave band, sample type and occlusion.

Example:  python train_v2.py --name v2_s8_pilot --s 8 --field-dir path/to/DIV2K_train_HR
The recipes of the four released models are in README.md.
"""
from __future__ import annotations

import argparse
import functools
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import RUNS  # noqa: F401  (adds the repo to sys.path)
import raft_core.raft as raft_mod
from corr_otf import CorrBlockOTF
from fields_v2 import TYPES, FieldCfgV2, NaturalFieldBank, SpeckleSynthV2
from losses import sequence_loss
from raft_ckpt import build_raft_ckpt


ALLPAIRS_MAX_GB = 8.0  # per-sample all-pairs volume incl. pyramid above which --otf auto uses OTF


def allpairs_gb(crop, s):
    """fp32 all-pairs correlation volume plus its 3 pooled levels, for one sample."""
    n = (crop // s) ** 2
    return n * n * 4 * (4 / 3) / 2 ** 30


def model_cfg(s):
    return {"downsample_factor": s, "corr_levels": 4, "corr_radius": 4, "small": False,
            "dropout": 0.0, "mixed_precision": True}


def get_valset(cfg, bank, n, dev, cache_dir: Path):
    tag = "" if cfg.min_edge == 4.0 else f"_e{cfg.min_edge:g}"          # default key kept for old runs
    tag += "" if cfg.p_detail == 0 else f"_d{cfg.p_detail:g}"
    tag += "" if cfg.max_target <= 0 else f"_t{cfg.max_target:g}"
    path = cache_dir / f"valsetV2_s{cfg.s}_c{cfg.crop}_n{n}{tag}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=False)
    syn = SpeckleSynthV2(cfg, bank, dev, seed=20261021)
    out = {k: [] for k in ("img1", "img2", "flow", "valid", "type", "occ", "peak")}
    for _ in range(n // 4):
        i1, i2, d, va, info = syn.sample(4)
        out["img1"].append(i1[:, :1].to(torch.uint8).cpu())
        out["img2"].append(i2[:, :1].to(torch.uint8).cpu())
        out["flow"].append(d.half().cpu())
        out["valid"].append(va.to(torch.uint8).cpu())
        out["type"].append(info["type"].cpu())
        out["occ"].append(info["occ_kind"].cpu())
        out["peak"].append(info["peak"].cpu())
    out = {k: torch.cat(v) for k, v in out.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return out


@torch.no_grad()
def evaluate(model, vs, cfg, dev, iters=20, bs=2):
    was = model.training
    model.eval()
    E, M, T, O = [], [], [], []
    for s in range(0, vs["img1"].shape[0], bs):
        i1 = vs["img1"][s:s + bs].to(dev).float().expand(-1, 3, -1, -1)
        i2 = vs["img2"][s:s + bs].to(dev).float().expand(-1, 3, -1, -1)
        fl = vs["flow"][s:s + bs].to(dev).float()
        va = vs["valid"][s:s + bs].to(dev) > 0
        _, pr = model(i1, i2, iters=iters, test_mode=True)
        e, mag = (pr - fl).norm(dim=1), fl.norm(dim=1)
        for j in range(i1.shape[0]):
            m = va[j]
            E.append(e[j][m].cpu().numpy()); M.append(mag[j][m].cpu().numpy())
            T.append(np.full(int(m.sum()), int(vs["type"][s + j])))
            O.append(np.full(int(m.sum()), int(vs["occ"][s + j]) > 0))
    if was:
        model.train()
    e, mg, ty, oc = map(np.concatenate, (E, M, T, O))
    res = {"epe": float(e.mean()), "1px": float((e < 1).mean())}
    edges = cfg.bins
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mg >= lo) & (mg < hi + (1e-3 if hi == edges[-1] else 0))
        res[f"epe_{lo:g}_{hi:g}"] = float(e[sel].mean()) if sel.any() else float("nan")
    for t, name in enumerate(TYPES):
        sel = ty == t
        res[f"epe_{name}"] = float(e[sel].mean()) if sel.any() else float("nan")
    res["epe_occluded_samples"] = float(e[oc].mean()) if oc.any() else float("nan")
    return res


def atomic_save(obj, path: Path):
    """Write to a temp file then rename, so a job killed mid-save never leaves a broken checkpoint."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--s", type=int, required=True, choices=[1, 2, 4, 8])
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--field-dir", required=True, metavar="path/to/DIV2K_train_HR",
                    help="folder of natural images for the smooth deformation fields (DIV2K_train_HR)")
    ap.add_argument("--field-max-images", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--total-iters", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--pct-start", type=float, default=0.05)
    ap.add_argument("--wdecay", type=float, default=1e-5)
    ap.add_argument("--gru-iters", type=int, default=12)
    ap.add_argument("--gamma", type=float, default=0.8)
    ap.add_argument("--otf", default="auto", choices=["auto", "yes", "no"],
                    help="on-the-fly correlation; auto = only if the all-pairs volume exceeds ALLPAIRS_MAX_GB")
    ap.add_argument("--otf-impl", default="loop", choices=["loop", "rows"],
                    help="CorrBlockOTF implementation (identical results; pick the faster one on your GPU)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--val-n", type=int, default=64)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--max-mem-gb", type=float, default=0.0, help="allocator cap (local Windows: avoid WDDM spill)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true", help="recompute each GRU iteration in backward")
    ap.add_argument("--runs-root", default=str(RUNS))
    ap.add_argument("--cache-root", default=str(RUNS.parent / "cache"))
    ap.add_argument("--fresh", action="store_true", help="ignore an existing latest.pth (default: auto-resume)")
    ap.add_argument("--min-edge", type=float, default=4.0,
                    help="first magnitude bin is [0, min_edge] px, octaves above (0.125 -> sub-pixel octaves)")
    ap.add_argument("--p-detail", type=float, default=0.0,
                    help="fraction of samples with a band-pass Gaussian detail layer added to w (fields_v2)")
    ap.add_argument("--max-target", type=float, default=0.0,
                    help="cap the target peak displacement (px); 0 = up to the variant's D_max")
    ap.add_argument("--init", default=None,
                    help="initialise the model weights from this checkpoint (fresh optimiser; ignored on resume)")
    ap.add_argument("--stop-at", type=int, default=0,
                    help="pause cleanly after this step (state saved, no DONE marker); 0 = run to the end")
    args = ap.parse_args()

    dev = "cuda"
    if args.max_mem_gb > 0:
        torch.cuda.set_per_process_memory_fraction(args.max_mem_gb * 2 ** 30 / torch.cuda.mem_get_info()[1])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    otf = args.otf == "yes" or (args.otf == "auto" and allpairs_gb(args.crop, args.s) > ALLPAIRS_MAX_GB)
    if otf:
        raft_mod.CorrBlock = functools.partial(CorrBlockOTF, impl=args.otf_impl)
    out = Path(args.runs_root) / args.name
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    torch.manual_seed(args.seed)
    mcfg = model_cfg(args.s)
    model = build_raft_ckpt(mcfg, grad_ckpt=args.grad_ckpt).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.total_iters + 10,
                                                pct_start=args.pct_start, cycle_momentum=False,
                                                anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda")
    step, best = 0, float("inf")
    resumed = False
    if (out / "ckpt" / "latest.pth").exists() and not args.fresh:
        # a latest.pth that exists but cannot be read must abort loudly (no silent restart)
        ck = torch.load(out / "ckpt" / "latest.pth", map_location=dev, weights_only=False)
        if int(ck["step"]) >= args.total_iters:
            print(f"[done] latest at step {ck['step']} >= {args.total_iters}: complete", flush=True)
            (out / "DONE").write_text(str(ck["step"]))
            return
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        step, best, resumed = ck["step"], ck["best"], True
        print(f"[resume] step {step}, best {best:.3f}", flush=True)
    elif args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["model"])
        print(f"[init] weights from {args.init}", flush=True)

    bank = NaturalFieldBank(args.field_dir, dev, side=768, max_images=args.field_max_images)
    cfg = FieldCfgV2(crop=args.crop, s=args.s, min_edge=args.min_edge, p_detail=args.p_detail,
                     max_target=args.max_target)
    syn = SpeckleSynthV2(cfg, bank, dev, seed=args.seed * 1000 + step + 1)
    vs = get_valset(cfg, bank, args.val_n, dev, Path(args.cache_root))
    print(f"[init] s{args.s} D_max {cfg.d_max:g} px bins {cfg.bins} crop {args.crop} otf={otf} "
          f"grad_ckpt={args.grad_ckpt} "
          f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M batch {args.batch}x{args.accum} "
          f"lr {args.lr} field images {len(bank.imgs)}", flush=True)

    band_keys = [f"epe_{lo:g}_{hi:g}" for lo, hi in zip(cfg.bins[:-1], cfg.bins[1:])]
    fields = (["step", "lr", "train_loss", "train_epe", "sec_per_step", "epe", "1px"] + band_keys +
              [f"epe_{t}" for t in TYPES] + ["epe_occluded_samples"])
    log_path = out / "log.csv"
    if not (resumed and log_path.exists()):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    run_loss, run_epe, n, t0 = 0.0, 0.0, 0, time.perf_counter()
    while step < args.total_iters:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            i1, i2, d, va, _ = syn.sample(args.batch)
            if step == 0 and n == 0:
                print(f"[shapes] img {tuple(i1.shape)} flow {tuple(d.shape)} valid {tuple(va.shape)}", flush=True)
            preds = model(i1, i2, iters=args.gru_iters)
            loss, met = sequence_loss(preds, d, va, gamma=args.gamma)
            scaler.scale(loss / args.accum).backward()
            run_loss += loss.item(); run_epe += met["epe"]; n += 1
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        prev = scaler.get_scale()
        scaler.step(opt); scaler.update()
        if scaler.get_scale() >= prev:
            sched.step()
        step += 1
        if step % args.eval_every == 0 or step == args.total_iters:
            sps = (time.perf_counter() - t0) / args.eval_every
            res = evaluate(model, vs, cfg, dev)
            torch.cuda.empty_cache()           # evaluation tensors differ in size from training ones
            row = {"step": step, "lr": sched.get_last_lr()[0], "train_loss": run_loss / n,
                   "train_epe": run_epe / n, "sec_per_step": sps, **res}
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([row.get(k, "") for k in fields])
            print(f"step {step:6d} | loss {run_loss / n:7.3f} | val {res['epe']:6.3f} | bands " +
                  " ".join(f"{res[k]:.2f}" for k in band_keys) +
                  " | " + " ".join(f"{t[:4]} {res['epe_' + t]:.2f}" for t in TYPES) +
                  f" | occ {res['epe_occluded_samples']:.2f}" + f" | {sps:.2f}s/step", flush=True)
            ck = {"model": model.state_dict(), "model_cfg": mcfg, "step": step, "otf": otf}
            if res["epe"] < best:
                best = res["epe"]
                atomic_save(ck, out / "ckpt" / "best.pth")
            if step % args.ckpt_every == 0:
                atomic_save(ck, out / "ckpt" / f"step{step:06d}.pth")
            atomic_save({**ck, "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "scaler": scaler.state_dict(), "best": best}, out / "ckpt" / "latest.pth")
            run_loss, run_epe, n, t0 = 0.0, 0.0, 0, time.perf_counter()
            if args.stop_at and step >= args.stop_at and step < args.total_iters:
                print(f"[pause] {args.name} stopped at step {step} (--stop-at); resume with the same command",
                      flush=True)
                return
    (out / "DONE").write_text(str(step))
    print(f"[done] {args.name} reached step {step}", flush=True)


if __name__ == "__main__":
    main()
