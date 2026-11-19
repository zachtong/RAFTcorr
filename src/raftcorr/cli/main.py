"""``raftcorr`` command line.

    raftcorr                         start the desktop application
    raftcorr gui [folder|project]    the same, opening a folder or a project
    raftcorr run FOLDER [options]    track a sequence and write results (HDF5 / MAT / NPZ, project)
    raftcorr suggest FOLDER          measure the displacement with s8 and suggest a model
    raftcorr models                  list the models and where their weights are
    raftcorr models --download       download the missing model files (per-user folder)
    raftcorr models --import FILE    install model files downloaded by hand (offline computers)
    raftcorr selftest                run every installed model on a synthetic pair
    raftcorr info                    versions, PyTorch, CUDA
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

from .._version import __version__


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="raftcorr", description="RAFTcorr: learned digital image correlation")
    ap.add_argument("--version", action="version", version=f"raftcorr {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    g = sub.add_parser("gui", help="start the desktop application")
    g.add_argument("path", nargs="?", help="image folder or .raftcorr project to open")

    r = sub.add_parser("run", help="track an image sequence")
    r.add_argument("folder", help="folder with the images (frame 0 = first file in natural order)")
    r.add_argument("--roi", help="ROI mask image (white = inside); default: the whole image")
    r.add_argument("--model", default="s4", help="s1, s2, s4, s8 or a checkpoint path (default s4)")
    r.add_argument("--config", help="RunConfig JSON (overrides the options below)")
    r.add_argument("--mode", default="accumulative", choices=["accumulative", "incremental", "every_n", "custom"])
    r.add_argument("--every", type=int, default=10, help="reference interval for --mode every_n")
    r.add_argument("--keys", default="", help="reference frames for --mode custom, e.g. 0,20,45")
    r.add_argument("--iters", type=int, default=20)
    r.add_argument("--device", default="auto")
    r.add_argument("--smooth", type=float, default=0.0, help="Gaussian smoothing of the displacement (px)")
    r.add_argument("--plain-sort", action="store_true", help="sort file names plainly instead of naturally")
    r.add_argument("--strain", action="store_true", help="also compute strain (default settings unless changed)")
    r.add_argument("--gauge", type=int, default=41, help="virtual strain gauge size (px, odd)")
    r.add_argument("--measure", default="green_lagrange", choices=["green_lagrange", "infinitesimal", "euler_almansi"])
    r.add_argument("--edge-trim", type=float, default=0.7, metavar="ALPHA",
                   help="leave strain empty within ALPHA x (gauge - 1) / 2 px of an edge (0 to 1; 0 = off)")
    r.add_argument("--out", help="results file: .h5, .mat or .npz")
    r.add_argument("--project", help="also save a .raftcorr project")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("--quiet", action="store_true")

    s = sub.add_parser("suggest", help="suggest a model from a quick s8 survey")
    s.add_argument("folder")
    s.add_argument("--roi")
    s.add_argument("--device", default="auto")

    m = sub.add_parser("models", help="list the models")
    m.add_argument("--download", nargs="*", metavar="KEY",
                   help="download the missing model files (all, or the keys given: s1 s2 s4 s8)")
    m.add_argument("--import", dest="import_files", nargs="+", metavar="FILE",
                   help="install .pth files downloaded by hand from the weights-v1 release; the SHA-256 "
                        "says which model each one is")
    st = sub.add_parser("selftest", help="run every installed model on a synthetic pair")
    st.add_argument("--device", default="auto")
    st.add_argument("--gui", action="store_true",
                    help="also open and close the main window offscreen (checks the GUI libraries, icons and "
                         "translations of an installation)")
    st.add_argument("--no-models", action="store_true", help="skip the models (with --gui: the GUI check only)")
    sub.add_parser("info", help="versions and devices")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd in (None, "gui"):
        from ..gui.app import main as gui_main  # the pyALDIC-style application

        extra = [args.path] if getattr(args, "path", None) else []
        sys.argv = [sys.argv[0], *extra]
        gui_main()  # exits the process when the window closes
        return 0
    return {"run": _run, "suggest": _suggest, "models": _models, "selftest": _selftest, "info": _info}[args.cmd](args)


def _load(folder: str, roi: str | None, natural: bool = True):
    from ..io.images import ImageSequence
    from ..roi.masks import load_mask

    seq = ImageSequence.from_folder(folder, natural=natural)
    mask = load_mask(roi, seq.shape) if roi else None
    return seq, mask


def _run(args) -> int:
    from ..callbacks import ConsoleCallbacks
    from ..config import RunConfig, StrainConfig, TrackingConfig
    from ..engine.pipeline import Pipeline

    cb = ConsoleCallbacks(min_level="warning" if args.quiet else "info")
    for target in (args.out, args.project):  # fail before the computation, not after it
        if target:
            t = Path(target)
            t.parent.mkdir(parents=True, exist_ok=True)
            if t.exists() and not args.overwrite:
                cb.log(f"{t} exists; pass --overwrite to replace it", "error")
                return 2
    if args.out and Path(args.out).suffix.lower() not in (".h5", ".hdf5", ".mat", ".npz"):
        cb.log("--out must end in .h5, .mat or .npz", "error")
        return 2
    seq, mask = _load(args.folder, args.roi, natural=not args.plain_sort)
    scfg = None
    if args.strain:  # strain settings are checked before the tracking, too
        import numpy as np

        from ..post.strain import check_edge_trim

        try:
            scfg = StrainConfig(measure=args.measure, gauge_px=args.gauge, edge_trim_alpha=args.edge_trim)
            check_edge_trim(mask if mask is not None else np.ones(seq.shape, dtype=bool), scfg)
        except ValueError as exc:
            cb.log(f"strain: {exc}", "error")
            return 2
    if args.config:
        cfg = RunConfig.load(args.config)
    else:
        keys = tuple(int(t) for t in args.keys.split(",") if t.strip())
        cfg = RunConfig(model=args.model, device=args.device, iters=args.iters, smoothing_sigma=args.smooth,
                        tracking=TrackingConfig(mode=args.mode, every_n=args.every, key_frames=keys))
    cb.log(f"{len(seq)} frames, {seq.shape[1]} x {seq.shape[0]} px ({seq.scale.description})", "info")
    series = Pipeline(cfg, cb).run(seq, mask)
    strain = None
    if scfg is not None:
        from ..post.strain import compute_strain

        strain = compute_strain(series, scfg, callbacks=cb)
    if args.out:
        from ..export.scientific import export_results

        p = export_results(args.out, series, strain, overwrite=args.overwrite, callbacks=cb)
        cb.log(f"wrote {p}", "info")
    if args.project:
        from ..io.project import Project, save_project

        prj = Project(image_folder=str(Path(args.folder).resolve()), image_files=tuple(seq.names),
                      natural_sort=not args.plain_sort, run_config=cfg, roi_masks={0: mask} if mask is not None else {},
                      series=series, strain=strain)
        p = save_project(args.project, prj, cb)
        cb.log(f"wrote {p}", "info")
    if not args.out and not args.project:
        cb.log("no --out or --project given: nothing was written", "warning")
    return 0


def _suggest(args) -> int:
    import numpy as np

    from ..callbacks import ConsoleCallbacks
    from ..config import RunConfig
    from ..engine.pipeline import Pipeline
    from ..models.advisor import sample_frames, survey

    cb = ConsoleCallbacks()
    seq, mask = _load(args.folder, args.roi)
    mask = np.ones(seq.shape, bool) if mask is None else mask
    pipe = Pipeline(RunConfig(model="s8", device=args.device), cb)
    advice = survey(pipe, seq, mask, [(0, k) for k in sample_frames(len(seq))], cb)
    print(f"suggested model: RAFTcorr-{advice.key}\n{advice.reason}")
    return 0


def _models(args) -> int:
    from ..models.zoo import MODELS, available_models, download_missing, import_weights, load_card, search_dirs

    failed = False
    for name in args.import_files or ():
        try:
            spec, dest = import_weights(name)
            print(f"{spec.name}: imported to {dest}")
        except (OSError, ValueError) as exc:
            print(f"{name}: {exc}")
            failed = True
    if args.download is not None:
        for key, outcome in download_missing(args.download or None).items():
            print(f"RAFTcorr-{key}: {outcome}")
            failed |= not outcome.startswith(("installed", "downloaded"))
        print()
    avail = available_models()
    for key, spec in MODELS.items():
        p = avail.get(key)
        card = load_card(p) if p else None
        extra = ""
        if card and card.spatial_resolution_px:
            extra = f", resolution ~{card.spatial_resolution_px:g} px"
        print(f"{spec.name:13s} s={spec.downsample_factor}  reach {spec.lookup_radius_px:4d} px{extra}  "
              f"{'installed: ' + str(p) if p else 'NOT INSTALLED'}")
        print(f"{'':13s} {spec.summary}")
    print("searched: " + "; ".join(str(d) for d in search_dirs()))
    return 3 if failed else 0


def _selftest(args) -> int:
    from ..callbacks import ConsoleCallbacks
    from ..selftest import run_selftest

    ok = True
    if args.gui:
        from ..selftest import gui_selftest

        ok, text = gui_selftest()
        print(text)
    if not args.no_models:
        rep = run_selftest(args.device, ConsoleCallbacks())
        print(rep.text)
        ok &= rep.ok
    return 0 if ok else 1


def _info(_args) -> int:
    print(f"raftcorr {__version__}, Python {platform.python_version()} on {platform.platform()}")
    try:
        import torch

        print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            from ..engine.memory import device_info

            print(device_info("cuda").describe())
    except ImportError:
        print("PyTorch is not installed")
    for mod in ("numpy", "scipy", "PySide6", "h5py", "tifffile", "PIL", "matplotlib"):
        try:
            m = __import__(mod)
            print(f"{mod} {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"{mod} not installed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
