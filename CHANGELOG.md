# Changelog

## Unreleased

### Models for a new user
- **Help > Models…** replaces the read-only list: each model installed or not with its size,
  Download and Download All (cancellable), Import Model Files… for files downloaded by hand on
  another computer (identified by their SHA-256, stored under their release name; any other file
  is refused), Open Model Folder, and the link to the release.
- A run or a model suggestion whose model can be neither found nor downloaded opens that window
  with the reason, instead of an "Analysis failed" box with the raw network error.
- The first-use download shows "Downloading model RAFTcorr-s4: 5.2 of 20.7 MB" in the progress
  area, and Stop cancels it. The progress bar then starts again from 0 for the tracking: it used
  to stay at 100 % for the whole run after a download.
- `raftcorr models --import FILE ...` on the command line.

### Distribution
- pip installs PySide6-Essentials instead of the full PySide6: every Qt module RAFTcorr uses is
  in it, and the WebEngine, Pdf and Addons packages (about 175 MB on Windows) are no longer
  downloaded. A test keeps the imports inside Essentials.

## 2.0.0 (2026-11-18)

A rewrite. One Python process replaces the Flask server and the React front end.

### Models
- The four RAFTcorr-s1, -s2, -s4 and -s8 models, ported from the research implementation
  (pixelwise identical results).
- On-demand correlation for s <= 4 (memory linear in the image size), stored volume for s8.
- Checkpoints must carry their architecture (`model_cfg`); they are read with
  `weights_only=True` and verified key by key. The 1.x models, their sign flip and the shape
  heuristics are gone.
- Model suggestion from a quick s8 survey of the sequence.
- The model files are assets of the GitHub release `weights-v1`: a model is downloaded and
  verified the first time it is used, or with `raftcorr models --download` (run by `install.bat`).

### Engine
- 20 iterations and raw displacements by default (as in the paper).
- Work areas grow by the model's reach so that displaced points stay inside the window; large
  regions are tiled with halos and feathered seams; a calibrated per-model memory model sizes the
  tiles; out-of-memory errors shrink them and retry.
- Accumulative, incremental, every-N and custom reference schedules; incremental results are
  composed into frame-0 coordinates; ROIs of later reference frames are carried forward by an
  inverse map that is robust to large rotations.
- Frame `k` of a result is image `k` (1.x shifted indices by one).
- Images: luminance for colour, effective-bit-depth scaling shared by the whole sequence, no
  8-bit re-quantisation.

### Application
- Native desktop application (PySide6) whose interface is ported from pyALDIC (same layout, light and
  dark themes, seven interface languages): images with drag and drop, workflow (tracking mode and
  model, with a model suggestion), regions of interest with per-frame masks, run with progress,
  pause and cancel, displacement on the reference or deformed frame, strain window, analysis with
  probes, export dialog, projects (`.raftcorr`).
- Intensity residual (tracking-quality map) for every frame, and a warning when the residual jumps.
- One definition of velocity everywhere.
- Strain: Green-Lagrange, infinitesimal or Euler-Almansi; uniform or Gaussian gauges; plane or
  quadratic fits; crack-aware; principal strains and angle, von Mises equivalent strain, rotation,
  area change. As in pyALDIC, points whose gauge an edge (region of interest, hole, crack) cuts
  within 0.7 of its half-width are left empty ("Trim low-confidence edges", 0 to 1); pictures can
  fill that band from the reliable points, data files keep it empty.
- Exports: HDF5, MAT v7.3 and NPZ written frame by frame (NPZ also one file per frame); figures;
  GIF / MP4; CSV; project files. Pictures follow the windows (hidden background white, black or
  transparent; hide untracked regions; fill trimmed edges), with pyALDIC's colorbar style (position,
  font, thickness, background, margin), a resolution cap and the JPEG quality.
- Command line (`raftcorr run | suggest | models | selftest | info`) and Python API
  (`raftcorr.track`, `raftcorr.compute_strain`).

### Distribution
- `pip install raftcorr` brings the whole application (the GUI, HDF5 and video libraries are no
  longer extras; `raftcorr[all]` still works). The `raftcorr` command starts the application, and
  its subcommands are the command line.
- Windows installer (`RAFTcorr-Windows-Setup.exe`, per user, no administrator rights): installs
  the PyTorch build that matches the NVIDIA driver (CUDA 12.8, 12.6, 11.8 or the CPU build), RAFTcorr
  and the model files; Start menu entries for RAFTcorr and "Repair RAFTcorr"; optional `raftcorr`
  on PATH; `.raftcorr` files open in RAFTcorr.
- macOS disk image (`RAFTcorr-macOS.dmg`, Apple silicon, macOS 14 or later), self-contained.
- `raftcorr selftest --gui` also opens the main window offscreen, to check an installation.
- `install.bat` reads the CUDA version of newer NVIDIA drivers ("CUDA UMD Version"), which it took
  for no GPU before.
