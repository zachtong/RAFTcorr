# Changelog

## 2.0.0 (unreleased)

A rewrite. One Python process replaces the Flask server and the React front end.

### Models
- The four RAFTcorr-s1, -s2, -s4 and -s8 models, ported from the research implementation
  (pixelwise identical results).
- On-demand correlation for s <= 4 (memory linear in the image size), stored volume for s8.
- Checkpoints must carry their architecture (`model_cfg`); they are read with
  `weights_only=True` and verified key by key. The 1.x models, their sign flip and the shape
  heuristics are gone.
- Model suggestion from a quick s8 survey of the sequence.

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
  area change.
- Exports: HDF5, MAT v7.3 and NPZ written frame by frame; figures; GIF / MP4; CSV; project files.
- Command line (`raftcorr run | suggest | models | selftest | info`) and Python API
  (`raftcorr.track`, `raftcorr.compute_strain`).
