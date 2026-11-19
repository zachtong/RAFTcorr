# Changelog

## 2.1.0 (2026-11-19)

### New
- Drag with the left button to pan the image, in the main window and in the strain window, as the
  Analysis tab already did. The middle button still pans at any time, and the drawing tools keep the
  left button for their shapes.

### Documentation
- A User Guide (PDF, LaTeX sources in `docs/user-guide/`) that walks through every panel, window
  and option with screenshots, plus the command line, the Python library and troubleshooting; and
  a two-page Quick Guide.
- A one-minute quick-start video in English and Chinese, and its GIF preview in the README.
- The README opens with a banner, badges and download buttons, and shows the main window, the
  strain window and the Analysis tab.
- `scripts/media/` makes every picture, the banner and the videos by driving the real application
  off-screen, so they are regenerated for each version.
- The README no longer mentions a brush or undo for the region of interest, which the application
  does not have.

### Fixes
- Charts in Chinese, Japanese and Korean (Analysis tab) showed boxes instead of the axis labels on
  Windows: matplotlib now gets the installed fonts as a fallback list.

Found while writing the User Guide:

**Region of interest, canvas, images**
- Invert and Save used the mask last opened for editing. After Batch Import, an import in the image list or
  Open Project, Invert could give a frame the whole image and Save could write another frame's mask. Both
  now work on the current frame's own mask.
- Batch Import lists the masks in natural order (mask2 before mask10). Auto-Match by Name pairs a mask with
  the image whose file name ends in the same number (mask6 with image6) instead of reading the number as a
  frame index counted from 0.
- The image list's mask import pairs the selected frames with the files in the natural order of their
  names, not in the file dialog's order.
- An empty region of interest on frame 1, e.g. an all-black mask from Batch Import, is refused before the
  run starts. It used to fail inside the run.
- Esc cancels a shape picked in the Add / Cut menu even before drawing starts, and Enter closes a polygon.
- Changing Natural Sort asks first when it would reorder the frames and discard regions of interest,
  results or probes. It used to discard them silently.
- Save proposes a file next to the image folder rather than inside it, so the next load does not take the
  mask for an image.
- An unreadable mask file is reported in the LOG instead of being ignored.
- The tooltips of Clear and of Batch Import's Assign Sequential were corrected.

**Workflow, projects, menus**
- After Open Project, the ADVANCED controls and the Interval box show the project's values. They kept
  their previous values, although the run used the project's.
- Custom reference frames are checked against frames 1 to N-1. Invalid input is marked and explained in
  the LOG, and a run is refused while the field holds it; invalid frames never reach the run.- Suggest a Model follows the tracking mode: in incremental tracking it measures the steps between each
  frame and its reference, not the motion from frame 1. Its progress counts frames from 1.
- The model list's tooltips are refreshed after a download, without a restart.
- The Language menu checks exactly one language.
- Export Results (Ctrl+E) without results says so in the LOG.
- Cancel while a project opens stops loading.
- The Tracking Mode tooltip said incremental tracking uses a later reference frame. It uses an earlier one.

**Run, LOG, strain window**
- Every warning is amber in the LOG. Warnings from the engine used to show in the plain colour.
- The out-of-memory message names the GPU Memory control in ADVANCED. A memory budget too small for the
  model's smallest window gets its own advice.
- A new run no longer shows the previous run's "Frame N/N".
- The strain window opens only after a run that produced results. It also opened after a failed run.
- With "Hide untracked regions" off, the main window, the strain window and exported pictures fill each
  part of the region of interest from itself, never across a crack, and leave the edge-trimmed band empty
  unless "Fill trimmed edges" is on.
- The frame-rate list under the canvas shows "2 fps" in full.
- Tooltips corrected: Hide untracked regions applies to the Reference frame view only; Output spacing does
  not make strain faster; the strain window's Export Results lists every format.

**Command line, library, export, installation**
- `raftcorr run --strain --project` stores the strain settings that were used (`--gauge`, `--measure`,
  `--edge-trim`) instead of the defaults.
- Frame numbers printed by the command line count from 1, as in the application: the `suggest` report and
  the strain progress. `--keys` stays 0-based and its help says so.
- Bad input on the command line prints one line on stderr and exits with code 2 instead of a Python
  traceback. Internal errors keep their traceback.
- Colour bars of exported pictures and report thumbnails use the names shown on screen (U, εxx, max shear)
  instead of internal names (`u`, `exx`, `max_shear`), and so does the export preview.
- The image-size error counts frames from 1 and names the files.
- A rotating log file is written for every installation, not only the frozen build:
  `RAFTcorr.log` in the `logs` folder of the user data folder.
- `install.bat` and Repair of the Windows installation replace an installed PyTorch of the wrong build,
  e.g. the CPU build after a driver update, and keep the right one without downloading it again.
- The `callbacks` docstring now describes stopping correctly: `Pipeline.run` returns the frames computed so
  far, and the other operations raise `StopRequested`.
- Removed an unused velocity-settings widget.

## 2.0.1 (2026-11-18)

### Models for a new user
- **Help > Models…** replaces the read-only list: each model installed or not with its size,
  Download and Download All (cancellable), Import Model Files… for files downloaded by hand on
  another computer (identified by their SHA-256, stored under their release name; any other file
  is refused), Open Model Folder, and the link to the release.
- A run or a model suggestion whose model can be neither found nor downloaded opens that window
  with the reason, instead of an "Analysis failed" box with the raw network error.
- The first-use download shows "Downloading model RAFTcorr-s4: 5.2 of 20.7 MB" in the progress
  area, and Cancel stops it. The progress bar then starts again from 0 for the tracking: it used
  to stay at 100 % for the whole run after a download.
- `raftcorr models --import FILE ...` on the command line.

### Fixes
- Cancel during "Suggest a Model" no longer shows a "StopRequested" error box.
- A download says when the model folder cannot be written instead of blaming the network, treats
  a dropped connection as a network failure, and writes its own temporary file, so a run and the
  Models window can fetch the same model at once. Checked model files are not hashed again until
  they change.

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
  cancel, displacement on the reference or deformed frame, strain window, analysis with
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
