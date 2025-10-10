# RAFT‑DIC GUI — User Manual

This manual explains how to install, configure, and use the 2D RAFT‑DIC GUI for displacement field analysis with a RAFT neural network.

## 1. Requirements

- Windows/macOS/Linux
- Python 3.8+
- NVIDIA GPU with CUDA 11.8+/12.x (no CPU mode)
- PyTorch with CUDA built

Verify GPU:
```bash
python verify_installation.py
```

## 2. Install and Launch

```bash
git clone https://github.com/zachtong/2D-RAFT-DIC-GUI.git
cd 2D-RAFT-DIC-GUI
pip install -e .
python main_GUI.py
```

Place the model checkpoint at `models/raft-dic_v1.pth`.

## 3. First Run — Basic Workflow

1. Select Input: choose a folder with images (`.tif/.tiff/.png/.jpg/.jpeg/.bmp`).
2. Select Output: choose a folder where results will be saved.
3. Draw ROI: use the ROI tools to draw a polygon and Confirm ROI.
4. Choose Mode: Accumulative (to first frame) or Incremental (to previous frame).
5. Advanced (optional): set tiling parameters — presets, Pixel Budget, Overlap Ratio, Show Tiles Overlay.
6. Run: click Run to process the sequence. The progress bar and counter update per frame.

## 4. ROI Metrics Panel

A text panel under the ROI canvas shows, in real time:
- ROI size and pixel count
- Expanded region (D) size and pixel count
- Tile size and number of tiles
- Budget check — single‑shot vs tiled

Tip: If expanded area ≤ Pixel Budget, processing runs in a single shot. Otherwise, tiles are used with feathered fusion.

## 5. Tiling Parameters

- Displacement Preset
  - Small (~100 px): sets `D_global=g_tile=100`
  - Large (~300 px): sets `D_global=g_tile=300`
  - Customized: you enter a value; both D and g are set to it

- Pixel Budget (`p_max_pixels`)
  - The maximum tile area (e.g., `1100*1100`, `1100x1100`, or integer)
  - If expanded area ≤ budget → single‑shot

- Overlap Ratio
  - Fraction of the tile interior used as overlap for blending (default 0.10)

- Show Tiles Overlay
  - Draws tiles (light blue) and valid interiors (orange) over the ROI preview

## 6. Smoothing & Visualization

- Use Smoothing: optional Gaussian smoothing of the fused displacement in ROI
  - Sigma range: 0.5–5
- Fixed Range: set fixed min/max for U and V colorbars
- Background options: show reference or deformed image underlay

## 7. Run Modes

- Accumulative: each frame compared to the first reference image
- Incremental: each frame compared to its previous frame

## 8. Outputs

Saved in the output folder:
- Displacement fields per frame (NumPy/MAT, depending on configuration)
- Optional figures/exports (when enabled)

## 9. Configuration Without Hardcoding

Create `assets/app_config.json` to customize app title, icons, and theme:

```json
{
  "app_title": "My DIC Pro",
  "appearance_mode": "dark",
  "color_theme": "blue",
  "icon_png": "assets/icons/app_icon.png",
  "icon_ico": "assets/icons/app_icon.ico"
}
```

Place your icons in `assets/icons/`. The GUI reads this file at startup.

## 10. Tips & Limits

- Choose Pixel Budget per your GPU VRAM; lower it for OOM safety
- For tiled runs, ensure `g_tile < min(tile)/2` to preserve interior
- Overlap Ratio around 0.10 is a good default for smooth blending

## 11. Troubleshooting

- CUDA not detected: install NVIDIA driver + CUDA Toolkit; confirm `nvidia-smi`
- PyTorch CPU build: reinstall with the correct CUDA wheel for your version
- “g_tile too large…”: reduce `g_tile` or increase Pixel Budget; not applicable in single‑shot
- Progress not updating: ensure at least 2 images are detected (extensions are matched case‑insensitively)

## 12. FAQ

- Single‑shot vs tiled?
  - If expanded ROI area ≤ Pixel Budget → single‑shot (no guard band)
  - Else → tiled with overlap and feathered fusion

- Can I force tiling even when single‑shot fits?
  - Not by default. Lower the Pixel Budget to force tiling if you need to debug fusion.

- How to change the app icon and name?
  - Use `assets/app_config.json` (see Section 9). No code edits required.

