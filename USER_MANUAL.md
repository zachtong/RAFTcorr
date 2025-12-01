# RAFTcorr — User Manual

This manual explains how to install, configure, and use RAFTcorr for displacement field analysis with a RAFT neural network.

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
git clone https://github.com/zachtong/RAFTcorr.git
cd RAFTcorr
pip install -e .
python main_GUI.py
```

## 3. First Run — Basic Workflow

1. Select Input: choose a folder with images (`.tif/.tiff/.png/.jpg/.jpeg/.bmp`).
2. Select Output: choose a folder where results will be saved.
3. Draw ROI: use the ROI tools to draw a polygon and Confirm ROI.
4. Choose Mode: Accumulative (to first frame) or Incremental (to previous frame).
5. Advanced (optional): set tiling parameters — Safety Factor, Overlap Ratio.
6. Run: click Run to process the sequence. The progress bar and counter update per frame.

## 4. ROI Metrics Panel

A text panel under the ROI canvas shows, in real time:
- ROI size and pixel count
- Expanded region (D) size and pixel count
- Tile size and number of tiles
- ROI size and pixel count
- Expanded region (D) size and pixel count
- Tile size and number of tiles
- Processing Mode: Single-shot vs Tiled (based on Safety Factor)

Tip: If expanded area ≤ Safe Pixel Limit (calculated from Safety Factor), processing runs in a single shot. Otherwise, tiles are used with feathered fusion.

## 5. Tiling Parameters

- **Safety Factor** (0.2 - 1.0)
  - Controls how aggressively VRAM is used.
  - **0.2 - 0.55**: Conservative. Good for stability.
  - **0.7 - 1.0**: Aggressive. Uses more VRAM, fewer tiles, faster speed.
  - **Note**: If VRAM is insufficient even at 1.0, the system automatically enables tiling.

- **Overlap Ratio**
  - Fraction of the tile interior used as overlap for blending (default 0.10).
  - Ensures smooth transitions between tiles.

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

- **Safety Factor**: Start with 0.55. If you encounter OOM errors, lower it to 0.4 or 0.2.
- **Tiling**: Tiling is automatic based on your Safety Factor and ROI size. You don't need to manually calculate tile sizes.
- Overlap Ratio around 0.10 is a good default for smooth blending

## 11. Troubleshooting

- CUDA not detected: install NVIDIA driver + CUDA Toolkit; confirm `nvidia-smi`
- PyTorch CPU build: reinstall with the correct CUDA wheel for your version
- “g_tile too large…”: reduce `g_tile` or increase Pixel Budget; not applicable in single‑shot
- Progress not updating: ensure at least 2 images are detected (extensions are matched case‑insensitively)

## 12. FAQ

- **Single‑shot vs tiled?**
  - If expanded ROI area ≤ Safe Pixel Limit → single‑shot (no guard band).
  - Else → tiled with overlap and feathered fusion.

- **Can I force tiling even when single‑shot fits?**
  - Yes, by lowering the **Safety Factor**. This reduces the "Safe Pixel Limit", forcing the system to split the image into tiles.

- How to change the app icon and name?
  - Use `assets/app_config.json` (see Section 9). No code edits required.

