# RAFTcorr

An interactive desktop GUI for Digital Image Correlation powered by a RAFT neural network (CUDA‑accelerated).

![Demo](assets/RAFT_DIC_demo.gif)

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA 11.8+/12.x (no CPU mode)
- PyTorch with CUDA build

## Install

```bash
git clone https://github.com/zachtong/RAFTcorr.git
cd RAFTcorr
pip install -e .
```

Verify your environment:
```bash
python verify_installation.py
```

## Run

```bash
python main_GUI.py
```

## Quick Usage

1. Select input folder (images) and output folder.
2. Draw and confirm ROI on the first image.
3. Choose processing mode (Accumulative / Incremental).
4. Optionally adjust Advanced parameters (Safety Factor, Overlap, Smoothing).
5. Click Run. The progress bar and counter update as frames process.

For a complete walkthrough and parameter explanations, see `USER_MANUAL.md`.

## Configure (No Hardcoding)

Customize branding, theme, and icons via `assets/app_config.json` (created in this repo):

```json
{
  "app_title": "RAFTcorr",
  "appearance_mode": "system",
  "color_theme": "blue",
  "icon_png": "assets/icons/app_icon.png",
  "icon_ico": "assets/icons/app_icon.ico"
}
```

The app applies these at startup (title, CTk appearance/theme, window icon).

Scaffolded assets:
- Icons folder: `assets/icons/` (auto‑generates a default icon if missing). Replace `app_icon.png` / `app_icon.ico` with your own.
- Sample theme: `assets/themes/my_theme.json`. To use, set `"color_theme": "assets/themes/my_theme.json"` in `assets/app_config.json`.

## Repository Layout

- `main_GUI.py` — GUI app: ROI tools, parameters, progress, visualization.
- `raft_dic_gui/processing.py` — ROI tiling, fusion, smoothing.
- `raft_dic_gui/model.py` — RAFT model loader + inference.
- `raft_dic_gui/preview.py` — tiling overlay helpers.
- `assets/` — icons, themes, `app_config.json`.
- `models/` — model checkpoint(s).

## Modify / Extend

- Presets (D_global and g_tile): `main_GUI.py:on_disp_preset_change`.
- Pixel budget / single‑shot vs tiled: `raft_dic_gui/processing._choose_tile_size`.
- Tiled fusion and guard band: `raft_dic_gui/processing.dic_over_roi_with_tiling`.
- ROI metrics panel and overlay: `main_GUI.py:update_roi_metrics_text`, `draw_tiles_overlay`.
- Progress UI: `main_GUI.py:process_images` (per‑frame updates with `update_idletasks`).

## Citation

If this software assists your work, please cite RAFT and this repository. Example:

```bibtex
TBD.
```

## License

MIT — see `LICENSE.md`.

## Acknowledgments

- RAFT: https://github.com/princeton-vl/RAFT
