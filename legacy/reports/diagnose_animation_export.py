"""End-to-end check: animation export follows per-frame ROI in deformed mode.

Produces two GIFs in reports/diagnose_output/:
  anim_reference.gif  — reference background, ROI stays at frame 0 position
  anim_deformed.gif   — deformed background, ROI should shrink/grow with
                        per-frame mask

Also asserts that the rendered GIF's ROI footprint correlates with the
per-frame mask by counting coloured (non-background) pixels.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from typing import List

import cv2
import numpy as np
from PIL import Image as PILImage

from reports.diagnose_large_bubble import make_scenario
from raft_dic_gui.export_animation import export_animation
from server.deformed_warp import InverseMapCache


OUT_DIR = os.path.join(os.path.dirname(__file__), "diagnose_output")
os.makedirs(OUT_DIR, exist_ok=True)


def _write_images(sc, tmp_dir: str) -> List[str]:
    """Dump the synthetic images to disk so image_loader can read them."""
    os.makedirs(tmp_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(sc.images):
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        p = os.path.join(tmp_dir, f"frame_{i:03d}.png")
        cv2.imwrite(p, rgb)
        paths.append(p)
    return paths


def _image_loader_factory(image_paths):
    def loader(idx):
        if 0 <= idx < len(image_paths):
            return cv2.imread(image_paths[idx])
        return None
    return loader


def _count_coloured_pixels(rgb: np.ndarray) -> np.ndarray:
    """Heuristic: a pixel is coloured (overlay active) when it deviates from
    the underlying greyscale background.  Returns a boolean mask.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    chroma = (r.astype(int) - g.astype(int)).__abs__() + \
             (g.astype(int) - b.astype(int)).__abs__() + \
             (r.astype(int) - b.astype(int)).__abs__()
    return chroma > 30


def run():
    radii = [30, 50, 70, 80, 70, 50, 30, 20, 15]
    sc = make_scenario(radii=radii)

    tmp_dir = os.path.join(OUT_DIR, "_anim_frames")
    image_paths = _write_images(sc, tmp_dir)
    loader = _image_loader_factory(image_paths)

    # Mirror the session's per_frame_rois keys:
    # 0 -> reference frame mask, 1..N -> frame N mask.
    per_frame_rois = {i: sc.per_frame_rois[i] for i in range(len(radii))}

    # Compute global range for colour consistency
    all_u = np.concatenate([d[:, :, 0].ravel() for d in sc.displacements])
    finite = all_u[np.isfinite(all_u)]
    vmin, vmax = float(finite.min()), float(finite.max())

    shared_settings = dict(
        colormap="turbo", alpha=1.0, vmin=vmin, vmax=vmax,
        log_scale=False, fps=1.0,
    )

    cache = InverseMapCache(max_size=10)
    for bg_mode, gif_name in [("reference", "anim_reference.gif"),
                              ("deformed", "anim_deformed.gif")]:
        out_path = os.path.join(OUT_DIR, gif_name)
        settings = {**shared_settings, "background": bg_mode}
        export_animation(
            output_path=out_path,
            fmt="gif",
            component="u",
            frame_range=(0, len(sc.displacements) - 1),
            displacement_results=sc.displacements,
            strain_results=[None] * len(sc.displacements),
            roi_rect=sc.roi_rect,
            roi_mask=sc.per_frame_rois[0],
            image_loader=loader,
            settings=settings,
            fps=2,
            loop=True,
            timestamp_overlay=True,
            include_colorbar=False,
            resize_factor=1.0,
            per_frame_rois=per_frame_rois,
            inverse_map_cache=cache,
        )
        print(f"Wrote {out_path}")

    # Analyse each frame: does overlay footprint correlate with per-frame mask?
    print("\nFrame-by-frame overlay-vs-mask correlation:")
    print(f"{'Frame':>5} {'R':>4} {'MaskPx':>8} {'OverlayPx (def)':>17} {'ratio':>7}")
    gif = PILImage.open(os.path.join(OUT_DIR, "anim_deformed.gif"))
    for i in range(gif.n_frames):
        gif.seek(i)
        arr = np.array(gif.convert("RGB"))
        overlay = _count_coloured_pixels(arr)
        mask = per_frame_rois[i + 1]
        # Overlay may be scaled vs full image — scale down to match aspect
        # only if sizes differ.  For 1.0 resize_factor they are identical.
        if overlay.shape != mask.shape:
            overlay = cv2.resize(
                overlay.astype(np.uint8), (mask.shape[1], mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        mask_px = int(mask.sum())
        overlay_px = int(overlay.sum())
        ratio = overlay_px / max(1, mask_px)
        print(f"{i:>5d} {radii[i+1]:>4.0f} {mask_px:>8d} {overlay_px:>17d} {ratio:>7.3f}")


if __name__ == "__main__":
    run()
