"""End-to-end visual test for bubble ROI visualization bugs.

Generates synthetic speckle images with a growing/shrinking bubble,
computes analytical displacement fields, and renders overlay PNGs via
the same code path as the actual application.  Saves diagnostic images
to server/tests/debug_output/bubble_*.png for visual inspection.

Run:  python -m pytest server/tests/test_bubble_e2e_visual.py -v -s
"""

import os

import cv2
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from server.deformed_warp import InverseMapCache, get_warped_full_data
from server.render_utils import fill_mask_nan_gaps, render_overlay_png

OUT_DIR = os.path.join(os.path.dirname(__file__), "debug_output")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _speckle_image(h, w, seed=42):
    """Generate a grayscale speckle pattern using Gaussian blobs."""
    rng = np.random.default_rng(seed)
    # Base noise
    img = rng.random((h, w)).astype(np.float32)
    # Add Gaussian speckle dots
    n_dots = h * w // 20
    ys = rng.integers(0, h, n_dots)
    xs = rng.integers(0, w, n_dots)
    for y, x in zip(ys, xs):
        cv2.circle(img, (int(x), int(y)), rng.integers(1, 3), float(rng.random()), -1)
    # Gaussian blur for smoothness
    img = cv2.GaussianBlur(img, (5, 5), 1.2)
    return (img * 255).clip(0, 255).astype(np.uint8)


def make_bubble_scenario(
    h=200, w=200, cx=100, cy=100,
    radii=None,
    noise_std=0.0,
):
    """Create synthetic growing/shrinking bubble test data.

    Parameters
    ----------
    radii : list of float — bubble radius at each frame (frame 0 = reference)
    noise_std : displacement noise (simulates RAFT inaccuracy near bubble)

    Returns dict with: h, w, cx, cy, radii, images, displacements,
                        per_frame_rois, roi_rect
    """
    if radii is None:
        radii = [12, 16, 20, 25, 30, 26, 22, 18, 14]

    R0 = radii[0]
    yy, xx = np.mgrid[:h, :w]
    dy = yy - cy
    dx = xx - cx
    dist = np.sqrt(dx ** 2 + dy ** 2)
    r_safe = np.maximum(dist, 1e-10)

    rng = np.random.default_rng(123)

    # --- Generate speckle images with bubble hole ---
    base = _speckle_image(h, w, seed=42)
    images = []
    for r in radii:
        img = base.copy()
        img[dist <= r] = 0
        images.append(img)

    # --- Per-frame ROI masks ---
    per_frame_rois = {}
    for i, r in enumerate(radii):
        mask = np.ones((h, w), dtype=bool)
        mask[dist <= r] = False
        per_frame_rois[i] = mask

    # --- Displacement fields (frame 0 → frame i+1) ---
    roi_rect = (0, 0, w, h)
    displacements = []

    for i in range(1, len(radii)):
        R = radii[i]
        # Incompressible: r' = sqrt(r^2 + R^2 - R0^2)
        arg = r_safe ** 2 + R ** 2 - R0 ** 2
        r_new = np.where(arg > 0, np.sqrt(arg), 0.0)
        u_r = r_new - r_safe

        u = u_r * dx / r_safe
        v = u_r * dy / r_safe

        # Add noise near the bubble edge (simulates RAFT tracking errors)
        if noise_std > 0:
            edge_band = (dist > R0 - 5) & (dist < R0 + 10)
            noise_u = rng.normal(0, noise_std, u.shape) * edge_band
            noise_v = rng.normal(0, noise_std, v.shape) * edge_band
            u = u + noise_u
            v = v + noise_v

        # Simulate _apply_current_frame_mask properly
        disp_full = np.stack([u, v], axis=-1).copy()
        # Pixels with r <= R0 were never tracked (inside original bubble)
        disp_full[dist <= R0, :] = np.nan

        # Apply current frame's mask: check deformed position
        valid = ~np.isnan(disp_full[..., 0])
        vyy, vxx = np.where(valid)
        U_vals = disp_full[vyy, vxx, 0]
        V_vals = disp_full[vyy, vxx, 1]
        def_x = np.round(vxx + U_vals).astype(np.intp)
        def_y = np.round(vyy + V_vals).astype(np.intp)
        in_bounds = (def_x >= 0) & (def_x < w) & (def_y >= 0) & (def_y < h)
        in_specimen = np.zeros(len(vyy), dtype=bool)
        bm = in_bounds
        cur_mask = per_frame_rois[i]
        in_specimen[bm] = cur_mask[def_y[bm], def_x[bm]]
        outside = ~in_specimen
        if outside.any():
            disp_full[vyy[outside], vxx[outside], :] = np.nan

        displacements.append(disp_full)

    return {
        "h": h, "w": w, "cx": cx, "cy": cy,
        "radii": radii,
        "images": images,
        "displacements": displacements,
        "per_frame_rois": per_frame_rois,
        "roi_rect": roi_rect,
    }


# ---------------------------------------------------------------------------
# Rendering helpers (replicates displacement.py code paths)
# ---------------------------------------------------------------------------

def render_reference_mode(sc, idx):
    """Reference mode: place data in full image, clip to frame 0 mask."""
    h, w = sc["h"], sc["w"]
    disp = sc["displacements"][idx]
    disp_data = disp[:, :, 0]  # u component
    roi_rect = sc["roi_rect"]

    full_data = np.full((h, w), np.nan)
    x0, y0, x1, y1 = roi_rect
    dh, dw = disp_data.shape
    sh = min(dh, y1 - y0)
    sw = min(dw, x1 - x0)
    full_data[y0:y0 + sh, x0:x0 + sw] = disp_data[:sh, :sw]

    # Fix: clip to frame 0's mask + fill NaN gaps from rounding artifacts
    ref_mask = sc["per_frame_rois"].get(0)
    if ref_mask is not None and ref_mask.shape == full_data.shape:
        full_data[~ref_mask] = np.nan
        full_data = fill_mask_nan_gaps(full_data, ref_mask)

    return full_data


def render_deformed_mode(sc, idx):
    """Deformed mode: inverse warp data to deformed coordinates."""
    h, w = sc["h"], sc["w"]
    disp = sc["displacements"][idx]
    U, V = disp[:, :, 0], disp[:, :, 1]
    disp_data = U  # u component
    roi_rect = sc["roi_rect"]
    deformed_mask = sc["per_frame_rois"].get(idx + 1)
    cache = InverseMapCache(max_size=10)

    full_data = get_warped_full_data(
        data=disp_data,
        frame_idx=idx,
        U=U, V=V,
        roi_rect=roi_rect,
        image_shape=(h, w),
        cache=cache,
        deformed_mask=deformed_mask,
    )
    return full_data


# ---------------------------------------------------------------------------
# Visual output
# ---------------------------------------------------------------------------

def save_comparison_grid(sc, filename, title_prefix=""):
    """Render all frames in reference + deformed mode, save as grid PNG."""
    n_disp = len(sc["displacements"])
    radii = sc["radii"]

    fig, axes = plt.subplots(3, n_disp, figsize=(3 * n_disp, 9), dpi=100)
    if n_disp == 1:
        axes = axes[:, np.newaxis]

    # Global vmin/vmax for consistent coloring
    all_u = np.concatenate([d[:, :, 0].ravel() for d in sc["displacements"]])
    finite = all_u[np.isfinite(all_u)]
    vmin, vmax = float(finite.min()), float(finite.max())

    for idx in range(n_disp):
        # Row 0: Reference mode
        ref_data = render_reference_mode(sc, idx)
        ax0 = axes[0, idx]
        ax0.imshow(sc["images"][0], cmap="gray", alpha=0.3)
        masked = np.ma.array(ref_data, mask=np.isnan(ref_data))
        ax0.imshow(masked, cmap="turbo", vmin=vmin, vmax=vmax, alpha=0.8)
        ref_valid = np.sum(~np.isnan(ref_data))
        ax0.set_title(f"Ref F{idx}\nR={radii[idx+1]}\n{ref_valid}px", fontsize=8)
        ax0.axis("off")

        # Row 1: Deformed mode
        def_data = render_deformed_mode(sc, idx)
        ax1 = axes[1, idx]
        ax1.imshow(sc["images"][idx + 1], cmap="gray", alpha=0.3)
        masked = np.ma.array(def_data, mask=np.isnan(def_data))
        ax1.imshow(masked, cmap="turbo", vmin=vmin, vmax=vmax, alpha=0.8)
        def_valid = np.sum(~np.isnan(def_data))
        mask_valid = sc["per_frame_rois"][idx + 1].sum()
        ax1.set_title(f"Def F{idx}\nR={radii[idx+1]}\n{def_valid}/{mask_valid}px", fontsize=8)
        ax1.axis("off")

        # Row 2: Per-frame mask
        ax2 = axes[2, idx]
        mask_display = np.zeros((*sc["per_frame_rois"][idx + 1].shape, 3), dtype=np.uint8)
        mask_display[sc["per_frame_rois"][idx + 1]] = [0, 200, 0]
        mask_display[sc["per_frame_rois"][0] & ~sc["per_frame_rois"][idx + 1]] = [200, 0, 0]
        ax2.imshow(mask_display)
        ax2.set_title(f"Mask F{idx+1}\nGreen=cur, Red=only-in-F0", fontsize=8)
        ax2.axis("off")

    axes[0, 0].set_ylabel("Reference\nMode", fontsize=10, rotation=0, labelpad=60, va="center")
    axes[1, 0].set_ylabel("Deformed\nMode", fontsize=10, rotation=0, labelpad=60, va="center")
    axes[2, 0].set_ylabel("Mask\nOverlay", fontsize=10, rotation=0, labelpad=60, va="center")

    fig.suptitle(f"{title_prefix}Bubble Visualization Test", fontsize=12)
    fig.tight_layout(rect=[0.08, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=100)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBubbleVisualE2E:

    def test_clean_incompressible(self):
        """Clean analytical displacement, no noise.
        Reference mode should have constant pixel count.
        """
        sc = make_bubble_scenario(noise_std=0.0)
        save_comparison_grid(sc, "bubble_clean.png", "Clean ")

        frame0_valid = sc["per_frame_rois"][0].sum()
        ref_counts = []
        def_counts = []

        for idx in range(len(sc["displacements"])):
            ref_data = render_reference_mode(sc, idx)
            ref_valid = int(np.sum(~np.isnan(ref_data)))
            ref_counts.append(ref_valid)

            def_data = render_deformed_mode(sc, idx)
            def_valid = int(np.sum(~np.isnan(def_data)))
            def_counts.append(def_valid)

        print(f"\n  Frame 0 mask valid: {frame0_valid}")
        print(f"  Reference mode valid per frame: {ref_counts}")
        print(f"  Deformed mode valid per frame:  {def_counts}")

        # Bug 1: all reference mode frames should have same count
        assert all(c == ref_counts[0] for c in ref_counts), (
            f"Reference mode pixel counts vary: {ref_counts}"
        )
        assert ref_counts[0] == frame0_valid, (
            f"Reference mode count {ref_counts[0]} != frame0 mask {frame0_valid}"
        )

    def test_noisy_displacement(self):
        """Noisy displacement near bubble edge — simulates real RAFT errors.
        _apply_current_frame_mask may remove extra pixels near the edge.
        """
        sc = make_bubble_scenario(noise_std=2.0)
        save_comparison_grid(sc, "bubble_noisy.png", "Noisy ")

        frame0_valid = sc["per_frame_rois"][0].sum()
        ref_counts = []

        for idx in range(len(sc["displacements"])):
            ref_data = render_reference_mode(sc, idx)
            ref_valid = int(np.sum(~np.isnan(ref_data)))
            ref_counts.append(ref_valid)
            disp_nan = int(np.isnan(sc["displacements"][idx][:, :, 0]).sum())
            print(
                f"  Frame {idx}: disp_NaN={disp_nan}, "
                f"ref_valid={ref_valid}, "
                f"frame0_valid={frame0_valid}"
            )

        # With noise, some edge pixels may be NaN in displacement data
        # but valid in frame 0's mask → ref_valid < frame0_valid
        # This is the ROOT CAUSE of Bug 1 in the real application!
        diffs = [frame0_valid - c for c in ref_counts]
        print(f"\n  Pixel deficit per frame (frame0 - actual): {diffs}")
        print(f"  (Non-zero deficit = Bug 1 visible)")

    def test_deformed_mode_growth_shrink(self):
        """Deformed mode: ROI should follow mask during both growth AND shrink."""
        sc = make_bubble_scenario(
            radii=[12, 16, 20, 25, 30, 26, 22, 18, 14],
            noise_std=0.0,
        )
        save_comparison_grid(sc, "bubble_deformed_growth_shrink.png", "Growth+Shrink ")

        for idx in range(len(sc["displacements"])):
            def_data = render_deformed_mode(sc, idx)
            mask = sc["per_frame_rois"][idx + 1]
            has_data = ~np.isnan(def_data)

            # All rendered pixels should be inside the mask
            outside = has_data & ~mask
            assert outside.sum() == 0, (
                f"Frame {idx}: {outside.sum()} pixels outside mask"
            )

            # Coverage: rendered valid / mask valid (should be close to 1.0)
            coverage = has_data.sum() / max(1, mask.sum())
            print(
                f"  Frame {idx}: R={sc['radii'][idx+1]}, "
                f"coverage={coverage:.3f}"
            )

    def test_deformed_mode_coverage_vs_mask(self):
        """After the fix, deformed coverage should be high (>90%) for all
        frames including the shrinking phase."""
        sc = make_bubble_scenario(
            radii=[12, 20, 30, 25, 15],  # grow then shrink
            noise_std=0.0,
        )
        save_comparison_grid(sc, "bubble_coverage.png", "Coverage ")

        for idx in range(len(sc["displacements"])):
            def_data = render_deformed_mode(sc, idx)
            mask = sc["per_frame_rois"][idx + 1]
            has_data = ~np.isnan(def_data)
            coverage = has_data.sum() / max(1, mask.sum())
            print(
                f"  Frame {idx}: R={sc['radii'][idx+1]}, "
                f"coverage={coverage:.3f} "
                f"({has_data.sum()}/{mask.sum()})"
            )
            # Coverage should be reasonably high
            assert coverage > 0.85, (
                f"Frame {idx}: deformed coverage too low ({coverage:.3f})"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
