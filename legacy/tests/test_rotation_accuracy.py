"""Automated accuracy test for incremental DIC on the rotation dataset.

Runs the DIC pipeline directly (no server) in incremental mode with
every-frame reference update, then compares against analytical ground truth.

Usage:
    python tests/test_rotation_accuracy.py [--device cuda|cpu] [--mask-dir auto]

Reports per-frame RMS error and generates a summary plot.
"""

import os
import sys
import argparse
import time
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "test_data", "rotation_90deg")
IMG_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")
GT_PATH = os.path.join(DATA_DIR, "ground_truth.npz")
RESULTS_DIR = os.path.join(DATA_DIR, "results")


def load_ground_truth():
    """Load ground truth displacement fields."""
    gt = np.load(GT_PATH)
    angles = gt["angles_deg"]
    center = gt["center"]
    mask_radius = int(gt["mask_radius"])

    displacements = {}
    for key in gt.files:
        if key.startswith("disp_"):
            frame_num = int(key.split("_")[1])
            displacements[frame_num] = gt[key]

    return angles, center, mask_radius, displacements


def run_dic(device: str, use_mask_dir: bool):
    """Run DIC processing using the controller directly."""
    from raft_dic_gui.config import DICConfig
    from raft_dic_gui.controller import DICProcessor
    import raft_dic_gui.processing as proc

    # Find model checkpoint
    model_path = None
    for candidate in [
        os.path.join(PROJECT_ROOT, "models", "active", "RAFTcorr_fine_parameter_more.pth"),
        os.path.join(PROJECT_ROOT, "models", "active", "RAFTcorr_fine_parameter_fewer.pth"),
        os.path.join(PROJECT_ROOT, "models", "RAFTcorr_fine_parameter_more.pth"),
        os.path.join(PROJECT_ROOT, "models", "RAFTcorr_fine_parameter_fewer.pth"),
    ]:
        if os.path.exists(candidate):
            model_path = candidate
            break

    if model_path is None:
        print("ERROR: No model checkpoint found. Searched:")
        print("  models/RAFTcorr_fine_parameter_more.pth")
        print("  models/RAFTcorr_fine_parameter_fewer.pth")
        sys.exit(1)

    print(f"Using model: {os.path.basename(model_path)}")
    print(f"Device: {device}")
    print(f"Mask source: {'folder' if use_mask_dir else 'auto-warp'}")

    # Load reference image to get ROI
    ref_path = os.path.join(IMG_DIR, "frame_0001.tif")
    ref_img = proc.load_and_convert_image(ref_path)
    H, W = ref_img.shape[:2]

    # Load frame 1 mask as ROI
    from PIL import Image
    mask_img = np.array(Image.open(os.path.join(MASK_DIR, "frame_0001.tif")))
    roi_mask = mask_img > 127

    # Compute bounding box
    y_idx, x_idx = np.where(roi_mask)
    roi_rect = (int(x_idx.min()), int(y_idx.min()),
                int(x_idx.max()) + 1, int(y_idx.max()) + 1)

    print(f"ROI rect: {roi_rect}")
    print(f"ROI area: {int(roi_mask.sum())} px")

    # Configure
    config = DICConfig(
        img_dir=IMG_DIR,
        project_root=RESULTS_DIR,
        model_path=model_path,
        mode="incremental",
        key_frame_interval=1,  # every frame update
        device=device,
        mask_dir=MASK_DIR if use_mask_dir else None,
        context_padding=64,
        tile_overlap=64,
        use_smooth=True,
        sigma=2.0,
    )

    def progress(pct, cur, total):
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "#" * filled + "." * (bar_len - filled)
        print(f"\r  [{bar}] {pct:5.1f}% ({cur}/{total})", end="", flush=True)

    processor = DICProcessor(
        update_progress_callback=progress,
    )

    print("\nRunning DIC...")
    t0 = time.time()
    results = processor.run(config, roi_mask, roi_rect)
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({len(results)} frames)")

    return results, roi_rect, roi_mask


def analyze_results(results, roi_rect, roi_mask, gt_displacements):
    """Compare DIC results against ground truth."""
    xmin, ymin, xmax, ymax = roi_rect

    print("\n" + "=" * 70)
    print("ACCURACY ANALYSIS")
    print("=" * 70)
    print(f"{'Frame':>6} {'Angle':>7} {'RMS_u':>8} {'RMS_v':>8} {'RMS_mag':>8} "
          f"{'MaxErr':>8} {'Valid%':>7}")
    print("-" * 70)

    rms_u_list = []
    rms_v_list = []
    rms_mag_list = []
    max_err_list = []
    valid_pct_list = []
    angles = []

    for i, disp in enumerate(results):
        frame_num = i + 2  # results[0] corresponds to frame 2

        if frame_num not in gt_displacements:
            continue

        gt_full = gt_displacements[frame_num]
        gt_crop = gt_full[ymin:ymax, xmin:xmax, :]

        # Valid pixels: both DIC result and GT are not NaN
        valid = ~np.isnan(disp[..., 0]) & ~np.isnan(gt_crop[..., 0])
        valid_count = int(np.sum(valid))
        total_roi = int(np.sum(~np.isnan(gt_crop[..., 0])))
        valid_pct = 100.0 * valid_count / max(1, total_roi)

        if valid_count == 0:
            print(f"{frame_num:6d} {(frame_num-1)*3:6.1f}°  — no valid pixels —")
            continue

        err_u = disp[valid, 0] - gt_crop[valid, 0]
        err_v = disp[valid, 1] - gt_crop[valid, 1]
        err_mag = np.sqrt(err_u**2 + err_v**2)

        rms_u = float(np.sqrt(np.mean(err_u**2)))
        rms_v = float(np.sqrt(np.mean(err_v**2)))
        rms_mag = float(np.sqrt(np.mean(err_mag**2)))
        max_err = float(np.max(err_mag))

        rms_u_list.append(rms_u)
        rms_v_list.append(rms_v)
        rms_mag_list.append(rms_mag)
        max_err_list.append(max_err)
        valid_pct_list.append(valid_pct)
        angles.append((frame_num - 1) * 3.0)

        print(f"{frame_num:6d} {angles[-1]:6.1f}° {rms_u:8.3f} {rms_v:8.3f} "
              f"{rms_mag:8.3f} {max_err:8.2f} {valid_pct:6.1f}%")

    print("-" * 70)

    if rms_mag_list:
        print(f"\nSummary:")
        print(f"  Mean RMS error:  {np.mean(rms_mag_list):.3f} px")
        print(f"  Max RMS error:   {np.max(rms_mag_list):.3f} px (at {angles[np.argmax(rms_mag_list)]:.0f}°)")
        print(f"  Mean valid coverage: {np.mean(valid_pct_list):.1f}%")
        print(f"  Min valid coverage:  {np.min(valid_pct_list):.1f}% (at {angles[np.argmin(valid_pct_list)]:.0f}°)")

    return {
        "angles": np.array(angles),
        "rms_u": np.array(rms_u_list),
        "rms_v": np.array(rms_v_list),
        "rms_mag": np.array(rms_mag_list),
        "max_err": np.array(max_err_list),
        "valid_pct": np.array(valid_pct_list),
    }


def save_plots(stats, out_dir):
    """Generate summary plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # RMS error vs angle
    ax1 = axes[0]
    ax1.plot(stats["angles"], stats["rms_u"], "b-o", ms=4, label="RMS u")
    ax1.plot(stats["angles"], stats["rms_v"], "r-s", ms=4, label="RMS v")
    ax1.plot(stats["angles"], stats["rms_mag"], "k-^", ms=4, label="RMS magnitude")
    ax1.set_ylabel("RMS Error (px)")
    ax1.set_title("Incremental DIC Accuracy — 90° Rotation Test")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Valid pixel coverage vs angle
    ax2 = axes[1]
    ax2.plot(stats["angles"], stats["valid_pct"], "g-o", ms=4)
    ax2.set_xlabel("Rotation Angle (°)")
    ax2.set_ylabel("Valid Pixel Coverage (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "rotation_accuracy.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Rotation accuracy test")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--mask-dir", default="folder",
                        choices=["folder", "auto"],
                        help="'folder' uses provided masks, 'auto' uses auto-warp")
    args = parser.parse_args()

    if not os.path.exists(GT_PATH):
        print("Ground truth not found. Run generate_rotation_test.py first.")
        sys.exit(1)

    angles, center, mask_radius, gt_displacements = load_ground_truth()
    print(f"Loaded ground truth: {len(gt_displacements)} frames, "
          f"center={center}, radius={mask_radius}")

    use_mask_dir = (args.mask_dir == "folder")
    results, roi_rect, roi_mask = run_dic(args.device, use_mask_dir)

    stats = analyze_results(results, roi_rect, roi_mask, gt_displacements)

    if stats["angles"].size > 0:
        save_plots(stats, RESULTS_DIR)

        # Save numeric results
        np.savez_compressed(
            os.path.join(RESULTS_DIR, "accuracy_stats.npz"),
            **stats,
        )


if __name__ == "__main__":
    main()
