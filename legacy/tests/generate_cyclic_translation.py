"""
Generate a synthetic cyclic TRANSLATION test dataset.

Pure uniform horizontal translation following a sinusoidal cycle.
Every pixel moves the same amount — simplest possible verification case.

The peak displacement (~40 px) exceeds the weak model's single-step
range (~10 px), but frame-to-frame steps (~8 px) are within range.

Usage:
    python tests/generate_cyclic_translation.py [--amplitude 40] [--num_frames 60]
"""

import argparse
import os
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndi_shift
from PIL import Image


def generate_speckle(H: int, W: int, seed: int = 42) -> np.ndarray:
    """Generate a speckle pattern with extra width for translation padding."""
    rng = np.random.default_rng(seed)

    # Generate wider pattern to avoid boundary artifacts during translation
    W_pad = W + 300  # Extra padding for max translation
    H_pad = H + 100

    fine = (rng.random((H_pad, W_pad)) < 0.3).astype(np.float64)
    coarse = (rng.random((H_pad // 2, W_pad // 2)) < 0.25).astype(np.float64)
    coarse_up = np.kron(coarse, np.ones((2, 2)))[:H_pad, :W_pad]

    pattern = 0.6 * fine + 0.4 * coarse_up
    pattern = gaussian_filter(pattern, sigma=1.8)

    pattern -= pattern.min()
    pattern /= pattern.max() + 1e-10
    pattern = (pattern * 255).astype(np.uint8)

    return pattern


def main():
    parser = argparse.ArgumentParser(description="Generate cyclic translation test")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--amplitude", type=float, default=40.0,
                        help="Peak translation in pixels (default: 40)")
    parser.add_argument("--period", type=int, default=30)
    args = parser.parse_args()

    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.output_dir = os.path.join(project_root, "test_data", "cyclic_translation")

    H = W = args.image_size
    N = args.num_frames
    A = args.amplitude
    T = args.period

    os.makedirs(args.output_dir, exist_ok=True)

    max_step = A * 2 * np.pi / T

    print(f"=== Cyclic Translation Test Dataset ===")
    print(f"  Output:     {args.output_dir}")
    print(f"  Image size: {H}x{W}")
    print(f"  Frames:     {N} ({N/T:.1f} cycles)")
    print(f"  Amplitude:  {A:.0f} px (uniform horizontal translation)")
    print(f"  Period:     {T} frames")
    print(f"  Max step:   ~{max_step:.1f} px (frame-to-frame)")
    print(f"  Max total:  ~{A:.0f} px (from reference)")
    print()

    # Generate oversized speckle (to crop from, avoiding edge artifacts)
    print("Generating speckle pattern...")
    big_speckle = generate_speckle(H, W)
    big_H, big_W = big_speckle.shape

    # Center crop region for reference frame
    pad_x = (big_W - W) // 2
    pad_y = (big_H - H) // 2

    gt_dir = os.path.join(args.output_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    print(f"Generating {N} frames...")
    displacements = []

    for i in range(N):
        # Uniform translation: u(t) = A * sin(2*pi*t/T)
        u_val = A * np.sin(2 * np.pi * i / T)
        v_val = 0.0

        displacements.append(u_val)

        # Crop from the big speckle with sub-pixel shift
        # Use scipy.ndimage.shift for sub-pixel accuracy
        shifted = ndi_shift(big_speckle.astype(np.float64), [0, -u_val], order=3, mode='reflect')
        frame = shifted[pad_y:pad_y + H, pad_x:pad_x + W]
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Save as RGB
        frame_rgb = np.stack([frame] * 3, axis=-1)
        fname = f"frame_{i+1:04d}.png"
        Image.fromarray(frame_rgb).save(os.path.join(args.output_dir, fname))

        # Save ground truth (uniform field)
        gt = np.zeros((H, W, 2), dtype=np.float32)
        gt[..., 0] = u_val
        gt[..., 1] = v_val
        np.save(os.path.join(gt_dir, f"gt_disp_{i+1:04d}.npy"), gt)

        if (i + 1) % 10 == 0 or i == 0 or i == N - 1:
            print(f"  Frame {i+1:3d}/{N}: u = {u_val:+7.2f} px")

    # Print displacement profile
    print()
    print("=== Displacement Profile ===")
    print("  Frame |    u (px)  | delta from prev")
    print("  ------+------------+----------------")
    for i in range(min(N, 35)):
        u = displacements[i]
        delta = displacements[i] - displacements[i-1] if i > 0 else 0
        marker = " <-- PEAK" if abs(u) > A * 0.99 else ""
        marker = " <-- zero crossing" if i > 0 and abs(u) < 1.0 and abs(displacements[i-1]) > 5 else marker
        print(f"  {i+1:5d} | {u:+10.2f} | {delta:+10.2f}{marker}")
    if N > 35:
        print(f"  ... ({N - 35} more frames)")

    print()
    print("=== Test Plan ===")
    print(f"  1. Accumulative mode: should FAIL at peak (frame ~{T//4+1}, u={A:.0f}px)")
    print(f"     - The 10px model cannot track {A:.0f}px in one step")
    print(f"  2. Incremental (Every Frame): should work but accumulates error")
    print(f"  3. Incremental (Every 5): should work, fewer error accumulation steps")
    print(f"  4. Compare all three against ground_truth/*.npy")
    print()
    print(f"Done! {N} images saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
