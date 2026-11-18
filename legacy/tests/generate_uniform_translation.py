"""
Generate a uniform constant-velocity translation test dataset.

Every frame translates 2 px to the right relative to the previous frame.
Frame 1: 0 px, Frame 2: 2 px, ..., Frame 31: 60 px.

Usage:
    python tests/generate_uniform_translation.py
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndi_shift
from PIL import Image


def generate_speckle(H: int, W: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W_pad = W + 200
    H_pad = H + 100

    fine = (rng.random((H_pad, W_pad)) < 0.3).astype(np.float64)
    coarse = (rng.random((H_pad // 2, W_pad // 2)) < 0.25).astype(np.float64)
    coarse_up = np.kron(coarse, np.ones((2, 2)))[:H_pad, :W_pad]

    pattern = 0.6 * fine + 0.4 * coarse_up
    pattern = gaussian_filter(pattern, sigma=1.8)
    pattern -= pattern.min()
    pattern /= pattern.max() + 1e-10
    return (pattern * 255).astype(np.uint8)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    output_dir = os.path.join(project_root, "test_data", "uniform_translation_2px")

    H = W = 512
    N = 31          # 31 frames: frame 1 (ref) + 30 steps
    step = 2.0      # px per frame

    os.makedirs(output_dir, exist_ok=True)
    gt_dir = os.path.join(output_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    print(f"=== Uniform Translation Test (2 px/frame, 30 steps) ===")
    print(f"  Output: {output_dir}")
    print(f"  Frames: {N} (ref + 30 deformed)")
    print(f"  Step:   {step} px/frame")
    print(f"  Total:  {step * (N - 1):.0f} px at last frame")
    print()

    big = generate_speckle(H, W)
    big_H, big_W = big.shape
    pad_x = (big_W - W) // 2
    pad_y = (big_H - H) // 2

    for i in range(N):
        u_val = step * i   # frame 1 → 0, frame 2 → 2, ...

        shifted = ndi_shift(big.astype(np.float64), [0, -u_val], order=3, mode='reflect')
        frame = np.clip(shifted[pad_y:pad_y + H, pad_x:pad_x + W], 0, 255).astype(np.uint8)
        frame_rgb = np.stack([frame] * 3, axis=-1)

        Image.fromarray(frame_rgb).save(os.path.join(output_dir, f"frame_{i+1:04d}.png"))

        gt = np.zeros((H, W, 2), dtype=np.float32)
        gt[..., 0] = u_val
        np.save(os.path.join(gt_dir, f"gt_disp_{i+1:04d}.npy"), gt)

        print(f"  Frame {i+1:2d}/{N}: u = {u_val:+6.1f} px")

    print(f"\nDone! {output_dir}")


if __name__ == "__main__":
    main()
