"""
Generate a synthetic cyclic-loading DIC test dataset.

Creates a speckle pattern reference image and warps it through a sinusoidal
displacement cycle. The max deformation (~120 px) is too large for a single
DIC step from the reference, forcing the use of incremental mode with
key frames.

Deformation model:
    u(x, y, t) = A * sin(2*pi*t/T) * (x - cx) / cx   (horizontal stretch)
    v(x, y, t) = -nu * A * sin(2*pi*t/T) * (y - cy) / cy  (Poisson contraction)

where A = peak edge displacement, T = period in frames, nu = Poisson's ratio.

Usage:
    python tests/generate_cyclic_test.py [--output_dir PATH] [--num_frames 60]
"""

import argparse
import os
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from PIL import Image


def generate_speckle(H: int, W: int, dot_density: float = 0.3,
                     blur_sigma: float = 1.8, seed: int = 42) -> np.ndarray:
    """Generate a realistic speckle pattern for DIC.

    Uses random binary dots with Gaussian smoothing to produce
    smooth intensity gradients that RAFT can track.
    """
    rng = np.random.default_rng(seed)

    # Multi-scale speckle: combine fine + coarse patterns
    fine = (rng.random((H, W)) < dot_density).astype(np.float64)
    coarse = (rng.random((H // 2, W // 2)) < dot_density * 0.8).astype(np.float64)

    # Upsample coarse
    coarse_up = np.kron(coarse, np.ones((2, 2)))[:H, :W]

    # Blend
    pattern = 0.6 * fine + 0.4 * coarse_up

    # Smooth to create trackable gradients
    pattern = gaussian_filter(pattern, sigma=blur_sigma)

    # Normalize to 0-255
    pattern -= pattern.min()
    pattern /= pattern.max() + 1e-10
    pattern = (pattern * 255).astype(np.uint8)

    return pattern


def apply_displacement(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image by displacement field (u, v) using inverse mapping.

    The displacement field maps reference coords to deformed coords:
        x_def = x + u(x, y)
        y_def = y + v(x, y)

    We use inverse mapping: for each output pixel (x_def, y_def),
    sample the reference at (x_def - u, y_def - v).
    Actually, we directly create forward map and invert via grid sampling.
    """
    H, W = image.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    # Inverse mapping: find where each deformed pixel came from in reference
    # For forward warp x_def = x + u, the inverse is x_ref = x - u
    # But that's only exact for small deformations.
    # Better: create the deformed coordinates and use scipy's map_coordinates
    # on the reference image.
    #
    # Forward: ref(x,y) -> def(x+u, y+v)
    # For rendering: sample ref at (x - u(x,y), y - v(x,y)) ← approximate inverse
    src_y = yy - v
    src_x = xx - u

    if image.ndim == 2:
        warped = map_coordinates(
            image.astype(np.float64), [src_y, src_x],
            order=3, mode='reflect'
        )
        return np.clip(warped, 0, 255).astype(np.uint8)
    else:
        channels = []
        for c in range(image.shape[2]):
            ch = map_coordinates(
                image[:, :, c].astype(np.float64), [src_y, src_x],
                order=3, mode='reflect'
            )
            channels.append(np.clip(ch, 0, 255).astype(np.uint8))
        return np.stack(channels, axis=-1)


def cyclic_displacement(H: int, W: int, t: float, T: float,
                        A: float, nu: float = 0.3) -> tuple:
    """Compute cyclic displacement field at time t.

    Parameters
    ----------
    H, W : int
        Image dimensions.
    t : float
        Current time (frame index, 0-based).
    T : float
        Period in frames.
    A : float
        Peak horizontal edge displacement in pixels.
    nu : float
        Poisson's ratio for transverse contraction.

    Returns
    -------
    u, v : ndarray (H, W)
        Horizontal and vertical displacement fields.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W / 2.0, H / 2.0

    # Sinusoidal strain cycle
    strain = np.sin(2 * np.pi * t / T)

    # Spatially varying horizontal displacement (stretch from center)
    u = A * strain * (xx - cx) / cx

    # Poisson contraction (vertical)
    v = -nu * A * strain * (yy - cy) / cy

    return u, v


def main():
    parser = argparse.ArgumentParser(description="Generate cyclic DIC test images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for images")
    parser.add_argument("--num_frames", type=int, default=60,
                        help="Number of frames (default: 60, ~2 cycles)")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Image size HxW (default: 512)")
    parser.add_argument("--amplitude", type=float, default=120.0,
                        help="Peak edge displacement in pixels (default: 120)")
    parser.add_argument("--period", type=int, default=30,
                        help="Period in frames (default: 30)")
    parser.add_argument("--poisson", type=float, default=0.3,
                        help="Poisson's ratio (default: 0.3)")
    args = parser.parse_args()

    # Default output dir
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.output_dir = os.path.join(project_root, "test_data", "cyclic_loading")

    H = W = args.image_size
    N = args.num_frames
    A = args.amplitude
    T = args.period
    nu = args.poisson

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Cyclic Loading Test Dataset Generator ===")
    print(f"  Output:     {args.output_dir}")
    print(f"  Image size: {H}x{W}")
    print(f"  Frames:     {N}")
    print(f"  Amplitude:  {A} px (peak edge displacement)")
    print(f"  Period:     {T} frames")
    print(f"  Poisson:    {nu}")
    print()

    # Max step-to-step displacement (at zero crossing, max velocity)
    max_step = A * 2 * np.pi / T
    print(f"  Max frame-to-frame step: ~{max_step:.1f} px")
    print(f"  Max total from ref:      ~{A:.1f} px")
    print(f"  -> Single-step DIC from ref will FAIL at peak deformation")
    print(f"  -> Incremental mode with key frames needed")
    print()

    # Generate speckle reference
    print("Generating speckle pattern...")
    speckle = generate_speckle(H, W)

    # Convert to RGB (3-channel) for realistic DIC images
    speckle_rgb = np.stack([speckle] * 3, axis=-1)

    # Save ground truth displacement for verification
    gt_dir = os.path.join(args.output_dir, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    # Generate frames
    print(f"Generating {N} frames...")
    for i in range(N):
        u, v = cyclic_displacement(H, W, t=i, T=T, A=A, nu=nu)

        if i == 0:
            # Frame 1 is the reference (zero displacement)
            frame = speckle_rgb.copy()
        else:
            frame = apply_displacement(speckle_rgb, u, v)

        # Save image
        fname = f"frame_{i+1:04d}.png"
        Image.fromarray(frame).save(os.path.join(args.output_dir, fname))

        # Save ground truth displacement
        gt = np.stack([u, v], axis=-1).astype(np.float32)
        np.save(os.path.join(gt_dir, f"gt_disp_{i+1:04d}.npy"), gt)

        # Progress
        if (i + 1) % 10 == 0 or i == 0 or i == N - 1:
            u_max = np.max(np.abs(u))
            v_max = np.max(np.abs(v))
            print(f"  Frame {i+1:3d}/{N}: |u|_max={u_max:6.1f} px, |v|_max={v_max:6.1f} px")

    # Print key frame suggestions
    print()
    print("=== Recommended Key Frame Settings ===")
    print(f"  Every 5 frames:  key frames at {list(range(1, N+1, 5))}")
    print(f"  Every 10 frames: key frames at {list(range(1, N+1, 10))}")
    print()

    # Print displacement at specific frames
    print("=== Displacement at Key Times ===")
    for label, t in [("t=0 (ref)", 0), ("t=T/4 (peak+)", T//4),
                     ("t=T/2 (zero crossing)", T//2), ("t=3T/4 (peak-)", 3*T//4),
                     ("t=T (full cycle)", T)]:
        if t < N:
            u, v = cyclic_displacement(H, W, t=t, T=T, A=A, nu=nu)
            print(f"  Frame {t+1:3d} ({label}): u_edge={u[H//2, -1]:.1f} px, "
                  f"v_edge={v[-1, W//2]:.1f} px")

    print()
    print(f"Done! Images saved to: {args.output_dir}")
    print(f"Ground truth saved to: {gt_dir}")
    print()
    print("To test in the GUI:")
    print("  1. Load images from the output directory")
    print("  2. Draw an ROI covering most of the image")
    print("  3. Set mode to 'Incremental'")
    print("  4. Set key frames: Every 5 (recommended)")
    print("  5. Run processing")
    print("  6. Compare results with ground truth .npy files")


if __name__ == "__main__":
    main()
