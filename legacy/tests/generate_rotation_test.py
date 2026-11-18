"""Generate a synthetic rotation test dataset for incremental DIC.

Creates:
  test_data/rotation_90deg/images/   — 31 frames (1000×1000, speckle rotated 0→90°)
  test_data/rotation_90deg/masks/    — 31 binary masks matching the speckle region
  test_data/rotation_90deg/ground_truth.npz — analytical displacement fields

The speckle pattern occupies a ~500×500 central region on a black background.
Rotation is applied in 30 equal steps of 3° each, centered on the image center.
"""

import os
import numpy as np
from PIL import Image, ImageDraw

# ── Parameters ──────────────────────────────────────────────────────────
IMG_SIZE = 1000
SPECKLE_SIZE = 500  # side length of speckle region
N_STEPS = 30        # number of deformation steps (total frames = N_STEPS + 1)
TOTAL_ANGLE_DEG = 90.0
SEED = 42

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "test_data", "rotation_90deg")
IMG_DIR = os.path.join(OUT_DIR, "images")
MASK_DIR = os.path.join(OUT_DIR, "masks")


def make_speckle_pattern(size: int, seed: int) -> np.ndarray:
    """Generate a realistic speckle pattern via random dots + Gaussian blur."""
    rng = np.random.default_rng(seed)

    # Start with mid-gray base
    pattern = np.full((size, size), 128, dtype=np.uint8)

    # Add random dots of varying size (simulates spray paint speckle)
    n_dots = size * size // 20  # density
    for _ in range(n_dots):
        cx = rng.integers(0, size)
        cy = rng.integers(0, size)
        radius = rng.integers(1, 5)
        brightness = rng.choice([0, 255])  # black or white dots

        # Draw filled circle
        y, x = np.ogrid[-cy:size - cy, -cx:size - cx]
        mask = x * x + y * y <= radius * radius
        pattern[mask] = brightness

    # Gaussian blur for smoother texture (prevents aliasing during rotation)
    from scipy.ndimage import gaussian_filter
    pattern = gaussian_filter(pattern.astype(np.float64), sigma=1.0)
    pattern = np.clip(pattern, 0, 255).astype(np.uint8)

    return pattern


def embed_speckle(speckle: np.ndarray, canvas_size: int) -> np.ndarray:
    """Place speckle pattern centered on a black canvas."""
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    s = speckle.shape[0]
    offset = (canvas_size - s) // 2
    canvas[offset:offset + s, offset:offset + s] = speckle
    return canvas


def make_circular_mask(canvas_size: int, radius: int) -> np.ndarray:
    """Create a circular mask centered on the canvas."""
    y, x = np.ogrid[:canvas_size, :canvas_size]
    cx, cy = canvas_size / 2.0, canvas_size / 2.0
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return dist <= radius


def rotate_image(img: np.ndarray, angle_deg: float, center: tuple) -> np.ndarray:
    """Rotate image around center using high-quality interpolation."""
    import cv2
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        img, M, (img.shape[1], img.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated


def rotate_mask(mask: np.ndarray, angle_deg: float, center: tuple) -> np.ndarray:
    """Rotate binary mask using nearest-neighbor interpolation."""
    import cv2
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        mask_uint8, M, (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated > 127


def compute_rotation_displacement(H: int, W: int, angle_deg: float,
                                  cx: float, cy: float) -> np.ndarray:
    """Compute analytical displacement field for rigid rotation.

    For a point (x, y) rotated by angle θ about (cx, cy):
      x' = cx + (x - cx) cos θ - (y - cy) sin θ
      y' = cy + (x - cx) sin θ + (y - cy) cos θ
      u = x' - x
      v = y' - y

    Returns (H, W, 2) array with [u, v] components.
    """
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    dx = x_grid - cx
    dy = y_grid - cy

    u = dx * (cos_t - 1) - dy * sin_t
    v = dx * sin_t + dy * (cos_t - 1)

    disp = np.stack([u, v], axis=-1)
    return disp


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(MASK_DIR, exist_ok=True)

    print(f"Generating rotation test: {N_STEPS} steps, {TOTAL_ANGLE_DEG}° total")
    print(f"Image size: {IMG_SIZE}×{IMG_SIZE}, speckle region: {SPECKLE_SIZE}×{SPECKLE_SIZE}")

    # Generate speckle pattern
    speckle = make_speckle_pattern(SPECKLE_SIZE, SEED)
    ref_image = embed_speckle(speckle, IMG_SIZE)

    # Use circular mask (radius = SPECKLE_SIZE/2) to avoid corner artifacts
    # during rotation. The circle inscribed in the 500×500 square has r=250.
    mask_radius = SPECKLE_SIZE // 2
    circular_mask = make_circular_mask(IMG_SIZE, mask_radius)

    # Apply circular mask to reference image (zero out corners of speckle square)
    ref_image[~circular_mask] = 0

    center = (IMG_SIZE / 2.0, IMG_SIZE / 2.0)
    angle_step = TOTAL_ANGLE_DEG / N_STEPS

    # Store ground truth
    gt_displacements = {}

    for i in range(N_STEPS + 1):
        angle = i * angle_step
        frame_num = i + 1  # 1-indexed

        # Rotate image
        if i == 0:
            frame_img = ref_image.copy()
            frame_mask = circular_mask.copy()
        else:
            # Rotate in the opposite direction for cv2 (positive = counter-clockwise)
            frame_img = rotate_image(ref_image, -angle, center)
            frame_mask = rotate_mask(circular_mask, -angle, center)

        # Save image
        img_path = os.path.join(IMG_DIR, f"frame_{frame_num:04d}.tif")
        Image.fromarray(frame_img).save(img_path)

        # Save mask (binary: 0 or 255)
        mask_path = os.path.join(MASK_DIR, f"frame_{frame_num:04d}.tif")
        mask_uint8 = (frame_mask.astype(np.uint8)) * 255
        Image.fromarray(mask_uint8).save(mask_path)

        # Ground truth displacement (from frame 1 coordinates)
        gt_disp = compute_rotation_displacement(
            IMG_SIZE, IMG_SIZE, -angle, center[0], center[1]
        )
        # Mask: NaN outside circular ROI of frame 1
        gt_disp[~circular_mask] = np.nan
        gt_displacements[frame_num] = gt_disp

        print(f"  Frame {frame_num:3d}: angle={angle:6.1f}°, "
              f"max|u|={np.nanmax(np.abs(gt_disp[...,0])):.1f}px, "
              f"max|v|={np.nanmax(np.abs(gt_disp[...,1])):.1f}px")

    # Save ground truth
    gt_path = os.path.join(OUT_DIR, "ground_truth.npz")
    np.savez_compressed(
        gt_path,
        angles_deg=np.array([i * angle_step for i in range(N_STEPS + 1)]),
        center=np.array(center),
        mask_radius=mask_radius,
        **{f"disp_{k}": v for k, v in gt_displacements.items()},
    )

    print(f"\nSaved {N_STEPS + 1} frames to {IMG_DIR}")
    print(f"Saved {N_STEPS + 1} masks to {MASK_DIR}")
    print(f"Ground truth saved to {gt_path}")

    # Summary of max displacements
    final_disp = gt_displacements[N_STEPS + 1]
    print(f"\nAt 90° rotation:")
    print(f"  Max displacement magnitude: "
          f"{np.nanmax(np.sqrt(final_disp[...,0]**2 + final_disp[...,1]**2)):.1f} px")
    print(f"  This is a challenging test — accumulative mode will likely fail")
    print(f"  because RAFT cannot track 90° rotation in one shot.")


if __name__ == "__main__":
    main()
