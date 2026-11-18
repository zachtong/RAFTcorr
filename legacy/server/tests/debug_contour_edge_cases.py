"""Diagnostic: test forward-mapped contour under challenging conditions.

Scenarios:
1. Internal holes (NaN regions from failed correlation)
2. Large deformation (contour self-intersection risk)
3. Complex non-convex boundary
4. Shear deformation (parallelogram shape)
"""

import numpy as np
import cv2
import os

from server.deformed_warp import compute_inverse_map, warp_data_inverse

OUT_DIR = os.path.join(os.path.dirname(__file__), "debug_output")
os.makedirs(OUT_DIR, exist_ok=True)


def render_case(U, V, data, roi_rect, image_shape, label, amplitude_info=""):
    """Render and save a test case, returning boundary error stats."""
    roi_h = roi_rect[3] - roi_rect[1]
    roi_w = roi_rect[2] - roi_rect[0]
    x0, y0 = roi_rect[0], roi_rect[1]

    inv_map = compute_inverse_map(U, V, roi_rect, image_shape)
    warped = warp_data_inverse(data, inv_map, image_shape)

    h, w = image_shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    valid = np.isfinite(warped)
    # Color by warped value
    valid_vals = warped[valid]
    if valid_vals.size > 0 and valid_vals.max() > valid_vals.min():
        norm = (warped - np.nanmin(warped)) / (np.nanmax(warped) - np.nanmin(warped) + 1e-10)
        norm = np.nan_to_num(norm, nan=0.0)
        vis[valid, 0] = (200 * (1 - norm[valid])).astype(np.uint8)
        vis[valid, 1] = (150 * norm[valid]).astype(np.uint8)
        vis[valid, 2] = (200 * norm[valid]).astype(np.uint8)
    else:
        vis[valid] = [200, 150, 50]

    # Draw theoretical boundary (forward-map ROI edges)
    # Top edge
    for col in range(roi_w):
        row = 0
        if np.isfinite(U[row, col]) and np.isfinite(V[row, col]):
            dx = int(round(x0 + col + U[row, col]))
            dy = int(round(y0 + row + V[row, col]))
            if 0 <= dy < h and 0 <= dx < w:
                vis[dy, dx] = [0, 255, 0]
    # Bottom edge
    for col in range(roi_w):
        row = roi_h - 1
        if np.isfinite(U[row, col]) and np.isfinite(V[row, col]):
            dx = int(round(x0 + col + U[row, col]))
            dy = int(round(y0 + row + V[row, col]))
            if 0 <= dy < h and 0 <= dx < w:
                vis[dy, dx] = [0, 255, 0]
    # Left edge
    for row in range(roi_h):
        col = 0
        if np.isfinite(U[row, col]) and np.isfinite(V[row, col]):
            dx = int(round(x0 + col + U[row, col]))
            dy = int(round(y0 + row + V[row, col]))
            if 0 <= dy < h and 0 <= dx < w:
                vis[dy, dx] = [0, 255, 0]
    # Right edge
    for row in range(roi_h):
        col = roi_w - 1
        if np.isfinite(U[row, col]) and np.isfinite(V[row, col]):
            dx = int(round(x0 + col + U[row, col]))
            dy = int(round(y0 + row + V[row, col]))
            if 0 <= dy < h and 0 <= dx < w:
                vis[dy, dx] = [0, 255, 0]

    # Count valid pixels
    source_valid = np.sum(np.isfinite(data))
    warped_valid = np.sum(valid)
    coverage = warped_valid / source_valid * 100 if source_valid > 0 else 0

    # Count holes in warped: valid pixels in source that became NaN in warped
    # (Check by looking for NaN "islands" inside the valid region)
    if valid.any():
        valid_u8 = valid.astype(np.uint8)
        # Fill from edges to find holes
        flood = valid_u8.copy()
        cv2.floodFill(flood, None, (0, 0), 2)
        holes = (flood != 2) & (~valid)
        hole_count = np.sum(holes)
    else:
        hole_count = 0

    cv2.putText(vis, f"Case: {label}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, amplitude_info, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, f"Coverage: {coverage:.1f}%", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, f"Interior holes: {hole_count}px", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, "Green=theoretical boundary", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    path = os.path.join(OUT_DIR, f"edge_{label}.png")
    cv2.imwrite(path, vis)
    print(f"  {label}: coverage={coverage:.1f}%, holes={hole_count}px")
    return coverage, hole_count


def test_internal_holes():
    """ROI with internal NaN holes simulating failed correlation."""
    print("\n=== Test 1: Internal Holes ===")
    roi_h, roi_w = 400, 200
    U = np.full((roi_h, roi_w), 15.0)
    V = np.zeros((roi_h, roi_w))

    # Add a bending component
    yy = np.arange(roi_h).reshape(-1, 1)
    U += 20.0 * (yy / roi_h) ** 2

    # Create holes (simulating correlation failure)
    data = np.ones((roi_h, roi_w), dtype=np.float64)
    # Hole 1: circular
    cy, cx = 150, 100
    yy2, xx2 = np.mgrid[:roi_h, :roi_w]
    hole1 = ((yy2 - cy)**2 + (xx2 - cx)**2) < 30**2
    U[hole1] = np.nan
    V[hole1] = np.nan
    data[hole1] = np.nan

    # Hole 2: rectangular
    U[250:280, 50:150] = np.nan
    V[250:280, 50:150] = np.nan
    data[250:280, 50:150] = np.nan

    roi_rect = (100, 50, 300, 450)
    image_shape = (550, 500)
    render_case(U, V, data, roi_rect, image_shape, "holes",
                "Circular + rect NaN holes")


def test_large_deformation():
    """Very large deformation that could cause contour issues."""
    print("\n=== Test 2: Large Deformation ===")
    roi_h, roi_w = 400, 150

    yy, xx = np.mgrid[:roi_h, :roi_w]

    # Large bending: 80px amplitude
    U = 80.0 * (yy / roi_h) ** 2
    V = np.zeros_like(U, dtype=np.float64)

    data = (yy / roi_h).astype(np.float64)  # gradient for visualization

    roi_rect = (50, 50, 200, 450)
    image_shape = (550, 400)
    render_case(U, V, data, roi_rect, image_shape, "large_deform",
                "80px bending amplitude")


def test_shear():
    """Pure shear deformation (parallelogram)."""
    print("\n=== Test 3: Shear Deformation ===")
    roi_h, roi_w = 300, 200

    yy, xx = np.mgrid[:roi_h, :roi_w]

    # Simple shear: U = shear * y
    shear = 0.3
    U = (shear * yy).astype(np.float64)
    V = np.zeros_like(U, dtype=np.float64)

    data = (xx / roi_w).astype(np.float64)

    roi_rect = (100, 50, 300, 350)
    image_shape = (450, 500)
    render_case(U, V, data, roi_rect, image_shape, "shear",
                f"Shear ratio={shear}")


def test_compression_necking():
    """Necking under compression - width narrows in the middle."""
    print("\n=== Test 4: Necking (Compression) ===")
    roi_h, roi_w = 500, 200

    yy, xx = np.mgrid[:roi_h, :roi_w]
    cy = roi_h / 2
    cx = roi_w / 2

    # Vertical compression
    V = -20.0 * (yy - cy) / cy
    # Lateral bulge at mid-height (necking)
    neck_factor = np.exp(-((yy - cy) / (roi_h * 0.15)) ** 2)
    U = 15.0 * (xx - cx) / cx * neck_factor
    U = U.astype(np.float64)
    V = V.astype(np.float64)

    data = np.ones((roi_h, roi_w), dtype=np.float64)

    roi_rect = (100, 50, 300, 550)
    image_shape = (650, 500)
    render_case(U, V, data, roi_rect, image_shape, "necking",
                "Compression with lateral necking")


def test_s_bend():
    """S-shaped bending (sinusoidal lateral displacement)."""
    print("\n=== Test 5: S-Bend ===")
    roi_h, roi_w = 500, 120

    yy, xx = np.mgrid[:roi_h, :roi_w]

    # S-curve: sinusoidal lateral displacement
    U = (40.0 * np.sin(2 * np.pi * yy / roi_h)).astype(np.float64)
    V = np.zeros_like(U, dtype=np.float64)

    data = (yy / roi_h).astype(np.float64)

    roi_rect = (100, 50, 220, 550)
    image_shape = (650, 400)
    render_case(U, V, data, roi_rect, image_shape, "s_bend",
                "Sinusoidal S-bend, A=40px")


def main():
    print("=== Forward-Mapped Contour Edge Case Tests ===")
    test_internal_holes()
    test_large_deformation()
    test_shear()
    test_compression_necking()
    test_s_bend()
    print(f"\nAll visualizations saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
