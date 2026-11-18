"""Diagnostic: visualize deformed ROI boundary for a bending specimen.

Creates a tall rectangular specimen with cantilever-like bending and renders
the deformed view to check if boundaries follow the smooth deformed shape.
"""

import numpy as np
import cv2
import os

from server.deformed_warp import compute_inverse_map, warp_data_inverse

OUT_DIR = os.path.join(os.path.dirname(__file__), "debug_output")
os.makedirs(OUT_DIR, exist_ok=True)


def make_bending_field(roi_h, roi_w, amplitude=30.0):
    """Create a bending displacement field for a vertical rectangular specimen.

    Simulates cantilever bending: horizontal displacement varies
    parabolically along height, zero at top (clamped), max at bottom.

    U(x, y) = amplitude * (y / H)^2    (horizontal bending)
    V(x, y) = 0                          (no vertical displacement)
    """
    yy, xx = np.mgrid[:roi_h, :roi_w]
    # Parabolic bending: U grows quadratically from top to bottom
    U = amplitude * (yy / roi_h) ** 2
    V = np.zeros_like(U)
    return U.astype(np.float64), V.astype(np.float64)


def render_deformed_boundary(U, V, roi_rect, image_shape, label):
    """Compute deformed view and save boundary visualization."""
    roi_h, roi_w = roi_rect[3] - roi_rect[1], roi_rect[2] - roi_rect[0]

    inv_map = compute_inverse_map(U, V, roi_rect, image_shape)

    # Source data: gradient value inside ROI
    data = np.ones((roi_h, roi_w), dtype=np.float64)
    warped = warp_data_inverse(data, inv_map, image_shape)

    # Create visualization
    h, w = image_shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw the deformed valid region in blue
    valid = np.isfinite(warped)
    vis[valid] = [200, 150, 50]  # blue-ish in BGR

    # Draw theoretical deformed boundary in green
    # Left boundary: x = roi_x0 + U(x=0, y), for each y
    x0, y0, x1, y1 = roi_rect
    for iy in range(roi_h):
        img_y = y0 + iy
        # Left edge
        left_x = int(round(x0 + U[iy, 0]))
        if 0 <= img_y < h and 0 <= left_x < w:
            cv2.circle(vis, (left_x, img_y), 1, (0, 255, 0), -1)
        # Right edge
        right_x = int(round(x0 + roi_w - 1 + U[iy, roi_w - 1]))
        if 0 <= img_y < h and 0 <= right_x < w:
            cv2.circle(vis, (right_x, img_y), 1, (0, 255, 0), -1)

    # Draw actual valid boundary in red
    # Find left and right edges per row
    boundary_errors = []
    for img_y in range(h):
        cols = np.where(valid[img_y, :])[0]
        if len(cols) > 0:
            # Mark actual edges
            cv2.circle(vis, (int(cols[0]), img_y), 1, (0, 0, 255), -1)
            cv2.circle(vis, (int(cols[-1]), img_y), 1, (0, 0, 255), -1)

            # Compute error vs theoretical right edge
            iy = img_y - y0
            if 0 <= iy < roi_h:
                expected_right = x0 + roi_w - 1 + U[iy, roi_w - 1]
                actual_right = cols[-1]
                boundary_errors.append(abs(actual_right - expected_right))

    boundary_errors = np.array(boundary_errors)

    # Add text info
    cv2.putText(vis, f"Amplitude: {U.max():.0f}px", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if len(boundary_errors) > 0:
        cv2.putText(vis, f"Max boundary error: {boundary_errors.max():.1f}px", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(vis, f"Mean boundary error: {boundary_errors.mean():.1f}px", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(vis, "Green=theoretical  Red=actual", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    path = os.path.join(OUT_DIR, f"bending_{label}.png")
    cv2.imwrite(path, vis)
    print(f"Saved: {path}")
    print(f"  {label}: max_err={boundary_errors.max():.1f}px, "
          f"mean_err={boundary_errors.mean():.1f}px")
    return boundary_errors


def main():
    # Tall rectangular specimen (like the user's tensile sample)
    roi_h, roi_w = 600, 150
    roi_rect = (100, 50, 250, 650)  # x0, y0, x1, y1
    image_shape = (750, 450)

    U, V = make_bending_field(roi_h, roi_w, amplitude=40.0)

    print("=== Bending Boundary Test ===")
    print(f"ROI: {roi_w}x{roi_h}, Bending amplitude: {U.max():.0f}px\n")

    render_deformed_boundary(U, V, roi_rect, image_shape, "bending")

    print(f"\nVisualizations saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
