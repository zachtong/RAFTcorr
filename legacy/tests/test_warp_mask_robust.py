"""Test robust contour warping (spatial-median + Gaussian smoothing)."""

import numpy as np
from raft_dic_gui.incremental import warp_mask_with_holes


def test_uniform_translation():
    """Uniform translation should preserve shape and shift center correctly."""
    H, W = 200, 200
    y, x = np.ogrid[:H, :W]
    mask = ((x - 100)**2 + (y - 100)**2) <= 80**2

    disp = np.zeros((H, W, 2))
    disp[..., 0] = 20.0

    warped = warp_mask_with_holes(mask, disp)
    oy, ox = np.where(mask)
    wy, wx = np.where(warped)
    shift_x = wx.mean() - ox.mean()
    area_ratio = warped.sum() / mask.sum()

    print(f"  Area ratio: {area_ratio:.3f}")
    print(f"  X shift: {shift_x:.1f} (expected 20)")
    assert abs(shift_x - 20) < 2, f"Shift {shift_x} too far from 20"
    assert abs(area_ratio - 1.0) < 0.05, f"Area ratio {area_ratio} too far from 1.0"
    print("  PASS")


def test_outlier_rejection():
    """Extreme outliers on ~40 contiguous boundary pixels should not corrupt mask."""
    H, W = 200, 200
    y, x = np.ogrid[:H, :W]
    mask = ((x - 100)**2 + (y - 100)**2) <= 80**2

    disp = np.zeros((H, W, 2))
    disp[..., 0] = 20.0

    # Inject outliers: 40 contiguous boundary pixels with extreme displacement
    for angle_deg in range(0, 40):
        a = np.deg2rad(angle_deg)
        bx = int(100 + 80 * np.cos(a))
        by = int(100 + 80 * np.sin(a))
        if 0 <= bx < W and 0 <= by < H:
            disp[by, bx, 0] = 500.0
            disp[by, bx, 1] = -300.0

    warped = warp_mask_with_holes(mask, disp)
    oy, ox = np.where(mask)
    wy, wx = np.where(warped)
    shift_x = wx.mean() - ox.mean()
    area_ratio = warped.sum() / mask.sum()

    print(f"  Area ratio: {area_ratio:.3f}")
    print(f"  X shift: {shift_x:.1f} (expected ~20)")
    assert abs(shift_x - 20) < 5, f"Shift {shift_x} corrupted by outliers!"
    assert abs(area_ratio - 1.0) < 0.10, f"Area ratio {area_ratio} corrupted!"
    print("  PASS")


def test_rotation_30deg():
    """30-degree rotation should preserve area reasonably well."""
    H, W = 200, 200
    y, x = np.ogrid[:H, :W]
    mask = ((x - 100)**2 + (y - 100)**2) <= 80**2

    disp = np.zeros((H, W, 2))
    theta = np.deg2rad(30)
    yg, xg = np.mgrid[:H, :W]
    dx = (xg - 100.0)
    dy = (yg - 100.0)
    disp[..., 0] = dx * (np.cos(theta) - 1) - dy * np.sin(theta)
    disp[..., 1] = dx * np.sin(theta) + dy * (np.cos(theta) - 1)
    disp[~mask] = np.nan

    warped = warp_mask_with_holes(mask, disp)
    area_ratio = warped.sum() / mask.sum()

    print(f"  Area ratio: {area_ratio:.3f}")
    assert area_ratio > 0.85, f"Area too small: {area_ratio}"
    print("  PASS")


def test_mask_with_hole():
    """Mask with internal hole: hole topology should be preserved after warp."""
    H, W = 200, 200
    y, x = np.ogrid[:H, :W]
    outer = ((x - 100)**2 + (y - 100)**2) <= 80**2
    inner = ((x - 100)**2 + (y - 100)**2) <= 30**2
    mask = outer & ~inner  # ring shape

    disp = np.zeros((H, W, 2))
    disp[..., 0] = 15.0

    warped = warp_mask_with_holes(mask, disp)

    # Check hole still exists: center of warped mask should be empty
    center_x, center_y = 115, 100  # shifted by 15
    assert not warped[center_y, center_x], "Hole was filled — topology broken!"

    area_ratio = warped.sum() / mask.sum()
    print(f"  Area ratio: {area_ratio:.3f}")
    assert abs(area_ratio - 1.0) < 0.10
    print("  PASS (hole preserved)")


def test_extreme_half_boundary_corrupted():
    """Half the boundary circle (~180 pixels) has extreme outliers."""
    H, W = 200, 200
    y, x = np.ogrid[:H, :W]
    mask = ((x - 100)**2 + (y - 100)**2) <= 80**2

    disp = np.zeros((H, W, 2))
    disp[..., 0] = 20.0

    count = 0
    for angle_deg in range(0, 180):
        a = np.deg2rad(angle_deg)
        bx = int(100 + 80 * np.cos(a))
        by = int(100 + 80 * np.sin(a))
        if 0 <= bx < W and 0 <= by < H:
            disp[by, bx, 0] = 500.0
            disp[by, bx, 1] = -300.0
            count += 1

    print(f"  Corrupted {count} boundary pixels (half the circle)")
    warped = warp_mask_with_holes(mask, disp)
    oy, ox = np.where(mask)
    wy, wx = np.where(warped)
    shift_x = wx.mean() - ox.mean()
    area_ratio = warped.sum() / mask.sum()

    print(f"  Area ratio: {area_ratio:.3f}")
    print(f"  X shift: {shift_x:.1f} (expected ~20)")
    assert abs(shift_x - 20) < 5, f"Shift {shift_x} corrupted by outliers!"
    assert abs(area_ratio - 1.0) < 0.10, f"Area ratio {area_ratio} corrupted!"
    print("  PASS")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"\n{name}:")
            fn()
    print("\nAll tests passed!")
