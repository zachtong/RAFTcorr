"""Tests for deformed-view inverse mapping (server/deformed_warp.py)."""

import numpy as np
import pytest

from server.deformed_warp import compute_inverse_map, warp_data_inverse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROI_RECT = (100, 100, 400, 400)  # x0, y0, x1, y1
ROI_H, ROI_W = 300, 300
IMAGE_SHAPE = (512, 512)


def _uniform_disp(u_val: float, v_val: float = 0.0):
    U = np.full((ROI_H, ROI_W), u_val, dtype=np.float64)
    V = np.full((ROI_H, ROI_W), v_val, dtype=np.float64)
    return U, V


def _rotation_disp(roi_h, roi_w, angle_deg, mask):
    """Create rotation displacement field within a mask."""
    cx, cy = roi_w / 2.0, roi_h / 2.0
    theta = np.deg2rad(angle_deg)
    y, x = np.mgrid[0:roi_h, 0:roi_w]
    dx = x.astype(np.float64) - cx
    dy = y.astype(np.float64) - cy
    U = np.full((roi_h, roi_w), np.nan)
    V = np.full((roi_h, roi_w), np.nan)
    U[mask] = dx[mask] * (np.cos(theta) - 1) - dy[mask] * np.sin(theta)
    V[mask] = dx[mask] * np.sin(theta) + dy[mask] * (np.cos(theta) - 1)
    return U, V


def _circular_mask(roi_h, roi_w, radius):
    y, x = np.mgrid[0:roi_h, 0:roi_w]
    cx, cy = roi_w / 2.0, roi_h / 2.0
    return ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeInverseMap:
    """Test that the inverse map produces correct results for various displacements."""

    def test_zero_displacement(self):
        """Zero displacement: deformed ROI = reference ROI."""
        U, V = _uniform_disp(0.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        assert inv.validity_mask.sum() >= ROI_H * ROI_W * 0.99

    def test_small_translation(self):
        """Small uniform translation should produce ~100% coverage."""
        U, V = _uniform_disp(-5.0, 3.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        coverage = inv.validity_mask.sum() / (ROI_H * ROI_W)
        assert coverage > 0.99, f"Coverage {coverage:.1%} too low for small translation"

    def test_large_translation_full_coverage(self):
        """Large uniform translation (-60px) must still produce near-full coverage.

        This is the exact scenario that triggered the original bug: the
        fixed-point iteration's initial guess landed outside the ROI,
        causing the leading edge to be marked invalid.
        """
        U, V = _uniform_disp(-60.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        valid = int(inv.validity_mask.sum())
        expected = ROI_H * ROI_W
        coverage = valid / expected
        assert coverage > 0.999, (
            f"Expected >99.9% coverage, got {valid} / {expected} "
            f"({coverage:.1%})"
        )

    def test_large_positive_translation(self):
        """Large rightward translation (+60px)."""
        U, V = _uniform_disp(60.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        coverage = inv.validity_mask.sum() / (ROI_H * ROI_W)
        assert coverage > 0.99

    def test_diagonal_translation(self):
        """Large diagonal translation."""
        U, V = _uniform_disp(-40.0, -30.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        coverage = inv.validity_mask.sum() / (ROI_H * ROI_W)
        assert coverage > 0.99

    def test_deformed_bbox_position(self):
        """Output bounding box should follow the displacement."""
        U, V = _uniform_disp(-60.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        # Deformed left edge ≈ x0 + U = 100 + (-60) = 40
        assert inv.out_x0 <= 40
        # Deformed right edge ≈ x1 + U = 400 + (-60) = 340
        assert inv.out_x1 >= 340


class TestWarpDataInverse:
    """Test that data is correctly warped to deformed coordinates."""

    def test_uniform_value_preserved(self):
        """Uniform data should remain uniform after warping."""
        U, V = _uniform_disp(-60.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        data = np.full((ROI_H, ROI_W), 42.0)
        warped = warp_data_inverse(data, inv, IMAGE_SHAPE)

        valid = ~np.isnan(warped)
        coverage = valid.sum() / (ROI_H * ROI_W)
        assert coverage > 0.999, f"Coverage {coverage:.1%} too low"
        np.testing.assert_allclose(warped[valid], 42.0, atol=1e-6)

    def test_gradient_field_warped_correctly(self):
        """A horizontal gradient should shift spatially after warping."""
        U, V = _uniform_disp(-20.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        # Create gradient: value = column index
        data = np.tile(np.arange(ROI_W, dtype=np.float64), (ROI_H, 1))
        warped = warp_data_inverse(data, inv, IMAGE_SHAPE)

        # Check that the warped data at the center of the deformed ROI
        # corresponds to the correct reference column
        x0, y0 = ROI_RECT[0], ROI_RECT[1]
        center_def_x = x0 + ROI_W // 2 + int(-20)  # deformed center
        center_def_y = y0 + ROI_H // 2
        val = warped[center_def_y, center_def_x]
        expected_col = ROI_W // 2  # reference column index
        assert abs(val - expected_col) < 2.0, (
            f"Expected ~{expected_col}, got {val}"
        )

    def test_nan_in_data_preserved(self):
        """NaN regions in source data should remain NaN after warping."""
        U, V = _uniform_disp(-10.0)
        inv = compute_inverse_map(U, V, ROI_RECT, IMAGE_SHAPE)

        data = np.ones((ROI_H, ROI_W))
        data[100:200, 100:200] = np.nan  # hole in the middle

        warped = warp_data_inverse(data, inv, IMAGE_SHAPE)

        # The NaN hole should appear in the warped output (shifted by U)
        total_valid = np.sum(~np.isnan(warped))
        # Original valid: 300*300 - 100*100 = 80000
        assert total_valid < ROI_H * ROI_W
        assert total_valid > 0


class TestRotationInverseMap:
    """Verify inverse map handles large rotations (griddata approach)."""

    @pytest.mark.parametrize("angle", [10, 30, 45, 60, 90])
    def test_rotation_coverage(self, angle):
        """Warped data should retain >90% of valid pixels for any rotation."""
        roi_h, roi_w = 150, 150
        roi_rect = (25, 25, 175, 175)
        image_shape = (200, 200)
        radius = 60
        mask = _circular_mask(roi_h, roi_w, radius)

        U, V = _rotation_disp(roi_h, roi_w, angle, mask)
        data = np.full((roi_h, roi_w), np.nan)
        y, x = np.mgrid[0:roi_h, 0:roi_w]
        cx, cy = roi_w / 2.0, roi_h / 2.0
        data[mask] = np.sqrt((x[mask] - cx) ** 2 + (y[mask] - cy) ** 2)

        inv = compute_inverse_map(U, V, roi_rect, image_shape)
        warped = warp_data_inverse(data, inv, image_shape)

        coverage = np.isfinite(warped).sum() / mask.sum() * 100
        assert coverage > 90, f"Coverage {coverage:.1f}% < 90% at {angle} deg"

    @pytest.mark.parametrize("angle", [10, 30, 45, 60, 90])
    def test_rotation_accuracy(self, angle):
        """Rotation-invariant data (distance from center) has low warp error."""
        roi_h, roi_w = 150, 150
        roi_rect = (25, 25, 175, 175)
        image_shape = (200, 200)
        radius = 60
        mask = _circular_mask(roi_h, roi_w, radius)

        U, V = _rotation_disp(roi_h, roi_w, angle, mask)
        y, x = np.mgrid[0:roi_h, 0:roi_w]
        cx, cy = roi_w / 2.0, roi_h / 2.0
        data = np.full((roi_h, roi_w), np.nan)
        data[mask] = np.sqrt((x[mask] - cx) ** 2 + (y[mask] - cy) ** 2)

        inv = compute_inverse_map(U, V, roi_rect, image_shape)
        warped = warp_data_inverse(data, inv, image_shape)

        valid = np.isfinite(warped)
        assert valid.any(), f"No valid pixels at {angle} deg"

        full_cx = roi_rect[0] + cx
        full_cy = roi_rect[1] + cy
        yf, xf = np.where(valid)
        expected = np.sqrt((xf - full_cx) ** 2 + (yf - full_cy) ** 2)
        error = np.abs(warped[valid] - expected)
        assert error.mean() < 1.5, (
            f"Mean error {error.mean():.3f}px at {angle} deg"
        )
