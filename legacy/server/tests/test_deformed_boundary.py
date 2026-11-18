"""Test that deformed view boundaries are smooth (no coarse staircase artifacts).

Regression test for the bug where the coarse query grid (~50 pts per dim)
produced staircase artifacts at the ROI boundary in deformed view.
The upsampled validity mask threshold of 0.5 created ~20px blocks.
"""

import numpy as np
import pytest

from server.deformed_warp import compute_inverse_map, warp_data_inverse


class TestBoundarySmoothnessUniform:
    """Test boundary smoothness with uniform translation (no shape change)."""

    @pytest.fixture
    def uniform_translation_setup(self):
        """Create a large ROI with 10px rightward uniform translation."""
        roi_h, roi_w = 400, 200
        U = np.full((roi_h, roi_w), 10.0)  # 10px rightward
        V = np.zeros((roi_h, roi_w))
        roi_rect = (100, 50, 300, 450)  # (x0, y0, x1, y1)
        image_shape = (600, 500)
        return U, V, roi_rect, image_shape

    def test_valid_pixel_count_near_expected(self, uniform_translation_setup):
        """Valid pixel count should be close to ROI area for uniform translation."""
        U, V, roi_rect, image_shape = uniform_translation_setup
        roi_h, roi_w = 400, 200

        inv_map = compute_inverse_map(U, V, roi_rect, image_shape)
        data = np.ones((roi_h, roi_w))  # uniform value 1.0
        warped = warp_data_inverse(data, inv_map, image_shape)

        valid_count = np.sum(np.isfinite(warped))
        expected = roi_h * roi_w
        # Allow up to 5% loss from boundary effects
        assert valid_count > expected * 0.95, (
            f"Too many pixels lost: {valid_count}/{expected} "
            f"({valid_count / expected:.1%})"
        )

    def test_boundary_not_staircased(self, uniform_translation_setup):
        """The right edge of the warped region should be smooth, not staircase.

        For a uniform rightward translation, the right boundary should be a
        near-vertical line (within 1-2px), not have ~20px staircase blocks.
        """
        U, V, roi_rect, image_shape = uniform_translation_setup
        roi_h, roi_w = 400, 200

        inv_map = compute_inverse_map(U, V, roi_rect, image_shape)
        data = np.ones((roi_h, roi_w))
        warped = warp_data_inverse(data, inv_map, image_shape)

        # For each row, find the rightmost valid pixel
        valid = np.isfinite(warped)
        right_edges = []
        for row in range(inv_map.out_y0, inv_map.out_y1):
            cols = np.where(valid[row, :])[0]
            if len(cols) > 0:
                right_edges.append(cols[-1])

        right_edges = np.array(right_edges)
        if len(right_edges) < 10:
            pytest.skip("Not enough rows with valid data")

        # For uniform translation, all right edges should be within a few px
        # of each other. A staircase would show ~20px jumps.
        edge_range = right_edges.max() - right_edges.min()
        assert edge_range <= 3, (
            f"Right boundary has {edge_range}px range, expected <= 3px. "
            f"This indicates staircase artifacts."
        )

    def test_boundary_not_staircased_left(self, uniform_translation_setup):
        """Left edge should also be smooth."""
        U, V, roi_rect, image_shape = uniform_translation_setup
        roi_h, roi_w = 400, 200

        inv_map = compute_inverse_map(U, V, roi_rect, image_shape)
        data = np.ones((roi_h, roi_w))
        warped = warp_data_inverse(data, inv_map, image_shape)

        valid = np.isfinite(warped)
        left_edges = []
        for row in range(inv_map.out_y0, inv_map.out_y1):
            cols = np.where(valid[row, :])[0]
            if len(cols) > 0:
                left_edges.append(cols[0])

        left_edges = np.array(left_edges)
        if len(left_edges) < 10:
            pytest.skip("Not enough rows with valid data")

        edge_range = left_edges.max() - left_edges.min()
        assert edge_range <= 3, (
            f"Left boundary has {edge_range}px range, expected <= 3px. "
            f"This indicates staircase artifacts."
        )


class TestBoundarySmoothnessCircularROI:
    """Test that circular ROI boundary warps smoothly."""

    def test_circular_roi_boundary_smooth(self):
        """A circular ROI with uniform translation should produce a smooth circle."""
        roi_h, roi_w = 300, 300
        # Create circular valid region (rest is NaN)
        cy, cx = roi_h // 2, roi_w // 2
        yy, xx = np.mgrid[:roi_h, :roi_w]
        radius = 120
        circle_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2

        U = np.full((roi_h, roi_w), 5.0)
        V = np.full((roi_h, roi_w), -3.0)
        # Mark outside circle as NaN (no displacement data)
        U[~circle_mask] = np.nan
        V[~circle_mask] = np.nan

        roi_rect = (50, 50, 350, 350)
        image_shape = (500, 500)

        inv_map = compute_inverse_map(U, V, roi_rect, image_shape)

        # Source data: 1.0 inside circle, NaN outside
        data = np.full((roi_h, roi_w), np.nan)
        data[circle_mask] = 1.0

        warped = warp_data_inverse(data, inv_map, image_shape)
        valid = np.isfinite(warped)

        # Check the equatorial band (middle 50% of circle) where the
        # boundary slope is gentle — exclude polar regions where dx/dy
        # is naturally large (up to sqrt(R/2) ~ 8px per row).
        center_y = cy + roi_rect[1] + int(V[cy, cx])
        band_start = center_y - radius // 2
        band_end = center_y + radius // 2
        right_edges = []
        for row in range(max(band_start, 0), min(band_end, image_shape[0])):
            cols = np.where(valid[row, :])[0]
            if len(cols) > 5:
                right_edges.append(cols[-1])

        right_edges = np.array(right_edges)
        if len(right_edges) < 20:
            pytest.skip("Not enough rows")

        # Boundary jumps should be <= 5px in the equatorial band.
        # Before the forward-contour fix, the coarse Delaunay hull
        # produced ~20px staircase blocks.
        diffs = np.abs(np.diff(right_edges))
        max_jump = diffs.max()
        assert max_jump <= 5, (
            f"Max boundary jump is {max_jump}px between consecutive rows "
            f"in equatorial band. Expected <= 5px. "
            f"This indicates staircase artifacts from coarse grid."
        )
