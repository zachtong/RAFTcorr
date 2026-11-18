"""Tests for full-image coordinate accumulation.

Verifies that accumulate_displacement works correctly with full-image-sized
arrays where NaN marks regions outside the valid ROI.
"""

import numpy as np
import pytest

from raft_dic_gui.incremental import accumulate_displacement


class TestFullImageAccumulation:

    def test_accumulate_with_different_valid_regions(self):
        """Accumulation works when u_prev and delta_u have different valid regions."""
        H, W = 100, 100

        # u_prev: valid in [20:60, 20:60], uniform 10px rightward
        u_prev = np.full((H, W, 2), np.nan)
        u_prev[20:60, 20:60, 0] = 10.0
        u_prev[20:60, 20:60, 1] = 0.0

        # delta_u: valid in [20:60, 30:70] (shifted right by 10px)
        delta_u = np.full((H, W, 2), np.nan)
        delta_u[20:60, 30:70, 0] = 5.0
        delta_u[20:60, 30:70, 1] = 0.0

        result = accumulate_displacement(u_prev, delta_u, debug=False)

        # For pixel (y=30, x=30) in u_prev:
        #   deformed coord = (30 + 0, 30 + 10) = (y=30, x=40)
        #   delta_u[30, 40] = (5.0, 0.0)  (in the valid region [30:70])
        #   total = (10 + 5, 0 + 0) = (15, 0)
        assert not np.isnan(result[30, 30, 0])
        np.testing.assert_allclose(result[30, 30, 0], 15.0, atol=0.1)
        np.testing.assert_allclose(result[30, 30, 1], 0.0, atol=0.1)

        # Pixel (y=30, x=20) in u_prev:
        #   deformed coord = (30 + 0, 20 + 10) = (y=30, x=30)
        #   delta_u[30, 30] = (5.0, 0.0)
        #   total = (10 + 5, 0 + 0) = (15, 0)
        assert not np.isnan(result[30, 20, 0])
        np.testing.assert_allclose(result[30, 20, 0], 15.0, atol=0.1)

    def test_first_segment_full_image(self):
        """First segment: accumulated is zeros inside ROI, NaN outside. Result = delta."""
        H, W = 80, 80

        u_prev = np.full((H, W, 2), np.nan)
        u_prev[10:50, 10:50, :] = 0.0  # ROI with zero displacement

        delta_u = np.full((H, W, 2), np.nan)
        delta_u[10:50, 10:50, 0] = 3.0
        delta_u[10:50, 10:50, 1] = 2.0

        result = accumulate_displacement(u_prev, delta_u, debug=False)

        # When accumulated is zero, deformed coords = original coords
        # Sampling delta at original coords = delta itself
        # total = 0 + delta = delta
        np.testing.assert_allclose(result[30, 30, 0], 3.0, atol=0.01)
        np.testing.assert_allclose(result[30, 30, 1], 2.0, atol=0.01)

        # Outside ROI should remain NaN
        assert np.isnan(result[5, 5, 0])
        assert np.isnan(result[5, 5, 1])

    def test_nan_outside_roi_preserved(self):
        """NaN outside the ROI region is preserved through accumulation."""
        H, W = 60, 60

        u_prev = np.full((H, W, 2), np.nan)
        u_prev[15:45, 15:45, :] = 0.0

        delta_u = np.full((H, W, 2), np.nan)
        delta_u[15:45, 15:45, 0] = 1.0
        delta_u[15:45, 15:45, 1] = -1.0

        result = accumulate_displacement(u_prev, delta_u, debug=False)

        # Valid region should have correct values
        np.testing.assert_allclose(result[30, 30, 0], 1.0, atol=0.01)
        np.testing.assert_allclose(result[30, 30, 1], -1.0, atol=0.01)

        # Everything outside [15:45, 15:45] should be NaN
        assert np.all(np.isnan(result[:15, :, :]))
        assert np.all(np.isnan(result[45:, :, :]))
        assert np.all(np.isnan(result[:, :15, :]))
        assert np.all(np.isnan(result[:, 45:, :]))

    def test_accumulate_two_steps_full_image(self):
        """Two-step accumulation with full-image arrays gives correct total."""
        H, W = 100, 100

        # Step 1: uniform 5px rightward in ROI [20:80, 20:80]
        u_step1 = np.full((H, W, 2), np.nan)
        u_step1[20:80, 20:80, :] = 0.0

        delta1 = np.full((H, W, 2), np.nan)
        delta1[20:80, 20:80, 0] = 5.0
        delta1[20:80, 20:80, 1] = 0.0

        result1 = accumulate_displacement(u_step1, delta1, debug=False)
        np.testing.assert_allclose(result1[50, 50, 0], 5.0, atol=0.01)

        # Step 2: another 5px rightward
        # delta2 valid in a region that overlaps with deformed positions
        delta2 = np.full((H, W, 2), np.nan)
        delta2[20:80, 25:80, 0] = 5.0  # shifted right by 5 to cover deformed coords
        delta2[20:80, 25:80, 1] = 0.0

        result2 = accumulate_displacement(result1, delta2, debug=False)

        # For pixel (50, 50): deformed from step1 = (50, 55)
        # delta2[50, 55] = 5.0 → total = 5 + 5 = 10
        np.testing.assert_allclose(result2[50, 50, 0], 10.0, atol=0.1)
