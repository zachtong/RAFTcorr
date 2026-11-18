"""Tests for NaN-aware bilinear interpolation (raft_dic_gui/nan_interp.py)."""

import numpy as np
import pytest

from raft_dic_gui.nan_interp import bilinear_sample_nan_aware


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBilinearSampleNanAware:
    """Verify NaN-aware bilinear interpolation behaviour."""

    def test_integer_coords_exact_lookup(self):
        """Integer coords return exact pixel values."""
        field = np.array([[1.0, 2.0, 3.0],
                          [4.0, 5.0, 6.0],
                          [7.0, 8.0, 9.0]])
        x = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        y = np.array([0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
        result = bilinear_sample_nan_aware(field, x, y)
        expected = np.array([1.0, 2.0, 3.0, 7.0, 8.0, 9.0])
        np.testing.assert_allclose(result, expected)

    def test_fractional_coords_standard_bilinear(self):
        """Fractional coords on all-valid field match standard bilinear."""
        field = np.array([[1.0, 2.0],
                          [3.0, 4.0]])
        # Sample at center (0.5, 0.5) -> average of all 4 = 2.5
        x = np.array([0.5])
        y = np.array([0.5])
        result = bilinear_sample_nan_aware(field, x, y)
        np.testing.assert_allclose(result, [2.5])

        # Sample at (0.25, 0.75) -> bilinear:
        # w00 = 0.75*0.25 = 0.1875, w10 = 0.25*0.25 = 0.0625
        # w01 = 0.75*0.75 = 0.5625, w11 = 0.25*0.75 = 0.1875
        # val = 1*0.1875 + 2*0.0625 + 3*0.5625 + 4*0.1875 = 2.75
        x2 = np.array([0.25])
        y2 = np.array([0.75])
        result2 = bilinear_sample_nan_aware(field, x2, y2)
        np.testing.assert_allclose(result2, [2.75])

    def test_nan_neighbor_excluded_and_renormalized(self):
        """One NaN neighbor: remaining neighbors re-weighted.

        field = [[nan, 10],
                 [  0, 10]]
        sample at (0.5, 0.5):
        corners: (0,0)=nan, (1,0)=10, (0,1)=0, (1,1)=10
        weights: w00=0.25(invalid), w10=0.25, w01=0.25, w11=0.25
        valid sum = 0.75, weighted = 10*0.25 + 0*0.25 + 10*0.25 = 5.0
        result = 5.0 / 0.75 = 20/3
        """
        field = np.array([[np.nan, 10.0],
                          [0.0, 10.0]])
        x = np.array([0.5])
        y = np.array([0.5])
        result = bilinear_sample_nan_aware(field, x, y)
        np.testing.assert_allclose(result, [20.0 / 3.0], rtol=1e-10)

    def test_all_nan_neighbors_returns_nan(self):
        """All NaN -> result NaN."""
        field = np.full((3, 3), np.nan)
        x = np.array([1.0, 0.5])
        y = np.array([1.0, 0.5])
        result = bilinear_sample_nan_aware(field, x, y)
        assert np.all(np.isnan(result))

    def test_fully_out_of_bounds_returns_nan(self):
        """Coords fully outside image -> NaN."""
        field = np.ones((5, 5))
        x = np.array([-2.0, 10.0, 0.0, -1.5])
        y = np.array([0.0, 0.0, -2.0, 8.0])
        result = bilinear_sample_nan_aware(field, x, y)
        assert np.all(np.isnan(result))

    def test_edge_partial_out_of_bounds(self):
        """Coords near edge where x1 >= W, use valid neighbors only.

        field shape (2, 3):
        [[0, 1, 2],
         [3, 4, 5]]
        Sample at x=2.5, y=0.5:
        x0=2, x1=3 (OOB, W=3)
        y0=0, y1=1
        corners: (2,0)=2, (3,0)=OOB, (2,1)=5, (3,1)=OOB
        fx=0.5, fy=0.5
        w00 = 0.5*0.5 = 0.25 (valid, val=2)
        w10 = 0.5*0.5 = 0.25 (invalid)
        w01 = 0.5*0.5 = 0.25 (valid, val=5)
        w11 = 0.5*0.5 = 0.25 (invalid)
        valid sum = 0.5, weighted = 2*0.25 + 5*0.25 = 1.75
        result = 1.75 / 0.5 = 3.5
        """
        field = np.array([[0.0, 1.0, 2.0],
                          [3.0, 4.0, 5.0]])
        x = np.array([2.5])
        y = np.array([0.5])
        result = bilinear_sample_nan_aware(field, x, y)
        np.testing.assert_allclose(result, [3.5])

    def test_large_field_vectorized(self):
        """512x512 field, 10000 random coords, verify no NaN and correct shape."""
        rng = np.random.default_rng(42)
        field = rng.random((512, 512))
        x = rng.uniform(0, 510, size=10000)
        y = rng.uniform(0, 510, size=10000)
        result = bilinear_sample_nan_aware(field, x, y)
        assert result.shape == (10000,)
        assert not np.any(np.isnan(result))

    def test_nan_at_boundary_no_contamination(self):
        """Valid 4x4 block in center of 10x10 NaN field.

        Sampling at the boundary of the valid block should return a value
        derived only from valid pixels, NOT contaminated by NaN regions.

        field: 10x10, all NaN except rows 3-6, cols 3-6 set to row index.
        field[3,3:7] = 3, field[4,3:7] = 4, field[5,3:7] = 5, field[6,3:7] = 6

        Sample at x=2.7, y=5.0:
        x0=2, x1=3, fx=0.7, fy=0.0
        y0=5, y1=5 (fy=0)
        corners: (2,5)=NaN, (3,5)=5, (2,5)=NaN(same), (3,5)=5(same)
        w00 = (1-0.7)*(1-0.0) = 0.3 -> invalid (NaN)
        w10 = 0.7 * 1.0 = 0.7 -> valid (val=5)
        w01 = 0.3 * 0.0 = 0.0 -> zero weight anyway
        w11 = 0.7 * 0.0 = 0.0 -> zero weight anyway
        valid sum = 0.7, weighted = 5 * 0.7 = 3.5
        result = 3.5 / 0.7 = 5.0
        """
        field = np.full((10, 10), np.nan)
        for r in range(3, 7):
            field[r, 3:7] = float(r)

        x = np.array([2.7])
        y = np.array([5.0])
        result = bilinear_sample_nan_aware(field, x, y)
        np.testing.assert_allclose(result, [5.0], rtol=1e-10)
