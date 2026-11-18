"""Test: per-frame mask coverage in incremental mode + deformed view.

Scenario: a cracking specimen with 4 images (I1-I4) and 4 masks (M1-M4).
M1 is full specimen, M2 has a crack (smaller), M3/M4 have bigger cracks.

We test both the old problem (coverage exceeding specimen) and the fix
(_apply_current_frame_mask removing crack-region pixels).
"""

import numpy as np
import pytest

from raft_dic_gui.controller import DICProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mask(h, w, valid_rows, valid_cols):
    """Create a boolean mask with True in the specified row/col range."""
    mask = np.zeros((h, w), dtype=bool)
    mask[valid_rows[0]:valid_rows[1], valid_cols[0]:valid_cols[1]] = True
    return mask


def simulate_dic_result(ref_mask, h, w, u=2.0, v=1.0):
    """Simulate dic_over_roi_with_tiling output: displacement with NaN outside mask."""
    disp = np.full((h, w, 2), np.nan)
    disp[ref_mask, 0] = u
    disp[ref_mask, 1] = v
    return disp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCurrentFrameMaskApplication:
    """Test _apply_current_frame_mask removes crack-region pixels."""

    H, W = 100, 100

    # M1: full specimen (rows 10-90, cols 10-90)  = 6400 px
    # M2: crack appeared (rows 10-90, cols 10-70) = 4800 px
    # M3: crack bigger   (rows 10-90, cols 10-50) = 3200 px
    # M4: crack even bigger (rows 10-90, cols 10-40) = 2400 px
    M1 = make_mask(100, 100, (10, 90), (10, 90))
    M2 = make_mask(100, 100, (10, 90), (10, 70))
    M3 = make_mask(100, 100, (10, 90), (10, 50))
    M4 = make_mask(100, 100, (10, 90), (10, 40))

    def test_mask_sizes_decreasing(self):
        """Sanity: masks get progressively smaller (cracking specimen)."""
        assert self.M1.sum() > self.M2.sum() > self.M3.sum() > self.M4.sum()

    # ------------------------------------------------------------------
    # Problem demonstration (without the fix)
    # ------------------------------------------------------------------

    def test_without_fix_coverage_exceeds_specimen(self):
        """Without current-frame masking, result covers crack region."""
        delta_12 = simulate_dic_result(self.M1, self.H, self.W)
        # No _apply_current_frame_mask called
        valid = ~np.isnan(delta_12[..., 0])

        # Coverage = M1 (6400 px), but specimen in I2 = M2 (4800 px)
        assert valid.sum() == self.M1.sum()
        assert valid.sum() > self.M2.sum(), \
            "Without fix: result extends into crack region of frame 2"

    # ------------------------------------------------------------------
    # Fix verification
    # ------------------------------------------------------------------

    def test_apply_current_frame_mask_removes_crack_pixels(self):
        """_apply_current_frame_mask removes pixels landing outside M2."""
        # Pair I1→I2: delta has M1 coverage, small displacement (2px, 1px)
        delta = simulate_dic_result(self.M1, self.H, self.W, u=2.0, v=1.0)

        # Apply M2 (current frame's mask)
        DICProcessor._apply_current_frame_mask(delta, self.M2, frame_num=2)

        valid = ~np.isnan(delta[..., 0])

        # Pixels in M1's right edge (cols 70-90) displaced by u=2
        # land at cols 72-92 in I2 → outside M2 (cols 10-70) → should be NaN
        crack_region = self.M1 & ~self.M2  # cols 70-90
        assert not valid[crack_region].any(), \
            "Crack-region pixels should be NaN after current-frame masking"

        # Valid pixels should be a SUBSET of M2
        # (pixels in M1 whose deformed position lands inside M2)
        assert valid.sum() < self.M1.sum(), \
            "Coverage should be smaller than M1 after masking"
        assert valid.sum() <= self.M2.sum(), \
            "Coverage should not exceed M2 (specimen boundary in frame 2)"

    def test_apply_current_frame_mask_zero_displacement(self):
        """With zero displacement, result coverage = M1 ∩ M2 exactly."""
        delta = simulate_dic_result(self.M1, self.H, self.W, u=0.0, v=0.0)
        DICProcessor._apply_current_frame_mask(delta, self.M2, frame_num=2)

        valid = ~np.isnan(delta[..., 0])
        expected = self.M1 & self.M2  # = M2 (since M2 ⊂ M1)
        assert np.array_equal(valid, expected), \
            "With zero displacement, coverage = M1 ∩ M2"

    def test_full_pipeline_simulation_3_frames(self):
        """Simulate 3 pairs with progressive cracking.

        Pair I1→I2: ref_mask=M1, cur_mask=M2
        Pair I2→I3: ref_mask=M2, cur_mask=M3
        Pair I3→I4: ref_mask=M3, cur_mask=M4

        With the fix, each result's coverage should not exceed
        the current frame's specimen boundary.
        """
        # Pair I1→I2
        delta_12 = simulate_dic_result(self.M1, self.H, self.W, u=0.0, v=0.0)
        DICProcessor._apply_current_frame_mask(delta_12, self.M2, frame_num=2)
        total_frame2 = delta_12.copy()

        valid_2 = ~np.isnan(total_frame2[..., 0])
        assert valid_2.sum() <= self.M2.sum(), \
            f"Frame 2 coverage ({valid_2.sum()}) should not exceed M2 ({self.M2.sum()})"

        # Pair I2→I3
        delta_23 = simulate_dic_result(self.M2, self.H, self.W, u=0.0, v=0.0)
        DICProcessor._apply_current_frame_mask(delta_23, self.M3, frame_num=3)
        # Simplified accumulation: NaN + anything = NaN
        total_frame3 = total_frame2 + delta_23

        valid_3 = ~np.isnan(total_frame3[..., 0])
        assert valid_3.sum() <= self.M3.sum(), \
            f"Frame 3 coverage ({valid_3.sum()}) should not exceed M3 ({self.M3.sum()})"

        # Pair I3→I4
        delta_34 = simulate_dic_result(self.M3, self.H, self.W, u=0.0, v=0.0)
        DICProcessor._apply_current_frame_mask(delta_34, self.M4, frame_num=4)
        total_frame4 = total_frame3 + delta_34

        valid_4 = ~np.isnan(total_frame4[..., 0])
        assert valid_4.sum() <= self.M4.sum(), \
            f"Frame 4 coverage ({valid_4.sum()}) should not exceed M4 ({self.M4.sum()})"

        print(f"\nFrame 2: {valid_2.sum()} px <= M2 {self.M2.sum()} px OK")
        print(f"Frame 3: {valid_3.sum()} px <= M3 {self.M3.sum()} px OK")
        print(f"Frame 4: {valid_4.sum()} px <= M4 {self.M4.sum()} px OK")

    def test_backward_compatible_no_current_mask(self):
        """If no current-frame mask exists, behavior is unchanged."""
        delta = simulate_dic_result(self.M1, self.H, self.W)
        original_valid_count = (~np.isnan(delta[..., 0])).sum()

        # user_masks only has reference frame masks, no current frame mask
        user_masks = {0: self.M1, 1: self.M2}
        # list_idx=2 (frame 3) not in user_masks → no current-frame masking
        list_idx = 2
        if list_idx in user_masks:
            DICProcessor._apply_current_frame_mask(delta, user_masks[list_idx], 3)

        valid_after = (~np.isnan(delta[..., 0])).sum()
        assert valid_after == original_valid_count, \
            "No current-frame mask → coverage unchanged"

    def test_deformed_view_coverage_with_fix(self):
        """After fix, deformed view coverage matches current specimen."""
        # Pair I1→I2 with both masks applied
        delta = simulate_dic_result(self.M1, self.H, self.W, u=0.0, v=0.0)
        DICProcessor._apply_current_frame_mask(delta, self.M2, frame_num=2)

        # Deformed view validity (same logic as deformed_warp.py)
        U, V = delta[..., 0], delta[..., 1]
        valid_mask = ~(np.isnan(U) | np.isnan(V))

        # Should now match M2, not M1
        assert valid_mask.sum() <= self.M2.sum(), \
            "Deformed view coverage should match current frame's specimen boundary"
        assert not np.array_equal(valid_mask, self.M1), \
            "Should NOT match the reference mask M1 (too large)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
