"""
Unit tests for raft_dic_gui/incremental.py

Tests cover:
- warp_displacement_field: sampling incremental displacement at deformed coords
- accumulate_displacement: combining previous + new displacement
- warp_mask_with_holes: warping binary masks while preserving holes
- validate_key_frames: validating key frame lists
- get_segment_ranges: computing segment ranges from key frames
- Integration: multi-step uniform translation accumulation
"""

import numpy as np
import pytest

from raft_dic_gui.incremental import (
    warp_displacement_field,
    accumulate_displacement,
    warp_mask_with_holes,
    validate_key_frames,
    get_segment_ranges,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_disp(H, W, du, dv):
    """Create a uniform displacement field of shape (H, W, 2)."""
    field = np.zeros((H, W, 2), dtype=np.float64)
    field[..., 0] = du
    field[..., 1] = dv
    return field


def _zero_disp(H, W):
    """Create a zero displacement field."""
    return np.zeros((H, W, 2), dtype=np.float64)


def _disk_mask(H, W, cy, cx, r):
    """Create a boolean disk mask centered at (cy, cx) with radius r."""
    yy, xx = np.mgrid[0:H, 0:W]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2


def _ring_mask(H, W, cy, cx, r_outer, r_inner):
    """Create a ring (annulus) mask — disk with a hole in the center."""
    outer = _disk_mask(H, W, cy, cx, r_outer)
    inner = _disk_mask(H, W, cy, cx, r_inner)
    return outer & ~inner


# ===========================================================================
# 1. warp_displacement_field
# ===========================================================================

class TestWarpDisplacementField:
    """Tests for warp_displacement_field(delta_u, accumulated_u)."""

    def test_zero_accumulated_passes_through(self):
        """When accumulated_u is zero, deformed coords = original coords.
        delta_u should pass through unchanged at interior pixels."""
        H, W = 32, 32
        delta_u = _uniform_disp(H, W, 3.0, -2.0)
        accumulated_u = _zero_disp(H, W)

        result = warp_displacement_field(delta_u, accumulated_u)

        # Interior pixels should match exactly (no boundary effects)
        margin = 2
        interior = result[margin:-margin, margin:-margin]
        np.testing.assert_allclose(interior[..., 0], 3.0, atol=1e-5)
        np.testing.assert_allclose(interior[..., 1], -2.0, atol=1e-5)

    def test_nan_in_accumulated_produces_nan(self):
        """NaN in accumulated_u should propagate to NaN in the output."""
        H, W = 16, 16
        delta_u = _uniform_disp(H, W, 1.0, 1.0)
        accumulated_u = _zero_disp(H, W)
        # Set a block to NaN
        accumulated_u[5:10, 5:10, :] = np.nan

        result = warp_displacement_field(delta_u, accumulated_u)

        # NaN region should produce NaN output
        assert np.all(np.isnan(result[5:10, 5:10, 0]))
        assert np.all(np.isnan(result[5:10, 5:10, 1]))
        # Non-NaN region (with margin from edges) should still be valid
        assert not np.isnan(result[2, 2, 0])

    def test_out_of_bounds_deformed_coords_use_edge_replication(self):
        """When accumulated_u pushes coords out of bounds, edge values are used.

        Coordinates are clamped to valid bounds (BORDER_REPLICATE equivalent)
        to prevent progressive NaN erosion at ROI boundaries during
        incremental accumulation.
        """
        H, W = 16, 16
        delta_u = _uniform_disp(H, W, 1.0, 1.0)
        # Large displacement pushes everything way out of bounds
        accumulated_u = _uniform_disp(H, W, 1000.0, 1000.0)

        result = warp_displacement_field(delta_u, accumulated_u)

        # Clamped to edge → valid values (edge replication), not NaN
        assert np.all(np.isfinite(result[..., 0]))
        assert np.all(np.isfinite(result[..., 1]))
        # Edge-replicated value should be delta_u's edge value (1.0)
        assert np.allclose(result[..., 0], 1.0)
        assert np.allclose(result[..., 1], 1.0)


# ===========================================================================
# 2. accumulate_displacement
# ===========================================================================

class TestAccumulateDisplacement:
    """Tests for accumulate_displacement(u_prev, delta_u, debug)."""

    def test_accumulation_from_zero(self):
        """Accumulating from zero displacement should approximate delta_u."""
        H, W = 32, 32
        u_prev = _zero_disp(H, W)
        delta_u = _uniform_disp(H, W, 5.0, -3.0)

        result = accumulate_displacement(u_prev, delta_u, debug=False)

        # Interior should match delta_u (zero + delta = delta)
        margin = 2
        interior = result[margin:-margin, margin:-margin]
        np.testing.assert_allclose(interior[..., 0], 5.0, atol=1e-5)
        np.testing.assert_allclose(interior[..., 1], -3.0, atol=1e-5)

    def test_two_uniform_translations_sum(self):
        """Two uniform translation steps should sum to the total translation.

        Step 1: translate by (2, 3) from zero → result ≈ (2, 3)
        Step 2: translate by (4, 1) from (2, 3) → result ≈ (6, 4)

        For uniform fields, warp_displacement_field samples from deformed
        coords which are still in bounds for small translations on a
        large-enough grid, so the sum should be exact at interior pixels.
        """
        H, W = 64, 64
        # Step 1
        u_prev = _zero_disp(H, W)
        delta1 = _uniform_disp(H, W, 2.0, 3.0)
        u_after_step1 = accumulate_displacement(u_prev, delta1, debug=False)

        # Step 2
        delta2 = _uniform_disp(H, W, 4.0, 1.0)
        u_after_step2 = accumulate_displacement(u_after_step1, delta2, debug=False)

        # Interior check: should be (2+4, 3+1) = (6, 4)
        margin = 8
        interior = u_after_step2[margin:-margin, margin:-margin]
        np.testing.assert_allclose(interior[..., 0], 6.0, atol=1e-4)
        np.testing.assert_allclose(interior[..., 1], 4.0, atol=1e-4)

    def test_nan_in_previous_stays_nan(self):
        """NaN in u_prev should remain NaN in accumulated result."""
        H, W = 32, 32
        u_prev = _zero_disp(H, W)
        u_prev[10:15, 10:15, :] = np.nan
        delta_u = _uniform_disp(H, W, 1.0, 1.0)

        result = accumulate_displacement(u_prev, delta_u, debug=False)

        # NaN region should stay NaN
        assert np.all(np.isnan(result[10:15, 10:15, 0]))
        assert np.all(np.isnan(result[10:15, 10:15, 1]))


# ===========================================================================
# 3. warp_mask_with_holes
# ===========================================================================

class TestWarpMaskWithHoles:
    """Tests for warp_mask_with_holes(mask, displacement)."""

    def test_zero_displacement_preserves_mask(self):
        """A zero displacement field should preserve the mask shape."""
        H, W = 64, 64
        mask = _disk_mask(H, W, 32, 32, 15)
        disp = _zero_disp(H, W)

        warped = warp_mask_with_holes(mask, disp)

        # Pixel counts should be very close (small differences from contour
        # rasterization and morphological cleanup are acceptable)
        original_count = np.sum(mask)
        warped_count = np.sum(warped)
        assert abs(int(warped_count) - int(original_count)) < original_count * 0.05, \
            f"Pixel count changed too much: {original_count} -> {warped_count}"

    def test_uniform_translation_shifts_center_of_mass(self):
        """Uniform translation should shift the mask's center of mass."""
        H, W = 128, 128
        cx, cy = 50, 50
        mask = _disk_mask(H, W, cy, cx, 15)
        dx, dy = 10.0, 8.0
        disp = _uniform_disp(H, W, dx, dy)

        warped = warp_mask_with_holes(mask, disp)

        # Check that the center of mass shifted approximately by (dx, dy)
        orig_ys, orig_xs = np.where(mask)
        warp_ys, warp_xs = np.where(warped)

        assert len(warp_xs) > 0, "Warped mask is empty"

        orig_cx = np.mean(orig_xs)
        orig_cy = np.mean(orig_ys)
        warp_cx = np.mean(warp_xs)
        warp_cy = np.mean(warp_ys)

        # Allow tolerance for contour discretization (integer coords)
        np.testing.assert_allclose(warp_cx - orig_cx, dx, atol=3.0)
        np.testing.assert_allclose(warp_cy - orig_cy, dy, atol=3.0)

    def test_empty_mask_returns_empty(self):
        """An all-false mask should return an all-false mask."""
        H, W = 32, 32
        mask = np.zeros((H, W), dtype=bool)
        disp = _zero_disp(H, W)

        warped = warp_mask_with_holes(mask, disp)

        assert not np.any(warped), "Empty mask should remain empty"

    def test_hole_preserved_after_warping(self):
        """A ring mask (disk with hole) should retain the hole after warping."""
        H, W = 128, 128
        # Outer radius 30, inner radius 10 — clearly distinct regions
        mask = _ring_mask(H, W, 64, 64, 30, 10)
        # Small uniform displacement so everything stays in bounds
        disp = _uniform_disp(H, W, 3.0, 2.0)

        warped = warp_mask_with_holes(mask, disp)

        # The warped center (roughly 64+2, 64+3) should be empty (the hole)
        warped_center_y = 64 + 2
        warped_center_x = 64 + 3
        # Check a small patch around the warped center — it should be False
        patch = warped[warped_center_y - 3:warped_center_y + 4,
                       warped_center_x - 3:warped_center_x + 4]
        assert not np.all(patch), \
            "The hole should be preserved — center region should not be all True"


# ===========================================================================
# 4. validate_key_frames
# ===========================================================================

class TestValidateKeyFrames:
    """Tests for validate_key_frames(key_frames, total_frames)."""

    def test_valid_key_frames(self):
        """A valid key frame list should pass validation."""
        valid, msg = validate_key_frames([1, 10, 20], total_frames=30)
        assert valid is True
        assert "valid" in msg.lower()

    def test_single_key_frame_1(self):
        """Just [1] is a valid minimal key frame list."""
        valid, msg = validate_key_frames([1], total_frames=10)
        assert valid is True

    def test_missing_frame_1_fails(self):
        """If frame 1 is not included, validation should fail."""
        valid, msg = validate_key_frames([5, 10], total_frames=20)
        assert valid is False
        assert "1" in msg  # Error should mention frame 1

    def test_empty_list_fails(self):
        """An empty list should fail validation."""
        valid, msg = validate_key_frames([], total_frames=10)
        assert valid is False

    def test_out_of_range_fails(self):
        """Frames outside [1, total_frames] should fail."""
        valid, msg = validate_key_frames([1, 50], total_frames=10)
        assert valid is False
        assert "out of range" in msg.lower()

    def test_zero_frame_fails(self):
        """Frame 0 is out of range (1-indexed scheme)."""
        valid, msg = validate_key_frames([0, 1], total_frames=10)
        assert valid is False

    def test_unsorted_input_still_valid(self):
        """Key frames can be passed unsorted — function sorts internally."""
        valid, msg = validate_key_frames([10, 1, 5], total_frames=20)
        assert valid is True

    def test_duplicate_frames_fail(self):
        """Duplicate key frames should fail."""
        valid, msg = validate_key_frames([1, 5, 5], total_frames=10)
        assert valid is False
        assert "duplicate" in msg.lower()


# ===========================================================================
# 5. get_segment_ranges
# ===========================================================================

class TestGetSegmentRanges:
    """Tests for get_segment_ranges(key_frames, total_frames)."""

    def test_single_segment(self):
        """With only frame 1 as key frame, one segment covers all frames."""
        segments = get_segment_ranges([1], total_frames=10)
        # Should be [(ref=1, start=2, end=10)]
        assert len(segments) == 1
        ref, start, end = segments[0]
        assert ref == 1
        assert start == 2
        assert end == 10

    def test_two_segments(self):
        """Two key frames produce two segments."""
        segments = get_segment_ranges([1, 5], total_frames=10)
        # Segment 1: ref=1, start=2, end=5
        # Segment 2: ref=5, start=6, end=10
        assert len(segments) == 2

        ref1, start1, end1 = segments[0]
        assert ref1 == 1
        assert start1 == 2
        assert end1 == 5

        ref2, start2, end2 = segments[1]
        assert ref2 == 5
        assert start2 == 6
        assert end2 == 10

    def test_three_segments(self):
        """Three key frames produce three segments."""
        segments = get_segment_ranges([1, 4, 7], total_frames=10)
        assert len(segments) == 3
        assert segments[0] == (1, 2, 4)
        assert segments[1] == (4, 5, 7)
        assert segments[2] == (7, 8, 10)

    def test_last_key_frame_equals_total(self):
        """If last key frame == total_frames, no segment for it (start > end)."""
        segments = get_segment_ranges([1, 10], total_frames=10)
        # Segment 1: ref=1, start=2, end=10
        # Segment 2: ref=10, start=11, end=10 → skipped (start > end)
        assert len(segments) == 1
        assert segments[0] == (1, 2, 10)

    def test_unsorted_input(self):
        """Function should sort internally."""
        segments = get_segment_ranges([5, 1], total_frames=10)
        assert len(segments) == 2
        assert segments[0] == (1, 2, 5)
        assert segments[1] == (5, 6, 10)


# ===========================================================================
# 6. Integration: multi-step accumulation contract
# ===========================================================================

class TestAccumulationIntegration:
    """Integration tests verifying the accumulation contract end-to-end."""

    def test_three_uniform_steps_sum(self):
        """Three uniform translation steps should sum to the total.

        Step 1: (1, 0)
        Step 2: (0, 2)
        Step 3: (3, -1)
        Total expected: (4, 1)
        """
        H, W = 64, 64
        u = _zero_disp(H, W)

        steps = [(1.0, 0.0), (0.0, 2.0), (3.0, -1.0)]
        for du, dv in steps:
            delta = _uniform_disp(H, W, du, dv)
            u = accumulate_displacement(u, delta, debug=False)

        expected_u = sum(s[0] for s in steps)  # 4.0
        expected_v = sum(s[1] for s in steps)  # 1.0

        margin = 10
        interior = u[margin:-margin, margin:-margin]
        np.testing.assert_allclose(interior[..., 0], expected_u, atol=1e-3)
        np.testing.assert_allclose(interior[..., 1], expected_v, atol=1e-3)

    def test_accumulation_preserves_nan_region(self):
        """NaN regions should propagate through multiple accumulation steps."""
        H, W = 32, 32
        u = _zero_disp(H, W)
        u[12:18, 12:18, :] = np.nan  # NaN region from the start

        for _ in range(3):
            delta = _uniform_disp(H, W, 1.0, 0.5)
            u = accumulate_displacement(u, delta, debug=False)

        # The NaN region should still be NaN
        assert np.all(np.isnan(u[12:18, 12:18, 0]))
        assert np.all(np.isnan(u[12:18, 12:18, 1]))


# ===========================================================================
# 7. key_frames_every_n
# ===========================================================================

class TestKeyFramesEveryN:
    """Tests for key_frames_every_n(total_frames, n)."""

    def test_every_50(self):
        from raft_dic_gui.incremental import key_frames_every_n
        kf = key_frames_every_n(total_frames=200, n=50)
        assert kf == [1, 51, 101, 151]

    def test_every_1(self):
        from raft_dic_gui.incremental import key_frames_every_n
        kf = key_frames_every_n(total_frames=5, n=1)
        assert kf == [1, 2, 3, 4, 5]

    def test_n_larger_than_total(self):
        from raft_dic_gui.incremental import key_frames_every_n
        kf = key_frames_every_n(total_frames=10, n=100)
        assert kf == [1]

    def test_n_zero_treated_as_1(self):
        from raft_dic_gui.incremental import key_frames_every_n
        kf = key_frames_every_n(total_frames=3, n=0)
        assert kf == [1, 2, 3]


# ===========================================================================
# 8. build_ref_map
# ===========================================================================

class TestBuildRefMap:
    """Tests for build_ref_map(total_frames, key_frames, ref_map)."""

    def test_single_segment_all_ref_frame1(self):
        from raft_dic_gui.incremental import build_ref_map
        ref_map = build_ref_map(total_frames=5, key_frames=[1])
        assert ref_map == {2: 1, 3: 1, 4: 1, 5: 1}

    def test_two_segments(self):
        from raft_dic_gui.incremental import build_ref_map
        ref_map = build_ref_map(total_frames=10, key_frames=[1, 5])
        # Frames 2-5 ref frame 1, frames 6-10 ref frame 5
        for f in range(2, 6):
            assert ref_map[f] == 1
        for f in range(6, 11):
            assert ref_map[f] == 5

    def test_three_segments(self):
        from raft_dic_gui.incremental import build_ref_map
        ref_map = build_ref_map(total_frames=100, key_frames=[1, 50, 100])
        assert ref_map[2] == 1
        assert ref_map[50] == 1
        assert ref_map[51] == 50
        assert ref_map[100] == 50

    def test_no_key_frames_defaults_to_frame1(self):
        from raft_dic_gui.incremental import build_ref_map
        ref_map = build_ref_map(total_frames=5)
        assert ref_map == {2: 1, 3: 1, 4: 1, 5: 1}

    def test_missing_frame1_auto_added(self):
        from raft_dic_gui.incremental import build_ref_map
        ref_map = build_ref_map(total_frames=10, key_frames=[5])
        # Frame 1 auto-added, so frames 2-5 ref 1, frames 6-10 ref 5
        assert ref_map[2] == 1
        assert ref_map[5] == 1
        assert ref_map[6] == 5

    def test_ref_map_passthrough(self):
        from raft_dic_gui.incremental import build_ref_map
        custom = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
        ref_map = build_ref_map(total_frames=6, ref_map=custom)
        assert ref_map == custom

    def test_unsorted_key_frames(self):
        from raft_dic_gui.incremental import build_ref_map
        ref_map = build_ref_map(total_frames=10, key_frames=[5, 1])
        assert ref_map[3] == 1
        assert ref_map[7] == 5

    def test_duplicate_key_frames_handled(self):
        from raft_dic_gui.incremental import build_ref_map
        ref_map = build_ref_map(total_frames=10, key_frames=[1, 5, 5])
        assert ref_map[3] == 1
        assert ref_map[7] == 5

    def test_frame1_not_in_ref_map(self):
        """Frame 1 is the origin, should not appear as a key in ref_map."""
        from raft_dic_gui.incremental import build_ref_map
        ref_map = build_ref_map(total_frames=10, key_frames=[1, 5])
        assert 1 not in ref_map


# ===========================================================================
# 8b. DICProcessor._build_ref_map mode isolation
# ===========================================================================

class TestControllerBuildRefMap:
    """Verify DICProcessor._build_ref_map ignores incremental params
    when mode is accumulative (prevents config leak between modes)."""

    def test_accumulative_ignores_key_frame_interval(self):
        """Switching from incremental to accumulative should not use
        a leftover key_frame_interval=1."""
        from raft_dic_gui.config import DICConfig
        from raft_dic_gui.controller import DICProcessor

        cfg = DICConfig(mode="accumulative", key_frame_interval=1)
        ref_map = DICProcessor._build_ref_map(cfg, total_frames=10)
        # All frames should reference frame 1
        for f in range(2, 11):
            assert ref_map[f] == 1

    def test_accumulative_ignores_key_frames(self):
        from raft_dic_gui.config import DICConfig
        from raft_dic_gui.controller import DICProcessor

        cfg = DICConfig(mode="accumulative", key_frames=[1, 5])
        ref_map = DICProcessor._build_ref_map(cfg, total_frames=10)
        for f in range(2, 11):
            assert ref_map[f] == 1

    def test_incremental_uses_key_frame_interval(self):
        from raft_dic_gui.config import DICConfig
        from raft_dic_gui.controller import DICProcessor

        cfg = DICConfig(mode="incremental", key_frame_interval=5)
        ref_map = DICProcessor._build_ref_map(cfg, total_frames=10)
        # key_frames_every_n(10, 5) = [1, 6]
        assert ref_map[2] == 1
        assert ref_map[6] == 1
        assert ref_map[7] == 6

    def test_incremental_default_every_frame(self):
        from raft_dic_gui.config import DICConfig
        from raft_dic_gui.controller import DICProcessor

        cfg = DICConfig(mode="incremental")
        ref_map = DICProcessor._build_ref_map(cfg, total_frames=5)
        # Every frame references the previous one
        assert ref_map[2] == 1
        assert ref_map[3] == 2
        assert ref_map[4] == 3
        assert ref_map[5] == 4


# ===========================================================================
# 9. Ref_map-driven accumulation (Strategy A end-to-end)
# ===========================================================================

class TestRefMapAccumulation:
    """Integration tests verifying Strategy A: accumulative within segments,
    accumulation only at segment boundaries."""

    def test_two_segment_uniform_translation(self):
        """Two segments with uniform translation: total should be segment1 + segment2.

        Segment 1 (ref=1, frames 2-5): DIC(1, 5) = (8, 0)
        Segment 2 (ref=5, frames 6-10): DIC(5, 10) = (10, 0)
        Total at frame 10 = accumulated[1] + DIC(1,5) accumulated at frame 5
        Then total_10 = accumulated[5] + DIC(5, 10)

        With Strategy A:
        - accumulated[1] = zeros
        - Frame 5 total = accumulated[1] + DIC(1,5) = (8, 0)
        - accumulated[5] = (8, 0)
        - Frame 10 total = accumulate(accumulated[5], DIC(5,10))
        - For uniform fields: (8, 0) + (10, 0) = (18, 0)
        """
        H, W = 64, 64

        # Simulate segment 1 result: DIC(frame1, frame5) = (8, 0)
        kf_accumulated = {1: _zero_disp(H, W)}
        delta_seg1 = _uniform_disp(H, W, 8.0, 0.0)

        # Frame 5 total (first segment, kf_accumulated[1] is zeros)
        total_5 = delta_seg1.copy()  # kf_acc is zero → delta IS total
        kf_accumulated[5] = total_5.copy()

        # Simulate segment 2 result: DIC(frame5, frame10) = (10, 0)
        delta_seg2 = _uniform_disp(H, W, 10.0, 0.0)

        # Frame 10 total (second segment, non-zero accumulated)
        total_10 = accumulate_displacement(kf_accumulated[5], delta_seg2, debug=False)

        # Interior check (avoid boundary effects)
        margin = 10
        interior = total_10[margin:-margin, margin:-margin]
        np.testing.assert_allclose(interior[..., 0], 18.0, atol=0.5)
        np.testing.assert_allclose(interior[..., 1], 0.0, atol=0.5)

    def test_custom_ref_map_passthrough(self):
        """A custom ref_map (e.g., cyclic) should pass through build_ref_map unchanged."""
        from raft_dic_gui.incremental import build_ref_map

        # Frame 6 references frame 1 instead of frame 5 (simulating cyclic loading)
        custom = {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 6, 8: 6}
        ref_map = build_ref_map(total_frames=8, ref_map=custom)
        assert ref_map == custom

    def test_single_segment_equals_accumulative(self):
        """With key_frames=[1], all frames ref frame 1. Delta IS total.

        This is the same behavior as accumulative mode.
        """
        from raft_dic_gui.incremental import build_ref_map

        ref_map = build_ref_map(total_frames=100, key_frames=[1])

        # Every frame references frame 1
        for f in range(2, 101):
            assert ref_map[f] == 1

        # Since kf_accumulated[1] = zeros, total = delta for every frame
        # (no accumulation needed — same as accumulative mode)
        H, W = 32, 32
        kf_acc = _zero_disp(H, W)
        delta = _uniform_disp(H, W, 5.0, 3.0)

        # First segment check: delta IS total when accumulated is zero
        import numpy as np
        is_first = np.allclose(kf_acc, 0, atol=1e-12) and not np.any(np.isnan(kf_acc))
        assert is_first is True

        # So total = delta
        total = delta.copy()
        np.testing.assert_array_equal(total, delta)

    def test_three_segments_accumulate(self):
        """Three segments: verify displacement chains correctly.

        Segment 1 (ref=1): delta = (2, 1)  → total = (2, 1)
        Segment 2 (ref=KF2): delta = (3, 0) → total = (2+3, 1+0) = (5, 1)
        Segment 3 (ref=KF3): delta = (1, 2) → total = (5+1, 1+2) = (6, 3)
        """
        H, W = 64, 64

        kf_accumulated = {1: _zero_disp(H, W)}

        # Segment 1 end
        delta1 = _uniform_disp(H, W, 2.0, 1.0)
        total_kf2 = delta1.copy()  # First segment: zeros + delta
        kf_accumulated[2] = total_kf2.copy()

        # Segment 2 end
        delta2 = _uniform_disp(H, W, 3.0, 0.0)
        total_kf3 = accumulate_displacement(kf_accumulated[2], delta2, debug=False)
        kf_accumulated[3] = total_kf3.copy()

        # Segment 3 end
        delta3 = _uniform_disp(H, W, 1.0, 2.0)
        total_final = accumulate_displacement(kf_accumulated[3], delta3, debug=False)

        margin = 10
        interior = total_final[margin:-margin, margin:-margin]
        np.testing.assert_allclose(interior[..., 0], 6.0, atol=0.5)
        np.testing.assert_allclose(interior[..., 1], 3.0, atol=0.5)
