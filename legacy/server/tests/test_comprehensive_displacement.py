"""Comprehensive displacement field tests covering complex ROI shapes,
large deformations, non-uniform fields, multi-step accumulation,
and boundary pixel loss scenarios.
"""

import numpy as np
import pytest
from raft_dic_gui.nan_interp import bilinear_sample_nan_aware
from raft_dic_gui.incremental import (
    warp_displacement_field,
    accumulate_displacement,
    warp_mask_with_holes,
    median_filter_displacement,
)
from raft_dic_gui.controller import DICProcessor


# ── Complex ROI shapes ──────────────────────────────────────


class TestComplexROIShapes:

    def test_L_shaped_roi_uniform_translation(self):
        """L-shaped mask should shift by exact displacement amount."""
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=bool)
        mask[10:60, 10:40] = True   # vertical arm
        mask[40:60, 10:70] = True   # horizontal arm
        orig_area = mask.sum()

        disp = np.full((H, W, 2), np.nan)
        disp[mask, 0] = 12.0
        disp[mask, 1] = -5.0

        warped = warp_mask_with_holes(mask, disp)
        orig_cx = np.where(mask)[1].mean()
        orig_cy = np.where(mask)[0].mean()
        warp_cx = np.where(warped)[1].mean()
        warp_cy = np.where(warped)[0].mean()

        np.testing.assert_allclose(warp_cx - orig_cx, 12.0, atol=3.0)
        np.testing.assert_allclose(warp_cy - orig_cy, -5.0, atol=3.0)
        assert abs(warped.sum() - orig_area) / orig_area < 0.15

    def test_annular_roi_preserves_hole(self):
        """Ring-shaped (annular) mask should preserve central hole after warp."""
        H, W = 120, 120
        yy, xx = np.mgrid[:H, :W]
        r = np.sqrt((xx - 60.0)**2 + (yy - 60.0)**2)
        mask = (r >= 15) & (r <= 40)

        disp = np.full((H, W, 2), np.nan)
        disp[mask, 0] = 20.0
        disp[mask, 1] = 10.0

        warped = warp_mask_with_holes(mask, disp)
        new_cx = np.where(warped)[1].mean()
        new_cy = np.where(warped)[0].mean()

        np.testing.assert_allclose(new_cx, 80.0, atol=5.0)
        np.testing.assert_allclose(new_cy, 70.0, atol=5.0)
        # Center of warped ring should be hollow
        assert not warped[70, 80], "Ring center should remain hollow"

    def test_multi_region_disconnected_mask(self):
        """Two separate regions should warp independently."""
        H, W = 100, 200
        mask = np.zeros((H, W), dtype=bool)
        mask[10:40, 10:50] = True    # Region A
        mask[60:90, 120:170] = True  # Region B

        disp = np.full((H, W, 2), np.nan)
        disp[10:40, 10:50, 0] = 15.0   # A moves right
        disp[10:40, 10:50, 1] = 0.0
        disp[60:90, 120:170, 0] = -10.0  # B moves left
        disp[60:90, 120:170, 1] = 5.0

        warped = warp_mask_with_holes(mask, disp)

        # Region A: center shifts from x=30 to x=45
        regionA_orig_xs = np.where(mask[10:40, :])[1]
        regionA_warp_xs = np.where(warped[10:40, :])[1]
        dx_A = regionA_warp_xs.mean() - regionA_orig_xs.mean()
        np.testing.assert_allclose(dx_A, 15.0, atol=3.0)

        # Total area preserved
        assert abs(warped.sum() - mask.sum()) / mask.sum() < 0.10

    def test_thin_strip_roi(self):
        """Thin horizontal strip (3px tall) should survive warp."""
        H, W = 50, 100
        mask = np.zeros((H, W), dtype=bool)
        mask[24:27, 10:90] = True  # 3 x 80

        disp = np.full((H, W, 2), np.nan)
        disp[mask, 0] = 7.0
        disp[mask, 1] = 3.0

        warped = warp_mask_with_holes(mask, disp)
        warp_cx = np.where(warped)[1].mean()
        warp_cy = np.where(warped)[0].mean()

        np.testing.assert_allclose(warp_cx, 49.5 + 7.0, atol=2.0)
        np.testing.assert_allclose(warp_cy, 25.0 + 3.0, atol=2.0)

    def test_mask_with_multiple_holes(self):
        """Mask with 3 holes should preserve all holes after warp."""
        H, W = 120, 120
        mask = np.zeros((H, W), dtype=bool)
        mask[10:110, 10:110] = True
        # Three holes
        mask[20:35, 20:35] = False
        mask[50:65, 50:65] = False
        mask[75:90, 75:90] = False
        hole_count_before = 3

        disp = np.full((H, W, 2), np.nan)
        disp[10:110, 10:110, 0] = 5.0
        disp[10:110, 10:110, 1] = 3.0

        warped = warp_mask_with_holes(mask, disp)

        # Check each hole center is still empty
        for cy, cx in [(27+3, 27+5), (57+3, 57+5), (82+3, 82+5)]:
            if 0 <= cy < H and 0 <= cx < W:
                assert not warped[cy, cx], f"Hole at ({cx},{cy}) should be empty"


# ── Non-uniform displacement fields ─────────────────────────


class TestNonUniformDisplacement:

    def test_pure_rotation_small_angle(self):
        """5-degree rotation about center: center pixel stays at 0, area preserved."""
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=bool)
        mask[20:80, 20:80] = True

        theta = np.deg2rad(5)
        cx, cy = 50.0, 50.0
        yy, xx = np.mgrid[:H, :W].astype(np.float64)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        du = (cos_t - 1) * (xx - cx) - sin_t * (yy - cy)
        dv = sin_t * (xx - cx) + (cos_t - 1) * (yy - cy)

        disp = np.full((H, W, 2), np.nan)
        disp[mask, 0] = du[mask]
        disp[mask, 1] = dv[mask]

        # Mask warp should preserve area
        warped = warp_mask_with_holes(mask, disp)
        area_change = abs(warped.sum() - mask.sum()) / mask.sum()
        assert area_change < 0.10

        # Accumulate: center pixel should have ~0 displacement
        u_prev = np.full((H, W, 2), np.nan)
        u_prev[mask] = 0.0
        u_acc = accumulate_displacement(u_prev, disp, debug=False)
        assert abs(u_acc[50, 50, 0]) < 0.1
        assert abs(u_acc[50, 50, 1]) < 0.1

    def test_uniaxial_stretch_10_percent(self):
        """10% stretch in x: displacement linear in x, v=0."""
        H, W = 80, 80
        mask = np.zeros((H, W), dtype=bool)
        mask[15:65, 15:65] = True

        strain = 0.10
        yy, xx = np.mgrid[:H, :W].astype(np.float64)
        cx = 40.0

        disp = np.full((H, W, 2), np.nan)
        disp[mask, 0] = strain * (xx[mask] - cx)
        disp[mask, 1] = 0.0

        u_prev = np.full((H, W, 2), np.nan)
        u_prev[mask] = 0.0
        u_acc = accumulate_displacement(u_prev, disp, debug=False)

        # Right edge: u = 0.10 * (64 - 40) = 2.40
        np.testing.assert_allclose(u_acc[40, 64, 0], 2.40, atol=0.5)
        # Left edge: u = 0.10 * (15 - 40) = -2.50
        np.testing.assert_allclose(u_acc[40, 15, 0], -2.50, atol=0.5)
        # Center: u ~ 0
        assert abs(u_acc[40, 40, 0]) < 0.1

    def test_simple_shear(self):
        """Simple shear gamma=0.05: u = gamma*(y - cy), v = 0."""
        H, W = 80, 80
        mask = np.zeros((H, W), dtype=bool)
        mask[15:65, 15:65] = True

        gamma = 0.05
        yy, xx = np.mgrid[:H, :W].astype(np.float64)
        cy = 40.0

        disp = np.full((H, W, 2), np.nan)
        disp[mask, 0] = gamma * (yy[mask] - cy)
        disp[mask, 1] = 0.0

        u_prev = np.full((H, W, 2), np.nan)
        u_prev[mask] = 0.0
        u_acc = accumulate_displacement(u_prev, disp, debug=False)

        # Top (y=64): u = 0.05*(64-40) = 1.20
        np.testing.assert_allclose(u_acc[64, 40, 0], 1.20, atol=0.3)
        # Bottom (y=15): u = 0.05*(15-40) = -1.25
        np.testing.assert_allclose(u_acc[15, 40, 0], -1.25, atol=0.3)

    def test_radial_expansion(self):
        """Radial expansion from center: displacement proportional to distance."""
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=bool)
        mask[20:80, 20:80] = True

        cx, cy = 50.0, 50.0
        yy, xx = np.mgrid[:H, :W].astype(np.float64)
        expansion = 0.05  # 5% radial expansion
        du = expansion * (xx - cx)
        dv = expansion * (yy - cy)

        disp = np.full((H, W, 2), np.nan)
        disp[mask, 0] = du[mask]
        disp[mask, 1] = dv[mask]

        u_prev = np.full((H, W, 2), np.nan)
        u_prev[mask] = 0.0
        u_acc = accumulate_displacement(u_prev, disp, debug=False)

        # Center stays at 0
        assert abs(u_acc[50, 50, 0]) < 0.1
        assert abs(u_acc[50, 50, 1]) < 0.1

        # Corner (20,20): u = 0.05*(20-50) = -1.5, v = 0.05*(20-50) = -1.5
        np.testing.assert_allclose(u_acc[20, 20, 0], -1.5, atol=0.3)
        np.testing.assert_allclose(u_acc[20, 20, 1], -1.5, atol=0.3)

    def test_vortex_displacement(self):
        """Vortex-like displacement (tangential, no radial): area should be preserved."""
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=bool)
        mask[25:75, 25:75] = True

        cx, cy = 50.0, 50.0
        yy, xx = np.mgrid[:H, :W].astype(np.float64)
        # Tangential: u = -k*(y-cy), v = k*(x-cx), with small k
        k = 0.02
        du = -k * (yy - cy)
        dv = k * (xx - cx)

        disp = np.full((H, W, 2), np.nan)
        disp[mask, 0] = du[mask]
        disp[mask, 1] = dv[mask]

        warped = warp_mask_with_holes(mask, disp)
        area_change = abs(warped.sum() - mask.sum()) / mask.sum()
        assert area_change < 0.10, f"Vortex should preserve area, got {area_change*100:.1f}%"


# ── Large deformation & boundary loss ────────────────────────


class TestLargeDeformation:

    def test_80px_translation_first_step_no_loss(self):
        """First accumulation step (u_prev=0) should never lose pixels,
        regardless of displacement magnitude."""
        H, W = 100, 200
        mask = np.zeros((H, W), dtype=bool)
        mask[20:80, 30:130] = True
        orig_area = mask.sum()

        disp = np.full((H, W, 2), np.nan)
        disp[mask, 0] = 80.0
        disp[mask, 1] = 0.0

        u_prev = np.full((H, W, 2), np.nan)
        u_prev[mask] = 0.0

        u_acc = accumulate_displacement(u_prev, disp, debug=False)
        valid = ~np.isnan(u_acc[..., 0])
        assert valid.sum() == orig_area, (
            f"First step should not lose pixels: {valid.sum()} != {orig_area}"
        )

    def test_second_step_loses_boundary_pixels(self):
        """Second step with large displacement: deformed coords that land
        outside the delta_u valid region (mask) produce NaN.

        mask x=[30,129] (100 cols).  After step 1, u_prev=80 everywhere.
        In step 2, deformed x = x + 80.  delta_u is valid only at x=[30,129],
        so only pixels with x+80 <= 129 (i.e. x <= 49) survive.
        That is 20 valid columns * 60 rows = 1200 pixels; 4800 lost.
        """
        H, W = 100, 200
        mask = np.zeros((H, W), dtype=bool)
        mask[20:80, 30:130] = True  # 60x100 = 6000

        disp = np.full((H, W, 2), np.nan)
        disp[mask, 0] = 80.0
        disp[mask, 1] = 0.0

        # First step
        u_prev = np.full((H, W, 2), np.nan)
        u_prev[mask] = 0.0
        u_step1 = accumulate_displacement(u_prev, disp, debug=False)

        # Second step
        u_step2 = accumulate_displacement(u_step1, disp, debug=False)
        valid = ~np.isnan(u_step2[..., 0])
        lost = 6000 - valid.sum()

        assert lost > 0, "Should lose boundary pixels on second step"
        # 80 columns fall outside mask valid region: 80*60 = 4800 lost
        assert abs(lost - 4800) < 200, f"Expected ~4800 lost, got {lost}"

        # Surviving pixels (x=[30,49]) should have u ~ 160
        mean_u = np.nanmean(u_step2[valid, 0])
        np.testing.assert_allclose(mean_u, 160.0, atol=2.0)

    def test_diagonal_large_displacement(self):
        """50px diagonal displacement: loss in both x and y directions."""
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=bool)
        mask[10:90, 10:90] = True  # 80x80 = 6400

        disp = np.full((H, W, 2), np.nan)
        disp[mask, 0] = 50.0  # right
        disp[mask, 1] = 50.0  # down

        u_prev = np.full((H, W, 2), np.nan)
        u_prev[mask] = 0.0
        u_step1 = accumulate_displacement(u_prev, disp, debug=False)

        # Second step: deformed coords = (x+50, y+50)
        # x+50 >= 90 (right boundary of mask) => x >= 40 lost from right
        # y+50 >= 90 => y >= 40 lost from bottom
        u_step2 = accumulate_displacement(u_step1, disp, debug=False)
        valid = ~np.isnan(u_step2[..., 0])
        print(f"Diagonal: {valid.sum()}/6400 valid after 2nd step")
        assert valid.sum() < 6400, "Should lose pixels in both directions"

    def test_large_displacement_mask_warp_clipped_to_image(self):
        """Mask warp with displacement pushing part of mask out of image bounds."""
        H, W = 80, 80
        mask = np.zeros((H, W), dtype=bool)
        mask[10:70, 50:75] = True  # Near right edge

        disp = np.full((H, W, 2), np.nan)
        disp[mask, 0] = 20.0  # Pushes rightward, partially OOB
        disp[mask, 1] = 0.0

        warped = warp_mask_with_holes(mask, disp)
        # Warped mask should be clipped to image bounds
        assert warped.shape == (H, W)
        # Some area may be lost (clipped at x=79)
        assert warped.sum() > 0, "Should have some valid pixels"
        assert warped.sum() <= mask.sum(), "Cannot gain pixels from clipping"


# ── Multi-step accumulation error propagation ────────────────


class TestMultiStepAccumulation:

    def test_10_steps_1_percent_stretch(self):
        """10 steps of 1% stretch should approximate 10% total stretch.

        Edge pixels may become NaN because stretch pushes deformed coords
        outside the valid mask region.  Check interior pixels instead.
        """
        H, W = 80, 80
        mask = np.zeros((H, W), dtype=bool)
        mask[10:70, 10:70] = True

        u_prev = np.full((H, W, 2), np.nan)
        u_prev[mask] = 0.0

        strain_per_step = 0.01
        yy, xx = np.mgrid[:H, :W].astype(np.float64)
        cx = 40.0

        for _ in range(10):
            delta = np.full((H, W, 2), np.nan)
            delta[mask, 0] = strain_per_step * (xx[mask] - cx)
            delta[mask, 1] = 0.0
            u_prev = accumulate_displacement(u_prev, delta, debug=False)

        # Center (x=40): u ~ 0  (stretch axis)
        assert abs(u_prev[40, 40, 0]) < 0.1

        # Interior point (x=55): u ~ 0.10*(55-40) = 1.5
        # Far enough from edge that deformed coords stay inside mask
        val_55 = u_prev[40, 55, 0]
        assert not np.isnan(val_55), "x=55 should survive 10% stretch"
        np.testing.assert_allclose(val_55, 1.5, atol=0.5)

        # Interior point (x=25): u ~ 0.10*(25-40) = -1.5
        val_25 = u_prev[40, 25, 0]
        assert not np.isnan(val_25), "x=25 should survive 10% stretch"
        np.testing.assert_allclose(val_25, -1.5, atol=0.5)

    def test_5_step_rotation_accumulation(self):
        """5 steps of 2-degree rotation should approximate 10-degree total."""
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=bool)
        mask[20:80, 20:80] = True

        u_prev = np.full((H, W, 2), np.nan)
        u_prev[mask] = 0.0

        theta_step = np.deg2rad(2)
        cx, cy = 50.0, 50.0
        yy, xx = np.mgrid[:H, :W].astype(np.float64)

        for _ in range(5):
            cos_t, sin_t = np.cos(theta_step), np.sin(theta_step)
            du = (cos_t - 1) * (xx - cx) - sin_t * (yy - cy)
            dv = sin_t * (xx - cx) + (cos_t - 1) * (yy - cy)

            delta = np.full((H, W, 2), np.nan)
            delta[mask, 0] = du[mask]
            delta[mask, 1] = dv[mask]
            u_prev = accumulate_displacement(u_prev, delta, debug=False)

        # Center should remain ~0
        assert abs(u_prev[50, 50, 0]) < 0.1
        assert abs(u_prev[50, 50, 1]) < 0.1

        # Corner (20,20) analytical: 10-degree rotation about (50,50)
        theta_total = np.deg2rad(10)
        expected_u = (np.cos(theta_total) - 1) * (20 - 50) - np.sin(theta_total) * (20 - 50)
        expected_v = np.sin(theta_total) * (20 - 50) + (np.cos(theta_total) - 1) * (20 - 50)

        if not np.isnan(u_prev[20, 20, 0]):
            np.testing.assert_allclose(u_prev[20, 20, 0], expected_u, atol=2.0)
            np.testing.assert_allclose(u_prev[20, 20, 1], expected_v, atol=2.0)

    def test_many_small_translations_sum_correctly(self):
        """20 steps of 3px translation should give 60px total."""
        H, W = 60, 200
        mask = np.zeros((H, W), dtype=bool)
        mask[10:50, 10:190] = True

        u_prev = np.full((H, W, 2), np.nan)
        u_prev[mask] = 0.0

        for _ in range(20):
            delta = np.full((H, W, 2), np.nan)
            delta[mask, 0] = 3.0
            delta[mask, 1] = 0.0
            u_prev = accumulate_displacement(u_prev, delta, debug=False)

        valid = ~np.isnan(u_prev[..., 0])
        # Some pixels lost at right boundary after large accumulated shift
        mean_u = np.nanmean(u_prev[valid, 0])
        np.testing.assert_allclose(mean_u, 60.0, atol=1.0)

    def test_alternating_direction_cancels(self):
        """Alternating +5 / -5 px steps should approximately cancel."""
        H, W = 60, 60
        mask = np.zeros((H, W), dtype=bool)
        mask[10:50, 10:50] = True

        u_prev = np.full((H, W, 2), np.nan)
        u_prev[mask] = 0.0

        for i in range(10):
            sign = 1.0 if i % 2 == 0 else -1.0
            delta = np.full((H, W, 2), np.nan)
            delta[mask, 0] = sign * 5.0
            delta[mask, 1] = 0.0
            u_prev = accumulate_displacement(u_prev, delta, debug=False)

        # After 10 steps (5 forward, 5 backward), net ~ 0
        valid = ~np.isnan(u_prev[..., 0])
        if valid.sum() > 0:
            mean_u = np.nanmean(u_prev[valid, 0])
            assert abs(mean_u) < 2.0, f"Alternating should cancel, got mean_u={mean_u}"


# ── NaN-aware interpolation edge cases ───────────────────────


class TestNanInterpEdgeCases:

    def test_single_valid_corner(self):
        """Only one corner valid in 2x2 neighborhood: should use that value."""
        field = np.array([[1.0, np.nan],
                          [np.nan, np.nan]])
        result = bilinear_sample_nan_aware(field, np.array([0.5]), np.array([0.5]))
        np.testing.assert_allclose(result[0], 1.0)

    def test_two_valid_corners(self):
        """Two valid corners: weighted average of those two."""
        field = np.array([[2.0, 8.0],
                          [np.nan, np.nan]])
        result = bilinear_sample_nan_aware(field, np.array([0.5]), np.array([0.0]))
        # y=0 exactly, so only top row matters: (2+8)/2 = 5
        np.testing.assert_allclose(result[0], 5.0, atol=0.01)

    def test_gradient_field_interpolation(self):
        """Linear gradient field: interpolation should be exact."""
        H, W = 10, 10
        yy, xx = np.mgrid[:H, :W].astype(np.float64)
        field = 3.0 * xx + 2.0 * yy  # linear gradient

        # Sample at fractional positions
        xs = np.array([2.5, 4.3, 7.8])
        ys = np.array([1.5, 6.7, 3.2])
        expected = 3.0 * xs + 2.0 * ys

        result = bilinear_sample_nan_aware(field, xs, ys)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_boundary_pixel_not_contaminated(self):
        """Pixels at NaN boundary should not be contaminated by NaN->0 substitution."""
        H, W = 20, 20
        field = np.full((H, W), np.nan)
        field[5:15, 5:15] = 10.0  # valid interior

        # Sample right at the inner boundary (x=5.0, y=10.0)
        result = bilinear_sample_nan_aware(field, np.array([5.0]), np.array([10.0]))
        np.testing.assert_allclose(result[0], 10.0, atol=0.01)

        # Sample at (4.5, 10.0) - straddles NaN boundary
        result2 = bilinear_sample_nan_aware(field, np.array([4.5]), np.array([10.0]))
        # Should still be 10.0 (NaN neighbors excluded, only valid neighbor = 10)
        np.testing.assert_allclose(result2[0], 10.0, atol=0.01)

    def test_large_vectorized_batch(self):
        """10000 sample points should work correctly in batch."""
        H, W = 50, 50
        field = np.random.RandomState(42).rand(H, W) * 100
        field[:, :5] = np.nan  # NaN strip on left

        xs = np.random.RandomState(43).uniform(5, 48, size=10000)
        ys = np.random.RandomState(44).uniform(0, 48, size=10000)

        result = bilinear_sample_nan_aware(field, xs, ys)
        # All should be valid (sampled away from NaN strip)
        assert np.isnan(result).sum() < 100  # very few near x=5 boundary


# ── Resolve mask fallback chain ──────────────────────────────


class TestResolveMaskChain:

    def _make_kf_acc(self, H, W, roi_mask, ref_disps):
        """Helper: build kf_accumulated dict."""
        kf_acc = {}
        for ref_num, (du, dv) in ref_disps.items():
            arr = np.full((H, W, 2), np.nan, dtype=np.float64)
            arr[roi_mask, 0] = du
            arr[roi_mask, 1] = dv
            kf_acc[ref_num] = arr
        return kf_acc

    def test_auto_warp_with_complex_roi(self):
        """Auto-warp on L-shaped ROI should produce shifted L-shape."""
        H, W = 100, 100
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[10:60, 10:40] = True
        roi_mask[40:60, 10:70] = True

        kf_acc = self._make_kf_acc(H, W, roi_mask, {
            1: (0.0, 0.0),
            50: (10.0, 0.0),
        })

        result = DICProcessor._resolve_mask(
            frame_num=55, list_idx=54, ref_num=50,
            roi_mask=roi_mask, kf_accumulated=kf_acc,
            user_masks={}, xmin=10, ymin=10, xmax=70, ymax=60,
            auto_warp=True,
        )

        # Warped mask center should be shifted ~10px right
        orig_cx = np.where(roi_mask)[1].mean()
        warp_cx = np.where(result)[1].mean()
        np.testing.assert_allclose(warp_cx - orig_cx, 10.0, atol=3.0)

    def test_user_mask_always_wins(self):
        """User mask overrides auto-warp regardless of displacement."""
        H, W = 60, 60
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[10:50, 10:50] = True

        user_mask = np.zeros((H, W), dtype=bool)
        user_mask[5:55, 5:55] = True  # different shape

        kf_acc = self._make_kf_acc(H, W, roi_mask, {
            1: (0.0, 0.0),
            20: (30.0, 0.0),  # large displacement
        })

        # _resolve_mask looks up user_masks by ref_idx = ref_num - 1 = 19
        result = DICProcessor._resolve_mask(
            frame_num=25, list_idx=24, ref_num=20,
            roi_mask=roi_mask, kf_accumulated=kf_acc,
            user_masks={19: user_mask},
            xmin=10, ymin=10, xmax=50, ymax=50,
            auto_warp=True,
        )
        np.testing.assert_array_equal(result, user_mask)

    def test_auto_warp_disabled_returns_original(self):
        """auto_warp=False always returns original ROI mask."""
        H, W = 60, 60
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[10:50, 10:50] = True

        kf_acc = self._make_kf_acc(H, W, roi_mask, {
            1: (0.0, 0.0),
            20: (15.0, 0.0),
        })

        result = DICProcessor._resolve_mask(
            frame_num=25, list_idx=24, ref_num=20,
            roi_mask=roi_mask, kf_accumulated=kf_acc,
            user_masks={}, xmin=10, ymin=10, xmax=50, ymax=50,
            auto_warp=False,
        )
        np.testing.assert_array_equal(result, roi_mask)


# ── Median filter on complex fields ──────────────────────────


class TestMedianFilterComplex:

    def test_preserves_smooth_gradient(self):
        """Median filter should not distort a smooth linear gradient."""
        H, W = 40, 40
        yy, xx = np.mgrid[:H, :W].astype(np.float64)
        disp = np.full((H, W, 2), np.nan)
        disp[5:35, 5:35, 0] = 2.0 * xx[5:35, 5:35]
        disp[5:35, 5:35, 1] = 1.0 * yy[5:35, 5:35]

        filtered = median_filter_displacement(disp, kernel_size=3)
        valid = ~np.isnan(disp[..., 0])

        # Interior pixels (away from boundary) should be unchanged
        interior = np.zeros_like(valid)
        interior[7:33, 7:33] = True
        interior &= valid

        diff_u = np.abs(filtered[interior, 0] - disp[interior, 0])
        assert diff_u.max() < 1.0, f"Gradient distortion: max_diff={diff_u.max()}"

    def test_removes_salt_and_pepper_noise(self):
        """Median filter should remove isolated outlier pixels."""
        H, W = 30, 30
        disp = np.full((H, W, 2), np.nan)
        disp[5:25, 5:25, 0] = 5.0
        disp[5:25, 5:25, 1] = 3.0

        # Add salt-and-pepper outliers
        rng = np.random.RandomState(42)
        for _ in range(10):
            y, x = rng.randint(7, 23), rng.randint(7, 23)
            disp[y, x, 0] = rng.choice([-50, 50])

        filtered = median_filter_displacement(disp, kernel_size=5)
        valid = ~np.isnan(filtered[..., 0])

        # All interior values should be close to 5.0 after filtering
        interior = np.zeros_like(valid)
        interior[8:22, 8:22] = True
        interior &= valid
        max_deviation = np.abs(filtered[interior, 0] - 5.0).max()
        assert max_deviation < 2.0, f"Outliers not removed: max_dev={max_deviation}"
