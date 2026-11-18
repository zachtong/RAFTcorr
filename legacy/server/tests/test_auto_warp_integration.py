"""Integration tests for auto-warp mask in _resolve_mask."""

import numpy as np
from raft_dic_gui.controller import DICProcessor
from raft_dic_gui.incremental import warp_mask_with_holes


class TestAutoWarpMask:

    def test_uniform_translation_warp(self):
        """Mask warped by uniform translation should shift by that amount."""
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=bool)
        mask[20:60, 20:60] = True

        disp = np.full((H, W, 2), np.nan)
        disp[20:60, 20:60, 0] = 15.0
        disp[20:60, 20:60, 1] = 10.0

        warped = warp_mask_with_holes(mask, disp)

        orig_ys, orig_xs = np.where(mask)
        warp_ys, warp_xs = np.where(warped)

        dx = warp_xs.mean() - orig_xs.mean()
        dy = warp_ys.mean() - orig_ys.mean()

        np.testing.assert_allclose(dx, 15.0, atol=2.0)
        np.testing.assert_allclose(dy, 10.0, atol=2.0)

    def test_warped_mask_preserves_hole(self):
        """A mask with a hole should still have a hole after warping."""
        H, W = 100, 100
        mask = np.zeros((H, W), dtype=bool)
        mask[20:80, 20:80] = True
        mask[40:60, 40:60] = False  # hole

        disp = np.full((H, W, 2), np.nan)
        disp[20:80, 20:80, 0] = 5.0
        disp[20:80, 20:80, 1] = 3.0

        warped = warp_mask_with_holes(mask, disp)

        center_y, center_x = 50 + 3, 50 + 5
        assert not warped[center_y, center_x], "Hole center should still be empty"

    def test_resolve_mask_auto_warp_enabled(self):
        """_resolve_mask with auto_warp=True warps mask for non-frame-1 references."""
        H, W = 100, 100
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[20:60, 20:60] = True

        kf_accumulated = {
            1: np.full((H, W, 2), np.nan, dtype=np.float64),
            50: np.full((H, W, 2), np.nan, dtype=np.float64),
        }
        kf_accumulated[1][20:60, 20:60] = 0.0
        kf_accumulated[50][20:60, 20:60, 0] = 10.0
        kf_accumulated[50][20:60, 20:60, 1] = 0.0

        result = DICProcessor._resolve_mask(
            frame_num=55, list_idx=54, ref_num=50,
            roi_mask=roi_mask, kf_accumulated=kf_accumulated,
            user_masks={}, xmin=20, ymin=20, xmax=60, ymax=60,
            auto_warp=True,
        )

        warp_ys, warp_xs = np.where(result)
        orig_ys, orig_xs = np.where(roi_mask)
        dx = warp_xs.mean() - orig_xs.mean()
        np.testing.assert_allclose(dx, 10.0, atol=2.0)

    def test_resolve_mask_auto_warp_ref1_uses_original(self):
        """For ref_num=1, auto-warp returns original mask (no displacement to warp)."""
        H, W = 50, 50
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[10:40, 10:40] = True

        kf_accumulated = {1: np.full((H, W, 2), np.nan, dtype=np.float64)}
        kf_accumulated[1][10:40, 10:40] = 0.0

        result = DICProcessor._resolve_mask(
            frame_num=5, list_idx=4, ref_num=1,
            roi_mask=roi_mask, kf_accumulated=kf_accumulated,
            user_masks={}, xmin=10, ymin=10, xmax=40, ymax=40,
            auto_warp=True,
        )

        np.testing.assert_array_equal(result, roi_mask)

    def test_resolve_mask_user_mask_overrides_auto_warp(self):
        """User mask takes priority over auto-warp."""
        H, W = 50, 50
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[10:40, 10:40] = True

        user_mask = np.zeros((H, W), dtype=bool)
        user_mask[5:45, 5:45] = True  # Different from roi_mask

        kf_accumulated = {
            1: np.full((H, W, 2), np.nan, dtype=np.float64),
            10: np.full((H, W, 2), np.nan, dtype=np.float64),
        }
        kf_accumulated[1][10:40, 10:40] = 0.0
        kf_accumulated[10][10:40, 10:40, 0] = 5.0
        kf_accumulated[10][10:40, 10:40, 1] = 0.0

        # _resolve_mask looks up user_masks by ref_idx = ref_num - 1 = 9
        result = DICProcessor._resolve_mask(
            frame_num=15, list_idx=14, ref_num=10,
            roi_mask=roi_mask, kf_accumulated=kf_accumulated,
            user_masks={9: user_mask},
            xmin=10, ymin=10, xmax=40, ymax=40,
            auto_warp=True,
        )

        np.testing.assert_array_equal(result, user_mask)

    def test_resolve_mask_auto_warp_disabled_returns_original(self):
        """When auto_warp=False, always returns original mask for non-user frames."""
        H, W = 50, 50
        roi_mask = np.zeros((H, W), dtype=bool)
        roi_mask[10:40, 10:40] = True

        kf_accumulated = {
            1: np.full((H, W, 2), np.nan, dtype=np.float64),
            10: np.full((H, W, 2), np.nan, dtype=np.float64),
        }
        kf_accumulated[1][10:40, 10:40] = 0.0
        kf_accumulated[10][10:40, 10:40, 0] = 5.0
        kf_accumulated[10][10:40, 10:40, 1] = 0.0

        result = DICProcessor._resolve_mask(
            frame_num=15, list_idx=14, ref_num=10,
            roi_mask=roi_mask, kf_accumulated=kf_accumulated,
            user_masks={}, xmin=10, ymin=10, xmax=40, ymax=40,
            auto_warp=False,  # disabled
        )

        np.testing.assert_array_equal(result, roi_mask)
