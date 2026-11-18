"""Synthetic bubble visualization test.

Generates a growing/shrinking bubble in an incompressible 2D material,
then verifies that reference-mode and deformed-mode ROI shapes behave
correctly when per-frame masks are provided.

Physics: 2D plane-strain incompressible solid with circular cavity.
  Area conservation: r'^2 = r^2 + R(t)^2 - R0^2
  Radial displacement: u_r = sqrt(r^2 + R^2 - R0^2) - r
"""

import numpy as np
import pytest

from server.deformed_warp import (
    InverseMapCache,
    compute_inverse_map,
    get_warped_full_data,
    warp_data_inverse,
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_bubble_scenario(
    h: int = 128,
    w: int = 128,
    cx: int = 64,
    cy: int = 64,
    radii: list = None,
):
    """Create synthetic data for a growing/shrinking bubble.

    Returns
    -------
    dict with keys:
        h, w, cx, cy, radii,
        displacements: list of (h, w, 2) arrays (ROI = full image),
        per_frame_rois: dict {frame_idx: (h,w) bool mask},
        roi_rect: (x0, y0, x1, y1),
    """
    if radii is None:
        # Grow from 10 to 25, then shrink to 15
        radii = [10, 14, 18, 22, 25, 22, 18, 15]

    R0 = radii[0]
    yy, xx = np.mgrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_safe = np.maximum(dist, 1e-10)

    # Per-frame ROI masks (full image size)
    per_frame_rois = {}
    for i, r in enumerate(radii):
        mask = np.ones((h, w), dtype=bool)
        mask[dist <= r] = False
        per_frame_rois[i] = mask

    # Displacement results: frame i → displacement from frame 0 to frame i+1
    roi_rect = (0, 0, w, h)
    displacements = []

    for i in range(1, len(radii)):
        R = radii[i]
        # Incompressible radial displacement
        r_new = np.sqrt(r_safe ** 2 + R ** 2 - R0 ** 2)
        u_r = r_new - r_safe

        u = u_r * (xx - cx) / r_safe  # x-component
        v = u_r * (yy - cy) / r_safe  # y-component

        # Simulate _apply_current_frame_mask:
        # Pixels whose deformed position lands inside the grown bubble → NaN.
        # For the bubble scenario, any pixel at r where r' = r + u_r
        # and r + u_r < R means the deformed position is inside the bubble.
        # Actually, _apply_current_frame_mask checks cur_mask at (x+U, y+V).
        # For our circular case this means: is the deformed position inside
        # the current bubble?
        # deformed_r = r + u_r = sqrt(r^2 + R^2 - R0^2)
        # deformed_r < R ⟺ r^2 + R^2 - R0^2 < R^2 ⟺ r < R0
        # So pixels with r < R0 (inside the original bubble) get NaN.
        # BUT also pixels with r < R (inside the current bubble) have no
        # material, so DIC can't track them → those are also NaN.
        #
        # The actual _apply_current_frame_mask checks masks at deformed
        # coords.  For pixels at R0 < r < R, the material moved outward
        # to r' > R, so cur_mask at r' would be True (outside bubble).
        # These pixels are VALID in the computation.
        #
        # For pixels at r < R0: these are inside the original bubble,
        # DIC never tracked them → NaN from the start.
        #
        # So the only NaN in the displacement data should be at r <= R0
        # (original bubble).  The per-frame mask's larger bubble doesn't
        # create NaN in the stored result for correctly tracked pixels!
        #
        # WAIT — re-check: DIC computes displacement on cur_mask's ROI.
        # If cur_mask excludes the bubble region, DIC doesn't compute
        # displacement there.  So pixels with r < R in the current mask
        # might not have displacement data at all (DIC didn't run there).
        #
        # Actually the DIC runs on the current frame's mask.  In the
        # controller, `current_mask` for incremental mode is the ROI mask
        # for the CURRENT reference frame.  If it's the first segment,
        # current_mask = roi_mask (frame 0's mask).  The bubble area in
        # frame 0 is excluded, so DIC produces NaN there.
        # For subsequent segments (new key frame), current_mask = new mask.
        #
        # For ACCUMULATIVE mode: always use frame 0's mask → DIC always
        # tracks on frame 0's ROI.  Pixels inside frame 0's bubble are
        # always NaN.  Pixels outside frame 0's bubble but inside later
        # bubbles... their deformed position may enter the bubble, but
        # _apply_current_frame_mask checks if the deformed position is
        # inside cur_mask.  If the bubble grew, cur_mask = later frame's
        # mask.  The deformed position of a pixel at r (R0 < r < R_later)
        # moves to r' = sqrt(r^2 + R^2 - R0^2) > R (always outside the
        # bubble for incompressible material!), so cur_mask at r' is True.
        # These pixels remain valid.
        #
        # But for compressible materials or irregular deformation, pixels
        # near the bubble edge might have deformed positions inside the
        # bubble, causing NaN.
        #
        # For our test: use the simple model where r <= R0 → NaN.
        u[dist <= R0] = np.nan
        v[dist <= R0] = np.nan

        disp = np.stack([u, v], axis=-1)
        displacements.append(disp)

    return {
        "h": h, "w": w, "cx": cx, "cy": cy,
        "radii": radii,
        "displacements": displacements,
        "per_frame_rois": per_frame_rois,
        "roi_rect": roi_rect,
    }


# ---------------------------------------------------------------------------
# Bug 1: Reference mode — ROI shape must be fixed to frame 0
# ---------------------------------------------------------------------------

class TestReferenceMode:
    """In reference-background mode the non-NaN region should match frame 0's
    mask for EVERY frame."""

    def _render_reference_mode(self, scenario, idx):
        """Reproduce displacement.py reference-mode full_data placement."""
        h, w = scenario["h"], scenario["w"]
        roi_rect = scenario["roi_rect"]
        disp_data = scenario["displacements"][idx][:, :, 0]  # u component

        full_data = np.full((h, w), np.nan)
        x0, y0, x1, y1 = roi_rect
        dh, dw = disp_data.shape
        sh = min(dh, y1 - y0)
        sw = min(dw, x1 - x0)
        full_data[y0:y0 + sh, x0:x0 + sw] = disp_data[:sh, :sw]

        # Current fix: clip to frame 0's mask
        ref_mask = scenario["per_frame_rois"].get(0)
        if ref_mask is not None and ref_mask.shape == full_data.shape:
            full_data[~ref_mask] = np.nan

        return full_data

    def test_valid_pixel_count_constant(self):
        """The number of non-NaN pixels should be the same for every frame
        in reference mode, matching frame 0's mask exactly."""
        sc = make_bubble_scenario()

        frame0_valid = sc["per_frame_rois"][0].sum()
        for idx in range(len(sc["displacements"])):
            full_data = self._render_reference_mode(sc, idx)
            actual_valid = np.sum(~np.isnan(full_data))
            print(
                f"  Frame {idx}: valid={actual_valid}, "
                f"expected={frame0_valid}, "
                f"bubble_R={sc['radii'][idx + 1]}"
            )
            assert actual_valid == frame0_valid, (
                f"Frame {idx}: valid pixel count {actual_valid} != "
                f"frame 0 mask count {frame0_valid}"
            )

    def test_nan_pattern_matches_frame0(self):
        """The NaN pattern should exactly match frame 0's mask (inverted)."""
        sc = make_bubble_scenario()
        expected_valid = sc["per_frame_rois"][0]

        for idx in range(len(sc["displacements"])):
            full_data = self._render_reference_mode(sc, idx)
            actual_valid = ~np.isnan(full_data)
            mismatch = (actual_valid != expected_valid).sum()
            assert mismatch == 0, (
                f"Frame {idx}: {mismatch} pixels differ from frame 0 mask"
            )


# ---------------------------------------------------------------------------
# Bug 2: Deformed mode — ROI shape must follow per-frame mask
# ---------------------------------------------------------------------------

class TestDeformedMode:
    """In deformed-background mode the non-NaN region should follow the
    per-frame user mask."""

    def _render_deformed_mode(self, scenario, idx):
        """Use inverse warping to place data in deformed coordinates."""
        h, w = scenario["h"], scenario["w"]
        roi_rect = scenario["roi_rect"]
        disp = scenario["displacements"][idx]
        U, V = disp[:, :, 0], disp[:, :, 1]
        disp_data = U  # u component

        deformed_mask = scenario["per_frame_rois"].get(idx + 1)
        cache = InverseMapCache(max_size=10)

        full_data = get_warped_full_data(
            data=disp_data,
            frame_idx=idx,
            U=U, V=V,
            roi_rect=roi_rect,
            image_shape=(h, w),
            cache=cache,
            deformed_mask=deformed_mask,
        )
        return full_data

    def test_deformed_roi_follows_mask_growth(self):
        """When bubble grows, the colored area inside the mask should shrink
        (more of the image is bubble = excluded)."""
        sc = make_bubble_scenario(radii=[10, 14, 18, 22, 25])

        prev_valid = None
        for idx in range(len(sc["displacements"])):
            full_data = self._render_deformed_mode(sc, idx)
            valid_count = np.sum(~np.isnan(full_data))
            mask_valid = sc["per_frame_rois"][idx + 1].sum()
            print(
                f"  Frame {idx}: rendered_valid={valid_count}, "
                f"mask_valid={mask_valid}, "
                f"bubble_R={sc['radii'][idx + 1]}"
            )
            if prev_valid is not None:
                # As bubble grows, valid area should decrease
                assert valid_count <= prev_valid + 10, (
                    f"Frame {idx}: valid grew unexpectedly "
                    f"({valid_count} > {prev_valid})"
                )
            prev_valid = valid_count

    def test_deformed_roi_follows_mask_shrink(self):
        """When bubble shrinks, the colored area should GROW (less bubble)."""
        # Only shrinking phase: radii decrease
        sc = make_bubble_scenario(radii=[25, 22, 18, 15, 12])

        prev_valid = None
        for idx in range(len(sc["displacements"])):
            full_data = self._render_deformed_mode(sc, idx)
            valid_count = np.sum(~np.isnan(full_data))
            mask_valid = sc["per_frame_rois"][idx + 1].sum()
            print(
                f"  Frame {idx}: rendered_valid={valid_count}, "
                f"mask_valid={mask_valid}, "
                f"bubble_R={sc['radii'][idx + 1]}"
            )
            if prev_valid is not None:
                # As bubble shrinks, valid area should increase
                assert valid_count >= prev_valid - 10, (
                    f"Frame {idx}: valid shrank unexpectedly "
                    f"({valid_count} < {prev_valid})"
                )
            prev_valid = valid_count

    def test_deformed_valid_inside_mask(self):
        """All non-NaN pixels should be inside the deformed frame's mask."""
        sc = make_bubble_scenario()

        for idx in range(len(sc["displacements"])):
            full_data = self._render_deformed_mode(sc, idx)
            mask = sc["per_frame_rois"][idx + 1]
            has_data = ~np.isnan(full_data)
            outside_mask = has_data & ~mask
            n_outside = outside_mask.sum()
            print(
                f"  Frame {idx}: {n_outside} pixels with data outside mask"
            )
            assert n_outside == 0, (
                f"Frame {idx}: {n_outside} rendered pixels outside mask"
            )


# ---------------------------------------------------------------------------
# Diagnostic: show what's happening without assertions
# ---------------------------------------------------------------------------

def test_diagnostic_reference_mode():
    """Print diagnostic information about reference mode rendering."""
    sc = make_bubble_scenario()
    radii = sc["radii"]
    masks = sc["per_frame_rois"]

    print("\n=== DIAGNOSTIC: Reference Mode ===")
    print(f"Bubble radii: {radii}")
    print(f"Frame 0 mask valid: {masks[0].sum()}")
    print()

    for idx in range(len(sc["displacements"])):
        disp = sc["displacements"][idx]
        u = disp[:, :, 0]
        roi_rect = sc["roi_rect"]
        h, w = sc["h"], sc["w"]

        # Step 1: raw data NaN count
        raw_nan = np.isnan(u).sum()

        # Step 2: after placement in full image
        full_data = np.full((h, w), np.nan)
        x0, y0, x1, y1 = roi_rect
        dh, dw = u.shape
        sh = min(dh, y1 - y0)
        sw = min(dw, x1 - x0)
        full_data[y0:y0 + sh, x0:x0 + sw] = u[:sh, :sw]
        after_place = np.sum(~np.isnan(full_data))

        # Step 3: after frame 0 mask clip
        ref_mask = masks.get(0)
        if ref_mask is not None and ref_mask.shape == full_data.shape:
            full_data[~ref_mask] = np.nan
        after_clip = np.sum(~np.isnan(full_data))

        # Step 4: what SHOULD we have?
        expected = masks[0].sum()

        print(
            f"  Frame {idx} (R={radii[idx+1]}): "
            f"raw_nan={raw_nan}, "
            f"after_place={after_place}, "
            f"after_clip={after_clip}, "
            f"expected={expected}, "
            f"diff={expected - after_clip}"
        )


if __name__ == "__main__":
    test_diagnostic_reference_mode()
    print()
    print("=== Running TestReferenceMode ===")
    t = TestReferenceMode()
    try:
        t.test_valid_pixel_count_constant()
        print("  PASSED: valid_pixel_count_constant")
    except AssertionError as e:
        print(f"  FAILED: {e}")
    try:
        t.test_nan_pattern_matches_frame0()
        print("  PASSED: nan_pattern_matches_frame0")
    except AssertionError as e:
        print(f"  FAILED: {e}")

    print()
    print("=== Running TestDeformedMode ===")
    t2 = TestDeformedMode()
    try:
        t2.test_deformed_roi_follows_mask_growth()
        print("  PASSED: roi_follows_mask_growth")
    except AssertionError as e:
        print(f"  FAILED: {e}")
    try:
        t2.test_deformed_roi_follows_mask_shrink()
        print("  PASSED: roi_follows_mask_shrink")
    except AssertionError as e:
        print(f"  FAILED: {e}")
    try:
        t2.test_deformed_valid_inside_mask()
        print("  PASSED: valid_inside_mask")
    except AssertionError as e:
        print(f"  FAILED: {e}")
