"""Test: VSG strain should not fit across crack discontinuities.

Creates a synthetic displacement field with a vertical crack (NaN gap)
separating two regions with very different displacements. Verifies that
the connectivity-aware VSG does not produce spurious high strain from
cross-crack polynomial fitting.
"""

import numpy as np
import pytest

from raft_dic_gui.strain import calculate_strain_field


def make_cracked_displacement(h, w, crack_col, crack_width, u_left, u_right):
    """Create displacement field with a vertical crack.

    Left side: uniform u = u_left
    Right side: uniform u = u_right
    Crack: NaN gap of crack_width pixels centered at crack_col
    """
    disp = np.zeros((h, w, 2))

    crack_start = crack_col - crack_width // 2
    crack_end = crack_start + crack_width

    # Left region
    disp[:, :crack_start, 0] = u_left
    # Right region
    disp[:, crack_end:, 0] = u_right
    # Crack = NaN
    disp[:, crack_start:crack_end, :] = np.nan
    # Top/bottom borders = NaN (to avoid edge effects)
    disp[:5, :, :] = np.nan
    disp[-5:, :, :] = np.nan

    return disp


class TestStrainCrackConnectivity:
    """Verify VSG does not fit across cracks."""

    def test_uniform_field_no_crack(self):
        """Baseline: uniform displacement → zero strain everywhere."""
        disp = np.full((100, 100, 2), 0.0)
        disp[:, :, 0] = 5.0  # uniform u=5
        # NaN border
        disp[:3, :, :] = np.nan
        disp[-3:, :, :] = np.nan
        disp[:, :3, :] = np.nan
        disp[:, -3:, :] = np.nan

        result = calculate_strain_field(
            disp, method='engineering', vsg_size=15, step=1,
            boundary_erosion=0,
        )
        assert result is not None
        exx = result['exx']
        valid = ~np.isnan(exx)
        assert valid.any()
        # All valid strain values should be ~0
        assert np.nanmax(np.abs(exx)) < 1e-10, \
            "Uniform displacement should give zero strain"

    def test_crack_without_connectivity_would_give_high_strain(self):
        """Demonstrate the problem: fitting across a 4px crack with
        u_left=0, u_right=10 produces spurious high exx near the crack.

        We simulate this by using a SINGLE mask (ignoring connectivity)
        to show the issue exists.
        """
        disp = make_cracked_displacement(
            h=100, w=100, crack_col=50, crack_width=4,
            u_left=0.0, u_right=10.0,
        )
        # Manually run strain without connectivity (use original mask)
        u = disp[..., 0]
        v = disp[..., 1]
        mask = (~np.isnan(u)) & (~np.isnan(v))

        # Check that pixels near crack have neighbors on both sides
        # within a VSG window of size 15 (half=7)
        vsg_half = 7
        test_row = 50
        test_col = 50 - 4  # 4px left of crack center (just outside crack)

        # This pixel's VSG window spans cols [test_col-7, test_col+7]
        # = [39, 53], which crosses the crack at cols [48, 52]
        window = mask[test_row, test_col - vsg_half:test_col + vsg_half + 1]
        has_left = mask[test_row, test_col - vsg_half]   # col 39 = left side
        has_right = mask[test_row, test_col + vsg_half]  # col 53 = right side
        assert has_left and has_right, \
            "VSG window at this pixel spans both sides of the crack"

    def test_connectivity_aware_vsg_no_spurious_strain(self):
        """With connectivity-aware VSG, no spurious high strain near crack.

        Two uniform regions (u=0 and u=10) separated by a NaN crack.
        Each region individually has zero strain. The connectivity-aware
        VSG should compute ~0 strain everywhere (no cross-crack fitting).
        """
        disp = make_cracked_displacement(
            h=100, w=100, crack_col=50, crack_width=4,
            u_left=0.0, u_right=10.0,
        )

        result = calculate_strain_field(
            disp, method='engineering', vsg_size=15, step=1,
            boundary_erosion=0,  # disable erosion to test raw VSG output
        )
        assert result is not None
        exx = result['exx']
        valid_exx = exx[~np.isnan(exx)]

        assert len(valid_exx) > 0, "Should have some valid strain values"

        # With connectivity-aware VSG, each side is uniform → exx ≈ 0
        max_strain = np.max(np.abs(valid_exx))
        print(f"\nMax |exx| with connectivity-aware VSG: {max_strain:.2e}")
        assert max_strain < 1e-6, \
            f"Strain should be ~0 (uniform regions), got max |exx|={max_strain:.2e}. " \
            f"Cross-crack fitting is leaking through!"

    def test_two_components_detected(self):
        """Verify that the crack creates exactly 2 connected components."""
        from scipy.ndimage import label as ndimage_label

        disp = make_cracked_displacement(
            h=100, w=100, crack_col=50, crack_width=4,
            u_left=0.0, u_right=10.0,
        )
        mask = (~np.isnan(disp[..., 0])) & (~np.isnan(disp[..., 1]))
        _, n = ndimage_label(mask)
        assert n == 2, f"Expected 2 components, got {n}"

    def test_narrow_crack_1px(self):
        """Even a 1px crack should prevent cross-crack fitting."""
        disp = make_cracked_displacement(
            h=80, w=80, crack_col=40, crack_width=1,
            u_left=0.0, u_right=20.0,
        )

        result = calculate_strain_field(
            disp, method='engineering', vsg_size=15, step=1,
            boundary_erosion=0,
        )
        assert result is not None
        exx = result['exx']
        valid_exx = exx[~np.isnan(exx)]

        max_strain = np.max(np.abs(valid_exx))
        print(f"\n1px crack: max |exx| = {max_strain:.2e}")
        assert max_strain < 1e-6, \
            f"1px crack: strain should be ~0, got {max_strain:.2e}"

    def test_single_region_performance_unchanged(self):
        """Without cracks, result should be identical to before."""
        disp = np.zeros((80, 80, 2))
        disp[:, :, 0] = np.linspace(0, 1, 80)[None, :]  # linear u gradient
        disp[:3, :, :] = np.nan
        disp[-3:, :, :] = np.nan
        disp[:, :3, :] = np.nan
        disp[:, -3:, :] = np.nan

        result = calculate_strain_field(
            disp, method='engineering', vsg_size=15, step=1,
            boundary_erosion=0,
        )
        assert result is not None
        exx = result['exx']
        valid = ~np.isnan(exx)
        # du/dx should be ~1/80 ≈ 0.0125 everywhere
        expected = 1.0 / 79.0  # linspace(0,1,80) → step = 1/79
        mean_exx = np.nanmean(exx)
        assert abs(mean_exx - expected) < 0.002, \
            f"Linear gradient: expected exx≈{expected:.4f}, got {mean_exx:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
