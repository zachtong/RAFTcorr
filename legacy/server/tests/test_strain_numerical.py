"""
Numerical correctness tests for strain calculation using analytic displacement fields.

Each test constructs a displacement field with a known analytic solution, runs it
through calculate_strain_field(), and verifies the interior region matches the
expected values within tolerance.  The interior crop [20:-20, 20:-20] avoids
boundary artifacts from the VSG window.
"""

import math

import numpy as np
import pytest

from raft_dic_gui.strain import calculate_strain_field

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_displacement(H, W, u_func, v_func):
    """Build (H, W, 2) displacement field from functions u(x,y), v(x,y)."""
    yy, xx = np.mgrid[0:H, 0:W]
    u = u_func(xx.astype(float), yy.astype(float))
    v = v_func(xx.astype(float), yy.astype(float))
    return np.stack([u, v], axis=-1).astype(np.float64)


def _interior(arr, margin=20):
    """Return the interior region of a 2D array, avoiding boundary effects."""
    return arr[margin:-margin, margin:-margin]


# ---------------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------------

SIZE = 128
VSG = 31
STEP = 1


# ===========================================================================
# 1. Uniform Extension: u = eps * x, v = 0
# ===========================================================================

class TestUniformExtension:
    """
    Uniform uniaxial extension along x.

    Displacement: u = eps * x, v = 0
    Gradients: du/dx = eps, du/dy = 0, dv/dx = 0, dv/dy = 0

    Engineering strain:      exx = eps,  eyy = 0,  exy = 0
    Green-Lagrange strain:   Exx = eps + 0.5*eps^2,  Eyy = 0,  Exy = 0
    """

    @pytest.mark.parametrize("eps", [0.01, 0.05, 0.10])
    def test_engineering(self, eps):
        disp = _make_displacement(SIZE, SIZE, lambda x, y: eps * x, lambda x, y: 0.0 * x)
        result = calculate_strain_field(disp, method='engineering', vsg_size=VSG, step=STEP)

        exx = _interior(result['exx'])
        eyy = _interior(result['eyy'])
        exy = _interior(result['exy'])

        # Mean should be very close to analytic value
        assert abs(np.nanmean(exx) - eps) < 1e-4, f"exx mean {np.nanmean(exx):.6f} != {eps}"
        assert abs(np.nanmean(eyy)) < 1e-4, f"eyy mean {np.nanmean(eyy):.6f} != 0"
        assert abs(np.nanmean(exy)) < 1e-4, f"exy mean {np.nanmean(exy):.6f} != 0"

        # Spatial uniformity: std should be near zero
        assert np.nanstd(exx) < 1e-6, f"exx std {np.nanstd(exx):.8f} too large"
        assert np.nanstd(eyy) < 1e-6, f"eyy std {np.nanstd(eyy):.8f} too large"
        assert np.nanstd(exy) < 1e-6, f"exy std {np.nanstd(exy):.8f} too large"

    @pytest.mark.parametrize("eps", [0.01, 0.05, 0.10])
    def test_green_lagrange(self, eps):
        disp = _make_displacement(SIZE, SIZE, lambda x, y: eps * x, lambda x, y: 0.0 * x)
        result = calculate_strain_field(disp, method='green_lagrange', vsg_size=VSG, step=STEP)

        expected_exx = eps + 0.5 * eps ** 2

        exx = _interior(result['exx'])
        eyy = _interior(result['eyy'])
        exy = _interior(result['exy'])

        assert abs(np.nanmean(exx) - expected_exx) < 1e-4, (
            f"GL exx mean {np.nanmean(exx):.6f} != {expected_exx:.6f}"
        )
        assert abs(np.nanmean(eyy)) < 1e-4, f"GL eyy mean {np.nanmean(eyy):.6f} != 0"
        assert abs(np.nanmean(exy)) < 1e-4, f"GL exy mean {np.nanmean(exy):.6f} != 0"

        assert np.nanstd(exx) < 1e-6, f"GL exx std {np.nanstd(exx):.8f} too large"
        assert np.nanstd(eyy) < 1e-6, f"GL eyy std {np.nanstd(eyy):.8f} too large"
        assert np.nanstd(exy) < 1e-6, f"GL exy std {np.nanstd(exy):.8f} too large"


# ===========================================================================
# 2. Pure Shear: u = gamma * y, v = 0
# ===========================================================================

class TestPureShear:
    """
    Simple shear.

    Displacement: u = gamma * y, v = 0
    Gradients: du/dx = 0, du/dy = gamma, dv/dx = 0, dv/dy = 0

    Engineering strain:      exx = 0,  eyy = 0,  exy = gamma/2
    Green-Lagrange strain:   Exx = 0,  Eyy = 0.5*gamma^2,  Exy = gamma/2
    """

    GAMMA = 0.04

    def test_engineering(self):
        gamma = self.GAMMA
        disp = _make_displacement(SIZE, SIZE, lambda x, y: gamma * y, lambda x, y: 0.0 * x)
        result = calculate_strain_field(disp, method='engineering', vsg_size=VSG, step=STEP)

        exx = _interior(result['exx'])
        eyy = _interior(result['eyy'])
        exy = _interior(result['exy'])

        assert abs(np.nanmean(exx)) < 1e-4, f"exx mean {np.nanmean(exx):.6f} != 0"
        assert abs(np.nanmean(eyy)) < 1e-4, f"eyy mean {np.nanmean(eyy):.6f} != 0"
        assert abs(np.nanmean(exy) - gamma / 2) < 1e-4, (
            f"exy mean {np.nanmean(exy):.6f} != {gamma / 2:.6f}"
        )

        assert np.nanstd(exx) < 1e-6
        assert np.nanstd(eyy) < 1e-6
        assert np.nanstd(exy) < 1e-6

    def test_green_lagrange(self):
        gamma = self.GAMMA
        disp = _make_displacement(SIZE, SIZE, lambda x, y: gamma * y, lambda x, y: 0.0 * x)
        result = calculate_strain_field(disp, method='green_lagrange', vsg_size=VSG, step=STEP)

        exx = _interior(result['exx'])
        eyy = _interior(result['eyy'])
        exy = _interior(result['exy'])

        # GL adds nonlinear terms: Eyy = 0.5 * gamma^2 from du/dy contribution
        expected_eyy = 0.5 * gamma ** 2

        assert abs(np.nanmean(exx)) < 1e-4, f"GL exx mean {np.nanmean(exx):.6f} != 0"
        assert abs(np.nanmean(eyy) - expected_eyy) < 1e-4, (
            f"GL eyy mean {np.nanmean(eyy):.6f} != {expected_eyy:.6f}"
        )
        assert abs(np.nanmean(exy) - gamma / 2) < 1e-4, (
            f"GL exy mean {np.nanmean(exy):.6f} != {gamma / 2:.6f}"
        )

        assert np.nanstd(exx) < 1e-6
        assert np.nanstd(eyy) < 1e-6
        assert np.nanstd(exy) < 1e-6


# ===========================================================================
# 3. Rigid Body Rotation: u = -theta * y, v = theta * x (small angle)
# ===========================================================================

class TestRigidBodyRotation:
    """
    Infinitesimal rigid body rotation.

    Displacement: u = -theta * y, v = theta * x
    Gradients: du/dx = 0, du/dy = -theta, dv/dx = theta, dv/dy = 0

    Engineering strain:      exx = 0, eyy = 0, exy = 0  (antisymmetric gradient)
    Green-Lagrange strain:   Exx = 0.5*theta^2, Eyy = 0.5*theta^2, Exy = 0
      (nonlinear terms: small but nonzero for finite theta)

    Rotation angle from polar decomposition should equal theta (in degrees).
    """

    THETA = 0.02  # ~1.15 degrees

    def test_engineering_strains_near_zero(self):
        theta = self.THETA
        disp = _make_displacement(
            SIZE, SIZE,
            lambda x, y: -theta * y,
            lambda x, y: theta * x,
        )
        result = calculate_strain_field(disp, method='engineering', vsg_size=VSG, step=STEP)

        exx = _interior(result['exx'])
        eyy = _interior(result['eyy'])
        exy = _interior(result['exy'])

        # All engineering strains should be ~0 for rigid rotation
        assert abs(np.nanmean(exx)) < 1e-4, f"exx mean {np.nanmean(exx):.6f} != 0"
        assert abs(np.nanmean(eyy)) < 1e-4, f"eyy mean {np.nanmean(eyy):.6f} != 0"
        assert abs(np.nanmean(exy)) < 1e-4, f"exy mean {np.nanmean(exy):.6f} != 0"

    def test_green_lagrange_strains(self):
        """GL strains for rigid rotation: small nonlinear terms proportional to theta^2."""
        theta = self.THETA
        disp = _make_displacement(
            SIZE, SIZE,
            lambda x, y: -theta * y,
            lambda x, y: theta * x,
        )
        result = calculate_strain_field(disp, method='green_lagrange', vsg_size=VSG, step=STEP)

        exx = _interior(result['exx'])
        eyy = _interior(result['eyy'])
        exy = _interior(result['exy'])

        # GL nonlinear terms: Exx = 0.5*theta^2, Eyy = 0.5*theta^2, Exy = 0
        expected_diag = 0.5 * theta ** 2

        assert abs(np.nanmean(exx) - expected_diag) < 1e-4, (
            f"GL exx mean {np.nanmean(exx):.6f} != {expected_diag:.6f}"
        )
        assert abs(np.nanmean(eyy) - expected_diag) < 1e-4, (
            f"GL eyy mean {np.nanmean(eyy):.6f} != {expected_diag:.6f}"
        )
        assert abs(np.nanmean(exy)) < 1e-4, f"GL exy mean {np.nanmean(exy):.6f} != 0"

    def test_rotation_angle(self):
        """Polar decomposition should recover the rotation angle."""
        theta = self.THETA
        expected_degrees = math.degrees(theta)  # ~1.146 degrees

        disp = _make_displacement(
            SIZE, SIZE,
            lambda x, y: -theta * y,
            lambda x, y: theta * x,
        )
        # Use engineering method; rotation is computed the same way for both methods
        result = calculate_strain_field(disp, method='engineering', vsg_size=VSG, step=STEP)

        rot = _interior(result['rotation'])

        assert abs(np.nanmean(rot) - expected_degrees) < 0.1, (
            f"Rotation mean {np.nanmean(rot):.4f} deg != {expected_degrees:.4f} deg"
        )
        # Spatial uniformity
        assert np.nanstd(rot) < 0.05, f"Rotation std {np.nanstd(rot):.6f} too large"


# ===========================================================================
# 4. Principal Strains: u = eps1 * x, v = eps2 * y
# ===========================================================================

class TestPrincipalStrains:
    """
    Biaxial extension aligned with axes.

    Displacement: u = eps1 * x, v = eps2 * y
    Gradients: du/dx = eps1, du/dy = 0, dv/dx = 0, dv/dy = eps2

    Engineering strain: exx = eps1, eyy = eps2, exy = 0
    Principal strains:  e1 = max(eps1, eps2), e2 = min(eps1, eps2)
    von_mises = sqrt(e1^2 - e1*e2 + e2^2)
    """

    EPS1 = 0.03
    EPS2 = -0.01

    def test_principal_and_vonmises(self):
        eps1, eps2 = self.EPS1, self.EPS2
        disp = _make_displacement(
            SIZE, SIZE,
            lambda x, y: eps1 * x,
            lambda x, y: eps2 * y,
        )
        result = calculate_strain_field(disp, method='engineering', vsg_size=VSG, step=STEP)

        e1 = _interior(result['e1'])
        e2 = _interior(result['e2'])
        vm = _interior(result['von_mises'])
        ms = _interior(result['max_shear'])

        expected_e1 = max(eps1, eps2)
        expected_e2 = min(eps1, eps2)
        expected_vm = math.sqrt(expected_e1 ** 2 - expected_e1 * expected_e2 + expected_e2 ** 2)
        expected_shear = (expected_e1 - expected_e2) / 2.0

        assert abs(np.nanmean(e1) - expected_e1) < 1e-4, (
            f"e1 mean {np.nanmean(e1):.6f} != {expected_e1}"
        )
        assert abs(np.nanmean(e2) - expected_e2) < 1e-4, (
            f"e2 mean {np.nanmean(e2):.6f} != {expected_e2}"
        )
        assert abs(np.nanmean(vm) - expected_vm) < 1e-4, (
            f"von_mises mean {np.nanmean(vm):.6f} != {expected_vm:.6f}"
        )
        assert abs(np.nanmean(ms) - expected_shear) < 1e-4, (
            f"max_shear mean {np.nanmean(ms):.6f} != {expected_shear:.6f}"
        )

        # Spatial uniformity
        assert np.nanstd(e1) < 1e-6
        assert np.nanstd(e2) < 1e-6
        assert np.nanstd(vm) < 1e-6

    def test_exx_eyy_values(self):
        """Verify raw strain components are correct too."""
        eps1, eps2 = self.EPS1, self.EPS2
        disp = _make_displacement(
            SIZE, SIZE,
            lambda x, y: eps1 * x,
            lambda x, y: eps2 * y,
        )
        result = calculate_strain_field(disp, method='engineering', vsg_size=VSG, step=STEP)

        exx = _interior(result['exx'])
        eyy = _interior(result['eyy'])
        exy = _interior(result['exy'])

        assert abs(np.nanmean(exx) - eps1) < 1e-4
        assert abs(np.nanmean(eyy) - eps2) < 1e-4
        assert abs(np.nanmean(exy)) < 1e-4

        assert np.nanstd(exx) < 1e-6
        assert np.nanstd(eyy) < 1e-6
        assert np.nanstd(exy) < 1e-6


# ===========================================================================
# 5. Step Downsampling: verify step > 1 produces smaller output with correct values
# ===========================================================================

class TestStepDownsampling:
    """
    Verify that step > 1 produces a spatially downsampled output whose values
    are still numerically correct.
    """

    EPS = 0.02

    def test_shape_reduction(self):
        """Output spatial dimensions should shrink proportional to step."""
        disp = _make_displacement(
            SIZE, SIZE,
            lambda x, y: self.EPS * x,
            lambda x, y: 0.0 * x,
        )

        r1 = calculate_strain_field(disp, method='engineering', vsg_size=VSG, step=1)
        r4 = calculate_strain_field(disp, method='engineering', vsg_size=VSG, step=4)

        h1, w1 = r1['exx'].shape
        h4, w4 = r4['exx'].shape

        # step=1 should give full size, step=4 should give ~1/4 per dim
        assert h1 == SIZE
        assert w1 == SIZE
        assert h4 == len(range(0, SIZE, 4))
        assert w4 == len(range(0, SIZE, 4))

    def test_values_correct_with_step(self):
        """Interior values should still match the analytic solution at step=4."""
        eps = self.EPS
        disp = _make_displacement(
            SIZE, SIZE,
            lambda x, y: eps * x,
            lambda x, y: 0.0 * x,
        )

        r4 = calculate_strain_field(disp, method='engineering', vsg_size=VSG, step=4)

        # Use a proportionally smaller margin for the downsampled grid
        # step=4 on 128 -> 32 pixels; margin of 5 pixels = 20 original pixels
        margin = 5
        exx = r4['exx'][margin:-margin, margin:-margin]
        eyy = r4['eyy'][margin:-margin, margin:-margin]
        exy = r4['exy'][margin:-margin, margin:-margin]

        assert abs(np.nanmean(exx) - eps) < 1e-4, (
            f"step=4 exx mean {np.nanmean(exx):.6f} != {eps}"
        )
        assert abs(np.nanmean(eyy)) < 1e-4
        assert abs(np.nanmean(exy)) < 1e-4

        assert np.nanstd(exx) < 1e-6
        assert np.nanstd(eyy) < 1e-6
        assert np.nanstd(exy) < 1e-6
