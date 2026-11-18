"""Tests for velocity calculation methods."""
import numpy as np
import pytest
from raft_dic_gui.velocity import calculate_velocity_field, calculate_velocity_central


class TestForwardDifference:
    """Existing forward difference: v[i] = |D[i] - D[i-1]| * fps"""

    def test_constant_velocity(self):
        """Linear displacement => constant velocity."""
        H, W = 4, 4
        u_curr = np.full((H, W), 10.0)
        v_curr = np.zeros((H, W))
        u_prev = np.full((H, W), 5.0)
        v_prev = np.zeros((H, W))
        vel = calculate_velocity_field(u_curr, v_curr, u_prev, v_prev, fps=2.0)
        np.testing.assert_allclose(vel, 10.0)  # |5| * 2


class TestCentralDifference:
    """Central difference: v[i] = |D[i+1] - D[i-1]| / (2*dt)"""

    def test_linear_ramp(self):
        """Linear u = t should give constant velocity = 1 * fps."""
        T, H, W = 10, 4, 4
        frames_u = [np.full((H, W), float(t)) for t in range(T)]
        frames_v = [np.zeros((H, W)) for _ in range(T)]
        vel = calculate_velocity_central(frames_u, frames_v, frame_idx=5, fps=1.0)
        np.testing.assert_allclose(vel, 1.0, atol=1e-10)

    def test_quadratic_ramp(self):
        """u = t^2 => velocity at t=5 should be 2*5=10 (derivative of t^2 is 2t)."""
        T, H, W = 10, 4, 4
        frames_u = [np.full((H, W), float(t**2)) for t in range(T)]
        frames_v = [np.zeros((H, W)) for _ in range(T)]
        vel = calculate_velocity_central(frames_u, frames_v, frame_idx=5, fps=1.0)
        np.testing.assert_allclose(vel, 10.0, atol=1e-10)

    def test_endpoint_frame0_forward(self):
        """Frame 0 falls back to forward difference."""
        T, H, W = 5, 4, 4
        frames_u = [np.full((H, W), float(t) * 2) for t in range(T)]
        frames_v = [np.zeros((H, W)) for _ in range(T)]
        vel = calculate_velocity_central(frames_u, frames_v, frame_idx=0, fps=1.0)
        np.testing.assert_allclose(vel, 2.0, atol=1e-10)

    def test_endpoint_last_backward(self):
        """Last frame falls back to backward difference."""
        T, H, W = 5, 4, 4
        frames_u = [np.full((H, W), float(t) * 2) for t in range(T)]
        frames_v = [np.zeros((H, W)) for _ in range(T)]
        vel = calculate_velocity_central(frames_u, frames_v, frame_idx=T - 1, fps=1.0)
        np.testing.assert_allclose(vel, 2.0, atol=1e-10)

    def test_single_frame(self):
        """Single frame => zero velocity."""
        frames_u = [np.ones((4, 4))]
        frames_v = [np.zeros((4, 4))]
        vel = calculate_velocity_central(frames_u, frames_v, frame_idx=0, fps=1.0)
        np.testing.assert_allclose(vel, 0.0)

    def test_fps_scaling(self):
        """fps=10 should multiply velocity by 10."""
        T, H, W = 5, 4, 4
        frames_u = [np.full((H, W), float(t)) for t in range(T)]
        frames_v = [np.zeros((H, W)) for _ in range(T)]
        vel = calculate_velocity_central(frames_u, frames_v, frame_idx=2, fps=10.0)
        np.testing.assert_allclose(vel, 10.0, atol=1e-10)
