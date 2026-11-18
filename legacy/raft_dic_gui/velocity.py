"""Velocity and displacement magnitude calculations for RAFT-DIC."""

import numpy as np


def calculate_displacement_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Calculate displacement magnitude: M = sqrt(u^2 + v^2)

    Args:
        u: Horizontal displacement component (H, W)
        v: Vertical displacement component (H, W)

    Returns:
        Magnitude array (H, W)
    """
    return np.sqrt(u**2 + v**2)


def calculate_velocity_field(u_curr: np.ndarray, v_curr: np.ndarray,
                            u_prev: np.ndarray, v_prev: np.ndarray,
                            fps: float = 1.0) -> np.ndarray:
    """
    Calculate velocity magnitude from frame-to-frame displacement difference.

    Args:
        u_curr, v_curr: displacement at current frame (H, W)
        u_prev, v_prev: displacement at previous frame (H, W)
        fps: frame rate (Hz), used to convert to physical velocity units

    Returns:
        Velocity magnitude = sqrt(du^2 + dv^2) * fps (H, W)
    """
    du = u_curr - u_prev
    dv = v_curr - v_prev
    return np.sqrt(du**2 + dv**2) * fps


def calculate_velocity_central(
    frames_u: list,
    frames_v: list,
    frame_idx: int,
    fps: float = 1.0,
) -> np.ndarray:
    """Central difference velocity: v[i] = |D[i+1] - D[i-1]| / (2*dt).

    Falls back to forward/backward difference at endpoints.
    Returns velocity MAGNITUDE (always non-negative).

    Args:
        frames_u: list of u-displacement arrays, one per frame
        frames_v: list of v-displacement arrays, one per frame
        frame_idx: which frame to compute velocity for
        fps: frame rate (Hz)

    Returns:
        Velocity magnitude array (H, W)
    """
    T = len(frames_u)
    dt = 1.0 / fps if fps > 0 else 1.0

    if T < 2:
        return np.zeros_like(frames_u[0])

    if frame_idx == 0:
        # Forward difference
        du = frames_u[1] - frames_u[0]
        dv = frames_v[1] - frames_v[0]
        return np.sqrt(du**2 + dv**2) / dt
    elif frame_idx >= T - 1:
        # Backward difference
        du = frames_u[T - 1] - frames_u[T - 2]
        dv = frames_v[T - 1] - frames_v[T - 2]
        return np.sqrt(du**2 + dv**2) / dt
    else:
        # Central difference
        du = frames_u[frame_idx + 1] - frames_u[frame_idx - 1]
        dv = frames_v[frame_idx + 1] - frames_v[frame_idx - 1]
        return np.sqrt(du**2 + dv**2) / (2.0 * dt)
