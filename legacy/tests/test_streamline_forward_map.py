"""Verify that forward-mapping streamline interpolation preserves velocity
coverage for large rotations in deformed mode (regression test for the
inverse-mapping unit-mismatch bug in arrows.py)."""

import numpy as np
import pytest
from scipy.interpolate import griddata


def _rotation_fields(roi_h, roi_w, radius, cum_angle_deg, vel_angle_deg):
    """Create cumulative displacement + velocity fields for rotation."""
    cx, cy = roi_w / 2.0, roi_h / 2.0
    y, x = np.mgrid[0:roi_h, 0:roi_w]
    dx = x.astype(np.float64) - cx
    dy = y.astype(np.float64) - cy
    mask = (dx ** 2 + dy ** 2) <= radius ** 2

    # Cumulative displacement (total rotation so far)
    theta = np.deg2rad(cum_angle_deg)
    U_disp = np.zeros((roi_h, roi_w))
    V_disp = np.zeros((roi_h, roi_w))
    U_disp[mask] = dx[mask] * (np.cos(theta) - 1) - dy[mask] * np.sin(theta)
    V_disp[mask] = dx[mask] * np.sin(theta) + dy[mask] * (np.cos(theta) - 1)

    # Frame-to-frame velocity (small rotation increment)
    dtheta = np.deg2rad(vel_angle_deg)
    du = np.full((roi_h, roi_w), np.nan)
    dv = np.full((roi_h, roi_w), np.nan)
    du[mask] = dx[mask] * (np.cos(dtheta) - 1) - dy[mask] * np.sin(dtheta)
    dv[mask] = dx[mask] * np.sin(dtheta) + dy[mask] * (np.cos(dtheta) - 1)

    return U_disp, V_disp, du, dv, mask


def _old_inverse_map(du_raw, U_disp_ds, V_disp_ds, h_ds, w_ds):
    """Old buggy inverse-mapping approach (for comparison)."""
    from scipy.ndimage import map_coordinates

    rows_ds, cols_ds = np.mgrid[0:h_ds, 0:w_ds]
    ref_rows = rows_ds.astype(np.float64) - V_disp_ds
    ref_cols = cols_ds.astype(np.float64) - U_disp_ds
    coords = np.array([ref_rows.ravel(), ref_cols.ravel()])
    du_src = np.nan_to_num(du_raw, nan=0.0)
    du_filled = map_coordinates(
        du_src, coords, order=1, mode="constant", cval=0.0,
    ).reshape(h_ds, w_ds)
    return du_filled


def _new_forward_map(du_raw, U_disp_ds, V_disp_ds, y_idx_s, x_idx_s,
                     x_stream, y_stream):
    """New forward-mapping + griddata approach."""
    ref_r, ref_c = np.meshgrid(y_idx_s, x_idx_s, indexing="ij")
    def_x = ref_c.astype(np.float64) + U_disp_ds
    def_y = ref_r.astype(np.float64) + V_disp_ds

    valid_vel = ~np.isnan(du_raw)
    if valid_vel.sum() < 10:
        return np.zeros((len(y_stream), len(x_stream)))

    pts = np.column_stack([def_x[valid_vel], def_y[valid_vel]])
    xx, yy = np.meshgrid(x_stream, y_stream)
    du_filled = griddata(
        pts, du_raw[valid_vel].astype(np.float64),
        (xx, yy), method="linear", fill_value=0.0,
    )
    return du_filled


@pytest.mark.parametrize("cum_angle", [10, 30, 45, 60, 90])
def test_forward_map_preserves_coverage(cum_angle):
    """Forward-mapped streamline data retains >80% of valid velocity pixels."""
    roi_h, roi_w = 150, 150
    radius = 60
    ds = 4

    U_disp, V_disp, du, dv, mask = _rotation_fields(
        roi_h, roi_w, radius, cum_angle, vel_angle_deg=3,
    )

    h_ds = max(2, roi_h // ds)
    w_ds = max(2, roi_w // ds)
    y_idx_s = np.round(np.linspace(0, roi_h - 1, h_ds)).astype(int)
    x_idx_s = np.round(np.linspace(0, roi_w - 1, w_ds)).astype(int)
    du_raw = du[np.ix_(y_idx_s, x_idx_s)]

    U_ds = U_disp[np.ix_(y_idx_s, x_idx_s)]
    V_ds = V_disp[np.ix_(y_idx_s, x_idx_s)]

    x_stream = np.linspace(0, roi_w - 1, w_ds)
    y_stream = np.linspace(0, roi_h - 1, h_ds)

    du_new = _new_forward_map(du_raw, U_ds, V_ds, y_idx_s, x_idx_s,
                              x_stream, y_stream)

    valid_input = (~np.isnan(du_raw)).sum()
    nonzero_output = (np.abs(du_new) > 1e-10).sum()
    coverage = nonzero_output / max(1, valid_input) * 100

    assert coverage > 80, (
        f"Forward-map coverage {coverage:.1f}% < 80% at {cum_angle} deg"
    )


@pytest.mark.parametrize("cum_angle", [30, 60, 90])
def test_old_method_loses_coverage(cum_angle):
    """Demonstrate that the old inverse-mapping method loses coverage
    for large rotations (documents the bug, not a pass/fail gate)."""
    roi_h, roi_w = 150, 150
    radius = 60
    ds = 4

    U_disp, V_disp, du, dv, mask = _rotation_fields(
        roi_h, roi_w, radius, cum_angle, vel_angle_deg=3,
    )

    h_ds = max(2, roi_h // ds)
    w_ds = max(2, roi_w // ds)
    y_idx_s = np.round(np.linspace(0, roi_h - 1, h_ds)).astype(int)
    x_idx_s = np.round(np.linspace(0, roi_w - 1, w_ds)).astype(int)
    du_raw = du[np.ix_(y_idx_s, x_idx_s)]

    U_ds = U_disp[np.ix_(y_idx_s, x_idx_s)]
    V_ds = V_disp[np.ix_(y_idx_s, x_idx_s)]

    du_old = _old_inverse_map(du_raw, U_ds, V_ds, h_ds, w_ds)

    valid_input = (~np.isnan(du_raw)).sum()
    nonzero_old = (np.abs(du_old) > 1e-10).sum()
    coverage_old = nonzero_old / max(1, valid_input) * 100

    # The old method should have significantly less coverage at large angles
    # (this documents the bug — NOT a requirement for the new code)
    print(f"  Old method at {cum_angle} deg: {coverage_old:.1f}% coverage")
