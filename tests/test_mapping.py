"""Unit tests for raftcorr.engine.mapping (sampling, composition, inverse map, warped masks)."""

from __future__ import annotations

import numpy as np
import pytest

from raftcorr.engine.mapping import (
    Grid,
    compose,
    forward_mask_check,
    gradients,
    inverse_map,
    sample_bilinear,
    to_deformed,
    warp_mask,
)


def _affine_field(grid: Grid, A: np.ndarray, t=(0.0, 0.0), c=(0.0, 0.0)) -> np.ndarray:
    """``U(X) = A (X - c) + t`` on ``grid`` as a (2, h, w) float64 array."""
    X, Y = grid.coords()
    dx, dy = X - c[0], Y - c[1]
    return np.stack([A[0, 0] * dx + A[0, 1] * dy + t[0], A[1, 0] * dx + A[1, 1] * dy + t[1]])


def _rotation(deg: float) -> np.ndarray:
    th = np.radians(deg)
    return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


# -- Grid -----------------------------------------------------------------------------------------
def test_grid_coords_and_round_trip():
    g = Grid(5, 7, 4, 3, 50, 60)
    X, Y = g.coords()
    assert X.shape == (3, 4) and X.dtype == np.float64
    assert X[0, 0] == 5 and Y[0, 0] == 7 and X[-1, -1] == 8 and Y[-1, -1] == 9
    assert g.shape == (3, 4) and g.image_shape == (50, 60)
    assert g.slices == (slice(7, 10), slice(5, 9))
    assert Grid.from_dict(g.to_dict()) == g


# -- sample_bilinear ------------------------------------------------------------------------------
def test_sample_bilinear_exact_on_linear_field():
    h, w = 20, 30
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    x0, y0 = 11, 4
    f = 2.0 + 0.3 * (xs + x0) - 0.7 * (ys + y0)  # linear in image coordinates
    rng = np.random.default_rng(0)
    x = rng.uniform(x0, x0 + w - 1, 500)
    y = rng.uniform(y0, y0 + h - 1, 500)
    got = sample_bilinear(f, x, y, x0, y0)
    np.testing.assert_allclose(got, 2.0 + 0.3 * x - 0.7 * y, rtol=0, atol=1e-12)


def test_sample_bilinear_multichannel_and_shape():
    h, w = 10, 12
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    f = np.stack([xs, ys, xs + ys])
    x = np.array([[0.25, 3.5, 7.75], [1.0, 2.0, 10.5]])
    y = np.array([[0.5, 4.25, 8.0], [9.0, 0.0, 3.3]])
    got = sample_bilinear(f, x, y)
    assert got.shape == (3, 2, 3)
    np.testing.assert_allclose(got[0], x, atol=1e-12)
    np.testing.assert_allclose(got[1], y, atol=1e-12)
    np.testing.assert_allclose(got[2], x + y, atol=1e-12)


def test_sample_bilinear_exact_at_grid_nodes_including_last_row_and_column():
    f = np.arange(12.0).reshape(3, 4)
    ys, xs = np.mgrid[0:3, 0:4].astype(np.float64)
    np.testing.assert_array_equal(sample_bilinear(f, xs, ys), f)


def test_sample_bilinear_ignores_nan_corner():
    f = np.array([[1.0, 2.0], [3.0, np.nan]])
    got = sample_bilinear(f, np.array([0.5]), np.array([0.5]))
    # the three valid corners have equal weight 1/4, renormalised to 1/3
    assert got[0] == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)
    got = sample_bilinear(f, np.array([0.25]), np.array([0.0]))
    assert got[0] == pytest.approx(0.75 * 1.0 + 0.25 * 2.0)


def test_sample_bilinear_nan_in_one_channel_drops_the_corner_for_all_channels():
    f = np.stack([np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[1.0, 2.0], [3.0, np.nan]])])
    got = sample_bilinear(f, np.array([0.5]), np.array([0.5]))
    assert got[0, 0] == pytest.approx(2.0)  # (1 + 2 + 3) / 3
    assert got[1, 0] == pytest.approx(2.0)


def test_sample_bilinear_all_nan_corners_give_nan():
    f = np.array([[np.nan, np.nan, 5.0], [np.nan, np.nan, 5.0]])
    got = sample_bilinear(f, np.array([0.5]), np.array([0.5]))
    assert np.isnan(got[0])


def test_sample_bilinear_outside_gives_nan():
    f = np.ones((5, 6))
    x = np.array([-1.0, -1.5, 6.0, 7.2, 2.0, 2.0, np.nan, 2.0])
    y = np.array([2.0, 2.0, 2.0, 2.0, -1.0, 5.0, 2.0, np.inf])
    got = sample_bilinear(f, x, y)
    assert np.isnan(got).all()


def test_sample_bilinear_within_half_pixel_of_edge_keeps_edge_value():
    f = np.tile(np.arange(6.0), (5, 1))  # value = column
    got = sample_bilinear(f, np.array([-0.5, 5.5]), np.array([2.0, 2.0]))
    np.testing.assert_allclose(got, [0.0, 5.0])


# -- compose --------------------------------------------------------------------------------------
def test_compose_translation_step():
    grid = Grid(10, 8, 30, 20, 60, 70)
    rng = np.random.default_rng(1)
    u_ref = rng.uniform(-2, 2, (2, 20, 30))
    a, b = 1.75, -3.25
    step = np.empty((2, 60, 70))
    step[0], step[1] = a, b
    got = compose(u_ref, grid, step, 0, 0)
    assert got.dtype == np.float32 and got.shape == u_ref.shape
    np.testing.assert_allclose(got[0], u_ref[0] + a, atol=1e-5)
    np.testing.assert_allclose(got[1], u_ref[1] + b, atol=1e-5)


def test_compose_affine_maps_matches_analytic_composition():
    H, W = 120, 140
    grid = Grid(30, 25, 60, 50, H, W)
    A1 = np.array([[0.04, 0.01], [-0.02, 0.03]])
    t1 = np.array([2.5, -1.25])
    u_ref = _affine_field(grid, A1, t1)
    # step field in frame-r coordinates, placed at (sx0, sy0)
    sx0, sy0 = 10, 5
    sgrid = Grid(sx0, sy0, 120, 105, H, W)
    A2 = np.array([[-0.03, 0.02], [0.015, 0.05]])
    t2 = np.array([-0.75, 1.5])
    step = _affine_field(sgrid, A2, t2)
    got = compose(u_ref, grid, step, sx0, sy0)

    X, Y = grid.coords()
    P = np.stack([X, Y]).reshape(2, -1)
    F1, F2 = np.eye(2) + A1, np.eye(2) + A2
    xk = F2 @ (F1 @ P + t1[:, None]) + t2[:, None]
    expected = (xk - P).reshape(2, *grid.shape)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-4)


def test_compose_nan_where_reference_is_nan_or_lands_off_the_step():
    grid = Grid(0, 0, 10, 10, 10, 10)
    u_ref = np.zeros((2, 10, 10))
    u_ref[:, 2, 3] = np.nan
    u_ref[0, 5, 5] = 50.0  # lands far outside the step field
    step = np.ones((2, 10, 10))
    got = compose(u_ref, grid, step)
    assert np.isnan(got[:, 2, 3]).all()
    assert np.isnan(got[:, 5, 5]).all()
    assert np.isfinite(got[:, 0, 0]).all()


def test_gradients_of_linear_field_are_exact():
    grid = Grid(0, 0, 15, 12, 12, 15)
    A = np.array([[0.1, -0.2], [0.3, 0.05]])
    g = gradients(_affine_field(grid, A))
    assert g.shape == (2, 2, 12, 15)
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(g[i, j], A[i, j], atol=1e-12)


# -- inverse_map ----------------------------------------------------------------------------------
def _check_inverse(grid: Grid, A: np.ndarray, t, c, tol_px: float = 0.25):
    u = _affine_field(grid, A, t, c)
    Xc, Yc, (x0, y0, x1, y1) = inverse_map(u, grid, tol_px=tol_px)
    assert Xc.shape == (y1 - y0, x1 - x0) == Yc.shape
    assert 0 <= x0 < x1 <= grid.image_width and 0 <= y0 < y1 <= grid.image_height
    ty, tx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    found = np.isfinite(Xc)
    assert np.array_equal(found, np.isfinite(Yc))

    # analytic inverse X* = c + F^-1 (x - c - t)
    Fi = np.linalg.inv(np.eye(2) + A)
    rx, ry = tx - c[0] - t[0], ty - c[1] - t[1]
    Xs = c[0] + Fi[0, 0] * rx + Fi[0, 1] * ry
    Ys = c[1] + Fi[1, 0] * rx + Fi[1, 1] * ry
    gx, gy = Xs - grid.x0, Ys - grid.y0

    # coverage: every pixel whose material point is >= 1 px inside the grid is found, exactly
    interior = (gx >= 1) & (gx <= grid.width - 2) & (gy >= 1) & (gy <= grid.height - 2)
    assert interior.sum() > 0.6 * grid.width * grid.height
    assert found[interior].all()
    np.testing.assert_allclose(Xc[interior], Xs[interior], atol=1e-6)
    np.testing.assert_allclose(Yc[interior], Ys[interior], atol=1e-6)

    # no false positives: every returned pixel sees a material point of the grid (pixel footprint)
    footprint = (gx >= -0.5) & (gx < grid.width - 0.5) & (gy >= -0.5) & (gy < grid.height - 0.5)
    assert not (found & ~footprint).any()

    # residual measured with the package's own sampling of U is within tol_px at every returned point
    uu = sample_bilinear(u, Xc, Yc, grid.x0, grid.y0)
    res = np.hypot(Xc + uu[0] - tx, Yc + uu[1] - ty)
    assert np.all(res[found] <= tol_px + 1e-9)

    # returned points whose X lies on the grid (no extrapolation) satisfy X + U(X) = x exactly
    Ux = A[0, 0] * (Xc - c[0]) + A[0, 1] * (Yc - c[1]) + t[0]
    Uy = A[1, 0] * (Xc - c[0]) + A[1, 1] * (Yc - c[1]) + t[1]
    true_res = np.hypot(Xc + Ux - tx, Yc + Uy - ty)
    on_grid = found & (Xc >= grid.x0) & (Xc <= grid.x0 + grid.width - 1) \
        & (Yc >= grid.y0) & (Yc <= grid.y0 + grid.height - 1)
    assert np.all(true_res[on_grid] < 1e-6)
    # in the half-pixel band just outside the grid the field is extrapolated with a constant, so
    # the analytic residual can exceed tol_px there, but the point stays within its pixel footprint
    assert np.all(np.hypot(Xc - Xs, Yc - Ys)[found] < 0.75)
    return found, interior


def test_inverse_map_affine():
    grid = Grid(30, 30, 40, 40, 100, 100)
    _check_inverse(grid, np.array([[0.05, 0.02], [-0.03, 0.04]]), (3.3, -2.1), (49.5, 49.5))


def test_inverse_map_translation_offset_grid():
    grid = Grid(12, 20, 35, 25, 80, 90)
    _check_inverse(grid, np.zeros((2, 2)), (7.4, -5.6), (0.0, 0.0))


@pytest.mark.parametrize("deg", [10, 45, 60, 90, 135, 180])
def test_inverse_map_large_rotation_about_grid_centre(deg):
    grid = Grid(30, 30, 40, 40, 100, 100)
    _check_inverse(grid, _rotation(deg) - np.eye(2), (0.0, 0.0), (49.5, 49.5))


@pytest.mark.parametrize("deg", [60, 90])
def test_inverse_map_large_rotation_rectangular_grid(deg):
    grid = Grid(30, 30, 41, 37, 100, 100)
    _check_inverse(grid, _rotation(deg) - np.eye(2), (0.0, 0.0), (50.0, 48.0))


def test_inverse_map_rotation_with_stretch():
    grid = Grid(30, 30, 40, 40, 110, 110)
    A = _rotation(60) @ np.diag([1.2, 0.9]) - np.eye(2)
    _check_inverse(grid, A, (2.0, -1.0), (49.5, 49.5))


def test_inverse_map_respects_mask():
    grid = Grid(10, 10, 30, 30, 60, 60)
    u = _affine_field(grid, np.zeros((2, 2)), (4.0, 3.0))
    mask = np.zeros(grid.shape, dtype=bool)
    mask[5:15, 5:20] = True
    Xc, Yc, (x0, y0, x1, y1) = inverse_map(u, grid, mask)
    found = np.isfinite(Xc)
    ty, tx = np.mgrid[y0:y1, x0:x1]
    # deformed region = mask translated by (4, 3)
    expected = (tx >= 10 + 5 + 4) & (tx < 10 + 20 + 4) & (ty >= 10 + 5 + 3) & (ty < 10 + 15 + 3)
    assert np.array_equal(found, expected)
    np.testing.assert_allclose(Xc[found], tx[found] - 4.0, atol=1e-9)


def test_inverse_map_empty_mask_or_all_nan():
    grid = Grid(0, 0, 10, 10, 20, 20)
    u = np.zeros((2, 10, 10))
    X, Y, win = inverse_map(u, grid, np.zeros((10, 10), dtype=bool))
    assert X.size == 0 and Y.size == 0 and win == (0, 0, 0, 0)
    X, Y, win = inverse_map(np.full((2, 10, 10), np.nan), grid)
    assert X.size == 0 and win == (0, 0, 0, 0)


# -- warp_mask / to_deformed / forward_mask_check -------------------------------------------------
@pytest.mark.parametrize("a,b", [(5.0, -3.0), (5.3, -2.7), (-4.6, 6.2)])
def test_warp_mask_translation_is_translated_rectangle(a, b):
    H, W = 80, 90
    grid = Grid(20, 25, 30, 20, H, W)
    mask = np.ones(grid.shape, dtype=bool)
    u = _affine_field(grid, np.zeros((2, 2)), (a, b))
    got = warp_mask(mask, u, grid)
    assert got.shape == (H, W) and got.dtype == bool
    yy, xx = np.mgrid[0:H, 0:W]
    # the translated rectangle by the pixel-centre rule, with a 1 px tolerance band at the edges
    inner = (xx >= 20 + a + 1) & (xx <= 20 + 29 + a - 1) & (yy >= 25 + b + 1) & (yy <= 25 + 19 + b - 1)
    outer = (xx >= 20 + a - 1) & (xx <= 20 + 29 + a + 1) & (yy >= 25 + b - 1) & (yy <= 25 + 19 + b + 1)
    assert got[inner].all()
    assert not got[~outer].any()
    # area preserved by the nearest-pixel footprint rule
    assert abs(int(got.sum()) - 30 * 20) <= 30 + 20


def test_warp_mask_integer_translation_is_exact():
    H, W = 60, 60
    grid = Grid(10, 12, 20, 15, H, W)
    mask = np.ones(grid.shape, dtype=bool)
    u = _affine_field(grid, np.zeros((2, 2)), (6.0, 4.0))
    expected = np.zeros((H, W), dtype=bool)
    expected[12 + 4 : 12 + 4 + 15, 10 + 6 : 10 + 6 + 20] = True
    assert np.array_equal(warp_mask(mask, u, grid), expected)


def test_to_deformed_constant_values():
    grid = Grid(10, 10, 30, 25, 70, 70)
    u = _affine_field(grid, np.array([[0.02, 0.01], [0.0, -0.03]]), (3.2, 1.7))
    vals = np.full(grid.shape, 7.5, dtype=np.float32)
    out, win = to_deformed(vals, u, grid)
    X, _Y, win2 = inverse_map(u, grid)
    assert win == win2
    assert out.shape == (win[3] - win[1], win[2] - win[0]) and out.dtype == np.float32
    assert np.array_equal(np.isfinite(out), np.isfinite(X))
    assert np.isfinite(out).sum() > 0.8 * 30 * 25
    np.testing.assert_allclose(out[np.isfinite(out)], 7.5, atol=1e-6)


def test_to_deformed_multichannel_and_linear_values():
    grid = Grid(10, 10, 30, 25, 70, 70)
    u = _affine_field(grid, np.zeros((2, 2)), (2.5, -1.5))
    X, Y = grid.coords()
    vals = np.stack([X, Y]).astype(np.float32)
    out, (x0, y0, x1, y1) = to_deformed(vals, u, grid)
    assert out.shape == (2, y1 - y0, x1 - x0)
    ty, tx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    fin = np.isfinite(out[0])
    # values carried by the material point: out(x) = X(x) = x - t (inside the grid)
    inside = fin & (tx - 2.5 >= grid.x0) & (tx - 2.5 <= grid.x0 + grid.width - 1)
    np.testing.assert_allclose(out[0][inside], tx[inside] - 2.5, atol=1e-4)


def test_to_deformed_empty():
    grid = Grid(0, 0, 8, 8, 16, 16)
    u = np.zeros((2, 8, 8))
    out, win = to_deformed(np.ones((3, 8, 8)), u, grid, np.zeros((8, 8), dtype=bool))
    assert out.shape == (3, 0, 0) and win == (0, 0, 0, 0)
    out, win = to_deformed(np.ones((8, 8)), u, grid, np.zeros((8, 8), dtype=bool))
    assert out.shape == (0, 0)


def test_forward_mask_check():
    H, W = 40, 50
    grid = Grid(5, 5, 20, 20, H, W)
    u = _affine_field(grid, np.zeros((2, 2)), (3.0, 2.0))
    frame_mask = np.ones((H, W), dtype=bool)
    frame_mask[15:20, 15:22] = False  # a hole in the deformed frame
    keep = forward_mask_check(u, grid, frame_mask)
    X, Y = grid.coords()
    in_hole = (X + 3 >= 15) & (X + 3 < 22) & (Y + 2 >= 15) & (Y + 2 < 20)
    assert keep.shape == grid.shape
    assert np.array_equal(keep, ~in_hole)


def test_forward_mask_check_off_image_and_nan_are_kept():
    H, W = 20, 20
    grid = Grid(0, 0, 10, 10, H, W)
    u = np.zeros((2, 10, 10))
    u[0, 0, 0] = -50.0       # off image
    u[:, 1, 1] = np.nan       # untracked
    frame_mask = np.zeros((H, W), dtype=bool)  # everything excluded on-image
    keep = forward_mask_check(u, grid, frame_mask)
    assert keep[0, 0] and keep[1, 1]
    assert keep.sum() == 2


def test_sample_bilinear_of_an_empty_field_is_nan():
    from raftcorr.engine.mapping import sample_bilinear

    out = sample_bilinear(np.zeros((0, 0), np.float32), np.array([1.0, 2.0]), np.array([0.5, 3.0]))
    assert out.shape == (2,) and np.isnan(out).all()
    out2 = sample_bilinear(np.zeros((2, 0, 5)), np.array([[1.0]]), np.array([[0.0]]))
    assert out2.shape == (2, 1, 1) and np.isnan(out2).all()
