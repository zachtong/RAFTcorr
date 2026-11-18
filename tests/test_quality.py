"""Tracking-quality helpers: intensity residual and the field-of-view check (CPU only)."""

from __future__ import annotations

import numpy as np

from raftcorr import synthetic as syn
from raftcorr.engine.mapping import Grid, inside_frame, intensity_residual

H, W = 128, 160


def _grid() -> Grid:
    return Grid(16, 16, W - 32, H - 32, H, W)


def test_residual_small_for_the_true_field_and_large_for_a_wrong_one():
    field = syn.affine(0.01, 0.0, 0.0, -0.005, tx=3.4, ty=-1.7, cx=W / 2, cy=H / 2)
    ref, dfm = syn.render_pair(H, W, field, seed=2)
    g = _grid()
    truth = syn.evaluate(field, H, W)[:, g.slices[0], g.slices[1]]
    good = intensity_residual(ref, dfm, truth, g)
    bad = intensity_residual(ref, dfm, np.zeros_like(truth), g)
    inner = (slice(8, -8), slice(8, -8))
    assert np.nanmedian(good[inner]) < 3.0          # interpolation + quantisation only
    assert np.nanmedian(bad[inner]) > 5 * np.nanmedian(good[inner])


def test_residual_ignores_a_uniform_brightness_change():
    field = syn.translation(2.0, 1.0)
    ref, dfm = syn.render_pair(H, W, field, seed=4)
    g = _grid()
    truth = syn.evaluate(field, H, W)[:, g.slices[0], g.slices[1]]
    a = intensity_residual(ref, dfm, truth, g)
    b = intensity_residual(ref, dfm + 20.0, truth, g)
    np.testing.assert_allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-3)


def test_residual_is_nan_where_the_field_is_nan():
    ref, dfm = syn.render_pair(H, W, syn.translation(1.0, 0.0), seed=1)
    g = _grid()
    u = np.zeros((2,) + g.shape, np.float32)
    u[:, :10, :] = np.nan
    r = intensity_residual(ref, dfm, u, g)
    assert np.isnan(r[:10]).all()
    assert np.isfinite(r[20:-20, 20:-20]).all()


def test_inside_frame_drops_points_that_leave_the_image():
    g = _grid()
    u = np.zeros((2,) + g.shape, np.float32)
    u[0] = 30.0  # everything moves 30 px right; the right 14 columns of the grid leave the image
    keep = inside_frame(u, g)
    X, _ = g.coords()
    np.testing.assert_array_equal(keep, X + 30.0 <= W - 0.5)
    u[0, 0, 0] = np.nan
    assert not inside_frame(u, g)[0, 0]


def test_suspect_frames_flags_a_jump_but_not_a_slow_rise():
    from raftcorr.post.quality import describe, suspect_frames

    nan = float("nan")
    accumulative = np.array([nan, 3.48, 4.41, 5.29, 6.3, 7.38, 9.58, 26.42, 25.85, 24.27])
    incremental = np.array([nan, 3.48, 4.43, 5.31, 6.19, 6.84, 7.88, 8.21, 9.22, 9.71])
    assert suspect_frames(accumulative) == [7, 8, 9]
    assert suspect_frames(incremental) == []
    assert "incremental" in describe([7, 8, 9], accumulative, accumulative=True)
    assert describe([], incremental, accumulative=True) is None
    assert suspect_frames(np.array([nan, 2.0])) == []
