"""Unit tests for raftcorr.synthetic (speckle pairs with a known displacement)."""

from __future__ import annotations

import numpy as np
import pytest

from raftcorr.synthetic import affine, evaluate, render_pair, sinusoid, speckle, translation


def test_speckle_shape_and_range():
    s = speckle(20, 30, seed=1)
    assert s.shape == (40, 60) and s.dtype == np.float64
    assert s.min() >= 128 - 107 and s.max() <= 128 + 107
    assert s.std() > 30  # real contrast
    assert speckle(10, 12, oversample=3).shape == (30, 36)
    np.testing.assert_array_equal(speckle(20, 30, seed=1), s)
    assert not np.array_equal(speckle(20, 30, seed=2), s)


def test_render_pair_zero_field_gives_identical_images():
    ref, dfm = render_pair(48, 64, translation(0.0, 0.0), seed=3)
    assert ref.shape == dfm.shape == (48, 64)
    assert ref.dtype == dfm.dtype == np.float32
    np.testing.assert_array_equal(ref, dfm)
    ref, dfm = render_pair(48, 64, translation(0.0, 0.0), seed=3, quantize=False)
    np.testing.assert_allclose(ref, dfm, atol=1e-6)


def test_render_pair_quantized_and_in_range():
    ref, dfm = render_pair(32, 32, affine(0.01, 0.0, 0.0, -0.01, cx=16, cy=16), seed=0)
    for img in (ref, dfm):
        assert img.min() >= 0 and img.max() <= 255
        np.testing.assert_array_equal(img, np.round(img))


@pytest.mark.parametrize("du,dv", [(3.0, 2.0), (-4.0, 1.0), (0.0, -5.0)])
def test_integer_translation_matches_roll_in_interior(du, dv):
    h, w = 48, 56
    ref, dfm = render_pair(h, w, translation(du, dv), seed=4)
    # a material point at X appears at x = X + d:  I_def(x) = I_ref(x - d)
    rolled = np.roll(ref, shift=(int(dv), int(du)), axis=(0, 1))
    m = 6
    np.testing.assert_allclose(dfm[m:-m, m:-m], rolled[m:-m, m:-m], atol=1e-3)


def test_subpixel_translation_is_not_an_integer_roll():
    ref, dfm = render_pair(40, 40, translation(0.5, 0.0), seed=5, quantize=False)
    inner = (slice(5, -5), slice(5, -5))
    for s in (0, 1):
        assert np.abs(dfm[inner] - np.roll(ref, s, axis=1)[inner]).mean() > 1.0


def test_noise_is_reproducible():
    a = render_pair(24, 24, translation(1.0, 0.0), seed=7, noise_std=2.0)
    b = render_pair(24, 24, translation(1.0, 0.0), seed=7, noise_std=2.0)
    c = render_pair(24, 24, translation(1.0, 0.0), seed=7, noise_std=0.0)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])


def test_evaluate_shape_and_values():
    f = affine(0.01, -0.02, 0.03, 0.04, tx=1.0, ty=-2.0, cx=5.0, cy=3.0)
    g = evaluate(f, 6, 9)
    assert g.shape == (2, 6, 9) and g.dtype == np.float32
    ys, xs = np.mgrid[0:6, 0:9].astype(np.float64)
    np.testing.assert_allclose(g[0], 0.01 * (xs - 5) - 0.02 * (ys - 3) + 1.0, atol=1e-6)
    np.testing.assert_allclose(g[1], 0.03 * (xs - 5) + 0.04 * (ys - 3) - 2.0, atol=1e-6)
    t = evaluate(translation(1.5, -0.5), 4, 5)
    assert np.all(t[0] == 1.5) and np.all(t[1] == -0.5)


def test_sinusoid_fields():
    gx = evaluate(sinusoid(2.0, 16.0, "x"), 8, 8)
    ys = np.arange(8.0)
    np.testing.assert_allclose(gx[0][:, 0], 2.0 * np.sin(2 * np.pi * ys / 16.0), atol=1e-6)
    assert np.all(gx[1] == 0)
    gy = evaluate(sinusoid(2.0, 16.0, "y"), 8, 8)
    np.testing.assert_allclose(gy[1][0], 2.0 * np.sin(2 * np.pi * np.arange(8.0) / 16.0), atol=1e-6)
    assert np.all(gy[0] == 0)
