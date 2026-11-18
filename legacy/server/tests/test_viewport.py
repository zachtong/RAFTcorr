"""Tests for viewport downsampling helper."""
import numpy as np
from server.viewport import downsample_for_viewport


def test_no_downsample_when_no_viewport():
    data = np.ones((100, 200), dtype=np.float64)
    bg = np.zeros((100, 200, 3), dtype=np.uint8)
    d2, b2, h2, w2 = downsample_for_viewport(data, bg, 0, 0)
    assert d2.shape == (100, 200)
    assert b2.shape == (100, 200, 3)


def test_no_downsample_when_viewport_larger():
    data = np.ones((100, 200), dtype=np.float64)
    bg = np.zeros((100, 200, 3), dtype=np.uint8)
    d2, b2, h2, w2 = downsample_for_viewport(data, bg, 400, 300)
    assert d2.shape == (100, 200)


def test_downsample_halves():
    data = np.random.rand(200, 400).astype(np.float64)
    bg = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)
    d2, b2, h2, w2 = downsample_for_viewport(data, bg, 200, 100)
    assert h2 == 100
    assert w2 == 200
    assert d2.shape == (100, 200)
    assert b2.shape == (100, 200, 3)


def test_nan_preserved():
    data = np.full((200, 400), np.nan, dtype=np.float64)
    data[50:150, 100:300] = 1.0
    bg = np.zeros((200, 400, 3), dtype=np.uint8)
    d2, b2, h2, w2 = downsample_for_viewport(data, bg, 200, 100)
    # Corners should still be NaN
    assert np.isnan(d2[0, 0])
    # Center should have data
    center = d2[d2.shape[0] // 2, d2.shape[1] // 2]
    assert np.isfinite(center)


def test_grayscale_bg():
    data = np.ones((100, 200), dtype=np.float64)
    bg = np.zeros((100, 200), dtype=np.uint8)  # 2D grayscale
    d2, b2, h2, w2 = downsample_for_viewport(data, bg, 100, 50)
    assert b2.ndim == 2
    assert b2.shape == (50, 100)
