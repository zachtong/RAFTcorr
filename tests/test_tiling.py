"""Unit tests for raftcorr.engine.tiling (work windows, tiles and blend weights)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from raftcorr.engine.tiling import Tile, TilePlan, Window, blend_weights, plan_tiles


# -- Window ---------------------------------------------------------------------------------------
def test_window_properties():
    w = Window(2, 3, 12, 8)
    assert (w.width, w.height, w.area) == (10, 5, 50)
    assert w.slices == (slice(3, 8), slice(2, 12))
    assert w.to_tuple() == (2, 3, 12, 8)


def test_grow_clips_to_image():
    w = Window(5, 5, 20, 30).grow(10, (40, 25))  # shape = (h, w)
    assert w.to_tuple() == (0, 0, 25, 40)
    w = Window(10, 10, 20, 20).grow(3, (100, 100))
    assert w.to_tuple() == (7, 7, 23, 23)


def test_at_least_grows_about_centre():
    w = Window(50, 50, 60, 60).at_least(30, (100, 100))
    assert w.to_tuple() == (40, 40, 70, 70)


def test_at_least_shifts_inside_image_at_borders():
    w = Window(0, 0, 10, 10).at_least(30, (100, 100))
    assert w.to_tuple() == (0, 0, 30, 30)
    w = Window(90, 92, 100, 100).at_least(30, (100, 100))
    assert w.to_tuple() == (70, 70, 100, 100)


def test_at_least_keeps_large_windows_and_caps_at_image():
    w = Window(10, 10, 60, 60)
    assert w.at_least(30, (100, 100)) == w
    # image narrower than the requested side: the whole axis
    w = Window(5, 5, 10, 10).at_least(64, (40, 50))
    assert w.to_tuple() == (0, 0, 50, 40)


# -- plan invariants ------------------------------------------------------------------------------
def _check_plan(plan: TilePlan, image_shape, bbox, max_pixels, min_side, margin_px, blend_px):
    h, w = image_shape
    out = plan.output
    bx, by, bw, bh = bbox
    # output covers the ROI bounding box and stays inside the image
    assert out.x0 <= bx and out.y0 <= by and out.x1 >= bx + bw and out.y1 >= by + bh
    assert 0 <= out.x0 < out.x1 <= w and 0 <= out.y0 < out.y1 <= h

    cover = np.zeros((out.height, out.width), dtype=np.int32)
    for t in plan.tiles:
        c, win = t.core, t.window
        # every core inside the output
        assert out.x0 <= c.x0 < c.x1 <= out.x1 and out.y0 <= c.y0 < c.y1 <= out.y1
        # window contains its core and lies inside the image
        assert win.x0 <= c.x0 and win.y0 <= c.y0 and win.x1 >= c.x1 and win.y1 >= c.y1
        assert 0 <= win.x0 and 0 <= win.y0 and win.x1 <= w and win.y1 <= h
        # the halo is at least the plan's margin unless the image edge cuts it
        m = plan.margin_px
        assert win.x0 <= max(0, c.x0 - m) and win.y0 <= max(0, c.y0 - m)
        assert win.x1 >= min(w, c.x1 + m) and win.y1 >= min(h, c.y1 + m)
        # minimum side whenever the image allows it
        assert win.width >= min(min_side, w) and win.height >= min(min_side, h)
        cover[c.y0 - out.y0 : c.y1 - out.y0, c.x0 - out.x0 : c.x1 - out.x0] += 1
    # cores cover the output (union == output)
    assert (cover > 0).all()
    if not plan.single:
        assert all(t.window.area <= max_pixels for t in plan.tiles)
        assert plan.max_window_pixels <= max_pixels
    return cover


CASES = [
    # image_shape, bbox, margin, max_pixels, min_side, blend
    ((512, 768), (100, 80, 500, 300), 32, 256 * 256, 64, 32),
    ((1000, 1300), (0, 0, 1300, 1000), 64, 300 * 300, 128, 32),
    ((1000, 1300), (17, 33, 1201, 911), 48, 250 * 250, 128, 16),
    ((700, 900), (300, 10, 30, 680), 32, 200 * 200, 64, 32),   # tall thin ROI
    ((300, 2000), (0, 100, 2000, 50), 32, 180 * 180, 64, 0),   # one row of tiles, no blending
    ((2048, 2048), (0, 0, 2048, 2048), 32, 512 * 512, 128, 32),
    ((513, 517), (1, 2, 509, 511), 32, 160 * 170, 96, 32),     # non-square budget
]


@pytest.mark.parametrize("image_shape,bbox,margin,max_pixels,min_side,blend", CASES)
def test_plan_invariants(image_shape, bbox, margin, max_pixels, min_side, blend):
    plan = plan_tiles(image_shape, bbox, margin, max_pixels, min_side, blend_px=blend)
    assert not plan.single
    assert not plan.reduced_margin
    assert plan.margin_px == margin
    _check_plan(plan, image_shape, bbox, max_pixels, min_side, margin, blend)


@pytest.mark.parametrize("image_shape,bbox,margin,max_pixels,min_side,blend", CASES)
def test_neighbouring_cores_overlap_by_blend(image_shape, bbox, margin, max_pixels, min_side, blend):
    plan = plan_tiles(image_shape, bbox, margin, max_pixels, min_side, blend_px=blend)
    xs = sorted({(t.core.x0, t.core.x1) for t in plan.tiles})
    ys = sorted({(t.core.y0, t.core.y1) for t in plan.tiles})
    for (a0, a1), (b0, b1) in zip(xs[:-1], xs[1:]):
        assert a1 - b0 >= blend
    for (a0, a1), (b0, b1) in zip(ys[:-1], ys[1:]):
        assert a1 - b0 >= blend


def test_single_window_when_it_fits():
    shape = (480, 640)
    bbox = (100, 120, 200, 150)
    plan = plan_tiles(shape, bbox, margin_px=32, max_pixels=10**7, min_side=64, out_pad=8)
    assert plan.single and not plan.reduced_margin
    t = plan.tiles[0]
    assert t.core == plan.output == Window(92, 112, 308, 278)
    assert t.window == Window(60, 80, 340, 310)
    assert "one window" in plan.describe()
    _check_plan(plan, shape, bbox, 10**7, 64, 32, 32)


def test_single_window_clipped_at_image_and_at_least_min_side():
    shape = (200, 300)
    bbox = (0, 0, 10, 10)
    plan = plan_tiles(shape, bbox, margin_px=5, max_pixels=10**6, min_side=64)
    t = plan.tiles[0]
    assert plan.single
    assert t.window.x0 == 0 and t.window.y0 == 0
    assert t.window.width >= 64 and t.window.height >= 64


def test_single_window_when_area_equals_budget():
    shape = (300, 300)
    bbox = (0, 0, 300, 300)
    plan = plan_tiles(shape, bbox, 32, max_pixels=300 * 300, min_side=64)
    assert plan.single


def test_multi_tile_when_budget_is_one_pixel_short():
    shape = (300, 300)
    plan = plan_tiles(shape, (0, 0, 300, 300), 32, max_pixels=300 * 300 - 1, min_side=64)
    assert not plan.single
    _check_plan(plan, shape, (0, 0, 300, 300), 300 * 300 - 1, 64, 32, 32)


def test_reduced_margin_when_budget_is_tiny():
    shape = (1000, 1000)
    bbox = (0, 0, 1000, 1000)
    max_pixels = 200 * 200
    plan = plan_tiles(shape, bbox, margin_px=128, max_pixels=max_pixels, min_side=64, blend_px=32)
    assert plan.reduced_margin
    assert plan.margin_px < 128
    assert "reduced" in plan.describe()
    cover = _check_plan(plan, shape, bbox, max_pixels, 64, 128, 32)
    assert (cover > 0).all()
    # the windows still respect the budget
    assert plan.max_window_pixels <= max_pixels


def test_memory_error_when_budget_below_min_side():
    with pytest.raises(MemoryError):
        plan_tiles((512, 512), (0, 0, 512, 512), 32, max_pixels=63 * 63, min_side=64)


def test_value_error_when_image_smaller_than_min_side():
    with pytest.raises(ValueError):
        plan_tiles((50, 100), (0, 0, 100, 50), 8, max_pixels=10**6, min_side=64)
    with pytest.raises(ValueError):
        plan_tiles((100, 50), (0, 0, 50, 100), 8, max_pixels=10**6, min_side=64)


def test_negative_margin_is_treated_as_zero():
    plan = plan_tiles((200, 200), (50, 50, 20, 20), -10, max_pixels=10**6, min_side=32, out_pad=0)
    assert plan.margin_px == 0
    assert plan.tiles[0].window.width >= 32


# -- blend weights --------------------------------------------------------------------------------
def test_blend_weights_single_tile_are_ones():
    plan = plan_tiles((480, 640), (100, 120, 200, 150), 32, 10**7, 64)
    t = plan.tiles[0]
    wts = blend_weights(t, plan.output, plan.blend_px)
    assert wts.shape == (t.core.height, t.core.width)
    assert np.array_equal(wts, np.ones_like(wts))


def test_blend_weights_ramp_only_on_inner_edges():
    # one row of tiles: no ramp in y, ramps in x only where another tile is next door
    plan = plan_tiles((300, 2000), (0, 100, 2000, 50), 32, 180 * 180, 64, blend_px=16)
    out = plan.output
    tiles = sorted(plan.tiles, key=lambda t: t.core.x0)
    assert len({t.core.y0 for t in tiles}) == 1 and len(tiles) >= 3
    b = plan.blend_px
    for i, t in enumerate(tiles):
        wts = blend_weights(t, out, b)
        assert wts.shape == (t.core.height, t.core.width)
        assert (wts > 0).all() and (wts <= 1).all()
        # no ramp along y: every row identical, interior columns exactly 1
        assert np.array_equal(wts, np.broadcast_to(wts[:1], wts.shape))
        row = wts[0]
        assert np.all(row[b:-b] == 1.0)
        left_inner = t.core.x0 > out.x0
        right_inner = t.core.x1 < out.x1
        assert left_inner == (i > 0)
        assert right_inner == (i < len(tiles) - 1)
        ramp = (np.arange(b) + 0.5) / b
        if left_inner:
            np.testing.assert_allclose(row[:b], ramp, rtol=0, atol=1e-7)
        else:
            assert np.all(row[:b] == 1.0)
        if right_inner:
            np.testing.assert_allclose(row[-b:], ramp[::-1], rtol=0, atol=1e-7)
        else:
            assert np.all(row[-b:] == 1.0)


def test_blend_weights_corner_tile_of_a_grid():
    plan = plan_tiles((1000, 1300), (0, 0, 1300, 1000), 64, 300 * 300, 128, blend_px=32)
    out = plan.output
    t = min(plan.tiles, key=lambda t: (t.core.y0, t.core.x0))  # top-left tile
    assert t.core.x0 == out.x0 and t.core.y0 == out.y0
    wts = blend_weights(t, out, plan.blend_px)
    b = plan.blend_px
    # outer edges (top, left) are not feathered
    assert np.all(wts[: -b, 0] == 1.0)
    assert np.all(wts[0, : -b] == 1.0)
    # inner edges (bottom, right) ramp down
    assert wts[-1, 0] < 1.0 and wts[0, -1] < 1.0
    assert wts[-1, -1] == pytest.approx((0.5 / b) ** 2)


@pytest.mark.parametrize("image_shape,bbox,margin,max_pixels,min_side,blend", CASES)
def test_blend_weight_sum_positive_everywhere(image_shape, bbox, margin, max_pixels, min_side, blend):
    plan = plan_tiles(image_shape, bbox, margin, max_pixels, min_side, blend_px=blend)
    out = plan.output
    wsum = np.zeros((out.height, out.width))
    for t in plan.tiles:
        c = t.core
        wts = blend_weights(t, out, plan.blend_px)
        wsum[c.y0 - out.y0 : c.y1 - out.y0, c.x0 - out.x0 : c.x1 - out.x0] += wts
    assert (wsum > 0).all()


def test_blend_weights_short_core_caps_ramp_at_half():
    out = Window(0, 0, 100, 10)
    t = Tile(Window(40, 0, 50, 10), Window(0, 0, 100, 10))  # 10 px wide core, both x edges inner
    wts = blend_weights(t, out, 32)
    row = wts[0]
    assert row.shape == (10,)
    np.testing.assert_allclose(row[:5], (np.arange(5) + 0.5) / 5, atol=1e-7)
    np.testing.assert_allclose(row[5:], ((np.arange(5) + 0.5) / 5)[::-1], atol=1e-7)


def test_window_count_is_consistent_with_budget():
    # a sanity bound: the number of tiles is not wildly larger than area / core area
    shape = (2048, 2048)
    plan = plan_tiles(shape, (0, 0, 2048, 2048), 32, 512 * 512, 128, blend_px=32)
    core = 512 - 2 * 32
    per_axis = 1 + math.ceil((plan.output.width - core) / (core - 32))
    assert len(plan.tiles) == per_axis**2


@pytest.mark.parametrize("seed", range(40))
def test_plan_invariants_random(seed):
    rng = np.random.default_rng(seed)
    h, w = int(rng.integers(64, 3000)), int(rng.integers(64, 3000))
    min_side = int(rng.choice([32, 64, 96, 128]))
    min_side = min(min_side, h, w)
    bw, bh = int(rng.integers(1, w + 1)), int(rng.integers(1, h + 1))
    bx, by = int(rng.integers(0, w - bw + 1)), int(rng.integers(0, h - bh + 1))
    margin = int(rng.integers(0, 200))
    blend = int(rng.integers(0, 64))
    side = int(rng.integers(min_side, max(min_side + 1, 1200)))
    max_pixels = side * side + int(rng.integers(0, side))
    plan = plan_tiles((h, w), (bx, by, bw, bh), margin, max_pixels, min_side, blend_px=blend)
    _check_plan(plan, (h, w), (bx, by, bw, bh), max_pixels, min_side, margin, blend)
    if plan.single:
        assert plan.tiles[0].core == plan.output
    else:
        xs = sorted({(t.core.x0, t.core.x1) for t in plan.tiles})
        ys = sorted({(t.core.y0, t.core.y1) for t in plan.tiles})
        assert len(plan.tiles) == len(xs) * len(ys)
        for seq in (xs, ys):
            for (a0, a1), (b0, b1) in zip(seq[:-1], seq[1:]):
                assert a1 - b0 >= min(plan.blend_px, a1 - a0)
