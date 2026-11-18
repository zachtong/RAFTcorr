"""Unit tests for raftcorr.roi.shapes and raftcorr.roi.masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from raftcorr.roi.masks import (
    MaskHistory,
    PackedMask,
    bbox_of,
    list_mask_files,
    load_mask,
    match_mask_files,
    remove_small_regions,
    resize_nearest,
    save_mask,
    smooth_edges,
)
from raftcorr.roi.shapes import Circle, Polygon, Rect, apply_shape, brush_stroke, rasterize, shape_from_dict


def _point_in_polygon(points, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Brute-force even-odd test (reference implementation)."""
    inside = np.zeros(px.shape, dtype=bool)
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        cond = (y1 > py) != (y2 > py)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (px < xint)
    return inside


def _shoelace(points) -> float:
    p = np.asarray(points, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# -- Rect -----------------------------------------------------------------------------------------
def test_rect_pixel_centre_rule_half_integer_bounds():
    m = rasterize(Rect(1.5, 1.5, 4.5, 3.5), 10, 10)
    expected = np.zeros((10, 10), dtype=bool)
    expected[2:4, 2:5] = True
    assert np.array_equal(m, expected)
    assert m.sum() == (4.5 - 1.5) * (3.5 - 1.5)


def test_rect_fractional_bounds_area():
    r = Rect(2.2, 3.7, 17.9, 11.4)
    m = rasterize(r, 30, 30)
    # centres in [2.2, 17.9] x [3.7, 11.4]: columns 3..17, rows 4..11
    assert m.sum() == 15 * 8
    assert abs(m.sum() - (17.9 - 2.2) * (11.4 - 3.7)) < 2 * ((17.9 - 2.2) + (11.4 - 3.7))


def test_rect_reversed_corners_are_normalised():
    a = rasterize(Rect(1.5, 1.5, 4.5, 3.5), 10, 10)
    b = rasterize(Rect(4.5, 3.5, 1.5, 1.5), 10, 10)
    assert np.array_equal(a, b)


def test_rect_partially_and_fully_outside():
    m = rasterize(Rect(-10.0, -10.0, 4.6, 2.2), 8, 8)
    expected = np.zeros((8, 8), dtype=bool)
    expected[0:3, 0:5] = True
    assert np.array_equal(m, expected)
    assert not rasterize(Rect(20.0, 20.0, 30.0, 30.0), 8, 8).any()
    assert not rasterize(Rect(-5.0, -5.0, -1.0, -1.0), 8, 8).any()


# -- Circle ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("cx,cy,r", [(30.0, 30.0, 20.0), (31.3, 28.7, 12.5), (25.5, 25.5, 7.0)])
def test_circle_pixel_centre_rule_and_area(cx, cy, r):
    m = rasterize(Circle(cx, cy, r), 64, 64)
    yy, xx = np.mgrid[0:64, 0:64]
    expected = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    assert np.array_equal(m, expected)
    assert abs(m.sum() - np.pi * r * r) < 2 * r


def test_circle_zero_radius_and_clipping():
    assert not rasterize(Circle(5.0, 5.0, 0.0), 10, 10).any()
    assert not rasterize(Circle(5.0, 5.0, -2.0), 10, 10).any()
    m = rasterize(Circle(0.0, 0.0, 5.0), 20, 20)  # quarter inside
    yy, xx = np.mgrid[0:20, 0:20]
    assert np.array_equal(m, xx**2 + yy**2 <= 25)
    assert not rasterize(Circle(100.0, 100.0, 5.0), 20, 20).any()


# -- Polygon --------------------------------------------------------------------------------------
def test_polygon_concave_l_shape_exact():
    pts = ((1.5, 1.5), (10.5, 1.5), (10.5, 4.5), (4.5, 4.5), (4.5, 10.5), (1.5, 10.5))
    m = rasterize(Polygon(pts), 16, 16)
    expected = np.zeros((16, 16), dtype=bool)
    expected[2:5, 2:11] = True   # horizontal bar: rows 2..4, cols 2..10
    expected[5:11, 2:5] = True   # vertical bar: rows 5..10, cols 2..4
    assert np.array_equal(m, expected)
    assert m.sum() == _shoelace(pts) == 45


def test_polygon_concave_arrow_matches_reference():
    pts = ((5.3, 5.1), (40.7, 20.2), (5.9, 35.6), (18.4, 20.3))  # concave "arrow head"
    m = rasterize(Polygon(pts), 48, 48)
    yy, xx = np.mgrid[0:48, 0:48].astype(np.float64)
    assert np.array_equal(m, _point_in_polygon(pts, xx, yy))
    # the notch is empty
    assert not m[20, 10]
    assert abs(m.sum() - _shoelace(pts)) < 0.5 * 4 * 40


def test_triangle_area_and_reference():
    pts = ((2.3, 3.1), (30.7, 5.9), (12.2, 27.4))
    m = rasterize(Polygon(pts), 32, 32)
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float64)
    assert np.array_equal(m, _point_in_polygon(pts, xx, yy))
    perimeter = sum(np.hypot(*(np.subtract(pts[i], pts[i - 1]))) for i in range(3))
    assert abs(m.sum() - _shoelace(pts)) < 0.5 * perimeter


def test_polygon_orientation_does_not_matter():
    pts = ((2.3, 3.1), (30.7, 5.9), (12.2, 27.4))
    a = rasterize(Polygon(pts), 32, 32)
    b = rasterize(Polygon(tuple(reversed(pts))), 32, 32)
    assert np.array_equal(a, b)


def test_polygon_points_out_of_image():
    pts = ((-10.2, -7.7), (25.6, 3.3), (14.1, 40.9), (-3.3, 18.8))
    m = rasterize(Polygon(pts), 20, 24)
    assert m.shape == (20, 24)
    yy, xx = np.mgrid[0:20, 0:24].astype(np.float64)
    assert np.array_equal(m, _point_in_polygon(pts, xx, yy))
    assert not rasterize(Polygon(((50.0, 50.0), (60.0, 50.0), (55.0, 60.0))), 20, 20).any()


@pytest.mark.parametrize("seed", range(5))
def test_random_polygons_match_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 9))
    pts = tuple(map(tuple, rng.uniform(-5, 45, (n, 2))))  # possibly self-intersecting: even-odd
    m = rasterize(Polygon(pts), 40, 40)
    yy, xx = np.mgrid[0:40, 0:40].astype(np.float64)
    assert np.array_equal(m, _point_in_polygon(pts, xx, yy))


def test_degenerate_polygon_is_empty():
    assert not rasterize(Polygon(((1.0, 1.0), (5.0, 5.0))), 10, 10).any()
    assert not rasterize(Polygon(()), 10, 10).any()


def test_rasterize_rejects_non_shapes():
    with pytest.raises(TypeError):
        rasterize("rect", 10, 10)  # type: ignore[arg-type]


# -- composition ----------------------------------------------------------------------------------
def test_apply_shape_add_and_cut_do_not_modify_input():
    base = np.zeros((20, 20), dtype=bool)
    base[5:15, 5:15] = True
    snapshot = base.copy()
    added = apply_shape(base, Rect(10.0, 10.0, 18.0, 18.0), "add")
    cut = apply_shape(base, Circle(10.0, 10.0, 3.0), "cut")
    assert np.array_equal(base, snapshot)
    assert added is not base and cut is not base
    assert added[17, 17] and added[6, 6]
    assert not cut[10, 10] and cut[6, 6]
    assert np.array_equal(added, base | rasterize(Rect(10.0, 10.0, 18.0, 18.0), 20, 20))
    assert np.array_equal(cut, base & ~rasterize(Circle(10.0, 10.0, 3.0), 20, 20))


def test_apply_shape_invalid_mode():
    with pytest.raises(ValueError):
        apply_shape(np.zeros((5, 5), dtype=bool), Rect(0, 0, 1, 1), "xor")


# -- brush ----------------------------------------------------------------------------------------
def test_brush_stroke_capsule():
    r = 3.0
    m = brush_stroke(30, 40, [(10.0, 10.0), (30.0, 10.0)], r)
    yy, xx = np.mgrid[0:30, 0:40].astype(np.float64)
    # distance to the segment from (10, 10) to (30, 10)
    dx = np.clip(xx, 10.0, 30.0) - xx
    dist = np.hypot(dx, yy - 10.0)
    assert m[dist <= 2.9].all()      # covered, allowing for the spacing of the stamps
    assert not m[dist > r].any()     # nothing beyond the brush radius


def test_brush_stroke_single_point_is_disc():
    m = brush_stroke(20, 20, [(9.5, 8.2)], 4.0)
    yy, xx = np.mgrid[0:20, 0:20]
    assert np.array_equal(m, (xx - 9.5) ** 2 + (yy - 8.2) ** 2 <= 16.0)


def test_brush_stroke_empty_cases():
    assert not brush_stroke(10, 10, [], 3.0).any()
    assert not brush_stroke(10, 10, [(5.0, 5.0)], 0.0).any()
    assert not brush_stroke(10, 10, [(50.0, 50.0), (60.0, 60.0)], 2.0).any()


# -- dict round trip ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [Rect(1.0, 2.5, 7.0, 9.25), Circle(3.5, 4.5, 2.0), Polygon(((0, 0), (5, 0), (2.5, 4)))],
)
def test_shape_dict_round_trip(shape):
    d = shape.to_dict()
    back = shape_from_dict(d)
    assert back == shape
    assert type(back) is type(shape)


def test_shape_from_dict_accepts_poly_and_rejects_unknown():
    p = shape_from_dict({"type": "poly", "points": [[0, 0], [4, 0], [0, 3]]})
    assert isinstance(p, Polygon) and p.points == ((0.0, 0.0), (4.0, 0.0), (0.0, 3.0))
    with pytest.raises(ValueError):
        shape_from_dict({"type": "ellipse"})
    with pytest.raises(ValueError):
        shape_from_dict({})


# -- masks ----------------------------------------------------------------------------------------
def test_bbox_of():
    m = np.zeros((20, 30), dtype=bool)
    m[3:8, 2:11] = True
    m[15, 25] = True
    assert bbox_of(m) == (2, 3, 24, 13)
    m2 = np.zeros((5, 5), dtype=bool)
    m2[2, 3] = True
    assert bbox_of(m2) == (3, 2, 1, 1)
    assert bbox_of(np.zeros((5, 5), dtype=bool)) is None


def test_resize_nearest():
    m = np.array([[True, False], [False, True]])
    up = resize_nearest(m, 4, 4)
    assert np.array_equal(up, np.kron(m, np.ones((2, 2), dtype=bool)))
    same = resize_nearest(m, 2, 2)
    assert np.array_equal(same, m) and same is not m
    big = np.zeros((4, 4), dtype=bool)
    big[1, 3] = True
    big[3, 1] = True
    down = resize_nearest(big, 2, 2)  # samples rows/cols 1 and 3 (pixel centres)
    assert np.array_equal(down, np.array([[False, True], [True, False]]))
    assert resize_nearest(m, 3, 5).shape == (3, 5)


def test_remove_small_regions_8_connected():
    m = np.zeros((20, 20), dtype=bool)
    m[1, 1] = m[2, 2] = m[3, 3] = True          # diagonal chain: one 8-connected region of 3
    m[10:12, 10:15] = True                       # region of 10
    m[18, 0] = True                              # region of 1
    out = remove_small_regions(m, 3)
    assert out[1, 1] and out[2, 2] and out[3, 3]
    assert out[10:12, 10:15].all()
    assert not out[18, 0]
    out = remove_small_regions(m, 5)
    assert not out[1:4, 1:4].any() and out[10:12, 10:15].all()
    snapshot = m.copy()
    out0 = remove_small_regions(m, 0)
    assert np.array_equal(out0, m) and out0 is not m
    assert np.array_equal(m, snapshot)


def test_smooth_edges():
    m = np.zeros((60, 60), dtype=bool)
    m[10:50, 10:50] = True
    m[5, 55] = True  # isolated pixel
    out = smooth_edges(m, 4.0)
    assert out.dtype == bool
    assert out[15:45, 15:45].all()            # interior kept
    assert not out[5, 55]                      # specks removed
    assert out[30, 10] and not out[30, 9]      # straight edges stay in place
    assert not out[10, 10]                     # corners rounded
    assert np.array_equal(smooth_edges(m, 0.0), m)


def test_save_and_load_mask_round_trip(tmp_path: Path):
    rng = np.random.default_rng(3)
    m = rng.random((37, 51)) > 0.5
    p = tmp_path / "mask.png"
    save_mask(p, m)
    back = load_mask(p)
    assert back.dtype == bool and np.array_equal(back, m)
    resized = load_mask(p, shape=(74, 102))
    assert np.array_equal(resized, np.kron(m, np.ones((2, 2), dtype=bool)))


def test_load_mask_16_bit_and_threshold(tmp_path: Path):
    from PIL import Image

    a = np.zeros((10, 10), dtype=np.uint16)
    a[2:5, 3:7] = 4095
    p = tmp_path / "m16.png"
    Image.fromarray(a).save(p)
    m = load_mask(p)
    assert np.array_equal(m, a > 0)


def test_list_mask_files(tmp_path: Path):
    for name in ("m10.png", "m2.png", "m1.tif", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sub.png").mkdir()
    assert [p.name for p in list_mask_files(tmp_path)] == ["m1.tif", "m2.png", "m10.png"]


def test_match_mask_files_sequential():
    files = [Path("m10.png"), Path("m1.png"), Path("m2.png")]
    got = match_mask_files(files, 5)
    assert {k: p.name for k, p in got.items()} == {0: "m1.png", 1: "m2.png", 2: "m10.png"}
    got = match_mask_files(files, 2)
    assert {k: p.name for k, p in got.items()} == {0: "m1.png", 1: "m2.png"}


def test_match_mask_files_by_number():
    files = [Path("cam2_mask_003.png"), Path("cam2_mask_0.png"), Path("roi.png"), Path("cam2_mask_9.png"),
             Path("s1_frame12.tif")]
    got = match_mask_files(files, 10, strategy="by_number")
    assert {k: p.name for k, p in got.items()} == {0: "cam2_mask_0.png", 3: "cam2_mask_003.png",
                                                   9: "cam2_mask_9.png"}
    with pytest.raises(ValueError):
        match_mask_files(files, 10, strategy="alphabetical")


@pytest.mark.parametrize("shape", [(1, 1), (7, 13), (8, 8), (31, 17)])
def test_packed_mask_round_trip(shape):
    rng = np.random.default_rng(sum(shape))
    m = rng.random(shape) > 0.4
    pk = PackedMask.pack(m)
    assert pk.shape == shape
    assert len(pk.bits) == -(-m.size // 8)
    back = pk.unpack()
    assert back.dtype == bool and back.shape == shape and np.array_equal(back, m)


def test_packed_mask_accepts_non_bool_input():
    m = np.array([[0, 255], [7, 0]], dtype=np.uint8)
    assert np.array_equal(PackedMask.pack(m).unpack(), m.astype(bool))


# -- MaskHistory ----------------------------------------------------------------------------------
def _apply(masks: dict, result) -> None:
    frame, mask = result
    if mask is None:
        masks.pop(frame, None)
    else:
        masks[frame] = mask


def _edit(history: MaskHistory, masks: dict, frame: int, new) -> None:
    history.record(frame, masks.get(frame))
    if new is None:
        masks.pop(frame, None)
    else:
        masks[frame] = new


def _rand(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((6, 9)) > 0.5


def test_mask_history_undo_redo_across_frames():
    A, B, C = _rand(1), _rand(2), _rand(3)
    h = MaskHistory()
    masks: dict[int, np.ndarray] = {}
    assert not h.can_undo and not h.can_redo
    assert h.undo(masks) is None and h.redo(masks) is None

    _edit(h, masks, 0, A)   # frame 0: none -> A
    _edit(h, masks, 0, B)   # frame 0: A -> B
    _edit(h, masks, 2, C)   # frame 2: none -> C
    assert h.can_undo and not h.can_redo

    r = h.undo(masks)
    assert r[0] == 2 and r[1] is None
    _apply(masks, r)
    assert 2 not in masks and np.array_equal(masks[0], B)

    r = h.undo(masks)
    assert r[0] == 0 and np.array_equal(r[1], A)
    _apply(masks, r)
    assert np.array_equal(masks[0], A)

    r = h.undo(masks)
    assert r == (0, None)
    _apply(masks, r)
    assert masks == {}
    assert not h.can_undo and h.can_redo
    assert h.undo(masks) is None

    # redo everything in order
    r = h.redo(masks)
    assert r[0] == 0 and np.array_equal(r[1], A)
    _apply(masks, r)
    r = h.redo(masks)
    assert r[0] == 0 and np.array_equal(r[1], B)
    _apply(masks, r)
    r = h.redo(masks)
    assert r[0] == 2 and np.array_equal(r[1], C)
    _apply(masks, r)
    assert not h.can_redo and h.redo(masks) is None
    assert np.array_equal(masks[0], B) and np.array_equal(masks[2], C)

    # and undo again after the redos
    r = h.undo(masks)
    assert r == (2, None)


def test_mask_history_removing_a_mask_is_undoable():
    A = _rand(4)
    h = MaskHistory()
    masks = {1: A}
    _edit(h, masks, 1, None)  # delete the mask of frame 1
    assert masks == {}
    r = h.undo(masks)
    assert r[0] == 1 and np.array_equal(r[1], A)
    _apply(masks, r)
    r = h.redo(masks)
    assert r == (1, None)


def test_mask_history_new_record_clears_redo():
    A, B, C = _rand(5), _rand(6), _rand(7)
    h = MaskHistory()
    masks: dict[int, np.ndarray] = {}
    _edit(h, masks, 0, A)
    _edit(h, masks, 0, B)
    _apply(masks, h.undo(masks))
    assert h.can_redo
    _edit(h, masks, 1, C)
    assert not h.can_redo and h.redo(masks) is None
    r = h.undo(masks)
    assert r == (1, None)


def test_mask_history_depth_and_clear():
    h = MaskHistory(depth=2)
    masks: dict[int, np.ndarray] = {}
    for k in range(4):
        _edit(h, masks, 0, _rand(10 + k))
    n = 0
    while h.can_undo:
        _apply(masks, h.undo(masks))
        n += 1
    assert n == 2
    assert np.array_equal(masks[0], _rand(11))  # oldest state still reachable
    h.clear()
    assert not h.can_undo and not h.can_redo
    assert MaskHistory(depth=0).depth == 1


def test_mask_history_stores_a_snapshot_not_a_reference():
    A = _rand(8)
    h = MaskHistory()
    h.record(0, A)
    A[:] = False  # caller mutates its array afterwards
    r = h.undo({0: A})
    assert np.array_equal(r[1], _rand(8))
