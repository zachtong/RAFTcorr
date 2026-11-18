"""Unit tests for raftcorr.io.images (discovery, reading, intensity scaling, sequences)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from raftcorr.io.images import (
    ImageSequence,
    IntensityScale,
    effective_bit_depth,
    image_size,
    lexicographic_differs,
    list_images,
    natural_key,
    read_grey,
    read_raw,
    scale_for,
    to_grey,
)


def _png(path: Path, arr: np.ndarray, mode: str | None = None) -> Path:
    Image.fromarray(arr, mode=mode).save(path) if mode else Image.fromarray(arr).save(path)
    return path


def _ramp(h: int = 12, w: int = 17, hi: int = 255, dtype=np.uint8) -> np.ndarray:
    return (np.arange(h * w).reshape(h, w) % (hi + 1)).astype(dtype)


# -- discovery ------------------------------------------------------------------------------------
def test_natural_key_orders_numbers():
    names = ["img10.png", "img2.png", "img1.png", "IMG3.png"]
    assert sorted(names, key=natural_key) == ["img1.png", "img2.png", "IMG3.png", "img10.png"]


def test_list_images_natural_vs_plain(tmp_path: Path):
    for name in ("img1.png", "img2.png", "img10.png"):
        _png(tmp_path / name, _ramp())
    (tmp_path / "notes.txt").write_text("x")
    _png(tmp_path / ".img3.png", _ramp())      # hidden: skipped
    (tmp_path / "dir.png").mkdir()            # not a file: skipped
    nat = list_images(tmp_path)
    plain = list_images(tmp_path, natural=False)
    assert [p.name for p in nat] == ["img1.png", "img2.png", "img10.png"]
    assert [p.name for p in plain] == ["img1.png", "img10.png", "img2.png"]
    assert lexicographic_differs(nat)


def test_list_images_extension_filter_is_case_insensitive(tmp_path: Path):
    _png(tmp_path / "a.PNG", _ramp())
    tifffile.imwrite(tmp_path / "b.TIF", _ramp())
    assert [p.name for p in list_images(tmp_path)] == ["a.PNG", "b.TIF"]
    assert [p.name for p in list_images(tmp_path, extensions=(".tif",))] == ["b.TIF"]


def test_list_images_not_a_folder(tmp_path: Path):
    with pytest.raises(NotADirectoryError):
        list_images(tmp_path / "missing")


def test_lexicographic_differs_false_for_padded_or_plain_names():
    assert not lexicographic_differs([Path("img001.png"), Path("img002.png"), Path("img010.png")])
    assert not lexicographic_differs([Path("a.png"), Path("b.png")])
    assert not lexicographic_differs([])


# -- reading --------------------------------------------------------------------------------------
def test_read_grey_8bit_png(tmp_path: Path):
    a = _ramp()
    out = read_grey(_png(tmp_path / "g.png", a))
    assert out.dtype == np.float32 and out.shape == a.shape and out.flags.c_contiguous
    np.testing.assert_array_equal(out, a.astype(np.float32))


def _luma(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def test_read_grey_rgb_png_is_luminance(tmp_path: Path):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (10, 14, 3), dtype=np.uint8)
    out = read_grey(_png(tmp_path / "c.png", rgb))
    np.testing.assert_allclose(out, _luma(rgb.astype(np.float64)), atol=1e-4)


def test_read_grey_rgba_png_drops_alpha(tmp_path: Path):
    rng = np.random.default_rng(1)
    rgba = rng.integers(0, 256, (10, 14, 4), dtype=np.uint8)
    out = read_grey(_png(tmp_path / "c.png", rgba))
    np.testing.assert_allclose(out, _luma(rgba[..., :3].astype(np.float64)), atol=1e-4)


def test_read_grey_grey_alpha_png(tmp_path: Path):
    la = np.stack([_ramp(), np.full((12, 17), 77, np.uint8)], axis=-1)
    out = read_grey(_png(tmp_path / "la.png", la, mode="LA"))
    np.testing.assert_array_equal(out, _ramp().astype(np.float32))


def test_read_grey_palette_png(tmp_path: Path):
    a = _ramp(hi=15)
    im = Image.fromarray(a).convert("P")
    im.save(tmp_path / "p.png")
    out = read_grey(tmp_path / "p.png")
    np.testing.assert_allclose(out, a.astype(np.float32), atol=0.6)


def test_read_grey_16bit_png_with_12bit_data(tmp_path: Path):
    a = _ramp(16, 256, hi=4095, dtype=np.uint16)
    a[0, 0] = 4095
    p = tmp_path / "g16.png"
    Image.fromarray(a).save(p)
    out = read_grey(p)
    np.testing.assert_allclose(out, a.astype(np.float64) * 255.0 / 4095.0, rtol=1e-6, atol=1e-4)
    assert out.max() == pytest.approx(255.0)


def test_read_grey_16bit_tiff_with_12bit_data(tmp_path: Path):
    a = _ramp(16, 256, hi=4095, dtype=np.uint16)
    p = tmp_path / "g16.tif"
    tifffile.imwrite(p, a)
    out = read_grey(p)
    np.testing.assert_allclose(out, a.astype(np.float64) * 255.0 / 4095.0, rtol=1e-6, atol=1e-4)


def test_read_grey_full_16bit_tiff(tmp_path: Path):
    a = np.array([[0, 1000], [40000, 65535]], dtype=np.uint16)
    p = tmp_path / "g16full.tif"
    tifffile.imwrite(p, a)
    np.testing.assert_allclose(read_grey(p), a * 255.0 / 65535.0, rtol=1e-6)


def test_read_grey_float_tiff_in_unit_range(tmp_path: Path):
    rng = np.random.default_rng(2)
    a = rng.random((9, 11)).astype(np.float32)
    p = tmp_path / "f.tif"
    tifffile.imwrite(p, a)
    np.testing.assert_allclose(read_grey(p), a * 255.0, rtol=1e-6)


def test_read_grey_float_tiff_scaled_by_maximum_and_nan(tmp_path: Path):
    a = np.array([[0.0, 2.0], [np.nan, 8.0]], dtype=np.float32)
    p = tmp_path / "f.tif"
    tifffile.imwrite(p, a)
    out = read_grey(p)
    np.testing.assert_allclose(out, [[0.0, 63.75], [0.0, 255.0]], atol=1e-4)


def test_read_grey_with_explicit_scale(tmp_path: Path):
    a = np.array([[0, 1000], [2000, 4000]], dtype=np.uint16)
    p = tmp_path / "s.tif"
    tifffile.imwrite(p, a)
    out = read_grey(p, IntensityScale(255.0 / 4095.0, "12-bit"))
    np.testing.assert_allclose(out, a * 255.0 / 4095.0, rtol=1e-6)


def test_read_raw_multipage_tiff_reads_first_page(tmp_path: Path):
    stack = np.stack([np.full((5, 6), 10, np.uint8), np.full((5, 6), 200, np.uint8)])
    p = tmp_path / "stack.tif"
    tifffile.imwrite(p, stack)
    assert np.array_equal(read_raw(p), stack[0])
    assert image_size(p) == (5, 6)


def test_image_size_png(tmp_path: Path):
    assert image_size(_png(tmp_path / "a.png", np.zeros((7, 13), np.uint8))) == (7, 13)


def test_to_grey_channel_first_and_errors():
    rgb = np.random.default_rng(3).integers(0, 256, (3, 5, 7)).astype(np.uint8)
    np.testing.assert_allclose(to_grey(rgb), _luma(np.moveaxis(rgb, 0, -1).astype(np.float64)))
    single = np.arange(20, dtype=np.uint8).reshape(5, 4, 1)
    assert np.array_equal(to_grey(single), single[..., 0])
    grey = np.arange(20, dtype=np.uint16).reshape(4, 5)
    assert to_grey(grey) is grey  # 2-D input is returned as is
    with pytest.raises(ValueError):
        to_grey(np.zeros((2, 3, 4, 5)))


@pytest.mark.parametrize("mx,bits", [(0, 8), (255, 8), (256, 10), (1023, 10), (4095, 12), (4096, 14),
                                     (16383, 14), (65535, 16), (10**6, 16)])
def test_effective_bit_depth(mx, bits):
    assert effective_bit_depth(mx) == bits


def test_scale_for():
    assert scale_for(np.zeros((2, 2), np.uint8)).gain == 1.0
    assert scale_for(np.array([[4095]], np.uint16)).gain == pytest.approx(255.0 / 4095.0)
    assert scale_for(np.array([[0.5]], np.float32)).gain == 255.0
    assert scale_for(np.array([[2.0, np.nan]], np.float64)).gain == pytest.approx(127.5)


def test_intensity_scale_clips():
    out = IntensityScale(2.0, "x").apply(np.array([[-5.0, 100.0, 200.0]]))
    np.testing.assert_array_equal(out, [[0.0, 200.0, 255.0]])
    assert out.dtype == np.float32


def test_intensity_scale_does_not_modify_its_input():
    a = np.array([[-5.0, 100.0, 300.0]], dtype=np.float32)
    before = a.copy()
    out = IntensityScale(1.0, "unit gain").apply(a)
    np.testing.assert_array_equal(out, [[0.0, 100.0, 255.0]])
    np.testing.assert_array_equal(a, before)


# -- ImageSequence --------------------------------------------------------------------------------
def test_image_sequence_shares_frame0_scale(tmp_path: Path):
    f0 = _ramp(8, 10, hi=4095, dtype=np.uint16)
    f0[0, 0] = 4095                                  # frame 0: 12-bit data
    f1 = np.full((8, 10), 1000, dtype=np.uint16)     # frame 1 alone would look 10-bit
    tifffile.imwrite(tmp_path / "f0.tif", f0)
    tifffile.imwrite(tmp_path / "f1.tif", f1)
    seq = ImageSequence.from_folder(tmp_path)
    assert len(seq) == 2 and seq.names == ["f0.tif", "f1.tif"] and seq.shape == (8, 10)
    assert seq.scale.gain == pytest.approx(255.0 / 4095.0)
    np.testing.assert_allclose(seq[0], f0 * 255.0 / 4095.0, rtol=1e-6, atol=1e-4)
    np.testing.assert_allclose(seq[1], f1 * 255.0 / 4095.0, rtol=1e-6)
    # reading frame 1 on its own uses its own (10-bit) scale: the sequence must not
    assert not np.allclose(read_grey(tmp_path / "f1.tif"), seq[1])


def test_image_sequence_size_mismatch(tmp_path: Path):
    _png(tmp_path / "a1.png", np.zeros((8, 10), np.uint8))
    _png(tmp_path / "a2.png", np.zeros((8, 11), np.uint8))
    with pytest.raises(ValueError, match="a2.png"):  # refused when the sequence is opened
        ImageSequence.from_folder(tmp_path)


def test_image_sequence_cache_is_read_only_and_reused(tmp_path: Path):
    for k in range(4):
        _png(tmp_path / f"im{k}.png", np.full((6, 7), 10 * k, np.uint8))
    seq = ImageSequence.from_folder(tmp_path, cache_size=2)
    a = seq[1]
    assert not a.flags.writeable
    with pytest.raises(ValueError):
        a[0, 0] = 1.0
    assert seq[1] is a                       # cache hit
    assert not seq[0].flags.writeable
    seq[2], seq[3]                           # evicts frame 1 (cache_size = 2)
    b = seq[1]
    assert b is not a and np.array_equal(b, a)


def test_image_sequence_negative_and_out_of_range_index(tmp_path: Path):
    for k in range(3):
        _png(tmp_path / f"im{k}.png", np.full((6, 7), 10 * k, np.uint8))
    seq = ImageSequence(sorted(tmp_path.iterdir(), key=lambda p: natural_key(p.name)))
    np.testing.assert_array_equal(seq[-1], seq[2])
    assert seq[-1][0, 0] == 20
    np.testing.assert_array_equal(seq[-3], seq[0])
    with pytest.raises(IndexError):
        seq[3]
    with pytest.raises(IndexError):
        seq[-4]


def test_image_sequence_empty(tmp_path: Path):
    with pytest.raises(ValueError):
        ImageSequence([])
    with pytest.raises(FileNotFoundError):
        ImageSequence.from_folder(tmp_path)
