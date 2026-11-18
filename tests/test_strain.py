"""Unit tests for raftcorr.post.strain (virtual strain gauge) and raftcorr.post.smoothing."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from raftcorr.config import StrainConfig
from raftcorr.engine.mapping import Grid
from raftcorr.post.smoothing import nan_gaussian, temporal_gaussian
from raftcorr.post.strain import (
    COMPONENTS,
    Gradients,
    compute_strain,
    edge_trimmed,
    fill_trimmed,
    frame_strain,
    strain_components,
    vsg_gradients,
)
from raftcorr.results import DisplacementSeries, RunInfo
from raftcorr.schedule import FrameSchedule

A_TEST = np.array([[0.012, -0.004], [0.007, -0.009]])


def _affine_uv(h: int, w: int, A: np.ndarray, t=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    return A[0, 0] * xs + A[0, 1] * ys + t[0], A[1, 0] * xs + A[1, 1] * ys + t[1]


def _interior(h: int, w: int, half: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[half:-half, half:-half] = True
    return m


def _grads(ux, uy, vx, vy) -> Gradients:
    return Gradients(*(np.atleast_1d(np.asarray(a, dtype=np.float64)) for a in (ux, uy, vx, vy)))


# -- vsg_gradients --------------------------------------------------------------------------------
@pytest.mark.parametrize("weighting", ["uniform", "gaussian"])
@pytest.mark.parametrize("order", [1, 2])
def test_vsg_affine_field_is_exact(weighting, order):
    h, w, gauge = 50, 61, 15
    u, v = _affine_uv(h, w, A_TEST, (3.0, -1.5))
    cfg = StrainConfig(gauge_px=gauge, weighting=weighting, poly_order=order)
    g = vsg_gradients(u, v, np.ones((h, w), dtype=bool), cfg)
    inner = _interior(h, w, gauge // 2)
    for name, (i, j) in (("dudx", (0, 0)), ("dudy", (0, 1)), ("dvdx", (1, 0)), ("dvdy", (1, 1))):
        arr = getattr(g, name)
        assert arr.shape == (h, w)
        assert np.isfinite(arr[inner]).all()
        np.testing.assert_allclose(arr[inner], A_TEST[i, j], rtol=0, atol=1e-9)
        # a plane is reproduced by any least-squares window, also the truncated ones at the border
        fin = np.isfinite(arr)
        np.testing.assert_allclose(arr[fin], A_TEST[i, j], rtol=0, atol=1e-8)


def test_vsg_gaussian_with_explicit_sigma():
    h, w = 40, 40
    u, v = _affine_uv(h, w, A_TEST)
    cfg = StrainConfig(gauge_px=21, weighting="gaussian", gaussian_sigma=3.0)
    g = vsg_gradients(u, v, np.ones((h, w), dtype=bool), cfg)
    inner = _interior(h, w, 10)
    np.testing.assert_allclose(g.dudx[inner], A_TEST[0, 0], atol=1e-9)


def _quadratic(h: int, w: int):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    a, b, c, d, e = 2e-4, -1.5e-4, 3e-4, 0.01, -0.02
    p, q, r, s, t = -1e-4, 2.5e-4, 1e-4, -0.005, 0.015
    u = a * xs**2 + b * xs * ys + c * ys**2 + d * xs + e * ys
    v = p * xs**2 + q * xs * ys + r * ys**2 + s * xs + t * ys
    grads = {
        "dudx": 2 * a * xs + b * ys + d, "dudy": b * xs + 2 * c * ys + e,
        "dvdx": 2 * p * xs + q * ys + s, "dvdy": q * xs + 2 * r * ys + t,
    }
    return u, v, grads


def test_vsg_quadratic_field_order_2_is_exact_including_border():
    h, w, gauge = 50, 60, 15
    u, v, exact = _quadratic(h, w)
    g = vsg_gradients(u, v, np.ones((h, w), dtype=bool), StrainConfig(gauge_px=gauge, poly_order=2))
    inner = _interior(h, w, gauge // 2)
    for name, ref in exact.items():
        arr = getattr(g, name)
        assert np.isfinite(arr[inner]).all()
        fin = np.isfinite(arr)
        np.testing.assert_allclose(arr[fin], ref[fin], rtol=0, atol=1e-8)


def test_vsg_quadratic_field_order_1_exact_only_on_symmetric_windows():
    h, w, gauge = 50, 60, 15
    u, v, exact = _quadratic(h, w)
    g = vsg_gradients(u, v, np.ones((h, w), dtype=bool), StrainConfig(gauge_px=gauge, poly_order=1))
    inner = _interior(h, w, gauge // 2)
    np.testing.assert_allclose(g.dudx[inner], exact["dudx"][inner], atol=1e-9)
    np.testing.assert_allclose(g.dvdy[inner], exact["dvdy"][inner], atol=1e-9)
    # a truncated (one-sided) window at the corner is biased by the curvature
    assert abs(g.dudx[0, 0] - exact["dudx"][0, 0]) > 1e-4


def test_vsg_nan_outside_valid_region():
    h, w = 40, 40
    u, v = _affine_uv(h, w, A_TEST)
    valid = np.zeros((h, w), dtype=bool)
    valid[5:35, 8:30] = True
    u[~valid] = np.nan
    v[~valid] = np.nan
    g = vsg_gradients(u, v, valid, StrainConfig(gauge_px=11))
    assert np.isnan(g.dudx[~valid]).all()
    assert np.isfinite(g.dudx[valid]).all()
    np.testing.assert_allclose(g.dudx[valid], A_TEST[0, 0], atol=1e-8)


def test_vsg_empty_valid_gives_all_nan():
    u = np.zeros((10, 10))
    g = vsg_gradients(u, u, np.zeros((10, 10), dtype=bool), StrainConfig(gauge_px=5))
    assert np.isnan(g.dudx).all() and np.isnan(g.dvdy).all()


def test_vsg_rejects_low_coverage():
    h, w = 60, 60
    u, v = _affine_uv(h, w, A_TEST)
    valid = np.zeros((h, w), dtype=bool)
    valid[28:33, 28:33] = True  # 25 points in a 41 x 41 gauge: far below 25 % coverage
    g = vsg_gradients(u, v, valid, StrainConfig(gauge_px=41))
    assert np.isnan(g.dudx).all()


def test_vsg_rejects_ill_conditioned_fit():
    h, w = 20, 30
    u, v = _affine_uv(h, w, A_TEST)
    valid = np.zeros((h, w), dtype=bool)
    valid[10, 2:28] = True  # one row: dy is undetermined, coverage 3/9 passes
    g = vsg_gradients(u, v, valid, StrainConfig(gauge_px=3))
    assert np.isnan(g.dudy).all() and np.isnan(g.dudx).all()


def test_connectivity_aware_fits_do_not_mix_across_a_crack():
    h, w, gauge = 60, 81, 21
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    left = xs <= 38
    crack = (xs >= 39) & (xs <= 41)  # 3 px crack
    u = np.where(left, 0.01 * xs, 0.03 * xs + 2.0)
    v = np.where(left, -0.005 * ys, 0.02 * ys - 1.0)
    u[crack] = np.nan
    v[crack] = np.nan
    valid = ~crack
    rows = slice(gauge // 2, h - gauge // 2)

    g = vsg_gradients(u, v, valid, StrainConfig(gauge_px=gauge, connectivity_aware=True))
    for col, (ex, ey) in ((38, (0.01, -0.005)), (42, (0.03, 0.02)), (37, (0.01, -0.005)), (43, (0.03, 0.02))):
        np.testing.assert_allclose(g.dudx[rows, col], ex, atol=1e-9)
        np.testing.assert_allclose(g.dvdy[rows, col], ey, atol=1e-9)
        np.testing.assert_allclose(g.dudy[rows, col], 0.0, atol=1e-9)
    assert np.isnan(g.dudx[:, 39:42]).all()

    # control: without connectivity the gauges next to the crack see both sides
    g2 = vsg_gradients(u, v, valid, StrainConfig(gauge_px=gauge, connectivity_aware=False))
    assert np.abs(g2.dudx[rows, 38] - 0.01).min() > 1e-3
    assert np.abs(g2.dudx[rows, 42] - 0.03).min() > 1e-3


# -- strain_components ----------------------------------------------------------------------------
H_TEST = np.array([[0.1, 0.05], [-0.02, 0.2]])  # grad u


def _measure_tensor(H: np.ndarray, measure: str) -> np.ndarray:
    F = np.eye(2) + H
    if measure == "green_lagrange":
        return 0.5 * (F.T @ F - np.eye(2))
    if measure == "infinitesimal":
        return 0.5 * (H + H.T)
    return 0.5 * (np.eye(2) - np.linalg.inv(F @ F.T))


@pytest.mark.parametrize("measure", ["green_lagrange", "infinitesimal", "euler_almansi"])
def test_strain_components_against_hand_computation(measure):
    H = H_TEST
    out = strain_components(_grads(H[0, 0], H[0, 1], H[1, 0], H[1, 1]), measure)
    assert set(out) == set(COMPONENTS)
    assert all(a.dtype == np.float32 for a in out.values())
    E = _measure_tensor(H, measure)
    assert out["exx"][0] == pytest.approx(E[0, 0], abs=1e-6)
    assert out["eyy"][0] == pytest.approx(E[1, 1], abs=1e-6)
    assert out["exy"][0] == pytest.approx(E[0, 1], abs=1e-6)
    e2, e1 = np.linalg.eigvalsh(E)
    assert out["e1"][0] == pytest.approx(e1, abs=1e-6)
    assert out["e2"][0] == pytest.approx(e2, abs=1e-6)
    assert out["max_shear"][0] == pytest.approx((e1 - e2) / 2, abs=1e-6)
    e3 = -(e1 + e2)
    assert out["von_mises"][0] == pytest.approx(np.sqrt(2 / 3 * (e1**2 + e2**2 + e3**2)), abs=1e-6)
    # principal direction of e1
    th = np.radians(float(out["principal_angle"][0]))
    n = np.array([np.cos(th), np.sin(th)])
    np.testing.assert_allclose(E @ n, e1 * n, atol=1e-6)
    F = np.eye(2) + H
    assert out["area_change"][0] == pytest.approx(np.linalg.det(F) - 1, abs=1e-6)
    assert out["rotation"][0] == pytest.approx(np.degrees(np.arctan2(F[1, 0] - F[0, 1], F[0, 0] + F[1, 1])),
                                               abs=1e-4)


def test_hand_values_green_lagrange():
    # ux = 0.1, uy = 0.05, vx = -0.02, vy = 0.2
    out = strain_components(_grads(0.1, 0.05, -0.02, 0.2), "green_lagrange")
    assert out["exx"][0] == pytest.approx(0.1 + 0.5 * (0.01 + 0.0004), abs=1e-7)
    assert out["eyy"][0] == pytest.approx(0.2 + 0.5 * (0.0025 + 0.04), abs=1e-7)
    assert out["exy"][0] == pytest.approx(0.5 * (0.05 - 0.02) + 0.5 * (0.1 * 0.05 - 0.02 * 0.2), abs=1e-7)
    assert out["area_change"][0] == pytest.approx(1.1 * 1.2 - 0.05 * -0.02 - 1, abs=1e-7)


@pytest.mark.parametrize("seed", range(4))
def test_principal_strains_ordered(seed):
    rng = np.random.default_rng(seed)
    g = _grads(*rng.uniform(-0.3, 0.3, (4, 200)))
    for measure in ("green_lagrange", "infinitesimal", "euler_almansi"):
        out = strain_components(g, measure)
        assert np.all(out["e1"] >= out["e2"])
        np.testing.assert_allclose(out["e1"] + out["e2"], out["exx"] + out["eyy"], atol=1e-5)
        assert np.all(out["max_shear"] >= 0)


def test_unknown_measure():
    with pytest.raises(ValueError):
        strain_components(_grads(0, 0, 0, 0), "hencky")


@pytest.mark.parametrize("deg", [5.0, 30.0, -20.0])
def test_pure_rotation_field(deg):
    h, w = 41, 41
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    u, v = _affine_uv(h, w, R - np.eye(2), (1.0, 2.0))
    cfg = StrainConfig(gauge_px=11, measure="green_lagrange")
    res = frame_strain(u, v, np.ones((h, w), dtype=bool), cfg)
    inner = _interior(h, w, 5)
    np.testing.assert_allclose(res["rotation"][inner], deg, atol=1e-4)
    for k in ("exx", "eyy", "exy", "e1", "e2", "von_mises", "area_change"):
        np.testing.assert_allclose(res[k][inner], 0.0, atol=1e-6)
    # the infinitesimal measure is not rotation-invariant: exx = cos(theta) - 1
    inf = frame_strain(u, v, np.ones((h, w), dtype=bool), StrainConfig(gauge_px=11, measure="infinitesimal"))
    np.testing.assert_allclose(inf["exx"][inner], np.cos(th) - 1, atol=1e-6)


def test_frame_strain_with_smoothing_keeps_affine_interior():
    h, w = 60, 60
    u, v = _affine_uv(h, w, A_TEST)
    cfg = StrainConfig(gauge_px=11, smoothing_sigma=1.5, measure="infinitesimal")
    res = frame_strain(u, v, np.ones((h, w), dtype=bool), cfg)
    inner = _interior(h, w, 15)
    np.testing.assert_allclose(res["exx"][inner], A_TEST[0, 0], atol=1e-6)
    np.testing.assert_allclose(res["eyy"][inner], A_TEST[1, 1], atol=1e-6)


# -- compute_strain on a DisplacementSeries -------------------------------------------------------
def _series(T: int = 3) -> tuple[DisplacementSeries, np.ndarray]:
    grid = Grid(5, 7, 33, 26, 60, 60)
    h, w = grid.shape
    mask = np.ones((h, w), dtype=bool)
    mask[:, -3:] = False
    u = np.full((T, h, w), np.nan, dtype=np.float32)
    v = np.full((T, h, w), np.nan, dtype=np.float32)
    u[0][mask] = 0.0
    v[0][mask] = 0.0
    uu, vv = _affine_uv(h, w, A_TEST, (0.5, -0.25))
    u[1] = np.where(mask, uu, np.nan)
    v[1] = np.where(mask, vv, np.nan)
    done = np.array([True, True] + [False] * (T - 2))
    info = RunInfo(software_version="test", model_name="none", model_sha256="", downsample_factor=1, iters=1,
                   corr_mode="volume", device="cpu", config={})
    s = DisplacementSeries(grid, mask, u, v, done, tuple(f"f{k}.png" for k in range(T)),
                           FrameSchedule.accumulative(T), info)
    return s, mask


@pytest.mark.parametrize("step", [1, 2, 4, 5])
def test_compute_strain_shape_and_values(step):
    series, mask = _series()
    cfg = StrainConfig(gauge_px=7, step=step, edge_trim_alpha=0.0)  # every gauge point, edges included
    st = compute_strain(series, cfg)
    hs, ws = -(-26 // step), -(-33 // step)
    assert st.step == step and st.grid == series.grid
    assert set(st.names) == set(COMPONENTS)
    for name in COMPONENTS:
        assert st.components[name].shape == (3, hs, ws)
        assert st.components[name].dtype == np.float32
    assert st.done.tolist() == [True, True, False]
    assert np.isnan(st.get("exx", 2)).all()
    # frame 0: zero strain
    f0 = st.get("exx", 0)
    np.testing.assert_allclose(f0[np.isfinite(f0)], 0.0, atol=1e-7)
    # frame 1: Green-Lagrange of the affine field at sample (i, j) = grid pixel (j * step, i * step)
    E = _measure_tensor(A_TEST, "green_lagrange")
    exx = st.get("exx", 1)
    sub_mask = mask[::step, ::step]
    assert np.isnan(exx[~sub_mask]).all()
    assert np.isfinite(exx[sub_mask]).all()
    np.testing.assert_allclose(exx[sub_mask], E[0, 0], atol=1e-6)
    np.testing.assert_allclose(st.get("exy", 1)[sub_mask], E[0, 1], atol=1e-6)
    assert st.settings.step == step and st.settings.gauge_px == 7 and st.settings.measure == "green_lagrange"


def test_compute_strain_frame_subset():
    series, _ = _series()
    st = compute_strain(series, StrainConfig(gauge_px=7), frames=[1, 2])
    assert st.done.tolist() == [False, True, False]  # frame 2 is not computed in the series
    assert np.isnan(st.get("exx", 0)).all()


def test_compute_strain_default_config():
    series, _ = _series()
    # the 26 x 30 px region is narrower than two default trim bands (2 x 0.7 x 20 px)
    with pytest.raises(ValueError, match="every strain point"):
        compute_strain(series)
    st = compute_strain(series, StrainConfig(edge_trim_alpha=0.0))
    assert st.settings.gauge_px == 41 and st.settings.measure == "green_lagrange"
    assert st.settings.edge_trim_alpha == 0.0 and st.trimmed is None
    assert st.components["exx"].shape == (3, 26, 33)


# -- edge trim (pyALDIC's "Trim low-confidence edges") ------------------------------------------------
def _trimmed_columns(alpha: float, gauge: int = 41) -> int:
    """Width of the trimmed band next to a straight edge (invalid columns on the left)."""
    valid = np.ones((200, 200), dtype=bool)
    valid[:, :50] = False
    cut = edge_trimmed(valid, gauge, alpha)
    row = cut[100]                                    # away from the top and bottom borders
    return int(row[50:100].sum())                     # next to the edge only, not the right border


def test_edge_trim_band_width_is_alpha_times_the_gauge_half_width():
    assert _trimmed_columns(0.0) == 0
    assert _trimmed_columns(0.7) == 14                # 0.7 x 20 px
    assert _trimmed_columns(1.0) == 20
    assert _trimmed_columns(0.5, gauge=21) == 5


def test_edge_trim_alpha_one_trims_exactly_the_cut_gauges():
    yy, xx = np.mgrid[0:150, 0:160]
    valid = np.hypot(xx - 80, yy - 75) > 25          # a hole
    valid[:, :8] = False
    full = ndimage.binary_erosion(valid, np.ones((41, 41), dtype=bool), border_value=0)
    np.testing.assert_array_equal(edge_trimmed(valid, 41, 1.0), valid & ~full)


def test_edge_trim_counts_the_grid_border_as_an_edge():
    cut = edge_trimmed(np.ones((100, 100), dtype=bool), 41, 0.7)
    assert cut[50, :14].all() and not cut[50, 14:86].any() and cut[50, 86:].all()


def _affine_series_with_hole(T: int = 2) -> tuple[DisplacementSeries, np.ndarray]:
    grid = Grid(0, 0, 160, 140, 160, 140)
    h, w = grid.shape
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.hypot(xx - 80, yy - 70) > 20
    u = np.zeros((T, h, w), dtype=np.float32)
    v = np.zeros((T, h, w), dtype=np.float32)
    uu, vv = _affine_uv(h, w, A_TEST, (0.5, -0.25))
    u[1], v[1] = uu, vv
    u[:, ~mask] = np.nan
    v[:, ~mask] = np.nan
    info = RunInfo(software_version="test", model_name="none", model_sha256="", downsample_factor=1, iters=1,
                   corr_mode="volume", device="cpu", config={})
    s = DisplacementSeries(grid, mask, u, v, np.ones(T, dtype=bool), tuple(f"f{k}.png" for k in range(T)),
                           FrameSchedule.accumulative(T), info)
    return s, mask


@pytest.mark.parametrize("step", [1, 3])
def test_compute_strain_trims_the_edge_band(step):
    series, mask = _affine_series_with_hole()
    st = compute_strain(series, StrainConfig(gauge_px=21, step=step))   # default alpha 0.7: 7 px band
    cut = edge_trimmed(mask, 21, 0.7)[::step, ::step]
    assert st.settings.edge_trim_alpha == 0.7
    np.testing.assert_array_equal(st.trimmed[1], cut)
    np.testing.assert_array_equal(st.trimmed_in(1), cut)
    kept = mask[::step, ::step] & ~cut
    E = _measure_tensor(A_TEST, "green_lagrange")
    for name in COMPONENTS:
        assert np.isnan(st.get(name, 1)[cut]).all(), name
    np.testing.assert_allclose(st.get("exx", 1)[kept], E[0, 0], atol=1e-6)


def test_compute_strain_without_trim_keeps_the_edge_values():
    series, mask = _affine_series_with_hole()
    st = compute_strain(series, StrainConfig(gauge_px=21, edge_trim_alpha=0.0))
    assert st.trimmed is None and st.trimmed_in(1) is None
    assert np.isfinite(st.get("exx", 1)[mask]).all()


def test_fill_trimmed_takes_values_from_its_own_side_only():
    vals = np.full((40, 60), np.nan, dtype=np.float32)
    trimmed = np.zeros((40, 60), dtype=bool)
    vals[:, :25] = 1.0                                # left region, a 5 px gap, right region
    vals[:, 30:] = 2.0
    trimmed[:, 20:25] = True                          # the band next to the gap, on both sides
    trimmed[:, 30:35] = True
    vals[trimmed] = np.nan
    island = np.zeros_like(trimmed)
    island[5:8, 40:43] = True                         # no reliable point of its own: stays NaN
    vals[4:9, 39:44] = np.nan
    trimmed |= island
    out = fill_trimmed(vals, trimmed)
    assert out.dtype == np.float32 and np.isnan(vals[trimmed]).all()      # the input is untouched
    assert (out[:, 20:25] == 1.0).all() and (out[:, 30:35] == 2.0).all()
    assert np.isnan(out[:, 25:30]).all()                                  # the gap stays empty
    assert np.isnan(out[island]).all()
    np.testing.assert_array_equal(out[~trimmed], vals[~trimmed])


def test_pictures_fill_the_trimmed_band_and_data_do_not():
    from raftcorr.export.figures import FrameStyle, compose
    from raftcorr.viz.fields import FieldProvider, Units
    from raftcorr.viz.render import ColorRange

    series, mask = _affine_series_with_hole()
    st = compute_strain(series, StrainConfig(gauge_px=21))
    E = _measure_tensor(A_TEST, "green_lagrange")[0, 0]
    prov = FieldProvider(series, st)
    cut = st.trimmed[1]
    raw, _ = prov.values("exx", 1)
    filled, _ = prov.values("exx", 1, fill_trimmed=True)
    assert np.isnan(raw[cut]).all()
    np.testing.assert_allclose(filled[cut], E, atol=1e-6)
    np.testing.assert_array_equal(filled[~cut], raw[~cut])
    assert np.isnan(filled[~mask]).all()                              # the hole stays empty
    assert np.isnan(prov.values("exx", 1)[0][cut]).all()              # data untouched afterwards
    d_raw, _ = prov.deformed("exx", 1)
    d_fill, _ = prov.deformed("exx", 1, fill_trimmed=True)
    assert np.isfinite(d_fill).sum() > np.isfinite(d_raw).sum()

    images = [np.full(series.grid.image_shape, 128, np.uint8)] * 2
    rng = ColorRange(E - 0.01, E + 0.01)
    plain = compose(prov, images, 1, FrameStyle(field="exx", background="none"), Units(), rng)
    full = compose(prov, images, 1, FrameStyle(field="exx", background="none", fill_trimmed_edges=True),
                   Units(), rng)
    y, x = np.argwhere(cut)[0]
    assert (plain[y, x, :3] == 255).all() and not (full[y, x, :3] == 255).all()


def test_trim_wider_than_the_region_is_refused():
    series, _ = _series()
    with pytest.raises(ValueError, match="Reduce the VSG size"):
        compute_strain(series, StrainConfig(gauge_px=41, edge_trim_alpha=0.7))
    compute_strain(series, StrainConfig(gauge_px=41, edge_trim_alpha=0.3))   # 6 px band: points remain


def test_strain_config_validates_edge_trim_alpha():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="edge_trim_alpha"):
            StrainConfig(edge_trim_alpha=bad)
    assert StrainConfig().edge_trim_alpha == 0.7


# -- smoothing ------------------------------------------------------------------------------------
def test_nan_gaussian_preserves_constant_and_keeps_nan_outside_mask():
    h, w = 50, 60
    yy, xx = np.mgrid[0:h, 0:w]
    mask = (xx - 30) ** 2 + (yy - 25) ** 2 <= 18**2
    f = np.full((h, w), 3.25)
    f[10, 30] = np.nan  # a hole inside the mask
    out = nan_gaussian(f, 2.0, mask)
    assert out.dtype == np.float32 and out.shape == (h, w)
    assert np.isnan(out[~mask]).all()
    assert np.isnan(out[10, 30])
    inside = mask.copy()
    inside[10, 30] = False
    np.testing.assert_allclose(out[inside], 3.25, rtol=1e-6)


def test_nan_gaussian_multichannel_and_linear_interior():
    h, w = 60, 60
    u, v = _affine_uv(h, w, A_TEST, (1.0, 2.0))
    out = nan_gaussian(np.stack([u, v]), 2.0)
    assert out.shape == (2, h, w)
    inner = _interior(h, w, 12)
    np.testing.assert_allclose(out[0][inner], u[inner], atol=1e-5)
    np.testing.assert_allclose(out[1][inner], v[inner], atol=1e-5)


def test_nan_gaussian_isolated_point_below_min_weight():
    f = np.full((21, 21), np.nan)
    f[10, 10] = 1.0
    out = nan_gaussian(f, 2.0)  # weight of one point = 1 / (2 pi sigma^2) ~ 0.04 < 0.05
    assert np.isnan(out).all()
    out = nan_gaussian(f, 2.0, min_weight=0.01)
    assert out[10, 10] == pytest.approx(1.0)


def test_nan_gaussian_zero_sigma_is_a_copy():
    f = np.arange(12.0).reshape(3, 4)
    out = nan_gaussian(f, 0.0)
    assert out.dtype == np.float32 and np.array_equal(out, f) and out is not f


def test_temporal_gaussian():
    stack = np.full((7, 4, 5), 2.0)
    stack[3, 1, 1] = np.nan
    out = temporal_gaussian(stack, 1.5)
    assert out.dtype == np.float32 and out.shape == stack.shape
    assert np.isnan(out[3, 1, 1])
    fin = np.isfinite(stack)
    np.testing.assert_allclose(out[fin], 2.0, rtol=1e-6)
    ramp = np.arange(9.0)[:, None, None] * np.ones((9, 2, 2))
    # sigma = 1 frame -> kernel radius 4 frames: only frame 4 is free of the boundary padding
    np.testing.assert_allclose(temporal_gaussian(ramp, 1.0)[4], ramp[4], atol=1e-5)
    np.testing.assert_array_equal(temporal_gaussian(ramp, 0.0), ramp.astype(np.float32))
