"""Strain from displacement with a virtual strain gauge (VSG).

At every output point a polynomial (plane by default, or quadratic) is fitted by (weighted) least
squares to each displacement component over an ``Lg x Lg`` window of valid points; the linear
coefficients are the displacement gradients. The paper setting is an unweighted plane fit and
the Green-Lagrange strain.

The normal equations are assembled with correlations: every entry of ``M = sum w phi_i phi_j m``
and ``b = sum w phi_i u m`` is the correlation of a mask-weighted image with a kernel
``w(dx, dy) phi_i(dx, dy) phi_j(dx, dy)``. Both weightings (uniform, Gaussian) and every monomial
``dx^a dy^b`` are separable, so each correlation is two 1-D passes (exact, ``O(N Lg)``).

Crack-aware mode fits each connected component of the valid region separately, so no gauge ever
spans a crack or a hole. Fits whose window holds too few valid points (``min_coverage``) or whose
normal matrix is ill-conditioned (``condition_max``) are rejected (NaN).

Strain measures (``F = I + grad u``, gradients in reference coordinates):

* ``green_lagrange``: ``E = (F^T F - I) / 2``;
* ``infinitesimal``: ``e = (grad u + grad u^T) / 2``;
* ``euler_almansi``: ``e = (I - (F F^T)^-1) / 2``.

Derived quantities: principal strains ``e1 >= e2``, principal angle (deg, of ``e1`` from +x,
clockwise on screen because +y points down), maximum in-plane shear ``(e1 - e2) / 2``, von Mises
equivalent strain ``sqrt(2/3 (e1^2 + e2^2 + e3^2))`` with ``e3 = -(e1 + e2)`` (incompressible),
rotation (deg, from the polar decomposition, ``atan2(F21 - F12, F11 + F22)``) and the area change
``det F - 1``. Shear components are tensor components (half the engineering shear).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from ..callbacks import Callbacks, check_stop, coerce
from ..config import StrainConfig
from ..results import DisplacementSeries, StrainSeries, StrainSettings
from .smoothing import nan_gaussian

COMPONENTS: tuple[str, ...] = (
    "exx", "eyy", "exy", "e1", "e2", "principal_angle", "max_shear", "von_mises", "rotation", "area_change",
)

#: The one name of each component, used by the GUI, the exported figures and the command line.
#: "max shear" is the tensor maximum shear (e1 - e2) / 2, not the engineering shear (often written
#: gamma, twice as large); "e1 angle" is the angle of the major principal direction from +x.
COMPONENT_LABELS: dict[str, str] = {
    "exx": "εxx", "eyy": "εyy", "exy": "εxy", "e1": "ε₁", "e2": "ε₂",
    "principal_angle": "ε₁ angle", "max_shear": "max shear", "von_mises": "von Mises",
    "rotation": "ω rot", "area_change": "ΔA/A₀",
}

COMPONENT_UNITS: dict[str, str] = {"principal_angle": "deg", "rotation": "deg"}


@dataclass(frozen=True)
class Gradients:
    """``dudx, dudy, dvdx, dvdy`` on the output grid (NaN where no fit)."""

    dudx: np.ndarray
    dudy: np.ndarray
    dvdx: np.ndarray
    dvdy: np.ndarray


def _monomials(order: int) -> list[tuple[int, int]]:
    """Exponents ``(a, b)`` of ``dx^a dy^b``: constant, x, y (+ x^2, x y, y^2)."""
    base = [(0, 0), (1, 0), (0, 1)]
    return base + [(2, 0), (1, 1), (0, 2)] if order == 2 else base


def _kernels_1d(gauge: int, weighting: str, sigma: float | None, max_power: int) -> list[np.ndarray]:
    half = gauge // 2
    d = np.arange(-half, half + 1, dtype=np.float64)
    if weighting == "gaussian":
        s = sigma if sigma is not None else gauge / 4.0
        w = np.exp(-(d**2) / (2 * s * s))
    else:
        w = np.ones_like(d)
    return [w * d**p for p in range(max_power + 1)]


def _corr2(img: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    tmp = ndimage.correlate1d(img, kx, axis=1, mode="constant", cval=0.0)
    return ndimage.correlate1d(tmp, ky, axis=0, mode="constant", cval=0.0)


def vsg_gradients(u: np.ndarray, v: np.ndarray, valid: np.ndarray, cfg: StrainConfig) -> Gradients:
    """Displacement gradients on the full grid of ``u`` / ``v`` (no subsampling), NaN where no fit."""
    h, w = u.shape
    out = {k: np.full((h, w), np.nan, dtype=np.float64) for k in ("dudx", "dudy", "dvdx", "dvdy")}
    if not valid.any():
        return Gradients(**out)
    if cfg.connectivity_aware:
        labels, n = ndimage.label(valid)
        objects = ndimage.find_objects(labels)
    else:
        labels, n, objects = valid.astype(np.int32), 1, [ndimage.find_objects(valid.astype(np.int32))[0]]
    half = cfg.gauge_px // 2
    min_points = max(len(_monomials(cfg.poly_order)) + 1, int(cfg.min_coverage * cfg.gauge_px**2 * 0.05))
    for lab in range(1, n + 1):
        sl = objects[lab - 1]
        if sl is None:
            continue
        y0, y1 = max(0, sl[0].start - half), min(h, sl[0].stop + half)
        x0, x1 = max(0, sl[1].start - half), min(w, sl[1].stop + half)
        comp = labels[y0:y1, x0:x1] == lab
        if comp.sum() < min_points:
            continue
        g = _fit_component(u[y0:y1, x0:x1], v[y0:y1, x0:x1], comp, cfg)
        for k in out:
            region = out[k][y0:y1, x0:x1]
            region[comp] = g[k][comp]
    return Gradients(**out)


def _fit_component(u: np.ndarray, v: np.ndarray, m: np.ndarray, cfg: StrainConfig) -> dict[str, np.ndarray]:
    mono = _monomials(cfg.poly_order)
    p = len(mono)
    max_pow = 2 * max(max(a, b) for a, b in mono)
    k1 = _kernels_1d(cfg.gauge_px, cfg.weighting, cfg.gaussian_sigma, max_pow)
    mf = m.astype(np.float64)
    uf = np.where(m, u, 0.0).astype(np.float64)
    vf = np.where(m, v, 0.0).astype(np.float64)
    M = np.empty(m.shape + (p, p))
    cache: dict[tuple[int, int], np.ndarray] = {}
    for i, (ai, bi) in enumerate(mono):
        for j in range(i, p):
            aj, bj = mono[j]
            key = (ai + aj, bi + bj)
            if key not in cache:
                cache[key] = _corr2(mf, k1[key[0]], k1[key[1]])
            M[..., i, j] = M[..., j, i] = cache[key]
    bu = np.stack([_corr2(uf, k1[a], k1[b]) for a, b in mono], axis=-1)
    bv = np.stack([_corr2(vf, k1[a], k1[b]) for a, b in mono], axis=-1)

    # coverage = weighted count of valid points / full-window weight
    full = float(k1[0].sum()) ** 2
    ok = m & (M[..., 0, 0] >= cfg.min_coverage * full)
    res = {k: np.full(m.shape, np.nan) for k in ("dudx", "dudy", "dvdx", "dvdy")}
    if not ok.any():
        return res
    Mo = M[ok]
    # scale-invariant conditioning check on the normalised matrix
    d = np.sqrt(np.clip(np.einsum("nii->ni", Mo), 1e-300, None))
    Mn = Mo / (d[:, :, None] * d[:, None, :])
    eig = np.linalg.eigvalsh(Mn)
    cond = eig[:, -1] / np.maximum(eig[:, 0], 1e-300)
    good = cond <= cfg.condition_max
    idx = np.flatnonzero(ok)[good]
    if idx.size == 0:
        return res
    rhs = np.stack([bu[ok][good], bv[ok][good]], axis=-1)        # (n, p, 2)
    coef = np.linalg.solve(Mo[good], rhs)                         # (n, p, 2)
    flat = {k: res[k].reshape(-1) for k in res}
    flat["dudx"][idx] = coef[:, 1, 0]
    flat["dudy"][idx] = coef[:, 2, 0]
    flat["dvdx"][idx] = coef[:, 1, 1]
    flat["dvdy"][idx] = coef[:, 2, 1]
    return res


def strain_components(g: Gradients, measure: str) -> dict[str, np.ndarray]:
    """Every component of :data:`COMPONENTS` from displacement gradients."""
    ux, uy, vx, vy = g.dudx, g.dudy, g.dvdx, g.dvdy
    F11, F12, F21, F22 = 1 + ux, uy, vx, 1 + vy
    if measure == "green_lagrange":
        exx = ux + 0.5 * (ux * ux + vx * vx)
        eyy = vy + 0.5 * (uy * uy + vy * vy)
        exy = 0.5 * (uy + vx) + 0.5 * (ux * uy + vx * vy)
    elif measure == "infinitesimal":
        exx, eyy, exy = ux, vy, 0.5 * (uy + vx)
    elif measure == "euler_almansi":
        # b = F F^T, e = (I - b^-1) / 2
        b11 = F11 * F11 + F12 * F12
        b12 = F11 * F21 + F12 * F22
        b22 = F21 * F21 + F22 * F22
        det = b11 * b22 - b12 * b12
        with np.errstate(invalid="ignore", divide="ignore"):
            exx = 0.5 * (1 - b22 / det)
            eyy = 0.5 * (1 - b11 / det)
            exy = 0.5 * (b12 / det)
    else:
        raise ValueError(f"unknown strain measure {measure!r}")
    mean = 0.5 * (exx + eyy)
    rad = np.sqrt((0.5 * (exx - eyy)) ** 2 + exy**2)
    e1, e2 = mean + rad, mean - rad
    e3 = -(e1 + e2)
    out = {
        "exx": exx,
        "eyy": eyy,
        "exy": exy,
        "e1": e1,
        "e2": e2,
        "principal_angle": np.degrees(0.5 * np.arctan2(2 * exy, exx - eyy)),
        "max_shear": rad,
        "von_mises": np.sqrt(2.0 / 3.0 * (e1 * e1 + e2 * e2 + e3 * e3)),
        "rotation": np.degrees(np.arctan2(F21 - F12, F11 + F22)),
        "area_change": F11 * F22 - F12 * F21 - 1.0,
    }
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def frame_strain(u: np.ndarray, v: np.ndarray, mask: np.ndarray, cfg: StrainConfig) -> dict[str, np.ndarray]:
    """Strain components of one frame on the grid subsampled by ``cfg.step``."""
    valid = mask & np.isfinite(u) & np.isfinite(v)
    if cfg.smoothing_sigma > 0:
        sm = nan_gaussian(np.stack([u, v]), cfg.smoothing_sigma, valid)
        u, v = sm[0], sm[1]
        valid &= np.isfinite(u) & np.isfinite(v)
    g = vsg_gradients(u, v, valid, cfg)
    if cfg.step > 1:
        s = cfg.step
        g = Gradients(g.dudx[::s, ::s], g.dudy[::s, ::s], g.dvdx[::s, ::s], g.dvdy[::s, ::s])
    return strain_components(g, cfg.measure)


def compute_strain(series: DisplacementSeries, cfg: StrainConfig | None = None, frames: list[int] | None = None,
                   callbacks: Callbacks | None = None) -> StrainSeries:
    """Strain of every computed frame of a displacement series."""
    cfg = cfg or StrainConfig()
    cb = coerce(callbacks)
    T = series.n_frames
    hs = -(-series.grid.height // cfg.step)
    ws = -(-series.grid.width // cfg.step)
    comps = {name: np.full((T, hs, ws), np.nan, dtype=np.float32) for name in COMPONENTS}
    done = np.zeros(T, dtype=bool)
    todo = [k for k in (frames if frames is not None else range(T)) if series.done[k]]
    for i, k in enumerate(todo):
        check_stop(cb)
        res = frame_strain(series.u[k], series.v[k], series.mask, cfg)
        for name in COMPONENTS:
            comps[name][k] = res[name]
        done[k] = True
        cb.progress((i + 1) / max(1, len(todo)), f"strain: frame {k} ({i + 1}/{len(todo)})")
    settings = StrainSettings(
        measure=cfg.measure, gauge_px=cfg.gauge_px, weighting=cfg.weighting, poly_order=cfg.poly_order,
        step=cfg.step, gaussian_sigma=cfg.gaussian_sigma, smoothing_sigma=cfg.smoothing_sigma,
        connectivity_aware=cfg.connectivity_aware,
    )
    return StrainSeries(series.grid, cfg.step, comps, done, settings)
