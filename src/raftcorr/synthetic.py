"""Synthetic speckle pairs with a known displacement field (tests, self-test, examples).

The deformed image follows the package convention: a material point at reference pixel ``X``
appears at ``x = X + d(X)``. It is rendered by inverse mapping, ``I_def(x) = I_ref(X)`` with
``X`` solved from ``x = X + d(X)`` by fixed-point iteration, on a 2x oversampled speckle that is
then area-averaged to the output resolution (anti-aliasing).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage

Field = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def speckle(h: int, w: int, seed: int = 0, grain_px: float = 2.5, oversample: int = 2) -> np.ndarray:
    """Random speckle (float64, grey levels about 20-235) at ``oversample`` x the size."""
    rng = np.random.default_rng(seed)
    hs, ws = h * oversample, w * oversample
    z = ndimage.gaussian_filter(rng.standard_normal((hs, ws)), grain_px * oversample / 2.0, mode="wrap")
    z /= z.std() + 1e-12
    return 128.0 + 107.0 * np.tanh(1.2 * z)


def _pool(a: np.ndarray, k: int) -> np.ndarray:
    h, w = a.shape[0] // k, a.shape[1] // k
    return a[: h * k, : w * k].reshape(h, k, w, k).mean(axis=(1, 3))


def render_pair(
    h: int,
    w: int,
    field: Field,
    seed: int = 0,
    grain_px: float = 2.5,
    noise_std: float = 0.0,
    quantize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference and deformed images (float32, 0-255) for a displacement ``field(x, y) -> (u, v)``.

    ``field`` receives reference pixel coordinates (output resolution) and returns the
    displacement in output pixels.
    """
    k = 2
    hi = speckle(h, w, seed, grain_px, oversample=k)
    hs, ws = hi.shape
    # pixel centres of the oversampled grid expressed in output pixels
    ys, xs = np.mgrid[0:hs, 0:ws].astype(np.float64)
    xo = (xs + 0.5) / k - 0.5
    yo = (ys + 0.5) / k - 0.5
    X, Y = xo.copy(), yo.copy()
    for _ in range(30):  # x = X + d(X)  =>  X = x - d(X)
        u, v = field(X, Y)
        Xn, Yn = xo - u, yo - v
        delta = max(float(np.abs(Xn - X).max()), float(np.abs(Yn - Y).max()))
        X, Y = Xn, Yn
        if delta < 1e-7:
            break
    rows = (Y + 0.5) * k - 0.5
    cols = (X + 0.5) * k - 0.5
    deformed_hi = ndimage.map_coordinates(hi, [rows, cols], order=3, mode="reflect")
    ref = _pool(hi, k)
    dfm = _pool(deformed_hi, k)
    if noise_std > 0:
        rng = np.random.default_rng(seed + 1)
        ref = ref + rng.normal(0, noise_std, ref.shape)
        dfm = dfm + rng.normal(0, noise_std, dfm.shape)
    if quantize:
        ref, dfm = np.round(ref), np.round(dfm)
    return np.clip(ref, 0, 255).astype(np.float32), np.clip(dfm, 0, 255).astype(np.float32)


def translation(du: float, dv: float) -> Field:
    return lambda x, y: (np.full_like(x, du), np.full_like(y, dv))


def affine(a11: float, a12: float, a21: float, a22: float, tx: float = 0.0, ty: float = 0.0,
           cx: float = 0.0, cy: float = 0.0) -> Field:
    """``d = A (X - c) + t``: homogeneous strain / rotation about ``c`` plus a translation."""
    return lambda x, y: (a11 * (x - cx) + a12 * (y - cy) + tx, a21 * (x - cx) + a22 * (y - cy) + ty)


def sinusoid(amplitude: float, wavelength: float, axis: str = "x") -> Field:
    """``u = A sin(2 pi y / lambda)`` (axis x) or ``v = A sin(2 pi x / lambda)`` (axis y)."""
    k = 2 * np.pi / wavelength
    if axis == "x":
        return lambda x, y: (amplitude * np.sin(k * y), np.zeros_like(y))
    return lambda x, y: (np.zeros_like(x), amplitude * np.sin(k * x))


def evaluate(field: Field, h: int, w: int) -> np.ndarray:
    """(2, h, w) ground-truth field at the reference pixels."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = field(xs, ys)
    return np.stack([u, v]).astype(np.float32)


def render_sequence(h: int, w: int, fields: list[Field], seed: int = 0, grain_px: float = 2.5,
                    noise_std: float = 0.0) -> list[np.ndarray]:
    """``[reference, deformed_1, ...]``: one deformed frame per field, all from the same speckle."""
    frames: list[np.ndarray] = []
    for i, f in enumerate(fields):
        ref, dfm = render_pair(h, w, f, seed=seed, grain_px=grain_px, noise_std=noise_std)
        if i == 0:
            frames.append(ref)
        frames.append(dfm)
    return frames
