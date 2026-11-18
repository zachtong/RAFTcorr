"""Field sampling, composition of displacements, and the inverse (deformed -> reference) map.

All functions are pure numpy / scipy, used by the pipeline (incremental composition, warped
ROIs) and by the views (fields drawn on the deformed frame).

Grids: a field stored on a *grid* has shape ``(2, h, w)`` and covers image pixels
``x = x0 + j``, ``y = y0 + i``. A *frame field* covers the whole image (``x0 = y0 = 0``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Grid:
    """Placement of a cropped field in the image: origin and size, plus the image size."""

    x0: int
    y0: int
    width: int
    height: int
    image_height: int
    image_width: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def image_shape(self) -> tuple[int, int]:
        return self.image_height, self.image_width

    @property
    def slices(self) -> tuple[slice, slice]:
        return slice(self.y0, self.y0 + self.height), slice(self.x0, self.x0 + self.width)

    def coords(self) -> tuple[np.ndarray, np.ndarray]:
        """Image coordinates ``(X, Y)`` of the grid points, float64, each ``(h, w)``."""
        ys, xs = np.mgrid[0 : self.height, 0 : self.width].astype(np.float64)
        return xs + self.x0, ys + self.y0

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "width": self.width, "height": self.height,
                "image_height": self.image_height, "image_width": self.image_width}

    @classmethod
    def from_dict(cls, d: dict) -> "Grid":
        return cls(int(d["x0"]), int(d["y0"]), int(d["width"]), int(d["height"]),
                   int(d["image_height"]), int(d["image_width"]))


def sample_bilinear(field: np.ndarray, x: np.ndarray, y: np.ndarray, x0: int = 0, y0: int = 0) -> np.ndarray:
    """NaN-aware bilinear sample of ``field`` (``(C, h, w)`` or ``(h, w)``) at image coords.

    Corners that are NaN or outside the field are dropped and the remaining weights renormalised;
    the result is NaN only where every corner is invalid (or the coordinate is NaN). This keeps
    valid values at the edge of a region instead of eroding it by one pixel per composition.
    """
    f = field[None] if field.ndim == 2 else field
    C, h, w = f.shape
    shp = np.shape(x)
    if h == 0 or w == 0:  # an empty field (e.g. a frame whose tracking was lost everywhere)
        out = np.full((C,) + tuple(shp), np.nan)
        return out[0] if field.ndim == 2 else out
    xf = np.asarray(x, dtype=np.float64).ravel() - x0
    yf = np.asarray(y, dtype=np.float64).ravel() - y0
    bad = ~(np.isfinite(xf) & np.isfinite(yf))
    xf = np.where(bad, -10.0, xf)
    yf = np.where(bad, -10.0, yf)
    xi = np.floor(xf).astype(np.int64)
    yi = np.floor(yf).astype(np.int64)
    fx = xf - xi
    fy = yf - yi
    out = np.zeros((C, xf.size), dtype=np.float64)
    wsum = np.zeros(xf.size, dtype=np.float64)
    finite = np.isfinite(f).all(axis=0)
    for dy, dx, wt in ((0, 0, (1 - fx) * (1 - fy)), (0, 1, fx * (1 - fy)), (1, 0, (1 - fx) * fy), (1, 1, fx * fy)):
        cx, cy = xi + dx, yi + dy
        inside = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        cxc, cyc = np.clip(cx, 0, w - 1), np.clip(cy, 0, h - 1)
        ok = inside & finite[cyc, cxc] & (wt > 0)
        wv = np.where(ok, wt, 0.0)
        wsum += wv
        vals = f[:, cyc, cxc]
        out += np.where(ok[None], vals, 0.0) * wv[None]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = out / wsum[None]
    out[:, (wsum <= 1e-12) | bad] = np.nan
    out = out.reshape((C,) + tuple(shp))
    return out[0] if field.ndim == 2 else out


def compose(u_ref: np.ndarray, grid: Grid, step: np.ndarray, step_x0: int = 0, step_y0: int = 0) -> np.ndarray:
    """``U_k(X) = U_r(X) + d_rk(X + U_r(X))`` on ``grid``.

    ``u_ref`` (2, h, w): displacement of frame-0 points to reference frame ``r`` (NaN outside).
    ``step`` (2, H', W'): displacement from frame ``r`` to frame ``k`` in frame-``r`` coordinates,
    placed at ``(step_x0, step_y0)``.
    """
    X, Y = grid.coords()
    xd = X + u_ref[0]
    yd = Y + u_ref[1]
    d = sample_bilinear(step, xd, yd, step_x0, step_y0)
    return (u_ref + d).astype(np.float32)


def gradients(u: np.ndarray) -> np.ndarray:
    """``(2, 2, h, w)``: ``g[i, j] = d u_i / d x_j`` (x = column, y = row), NaN-propagating."""
    gy_u, gx_u = np.gradient(u[0])
    gy_v, gx_v = np.gradient(u[1])
    return np.array([[gx_u, gy_u], [gx_v, gy_v]])


def inverse_map(u: np.ndarray, grid: Grid, mask: np.ndarray | None = None, newton_iters: int = 6,
                tol_px: float = 0.25) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Reference coordinates of the material points seen at each deformed-frame pixel.

    Solves ``X + U(X) = x`` for every pixel ``x`` of the deformed region: forward splatting gives
    an initial guess (robust to large rotations), Newton steps with the local deformation gradient
    refine it. Returns ``(X, Y, window)`` where ``X``, ``Y`` are float arrays over ``window`` =
    ``(x0, y0, x1, y1)`` (half-open, image pixels) and NaN where no material point of ``mask``
    lands.
    """
    h, w = grid.shape
    valid = np.isfinite(u).all(axis=0)
    if mask is not None:
        valid &= mask
    if not valid.any():
        return np.full((0, 0), np.nan), np.full((0, 0), np.nan), (0, 0, 0, 0)
    X, Y = grid.coords()
    xd = np.where(valid, X + u[0], np.nan)
    yd = np.where(valid, Y + u[1], np.nan)
    H, W = grid.image_shape
    wx0 = max(0, int(np.floor(np.nanmin(xd))) - 1)
    wy0 = max(0, int(np.floor(np.nanmin(yd))) - 1)
    wx1 = min(W, int(np.ceil(np.nanmax(xd))) + 2)
    wy1 = min(H, int(np.ceil(np.nanmax(yd))) + 2)
    if wx1 <= wx0 or wy1 <= wy0:
        return np.full((0, 0), np.nan), np.full((0, 0), np.nan), (0, 0, 0, 0)
    th, tw = wy1 - wy0, wx1 - wx0

    # forward splat (nearest) of the reference coordinates
    ti = np.rint(yd[valid]).astype(np.int64) - wy0
    tj = np.rint(xd[valid]).astype(np.int64) - wx0
    keep = (ti >= 0) & (ti < th) & (tj >= 0) & (tj < tw)
    initX = np.full((th, tw), np.nan)
    initY = np.full((th, tw), np.nan)
    initX[ti[keep], tj[keep]] = X[valid][keep]
    initY[ti[keep], tj[keep]] = Y[valid][keep]
    hit = np.isfinite(initX)
    # fill splat holes (stretching) from the nearest hit within a few pixels
    dist, (ii, jj) = ndimage.distance_transform_edt(~hit, return_indices=True)
    near = dist <= 3.0
    initX = np.where(near, initX[ii, jj], np.nan)
    initY = np.where(near, initY[ii, jj], np.nan)

    ty, tx = np.mgrid[wy0:wy1, wx0:wx1].astype(np.float64)
    g = gradients(np.where(valid[None], u, np.nan))
    Xc, Yc = initX, initY
    for _ in range(newton_iters):
        uu = sample_bilinear(u, Xc, Yc, grid.x0, grid.y0)
        rx = Xc + uu[0] - tx
        ry = Yc + uu[1] - ty
        gg = sample_bilinear(g.reshape(4, h, w), Xc, Yc, grid.x0, grid.y0)
        a, b, c, d = 1 + gg[0], gg[1], gg[2], 1 + gg[3]   # F = I + grad U
        det = a * d - b * c
        ok = np.isfinite(det) & (np.abs(det) > 1e-6)
        det = np.where(ok, det, 1.0)
        dX = np.where(ok, (d * rx - b * ry) / det, rx)  # fall back to a fixed-point step
        dY = np.where(ok, (-c * rx + a * ry) / det, ry)
        Xc, Yc = Xc - dX, Yc - dY
    uu = sample_bilinear(u, Xc, Yc, grid.x0, grid.y0)
    res = np.hypot(Xc + uu[0] - tx, Yc + uu[1] - ty)
    inside = _lookup_mask(valid, Xc - grid.x0, Yc - grid.y0)
    good = np.isfinite(res) & (res <= tol_px) & inside
    Xc = np.where(good, Xc, np.nan)
    Yc = np.where(good, Yc, np.nan)
    return Xc, Yc, (wx0, wy0, wx1, wy1)


def _lookup_mask(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nearest-pixel lookup of a boolean mask at fractional coordinates (False outside / NaN)."""
    h, w = mask.shape
    fin = np.isfinite(x) & np.isfinite(y)
    xi = np.rint(np.where(fin, x, -1)).astype(np.int64)
    yi = np.rint(np.where(fin, y, -1)).astype(np.int64)
    inside = fin & (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros(x.shape, dtype=bool)
    out[inside] = mask[yi[inside], xi[inside]]
    return out


def warp_mask(mask: np.ndarray, u: np.ndarray, grid: Grid) -> np.ndarray:
    """Frame-0 ROI (on ``grid``) carried to the deformed frame: full-image boolean mask."""
    X, _Y, (x0, y0, x1, y1) = inverse_map(u, grid, mask)
    out = np.zeros(grid.image_shape, dtype=bool)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = np.isfinite(X)
    return out


def to_deformed(values: np.ndarray, u: np.ndarray, grid: Grid, mask: np.ndarray | None = None,
                ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Values defined on the reference grid (``(C, h, w)`` or ``(h, w)``) drawn on the deformed
    frame: ``out(x) = values(X(x))``. Returns the array over ``window`` = ``(x0, y0, x1, y1)``."""
    X, Y, win = inverse_map(u, grid, mask)
    if X.size == 0:
        shape = (0, 0) if values.ndim == 2 else (values.shape[0], 0, 0)
        return np.full(shape, np.nan, dtype=np.float32), win
    out = sample_bilinear(values, X, Y, grid.x0, grid.y0)
    return out.astype(np.float32), win


def forward_mask_check(u: np.ndarray, grid: Grid, frame_mask: np.ndarray) -> np.ndarray:
    """True where the deformed position ``X + U(X)`` lies inside ``frame_mask`` (or off-image).

    Used to drop material points that end up in a region excluded in the deformed frame
    (a crack, a hole): their correlation is meaningless there.
    """
    X, Y = grid.coords()
    xd = X + u[0]
    yd = Y + u[1]
    H, W = frame_mask.shape
    fin = np.isfinite(xd) & np.isfinite(yd)
    xi = np.rint(np.where(fin, xd, -1)).astype(np.int64)
    yi = np.rint(np.where(fin, yd, -1)).astype(np.int64)
    on_image = fin & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    keep = np.ones(xd.shape, dtype=bool)
    keep[on_image] = frame_mask[yi[on_image], xi[on_image]]
    return keep


def inside_frame(u: np.ndarray, grid: Grid) -> np.ndarray:
    """True where the deformed position ``X + U(X)`` lies inside the image (NaN counts as False).

    A point that leaves the field of view cannot be matched; the network still returns a number
    for it, so the pipeline drops these points.
    """
    X, Y = grid.coords()
    xd, yd = X + u[0], Y + u[1]
    H, W = grid.image_shape
    with np.errstate(invalid="ignore"):
        return (xd >= -0.5) & (xd <= W - 0.5) & (yd >= -0.5) & (yd <= H - 0.5)


def intensity_residual(reference: np.ndarray, deformed: np.ndarray, u: np.ndarray, grid: Grid,
                       sigma: float = 4.0) -> np.ndarray:
    """Local RMS (grey levels) of ``I_def(X + U(X)) - I_ref(X)`` after removing its local mean.

    Near zero where the field maps the deformed image back onto the reference (up to noise);
    large where tracking failed. Removing the local mean makes it insensitive to slow changes of
    illumination between frames. NaN where ``u`` is NaN.
    """
    X, Y = grid.coords()
    valid = np.isfinite(u).all(axis=0)
    xd = np.where(valid, X + u[0], 0.0)
    yd = np.where(valid, Y + u[1], 0.0)
    # cubic spline: bilinear interpolation alone leaves several grey levels on fine speckle
    warped = ndimage.map_coordinates(np.asarray(deformed, dtype=np.float64), [yd, xd], order=3, mode="nearest")
    d = warped - np.asarray(reference, dtype=np.float64)[grid.slices]
    w = valid.astype(np.float64)
    den = ndimage.gaussian_filter(w, sigma, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = ndimage.gaussian_filter(np.where(valid, d, 0.0), sigma, mode="constant") / den
        m2 = ndimage.gaussian_filter(np.where(valid, d * d, 0.0), sigma, mode="constant") / den
        res = np.sqrt(np.clip(m2 - m1 * m1, 0.0, None))
    res[~valid | (den < 0.05)] = np.nan
    return res.astype(np.float32)
