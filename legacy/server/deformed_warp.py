"""Inverse mapping (backward warping) for deformed-mode visualization.

When background == "deformed", overlay data (displacement/strain) must be
transformed from the reference coordinate system to the deformed coordinate
system so geometry matches the deformed background image.

Core algorithm — for each output pixel (x', y') in deformed space, solve for
its reference coordinate (x, y) via fixed-point iteration:
    x_ref^(k+1) = x_def - U_interp(x_ref^k, y_ref^k)
    y_ref^(k+1) = y_def - V_interp(x_ref^k, y_ref^k)
Then sample the data at (x, y) using bilinear interpolation.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import map_coordinates


# ---------------------------------------------------------------------------
# Deformed-warp constants
# ---------------------------------------------------------------------------

# Target number of Delaunay triangulation source points.  More points
# improve interpolation fidelity but increase scipy.spatial overhead.
_DELAUNAY_TARGET_POINTS: int = 3000

# Maximum grid dimension for coarse inverse-map query.  The coarse grid
# is later upsampled with cv2.resize, trading slight smoothing for a
# large speed-up on high-res images.
_MAX_QUERY_PER_DIM: int = 50


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InverseMapResult:
    """Precomputed inverse mapping from deformed to reference coordinates."""
    frame_idx: int
    ref_row_coords: np.ndarray   # (out_h, out_w) float64 — ROI-local y coords
    ref_col_coords: np.ndarray   # (out_h, out_w) float64 — ROI-local x coords
    validity_mask: np.ndarray    # (out_h, out_w) bool
    out_x0: int                  # output bounding box in full-image coords
    out_y0: int
    out_x1: int
    out_y1: int
    has_user_mask: bool = False   # True when validity_mask comes from user mask


def auto_inverse_cache_size() -> int:
    """Compute inverse map cache size. Each entry is ~2x image size in float64."""
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        # Each inverse map is ~2 * H * W * 8 bytes. For 1024x1024: ~16MB
        # Use 2% of available RAM for inverse maps
        budget_mb = min(100, available_mb * 0.02)
        return max(3, int(budget_mb / 16))
    except (ImportError, Exception):
        return 5


class InverseMapCache:
    """Thread-safe LRU cache for InverseMapResult (keyed by frame index)."""

    def __init__(self, max_size: int = None):
        if max_size is None:
            max_size = auto_inverse_cache_size()
        self._max_size = max_size
        self._cache: OrderedDict[int, InverseMapResult] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, frame_idx: int) -> Optional[InverseMapResult]:
        with self._lock:
            if frame_idx in self._cache:
                self._cache.move_to_end(frame_idx)
                return self._cache[frame_idx]
            return None

    def put(self, frame_idx: int, result: InverseMapResult) -> None:
        with self._lock:
            if frame_idx in self._cache:
                self._cache.move_to_end(frame_idx)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
            self._cache[frame_idx] = result

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------

def _auto_K(roi_h: int, roi_w: int) -> int:
    """Compute adaptive coarse-grid spacing K for Delaunay interpolation.

    Targets ~3000 scatter points — enough for sub-pixel accuracy on smooth
    DIC displacement fields while keeping Delaunay build + query fast.
    """
    target_points = _DELAUNAY_TARGET_POINTS
    K = max(1, int(np.ceil(np.sqrt(roi_h * roi_w / target_points))))
    # Ensure at least 30 points per dimension (avoid excessive coarseness)
    K = min(K, max(1, roi_h // 30), max(1, roi_w // 30))
    return K


def compute_inverse_map(
    U: np.ndarray,
    V: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    deformed_frame_mask: Optional[np.ndarray] = None,
) -> InverseMapResult:
    """Compute the inverse mapping from deformed to reference coordinates.

    Uses the forward displacement map to build scattered correspondences
    (ref_pixel -> deformed_pixel), then interpolates the inverse via Delaunay
    triangulation on a coarse grid and upsamples to full resolution.

    The coarse grid spacing K is computed automatically from the ROI size,
    targeting ~3000 Delaunay points for a good speed/accuracy balance.

    Parameters
    ----------
    U, V : (roi_h, roi_w) displacement fields (may contain NaN)
    roi_rect : (x0, y0, x1, y1) ROI bounding box in full-image coords
    image_shape : (H, W) of the full image
    deformed_frame_mask : optional (img_H, img_W) bool array — user-provided
        mask for the deformed frame.  When given, the validity boundary is
        intersected with this mask so the deformed view respects user-defined
        specimen geometry.

    Returns
    -------
    InverseMapResult with precomputed reference coordinates for each output pixel.
    """
    from scipy.interpolate import LinearNDInterpolator

    x0, y0, x1, y1 = roi_rect
    roi_h, roi_w = y1 - y0, x1 - x0
    img_h, img_w = image_shape
    K = _auto_K(roi_h, roi_w)

    # --- Step 1: Forward map — subsample reference grid by K ---
    valid_mask = ~(np.isnan(U) | np.isnan(V))
    if not valid_mask.any():
        return InverseMapResult(
            frame_idx=-1,
            ref_row_coords=np.empty((0, 0)),
            ref_col_coords=np.empty((0, 0)),
            validity_mask=np.empty((0, 0), dtype=bool),
            out_x0=x0, out_y0=y0, out_x1=x1, out_y1=y1,
        )

    # Subsample the displacement field on a regular coarse grid.
    # Always include boundary rows/columns so the Delaunay convex hull
    # covers the full ROI extent.
    row_idx = np.unique(np.append(np.arange(0, roi_h, K), roi_h - 1))
    col_idx = np.unique(np.append(np.arange(0, roi_w, K), roi_w - 1))
    row_grid, col_grid = np.meshgrid(row_idx, col_idx, indexing='ij')
    U_coarse = U[np.ix_(row_idx, col_idx)]
    V_coarse = V[np.ix_(row_idx, col_idx)]
    valid_coarse = ~(np.isnan(U_coarse) | np.isnan(V_coarse))

    if not valid_coarse.any():
        return InverseMapResult(
            frame_idx=-1,
            ref_row_coords=np.empty((0, 0)),
            ref_col_coords=np.empty((0, 0)),
            validity_mask=np.empty((0, 0), dtype=bool),
            out_x0=x0, out_y0=y0, out_x1=x1, out_y1=y1,
        )

    ref_rows = row_grid[valid_coarse].astype(np.float64)
    ref_cols = col_grid[valid_coarse].astype(np.float64)
    def_y = ref_rows + y0 + V_coarse[valid_coarse]
    def_x = ref_cols + x0 + U_coarse[valid_coarse]

    # --- Step 2: Forward-mapped contour for precise deformed boundary ---
    # Extract contour of the valid displacement region at full pixel
    # resolution, then forward-map it using the displacement field.
    # This gives pixel-accurate deformed boundaries instead of the coarse
    # Delaunay convex hull which caused staircase artifacts on curved shapes.
    valid_mask_u8 = valid_mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        valid_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    deformed_contours = []
    all_def_x = [def_x]  # include coarse points for bbox fallback
    all_def_y = [def_y]
    for contour in contours:
        pts = contour.reshape(-1, 2)  # (N, 2) as (col, row)
        c_cols, c_rows = pts[:, 0], pts[:, 1]
        u_vals = U[c_rows, c_cols]
        v_vals = V[c_rows, c_cols]
        ok = np.isfinite(u_vals) & np.isfinite(v_vals)
        if ok.sum() < 3:
            continue
        dx = (c_cols[ok] + x0 + u_vals[ok]).astype(np.float64)
        dy = (c_rows[ok] + y0 + v_vals[ok]).astype(np.float64)
        all_def_x.append(dx)
        all_def_y.append(dy)
        deformed_contours.append(
            np.column_stack([dx, dy]).astype(np.int32)
        )

    # --- Step 2b: Deformed bounding box (from full-res contour) ---
    all_dx = np.concatenate(all_def_x)
    all_dy = np.concatenate(all_def_y)
    out_x0 = max(0, int(np.floor(all_dx.min())))
    out_y0 = max(0, int(np.floor(all_dy.min())))
    out_x1 = min(img_w, int(np.ceil(all_dx.max())) + 1)
    out_y1 = min(img_h, int(np.ceil(all_dy.max())) + 1)

    # When a user mask is provided, expand the bounding box to include
    # the full extent of the user mask.  The forward-mapped contour only
    # covers the displacement-data footprint; the user mask may extend
    # beyond it (e.g. when a bubble shrinks and new material appears).
    if deformed_frame_mask is not None:
        rows_any = np.any(deformed_frame_mask, axis=1)
        cols_any = np.any(deformed_frame_mask, axis=0)
        if rows_any.any():
            mask_y0 = int(np.argmax(rows_any))
            mask_y1 = len(rows_any) - int(np.argmax(rows_any[::-1]))
            mask_x0 = int(np.argmax(cols_any))
            mask_x1 = len(cols_any) - int(np.argmax(cols_any[::-1]))
            out_x0 = min(out_x0, mask_x0)
            out_y0 = min(out_y0, mask_y0)
            out_x1 = max(out_x1, mask_x1)
            out_y1 = max(out_y1, mask_y1)

    out_h = out_y1 - out_y0
    out_w = out_x1 - out_x0

    # --- Step 3: Build interpolators ---
    # LinearND for accurate interior interpolation + NearestND to
    # extrapolate coordinates beyond the Delaunay convex hull (boundary
    # pixels that are inside the true deformed contour but outside the
    # coarse-grid convex hull).
    from scipy.interpolate import NearestNDInterpolator

    points = np.column_stack([def_y, def_x])
    use_linear = True
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        interp_row_lin = LinearNDInterpolator(tri, ref_rows, fill_value=np.nan)
        interp_col_lin = LinearNDInterpolator(tri, ref_cols, fill_value=np.nan)
    except Exception:
        use_linear = False

    interp_row_nn = NearestNDInterpolator(points, ref_rows)
    interp_col_nn = NearestNDInterpolator(points, ref_cols)

    # --- Step 4: Query on coarse output grid, then upsample ---
    query_step_h = max(K, out_h // _MAX_QUERY_PER_DIM) if out_h > _MAX_QUERY_PER_DIM else K
    query_step_w = max(K, out_w // _MAX_QUERY_PER_DIM) if out_w > _MAX_QUERY_PER_DIM else K
    coarse_h = max(1, (out_h + query_step_h - 1) // query_step_h)
    coarse_w = max(1, (out_w + query_step_w - 1) // query_step_w)
    coarse_y = np.linspace(out_y0, out_y1 - 1, coarse_h)
    coarse_x = np.linspace(out_x0, out_x1 - 1, coarse_w)
    cg_y, cg_x = np.meshgrid(coarse_y, coarse_x, indexing='ij')
    query = np.column_stack([cg_y.ravel(), cg_x.ravel()])

    if use_linear:
        ref_row_coarse = interp_row_lin(query).reshape(coarse_h, coarse_w)
        ref_col_coarse = interp_col_lin(query).reshape(coarse_h, coarse_w)
        # Fill Delaunay-hull-exterior NaN with nearest-neighbor extrapolation
        nan_mask = np.isnan(ref_row_coarse)
        if nan_mask.any():
            nan_q = query[nan_mask.ravel()]
            ref_row_coarse[nan_mask] = interp_row_nn(nan_q)
            ref_col_coarse[nan_mask] = interp_col_nn(nan_q)
    else:
        ref_row_coarse = interp_row_nn(query).reshape(coarse_h, coarse_w)
        ref_col_coarse = interp_col_nn(query).reshape(coarse_h, coarse_w)

    # Upsample to full output resolution (all coarse cells now have values)
    if coarse_h < out_h or coarse_w < out_w:
        ref_row_interp = cv2.resize(
            ref_row_coarse.astype(np.float64), (out_w, out_h),
            interpolation=cv2.INTER_LINEAR,
        )
        ref_col_interp = cv2.resize(
            ref_col_coarse.astype(np.float64), (out_w, out_h),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        ref_row_interp = ref_row_coarse
        ref_col_interp = ref_col_coarse

    # --- Step 5: Forward-mapped contour validity mask ---
    # Use the pixel-accurate deformed contour instead of the coarse
    # Delaunay hull for the validity boundary.
    if deformed_contours:
        contour_mask = np.zeros((out_h, out_w), dtype=np.uint8)
        shifted = [dc - np.array([[out_x0, out_y0]]) for dc in deformed_contours]
        cv2.fillPoly(contour_mask, shifted, 1)
        validity = contour_mask.astype(bool)
    else:
        # Fallback: coordinate bounds check
        validity = (
            (ref_row_interp >= 0) & (ref_row_interp <= roi_h - 1) &
            (ref_col_interp >= 0) & (ref_col_interp <= roi_w - 1)
        )

    # When a user-provided mask exists for the deformed frame, use it as the
    # authoritative validity boundary (replaces the auto-derived contour).
    # The user's mask represents ground-truth geometry; the contour is only an
    # approximation.  Data-quality gating (valid_warped > 0.5 in
    # warp_data_inverse) still filters pixels with no source displacement data.
    if deformed_frame_mask is not None:
        mask_crop = deformed_frame_mask[out_y0:out_y1, out_x0:out_x1]
        if mask_crop.shape == validity.shape:
            validity = mask_crop.copy()
        elif mask_crop.size > 0:
            # Handle potential shape mismatch from rounding in viewport scaling
            import warnings
            warnings.warn(
                f"deformed_frame_mask crop shape {mask_crop.shape} != "
                f"validity shape {validity.shape}; resizing mask"
            )
            mask_resized = cv2.resize(
                mask_crop.astype(np.uint8),
                (validity.shape[1], validity.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            validity = mask_resized

    return InverseMapResult(
        frame_idx=-1,  # caller sets this
        ref_row_coords=ref_row_interp,
        ref_col_coords=ref_col_interp,
        validity_mask=validity,
        out_x0=out_x0, out_y0=out_y0,
        out_x1=out_x1, out_y1=out_y1,
        has_user_mask=deformed_frame_mask is not None,
    )


def _nearest_fill_nan(data: np.ndarray) -> np.ndarray:
    """Fill NaN entries with the nearest finite neighbour (L2 distance).

    Mirrors the pyALDIC scatter-interpolation semantics: once a user supplies
    a per-frame ROI, every pixel inside that mask should receive *some* value,
    even if the reference frame lacks data at the corresponding back-warped
    location.  Nearest-neighbour extrapolation from adjacent valid reference
    pixels is visually smooth and keeps the rendered ROI outline identical to
    the user mask.
    """
    valid = np.isfinite(data)
    if valid.all() or not valid.any():
        return np.nan_to_num(data, nan=0.0).astype(np.float64)
    from scipy.ndimage import distance_transform_edt
    _, idx = distance_transform_edt(~valid, return_indices=True)
    return data[tuple(idx)].astype(np.float64)


def warp_data_inverse(
    data: np.ndarray,
    inv_map: InverseMapResult,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Warp ROI-sized data to deformed coordinates using a precomputed inverse map.

    Parameters
    ----------
    data : (roi_h, roi_w) array (may contain NaN)
    inv_map : precomputed InverseMapResult
    image_shape : (H, W) of the full output image

    Returns
    -------
    (H, W) array with warped data placed at the correct position; NaN elsewhere.
    """
    if inv_map.ref_row_coords.size == 0:
        return np.full(image_shape, np.nan)

    out_h = inv_map.out_y1 - inv_map.out_y0
    out_w = inv_map.out_x1 - inv_map.out_x0

    if inv_map.has_user_mask:
        # User-mask semantics: the mask is the sole shape authority.  Pre-fill
        # NaN on the reference side via nearest-neighbour so that per-frame ROI
        # pixels whose back-warp lands in a reference NaN region (e.g. newly
        # exposed specimen when a bubble shrinks below its reference size)
        # still receive a plausible extrapolated value — matching pyALDIC's
        # behaviour where rendering is driven by scatter values that can be
        # extrapolated into those regions.
        data_clean = _nearest_fill_nan(data)
        src_valid = np.ones_like(data, dtype=np.float64)
    else:
        valid_src = ~np.isnan(data)
        data_clean = np.nan_to_num(data, nan=0.0).astype(np.float64)
        src_valid = valid_src.astype(np.float64)

    coords = np.array([
        inv_map.ref_row_coords.ravel(),
        inv_map.ref_col_coords.ravel(),
    ])
    warped = map_coordinates(
        data_clean, coords, order=1, mode='constant', cval=0.0
    ).reshape(out_h, out_w)

    valid_warped = map_coordinates(
        src_valid, coords, order=1, mode='constant', cval=0.0
    ).reshape(out_h, out_w)

    has_data = valid_warped > 0.5
    if inv_map.has_user_mask:
        # Mask shape strictly follows the user-supplied per-frame ROI.
        # Data-quality gating is disabled here because we pre-filled NaN on
        # the reference side above, so every mask pixel has a defined value.
        final_valid = inv_map.validity_mask
    else:
        final_valid = has_data & inv_map.validity_mask

    # Place into full image
    full = np.full(image_shape, np.nan)
    patch = np.where(final_valid, warped, np.nan)
    full[inv_map.out_y0:inv_map.out_y1, inv_map.out_x0:inv_map.out_x1] = patch

    return full


# ---------------------------------------------------------------------------
# Strain upsampling utility
# ---------------------------------------------------------------------------

def upsample_strain_to_roi(
    strain_data: np.ndarray,
    roi_h: int,
    roi_w: int,
) -> np.ndarray:
    """Upsample (potentially downsampled) strain data to ROI dimensions.

    NaN-safe: uses INTER_LINEAR for values and INTER_NEAREST for the mask.
    """
    mask_valid = ~np.isnan(strain_data)
    data_clean = np.nan_to_num(strain_data, nan=0.0)
    data_resized = cv2.resize(
        data_clean, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR
    )
    mask_resized = cv2.resize(
        mask_valid.astype(np.uint8), (roi_w, roi_h),
        interpolation=cv2.INTER_NEAREST,
    )
    data_resized[mask_resized == 0] = np.nan
    return data_resized


# ---------------------------------------------------------------------------
# Unified entry point for render endpoints
# ---------------------------------------------------------------------------

def _nan_safe_resize(arr: np.ndarray, target_wh: Tuple[int, int]) -> np.ndarray:
    """NaN-safe downsample a 2D array using weighted averaging.

    Uses INTER_AREA for proper block averaging; normalizes by the valid-pixel
    weight so NaN regions don't bias neighboring averages toward zero.
    """
    mask = np.isfinite(arr).astype(np.float32)
    clean = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    data_r = cv2.resize(clean, target_wh, interpolation=cv2.INTER_AREA)
    mask_r = cv2.resize(mask, target_wh, interpolation=cv2.INTER_AREA)
    valid = mask_r > 0.5
    result = np.full(data_r.shape, np.nan, dtype=np.float64)
    result[valid] = data_r[valid].astype(np.float64) / mask_r[valid]
    return result


def _warp_at_viewport_scale(
    data: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    scale: float,
    needs_upsample: bool = False,
    roi_h: int = 0,
    roi_w: int = 0,
    deformed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fast path: downsample everything to viewport resolution before warping.

    For a 5000×4000 image viewed in a 1200×800 viewport (scale ≈ 0.2), this
    reduces Delaunay input from ~550K to ~22K points and map_coordinates from
    20M to ~800K pixels — typically 10–50× faster than full-resolution warping.

    The displacement values U, V are scaled by ``scale`` because they represent
    pixel-unit offsets: a 10 px displacement in the original equals 2 px in a
    0.2× downsampled coordinate system.
    """
    img_h, img_w = image_shape
    ds_w = max(1, round(img_w * scale))
    ds_h = max(1, round(img_h * scale))

    x0, y0, x1, y1 = roi_rect
    ds_x0, ds_y0 = round(x0 * scale), round(y0 * scale)
    ds_x1 = max(ds_x0 + 2, round(x1 * scale))
    ds_y1 = max(ds_y0 + 2, round(y1 * scale))
    ds_roi_w, ds_roi_h = ds_x1 - ds_x0, ds_y1 - ds_y0

    # Upsample strain data to ROI size first (if needed), then downsample
    if needs_upsample and roi_h > 0 and roi_w > 0:
        data = upsample_strain_to_roi(data, roi_h, roi_w)

    # Downsample data, U, V to viewport-proportional ROI.
    # Fast path when no NaN (common for DIC displacement fields).
    has_nan = np.isnan(data).any() or np.isnan(U).any() or np.isnan(V).any()

    if not has_nan:
        # No NaN: direct resize (avoids expensive mask computation on large arrays)
        data_ds = cv2.resize(
            data.astype(np.float32), (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA,
        ).astype(np.float64)
        U_ds = cv2.resize(
            U.astype(np.float32), (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA,
        ).astype(np.float64) * scale
        V_ds = cv2.resize(
            V.astype(np.float32), (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA,
        ).astype(np.float64) * scale
    else:
        # NaN-safe: weighted average with mask normalization
        combined_mask = (
            np.isfinite(data) & np.isfinite(U) & np.isfinite(V)
        ).astype(np.float32)
        data_f = np.where(combined_mask > 0, data, 0.0).astype(np.float32)
        U_f = np.where(combined_mask > 0, U, 0.0).astype(np.float32)
        V_f = np.where(combined_mask > 0, V, 0.0).astype(np.float32)

        mask_ds = cv2.resize(combined_mask, (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA)
        data_r = cv2.resize(data_f, (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA)
        U_r = cv2.resize(U_f, (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA)
        V_r = cv2.resize(V_f, (ds_roi_w, ds_roi_h), interpolation=cv2.INTER_AREA)

        valid = mask_ds > 0.5
        inv_w = np.where(valid, 1.0 / mask_ds, 0.0)
        data_ds = np.where(valid, data_r * inv_w, np.nan).astype(np.float64)
        U_ds = np.where(valid, U_r * inv_w * scale, np.nan).astype(np.float64)
        V_ds = np.where(valid, V_r * inv_w * scale, np.nan).astype(np.float64)

    # Downsample deformed-frame mask if provided
    ds_deformed_mask = None
    if deformed_mask is not None:
        ds_deformed_mask = cv2.resize(
            deformed_mask.astype(np.uint8), (ds_w, ds_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    # Compute inverse map at viewport resolution (no caching — it's fast;
    # _auto_K adapts to the downsampled ROI size automatically)
    ds_roi_rect = (ds_x0, ds_y0, ds_x1, ds_y1)
    inv_map = compute_inverse_map(
        U_ds, V_ds, ds_roi_rect, (ds_h, ds_w),
        deformed_frame_mask=ds_deformed_mask,
    )

    return warp_data_inverse(data_ds, inv_map, (ds_h, ds_w))


def get_warped_full_data(
    data: np.ndarray,
    frame_idx: int,
    U: np.ndarray,
    V: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    cache: InverseMapCache,
    needs_upsample: bool = False,
    roi_h: int = 0,
    roi_w: int = 0,
    vw: int = 0,
    vh: int = 0,
    deformed_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convenience function for render endpoints: upsample if needed, compute
    or retrieve cached inverse map, then warp data to deformed coordinates.

    When ``vw``/``vh`` indicate a viewport smaller than the full image, all
    inputs are downsampled to viewport resolution before warping, which is
    dramatically faster for large images (e.g. 5000×4000 → 1200×800).

    Parameters
    ----------
    data : ROI-sized (or downsampled) data array
    frame_idx : displacement frame index (for caching)
    U, V : (roi_h, roi_w) displacement fields
    roi_rect : (x0, y0, x1, y1)
    image_shape : (H, W)
    cache : InverseMapCache instance
    needs_upsample : if True, upsample data to (roi_h, roi_w) first
    roi_h, roi_w : target ROI dimensions (required if needs_upsample)
    vw, vh : viewport dimensions (0 = full resolution)
    deformed_mask : optional (H, W) bool mask for the deformed frame

    Returns
    -------
    (out_H, out_W) array with warped data; NaN outside valid region.
    When viewport downsampling is active, out_H/out_W are viewport-proportional
    (smaller than H/W), which is what render_overlay_png expects.
    """
    img_h, img_w = image_shape
    scale = min(vw / img_w, vh / img_h, 1.0) if vw > 0 and vh > 0 else 1.0

    # Fast path: warp at viewport resolution
    if scale < 1.0:
        return _warp_at_viewport_scale(
            data, U, V, roi_rect, image_shape, scale,
            needs_upsample, roi_h, roi_w,
            deformed_mask=deformed_mask,
        )

    # Full-resolution path (for downloads and when viewport >= image)
    if needs_upsample and roi_h > 0 and roi_w > 0:
        data = upsample_strain_to_roi(data, roi_h, roi_w)

    # When a deformed mask is provided, skip cache (mask may change between
    # requests for the same frame) and compute fresh.
    inv_map = None if deformed_mask is not None else cache.get(frame_idx)
    if inv_map is None:
        inv_map = compute_inverse_map(
            U, V, roi_rect, image_shape,
            deformed_frame_mask=deformed_mask,
        )
        inv_map.frame_idx = frame_idx
        if deformed_mask is None:
            cache.put(frame_idx, inv_map)

    return warp_data_inverse(data, inv_map, image_shape)
