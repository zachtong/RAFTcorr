"""
Strain calculation utilities for RAFT-DIC.

- Virtual Strain Gauge (VSG) method with weighted least-squares
- Condition number pre-filtering via eigenvalue analysis
- Boundary erosion to mask unreliable edge strain
- Rotation field via polar decomposition (Numba-optimized)
"""

import time

import numpy as np
from scipy.signal import fftconvolve

# Numba-optimized rotation angle calculation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

@jit(nopython=True, parallel=True, cache=True)
def _compute_rotation_field_numba(du_dx, du_dy, dv_dx, dv_dy):
    """
    Compute rotation angle field using polar decomposition with analytical 2x2 solutions.
    Numba JIT-compiled for high performance.

    For each pixel:
    1. Build deformation gradient: F = I + grad(u)
    2. Compute right Cauchy-Green tensor: C = F^T * F
    3. Analytical eigendecomposition of 2x2 symmetric matrix C
    4. Compute U = sqrt(C) and R = F * U^(-1)
    5. Extract rotation angle: theta = atan2(R21, R11)

    Returns rotation angle in degrees.
    """
    H, W = du_dx.shape
    rotation = np.full((H, W), np.nan)

    for i in prange(H):
        for j in range(W):
            # Skip invalid pixels
            ux = du_dx[i, j]
            uy = du_dy[i, j]
            vx = dv_dx[i, j]
            vy = dv_dy[i, j]

            if np.isnan(ux) or np.isnan(uy) or np.isnan(vx) or np.isnan(vy):
                continue

            # Deformation gradient tensor: F = I + grad(u)
            # F = [[F11, F12], [F21, F22]]
            F11 = 1.0 + ux
            F12 = uy
            F21 = vx
            F22 = 1.0 + vy

            # Right Cauchy-Green tensor: C = F^T * F
            # C = [[C11, C12], [C12, C22]] (symmetric)
            C11 = F11 * F11 + F21 * F21
            C12 = F11 * F12 + F21 * F22
            C22 = F12 * F12 + F22 * F22

            # Analytical eigenvalue decomposition for 2x2 symmetric matrix
            # eigenvalues: lambda = (trace/2) +/- sqrt((trace/2)^2 - det)
            trace = C11 + C22
            det = C11 * C22 - C12 * C12

            half_trace = trace * 0.5
            discriminant = half_trace * half_trace - det

            if discriminant < 0:
                continue

            sqrt_disc = np.sqrt(discriminant)
            lambda1 = half_trace + sqrt_disc
            lambda2 = half_trace - sqrt_disc

            if lambda1 < 0 or lambda2 < 0:
                continue

            # Compute sqrt of eigenvalues
            sqrt_l1 = np.sqrt(lambda1)
            sqrt_l2 = np.sqrt(lambda2)

            # For 2x2 symmetric matrix, we can compute U = sqrt(C) directly
            # Using the formula: U = (C + sqrt(det)*I) / (sqrt(lambda1) + sqrt(lambda2))
            # This avoids explicit eigenvector computation
            sqrt_det = np.sqrt(det)
            denom = sqrt_l1 + sqrt_l2

            if abs(denom) < 1e-12:
                continue

            U11 = (C11 + sqrt_det) / denom
            U12 = C12 / denom
            U22 = (C22 + sqrt_det) / denom

            # Compute U^(-1) for 2x2 matrix
            det_U = U11 * U22 - U12 * U12
            if abs(det_U) < 1e-12:
                continue

            inv_det_U = 1.0 / det_U
            U_inv11 = U22 * inv_det_U
            U_inv12 = -U12 * inv_det_U
            U_inv22 = U11 * inv_det_U

            # Rotation matrix: R = F * U^(-1)
            R11 = F11 * U_inv11 + F12 * U_inv12
            R12 = F11 * U_inv12 + F12 * U_inv22
            R21 = F21 * U_inv11 + F22 * U_inv12
            R22 = F21 * U_inv12 + F22 * U_inv22

            # Ensure proper rotation (det(R) = 1)
            # For 2D, we can just use atan2 directly if R is close to orthogonal
            # The SVD orthogonalization is replaced by direct angle extraction
            # since the analytical solution should give a proper rotation matrix

            # Extract rotation angle (2D) in degrees
            rotation[i, j] = np.degrees(np.arctan2(R21, R11))

    return rotation



def _vsg_gradients(u_filled, v_filled, mask_float, vsg_size, poly_order,
                   weighting, step, gaussian_sigma, min_weight, cond_threshold):
    """Compute displacement gradients via single-pass VSG with quality filtering.

    Args:
        u_filled, v_filled: NaN-free displacement arrays (H, W).
        mask_float: Validity mask as float64 (1=valid, 0=invalid).
        vsg_size: Window size (odd).
        poly_order: 1 or 2.
        weighting: 'Gaussian' or 'Uniform'.
        step: Output downsampling stride.
        gaussian_sigma: Gaussian weight sigma (None -> vsg_size/4).
        min_weight: Minimum weighted pixel count to attempt solve.
        cond_threshold: Max condition number (0 to skip check).

    Returns:
        (du_dx, du_dy, dv_dx, dv_dy, coverage, n_solved, n_filtered)
        Gradients are 2D (H_down, W_down), NaN where invalid.
        coverage: ratio of weighted valid pixels to full-window weight (0..1).
    """
    if vsg_size % 2 == 0:
        raise ValueError("VSG size must be odd.")

    half = vsg_size // 2
    x_range = np.arange(-half, half + 1)
    y_range = np.arange(-half, half + 1)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    # Weight kernel
    if weighting.lower() == 'gaussian':
        sigma = gaussian_sigma if gaussian_sigma is not None else vsg_size / 4.0
        G = np.exp(-(X_grid**2 + Y_grid**2) / (2 * sigma**2))
    else:
        G = np.ones((vsg_size, vsg_size))

    # Basis functions
    basis_funcs = [np.ones_like(X_grid), X_grid, Y_grid]
    if poly_order == 2:
        basis_funcs.extend([X_grid**2, X_grid * Y_grid, Y_grid**2])
    num_params = len(basis_funcs)

    s_slice = slice(None, None, step)

    # Build M matrix (symmetric) via FFT convolution
    M_elements = [[None] * num_params for _ in range(num_params)]
    for i in range(num_params):
        for j in range(i, num_params):
            kernel = G * basis_funcs[i] * basis_funcs[j]
            res = fftconvolve(mask_float, kernel[::-1, ::-1], mode='same')
            down = res[s_slice, s_slice]
            M_elements[i][j] = down
            if i != j:
                M_elements[j][i] = down

    # Build b vectors
    u_masked = u_filled * mask_float
    v_masked = v_filled * mask_float
    b_u_els = []
    b_v_els = []
    for k in range(num_params):
        kernel = G * basis_funcs[k]
        kf = kernel[::-1, ::-1]
        b_u_els.append(fftconvolve(u_masked, kf, mode='same')[s_slice, s_slice])
        b_v_els.append(fftconvolve(v_masked, kf, mode='same')[s_slice, s_slice])

    # Stack into batch arrays
    H_down, W_down = M_elements[0][0].shape
    N = H_down * W_down

    M_stack = np.empty((N, num_params, num_params))
    b_u_stack = np.empty((N, num_params))
    b_v_stack = np.empty((N, num_params))

    for i in range(num_params):
        b_u_stack[:, i] = b_u_els[i].ravel()
        b_v_stack[:, i] = b_v_els[i].ravel()
        for j in range(num_params):
            M_stack[:, i, j] = M_elements[i][j].ravel()

    # --- Quality filtering: min_weight + condition number ---
    valid = M_stack[:, 0, 0] > min_weight

    # Condition number pre-filtering (eigvalsh is fast for symmetric matrices)
    if cond_threshold > 0 and np.any(valid):
        M_check = M_stack[valid]
        eigvals = np.linalg.eigvalsh(M_check)  # (N_valid, p) sorted ascending
        cond = eigvals[:, -1] / np.maximum(eigvals[:, 0], 1e-15)
        ill = cond > cond_threshold
        if np.any(ill):
            valid_indices = np.where(valid)[0]
            valid[valid_indices[ill]] = False

    n_filtered = int(np.sum(M_stack[:, 0, 0] > min_weight)) - int(np.sum(valid))

    # --- Batch solve ---
    coeffs_u = np.full((N, num_params), np.nan)
    coeffs_v = np.full((N, num_params), np.nan)

    if np.any(valid):
        M_valid = M_stack[valid]
        b_u_valid = b_u_stack[valid][..., None]
        b_v_valid = b_v_stack[valid][..., None]

        try:
            coeffs_u[valid] = np.linalg.solve(M_valid, b_u_valid).squeeze(-1)
            coeffs_v[valid] = np.linalg.solve(M_valid, b_v_valid).squeeze(-1)
        except np.linalg.LinAlgError:
            M_inv = np.linalg.pinv(M_valid)
            coeffs_u[valid] = (M_inv @ b_u_valid).squeeze(-1)
            coeffs_v[valid] = (M_inv @ b_v_valid).squeeze(-1)

    du_dx = coeffs_u[:, 1].reshape(H_down, W_down)
    du_dy = coeffs_u[:, 2].reshape(H_down, W_down)
    dv_dx = coeffs_v[:, 1].reshape(H_down, W_down)
    dv_dy = coeffs_v[:, 2].reshape(H_down, W_down)

    # Coverage: ratio of weighted valid pixels to full-window weight
    g_sum = float(G.sum())
    coverage = (M_stack[:, 0, 0] / g_sum).reshape(H_down, W_down)

    return du_dx, du_dy, dv_dx, dv_dy, coverage, int(np.sum(valid)), n_filtered


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def calculate_strain_field(displacement_field: np.ndarray, method: str = 'green_lagrange',
                          vsg_size: int = 31, poly_order: int = 1, weighting: str = 'Gaussian',
                          step: int = 1, gaussian_sigma=None,
                          cond_threshold: float = 1000.0,
                          boundary_erosion: int = -1):
    """
    Calculate strain field with VSG and quality filtering.

    Args:
        displacement_field: (H, W, 2) array with (u, v) displacements.
        method: 'green_lagrange' (default) or 'engineering'.
        vsg_size: Primary VSG window size (odd int).
        poly_order: Polynomial order (1 or 2).
        weighting: 'Uniform' or 'Gaussian'.
        step: Calculation stride (downsampling factor).
        gaussian_sigma: Custom Gaussian sigma. If None, defaults to vsg_size / 4.
        cond_threshold: Max condition number; pixels above are NaN (0 to disable).
        boundary_erosion: Pixels to erode from NaN boundaries (masks out unreliable
            border strain). -1 = auto (vsg_size // 2), 0 = disabled.

    Returns:
        strain_dict: Dictionary containing strain components, or None.
    """
    t_start = time.perf_counter()

    u = displacement_field[..., 0]
    v = displacement_field[..., 1]
    H, W = u.shape

    num_params = 3 if poly_order == 1 else 6
    min_weight = num_params * 3.0

    print(f"[TIMING] Strain calc: {H}x{W}, VSG={vsg_size}, step={step}, "
          f"cond_thresh={cond_threshold}, min_weight={min_weight}")

    # Validity mask
    mask = (~np.isnan(u)) & (~np.isnan(v))
    mask_float = mask.astype(np.float64)

    if not np.any(mask):
        return None

    u_filled = np.nan_to_num(u)
    v_filled = np.nan_to_num(v)

    s_slice = slice(None, None, step)
    mask_down = mask[s_slice, s_slice]

    # --- Primary VSG pass (connectivity-aware) ---
    t1 = time.perf_counter()

    from scipy.ndimage import label as ndimage_label
    labels, n_components = ndimage_label(mask)

    if n_components <= 1:
        # Single connected region (or empty) — fast path
        du_dx, du_dy, dv_dx, dv_dy, coverage, n_solved, n_cond_filtered = (
            _vsg_gradients(
                u_filled, v_filled, mask_float, vsg_size, poly_order,
                weighting, step, gaussian_sigma, min_weight, cond_threshold,
            )
        )
    else:
        # Multiple components (e.g. crack splits specimen).  Run VSG per
        # component so that no window mixes data from across the crack.
        labels_down = labels[s_slice, s_slice]
        H_down, W_down = labels_down.shape

        du_dx = np.full((H_down, W_down), np.nan)
        du_dy = np.full((H_down, W_down), np.nan)
        dv_dx = np.full((H_down, W_down), np.nan)
        dv_dy = np.full((H_down, W_down), np.nan)
        coverage = np.zeros((H_down, W_down))
        n_solved = 0
        n_cond_filtered = 0

        print(f"[STRAIN] {n_components} connected components detected — "
              f"running per-component VSG")

        for comp_id in range(1, n_components + 1):
            comp_mask_float = (labels == comp_id).astype(np.float64)
            if comp_mask_float.sum() < min_weight:
                continue

            (c_dudx, c_dudy, c_dvdx, c_dvdy,
             c_cov, c_solved, c_filt) = _vsg_gradients(
                u_filled, v_filled, comp_mask_float, vsg_size, poly_order,
                weighting, step, gaussian_sigma, min_weight, cond_threshold,
            )

            # Keep results only for pixels belonging to this component
            sel = labels_down == comp_id
            du_dx[sel] = c_dudx[sel]
            du_dy[sel] = c_dudy[sel]
            dv_dx[sel] = c_dvdx[sel]
            dv_dy[sel] = c_dvdy[sel]
            coverage[sel] = c_cov[sel]
            n_solved += c_solved
            n_cond_filtered += c_filt

    t2 = time.perf_counter()
    n_valid_down = int(np.sum(mask_down))
    n_got = int(np.sum(~np.isnan(du_dx) & mask_down))
    print(f"[TIMING] Primary pass (vsg={vsg_size}): {t2-t1:.3f}s, "
          f"solved={n_solved}, cond_filtered={n_cond_filtered}, "
          f"output={n_got}/{n_valid_down}")

    # --- Strain computation ---
    if method == 'green_lagrange':
        exx = du_dx + 0.5 * (du_dx**2 + dv_dx**2)
        eyy = dv_dy + 0.5 * (du_dy**2 + dv_dy**2)
        exy = 0.5 * (du_dy + dv_dx + du_dx * du_dy + dv_dx * dv_dy)
    elif method == 'engineering':
        exx = du_dx
        eyy = dv_dy
        exy = 0.5 * (du_dy + dv_dx)
    else:
        raise ValueError(f"Unknown strain method: {method}")

    # Principal strains
    center = (exx + eyy) / 2.0
    radius = np.sqrt(((exx - eyy) / 2.0)**2 + exy**2)
    e1 = center + radius
    e2 = center - radius
    max_shear = radius
    von_mises = np.sqrt(e1**2 - e1 * e2 + e2**2)

    # Rotation via polar decomposition (Numba JIT)
    rotation = _compute_rotation_field_numba(du_dx, du_dy, dv_dx, dv_dy)

    # Apply original mask to prevent VSG "expansion" beyond valid region
    for comp in [exx, eyy, exy, e1, e2, max_shear, von_mises, rotation]:
        comp[~mask_down] = np.nan

    # Boundary erosion: mask out pixels too close to NaN regions (holes/edges)
    erosion_px = boundary_erosion
    if erosion_px < 0:
        # Auto: use half the VSG window (where asymmetric weighting is severe)
        erosion_px = vsg_size // 2

    if erosion_px > 0:
        from scipy.ndimage import distance_transform_edt
        # Use the displacement validity mask (not VSG-valid mask) so that
        # erosion follows the ROI shape faithfully.  VSG edge failures already
        # produce NaN and don't need extra erosion.
        # Pad with a 1-pixel False border so distance_transform_edt always has
        # background pixels — without padding, an all-True array (rectangular
        # ROI cropped to its bounding box) triggers degenerate EDT behaviour
        # that only erodes a quarter-circle at one corner.
        padded = np.pad(mask_down, pad_width=1, mode='constant',
                        constant_values=False)
        dist_to_boundary = distance_transform_edt(padded)[1:-1, 1:-1]
        # Erosion radius in downsampled pixels
        erosion_down = max(1, erosion_px // step)
        erode_mask = dist_to_boundary < erosion_down
        n_eroded = int(np.sum(erode_mask & mask_down))
        for comp in [exx, eyy, exy, e1, e2, max_shear, von_mises, rotation]:
            comp[erode_mask] = np.nan
        print(f"[TIMING] Boundary erosion: {erosion_px}px "
              f"(downsampled={erosion_down}px), eroded {n_eroded} pixels")

    t_end = time.perf_counter()
    print(f"[TIMING] Strain calc total: {t_end-t_start:.3f}s")

    return {
        'exx': exx, 'eyy': eyy, 'exy': exy,
        'e1': e1, 'e2': e2,
        'max_shear': max_shear, 'von_mises': von_mises,
        'rotation': rotation,
    }


def smooth_strain_temporal(strain_results: list, sigma_t: float = 1.0) -> list:
    """Apply Gaussian smoothing along the time axis for each strain component.

    Args:
        strain_results: List of strain dicts (one per frame).
        sigma_t: Gaussian sigma in frame units (e.g. 1.0 = smooth over ~3 frames).

    Returns:
        New list of strain dicts with temporally smoothed values.
    """
    if not strain_results or sigma_t <= 0:
        return strain_results

    from scipy.ndimage import gaussian_filter1d

    keys = [k for k in strain_results[0].keys() if strain_results[0][k] is not None]
    T = len(strain_results)

    smoothed = [{} for _ in range(T)]
    for key in keys:
        stack = np.array([s[key] for s in strain_results])  # (T, H, W)
        valid = np.isfinite(stack)
        filled = np.nan_to_num(stack, nan=0.0)
        weight = valid.astype(np.float64)
        smoothed_data = gaussian_filter1d(filled, sigma=sigma_t, axis=0)
        smoothed_weight = gaussian_filter1d(weight, sigma=sigma_t, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(smoothed_weight > 0.01, smoothed_data / smoothed_weight, np.nan)
        for t in range(T):
            smoothed[t][key] = result[t].astype(np.float32)

    return smoothed


def calculate_strain_rate(strain_results: list, fps: float = 1.0) -> list:
    """Compute strain rate de/dt using central difference.

    For each frame, computes the time derivative of exx, eyy, exy.
    Uses central difference for interior frames, forward/backward at endpoints.

    Returns list of dicts with keys: 'dexx_dt', 'deyy_dt', 'dexy_dt'.
    These are merged INTO the existing strain_results dicts (modified in-place).
    """
    T = len(strain_results)
    if T < 2:
        return strain_results

    dt = 1.0 / fps if fps > 0 else 1.0

    for i in range(T):
        for comp in ['exx', 'eyy', 'exy']:
            rate_key = f'd{comp}_dt'
            if comp not in strain_results[i]:
                continue

            if i == 0:
                # Forward difference
                if comp in strain_results[1]:
                    strain_results[i][rate_key] = (strain_results[1][comp] - strain_results[0][comp]) / dt
            elif i == T - 1:
                # Backward difference
                if comp in strain_results[T - 2]:
                    strain_results[i][rate_key] = (strain_results[T - 1][comp] - strain_results[T - 2][comp]) / dt
            else:
                # Central difference
                if comp in strain_results[i - 1] and comp in strain_results[i + 1]:
                    strain_results[i][rate_key] = (strain_results[i + 1][comp] - strain_results[i - 1][comp]) / (2.0 * dt)

    return strain_results
