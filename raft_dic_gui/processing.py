"""
Processing helpers for RAFT-DIC
- Robust image loading/normalization
- Windowed and full-image processing
- Smoothing and result saving
"""

import os
import time
import numpy as np
import cv2
import tifffile
import scipy.io as sio
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
import matplotlib.cm as cm

from .model import inference


def load_and_convert_image(img_path: str) -> np.ndarray:
    """Load image and convert to 8-bit RGB with sensible handling for TIFF/float/uint16."""
    ext = os.path.splitext(img_path)[1].lower()

    if ext in ['.tif', '.tiff']:
        frame = tifffile.imread(img_path)
    else:
        frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if frame is None:
        raise Exception(f"Failed to load image from {img_path}")

    if frame.dtype == np.float32:
        min_val = np.nanmin(frame)
        max_val = np.nanmax(frame)
        if max_val != min_val:
            frame = np.clip((frame - min_val) / (max_val - min_val), 0, 1)
        else:
            frame = np.zeros_like(frame)
        frame = (frame * 255).astype(np.uint8)
    elif frame.dtype == np.uint16:
        frame = (frame / 256).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] > 3:
        frame = frame[:, :, :3]

    if ext in ['.tif', '.tiff'] and len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def calculate_window_positions(image_size: int, crop_size: int, shift: int):
    positions = []
    current = 0
    while current <= image_size - crop_size:
        positions.append(current)
        current += shift
    if current < image_size - crop_size:
        positions.append(image_size - crop_size)
    elif positions and positions[-1] + crop_size < image_size:
        positions.append(image_size - crop_size)
    return positions


def smooth_displacement_field(displacement_field: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    smoothed = np.zeros_like(displacement_field)
    for i in range(2):
        comp = displacement_field[..., i]
        valid_mask = ~np.isnan(comp)
        if not np.any(valid_mask):
            smoothed[..., i] = comp
            continue
        filled = np.where(valid_mask, comp, 0)
        smoothed_data = gaussian_filter(filled, sigma)
        weight = gaussian_filter(valid_mask.astype(float), sigma)
        with np.errstate(divide='ignore', invalid='ignore'):
            smoothed[..., i] = np.where(weight > 0.01, smoothed_data / weight, np.nan)
    return smoothed


def cut_image_pair_with_flow(ref_img: np.ndarray, def_img: np.ndarray, project_root: str, model, device: str,
                             crop_size=(128, 128), shift=64, plot_windows=False,
                             roi_mask=None, use_smooth=True, sigma=2.0):
    start_total = time.time()

    windows_dir = os.path.join(project_root, "windows")
    os.makedirs(windows_dir, exist_ok=True)

    H, W, _ = ref_img.shape
    crop_h, crop_w = crop_size

    x_positions = calculate_window_positions(W, crop_w, shift)
    y_positions = calculate_window_positions(H, crop_h, shift)

    windows = []
    global_flow = np.zeros((H, W, 2), dtype=np.float64)
    count_field = np.zeros((H, W), dtype=np.float64)

    if roi_mask is not None:
        roi_mask = roi_mask.astype(bool)
    else:
        roi_mask = np.ones((H, W), dtype=bool)

    # Smooth fusion weights: raised-cosine (Hanning) window to taper borders
    wy = np.hanning(max(crop_h, 2))  # ensure length >= 2
    wx = np.hanning(max(crop_w, 2))
    weight2d = np.outer(wy, wx).astype(np.float64)
    # Avoid extremely small weights causing numerical issues
    weight2d = np.clip(weight2d, 1e-6, None)

    count = 0
    inference_time = 0
    for y in y_positions:
        for x in x_positions:
            ref_window = ref_img[y:y+crop_h, x:x+crop_w, :]
            def_window = def_img[y:y+crop_h, x:x+crop_w, :]

            windows.append({'index': count, 'position': (x, y, x+crop_w, y+crop_h)})

            t0 = time.time()
            flow_low, flow_up = inference(model, ref_window, def_window, device, test_mode=True)
            flow_up = flow_up.squeeze(0)
            inference_time += time.time() - t0

            u = flow_up[0].cpu().numpy()
            v = flow_up[1].cpu().numpy()
            window_flow = np.stack((u, v), axis=-1)

            # Local ROI weighting
            if roi_mask is not None:
                local_roi = roi_mask[y:y+crop_h, x:x+crop_w].astype(np.float64)
                local_weight = weight2d * local_roi
            else:
                local_weight = weight2d
            global_flow[y:y+crop_h, x:x+crop_w, :] += window_flow * local_weight[..., None]
            count_field[y:y+crop_h, x:x+crop_w] += local_weight

            count += 1

    final_flow = np.where(count_field[..., None] > 0,
                          global_flow / count_field[..., None],
                          np.nan)
    final_flow[~roi_mask] = np.nan

    displacement_field = smooth_displacement_field(final_flow, sigma=sigma) if use_smooth else final_flow

    if plot_windows:
        try:
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(ref_gray, cmap='gray')
            ax.axis('off')
            colormap = cm.get_cmap('hsv', len(windows))
            patches_list = []
            for w in windows:
                x0, y0, x1, y1 = w['position']
                rect = patches.Rectangle((x0, y0), x1-x0, y1-y0, linewidth=1,
                                         edgecolor=colormap(w['index']), facecolor='none')
                patches_list.append(rect)
            ax.add_collection(collections.PatchCollection(patches_list, match_original=True))
            plt.title("Sliding Windows Layout")
            plt.savefig(os.path.join(windows_dir, "windows_layout.png"), bbox_inches='tight', dpi=300)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Failed to save windows layout: {str(e)}")

    total_time = time.time() - start_total
    print("\nTime statistics:")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"RAFT inference time: {inference_time:.2f} seconds")
    if total_time > 0:
        print(f"RAFT inference percentage: {(inference_time/total_time*100):.1f}%")
        print(f"Other operations time: {(total_time-inference_time):.2f} seconds")

    return displacement_field, windows


def process_image_pair(ref_img: np.ndarray, def_img: np.ndarray, project_root: str, model, device: str,
                       roi_mask=None, use_smooth: bool = True, sigma: float = 2.0, iters: int = 12):
    start_total = time.time()
    H, W, _ = ref_img.shape
    if roi_mask is not None:
        roi_mask = roi_mask.astype(bool)
    else:
        roi_mask = np.ones((H, W), dtype=bool)

    t0 = time.time()
    try:
        flow_low, flow_up = inference(model, ref_img, def_img, device, test_mode=True, iters=iters)
    except torch.cuda.OutOfMemoryError:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # Retry with fewer iterations to reduce memory footprint
        flow_low, flow_up = inference(model, ref_img, def_img, device, test_mode=True, iters=max(4, iters // 2))
    flow_up = flow_up.squeeze(0)
    inference_time = time.time() - t0

    u = flow_up[0].cpu().numpy()
    v = flow_up[1].cpu().numpy()
    displacement_field = np.stack((u, v), axis=-1)
    displacement_field[~roi_mask] = np.nan

    if use_smooth:
        displacement_field = smooth_displacement_field(displacement_field, sigma=sigma)

    total_time = time.time() - start_total
    print("\nTime statistics:")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"RAFT inference time: {inference_time:.2f} seconds")
    if total_time > 0:
        print(f"RAFT inference percentage: {(inference_time/total_time*100):.1f}%")
        print(f"Other operations time: {(total_time-inference_time):.2f} seconds")

    return displacement_field, None


# ------------------------- New tiled ROI processing -------------------------
import torch


def _compute_min_bounding_box(roi_mask_full: np.ndarray):
    ys, xs = np.where(roi_mask_full)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    m = int(xs.max() - x0 + 1)
    n = int(ys.max() - y0 + 1)
    return x0, y0, m, n


def _expand_and_clamp(bbox, pad, width, height):
    x0, y0, m, n = bbox
    xC = max(0, x0 - pad)
    yC = max(0, y0 - pad)
    xR = min(width, x0 + m + pad)
    yB = min(height, y0 + n + pad)
    wC = int(xR - xC)
    hC = int(yB - yC)
    return xC, yC, wC, hC


def _choose_tile_size(wC, hC, p_max_pixels=1100*1100, prefer_square=False):
    """Choose tile size for context C.

    If the entire expanded region fits within the pixel budget, return a single-tile
    size equal to the region itself (wC, hC) so processing can run in one shot.
    Otherwise, choose a tile size under the pixel budget with optional square preference.
    """
    # One-shot if area fits budget
    if max(1, wC) * max(1, hC) <= int(p_max_pixels):
        return max(1, int(wC)), max(1, int(hC))

    if prefer_square:
        # Choose the largest square up to per-axis cap and area budget
        T = min(1100, wC, hC)
        Tx = Ty = max(1, int(T))
    else:
        # Aspect-ratio aware under per-axis cap and area budget
        if wC >= hC:
            Tx = min(1100, int(wC))
            Ty = min(1100, max(1, int(p_max_pixels) // max(1, Tx)))
        else:
            Ty = min(1100, int(hC))
            Tx = min(1100, max(1, int(p_max_pixels) // max(1, Ty)))
    # Ensure product within budget
    while Tx * Ty > int(p_max_pixels):
        if Tx >= Ty:
            Tx -= 1
        else:
            Ty -= 1
        if Tx <= 1 and Ty <= 1:
            break
    return max(1, int(Tx)), max(1, int(Ty))


def _make_starts_with_tail_snap(total, tile, stride):
    starts = [0]
    pos = 0
    if tile >= total:
        return [0]
    while pos + tile < total:
        pos = min(pos + stride, total - tile)
        if not starts or starts[-1] != pos:
            starts.append(pos)
        if pos == total - tile:
            break
    # unique sorted
    return sorted(set(starts))


def _build_weight_window(Tx, Ty, g_tile):
    if g_tile <= 0:
        return np.ones((Ty, Tx), dtype=np.float64)
    wx = np.ones(Tx, dtype=np.float64)
    wy = np.ones(Ty, dtype=np.float64)
    # cosine ramps in guard bands
    for x in range(Tx):
        if x < g_tile:
            t = (x + 0.5) / max(1, g_tile)
            wx[x] = 0.5 * (1 - np.cos(np.pi * t))
        elif x >= Tx - g_tile:
            t = (Tx - 0.5 - x) / max(1, g_tile)
            wx[x] = 0.5 * (1 - np.cos(np.pi * t))
        else:
            wx[x] = 1.0
    for y in range(Ty):
        if y < g_tile:
            t = (y + 0.5) / max(1, g_tile)
            wy[y] = 0.5 * (1 - np.cos(np.pi * t))
        elif y >= Ty - g_tile:
            t = (Ty - 0.5 - y) / max(1, g_tile)
            wy[y] = 0.5 * (1 - np.cos(np.pi * t))
        else:
            wy[y] = 1.0
    return np.outer(wy, wx).astype(np.float64)


def dic_over_roi_with_tiling(ref_img: np.ndarray,
                             def_img: np.ndarray,
                             roi_mask_full: np.ndarray,
                             model,
                             device: str,
                             D_global: int,
                             g_tile: int,
                             overlap_ratio: float = 0.10,
                             p_max_pixels: int = 1100*1100,
                             prefer_square: bool = False,
                             use_smooth: bool = True,
                             sigma: float = 2.0,
                             iters: int = 12):
    H, W, _ = ref_img.shape
    if roi_mask_full is None or roi_mask_full.shape[:2] != (H, W):
        raise ValueError("roi_mask_full must be provided and match image size")

    bbox = _compute_min_bounding_box(roi_mask_full)
    if bbox is None:
        raise ValueError("ROI mask is empty")

    # Expand bounding box by D_global on all sides (clamped)
    xC, yC, wC, hC = _expand_and_clamp(bbox, int(D_global), W, H)

    # Prepare accumulation buffers over C
    accum = np.zeros((hC, wC, 2), dtype=np.float64)
    wsum = np.zeros((hC, wC), dtype=np.float64)

    # ROI mask cropped to C
    roiC = roi_mask_full[yC:yC+hC, xC:xC+wC]
    roiC = roiC.astype(np.float64)

    # Choose tile size under pixel budget
    Tx, Ty = _choose_tile_size(wC, hC, p_max_pixels=p_max_pixels, prefer_square=prefer_square)
    # Single-shot if the tile equals the expanded region
    single_shot = (int(Tx) == int(wC) and int(Ty) == int(hC))

    if single_shot:
        # No guard-band constraints or overlaps needed
        starts_x = [0]
        starts_y = [0]
        W2D = np.ones((Ty, Tx), dtype=np.float64)
    else:
        # Ensure effective interior positive
        Ex = Tx - 2*int(g_tile)
        Ey = Ty - 2*int(g_tile)
        if Ex <= 0 or Ey <= 0:
            raise ValueError("g_tile too large for tile size under pixel limit; reduce g_tile or increase pixel budget")

        # Overlap in effective interior
        Ex_olap = max(1, int(round(overlap_ratio * Ex)))
        Ey_olap = max(1, int(round(overlap_ratio * Ey)))
        stride_x = max(1, Ex - Ex_olap)
        stride_y = max(1, Ey - Ey_olap)

        starts_x = _make_starts_with_tail_snap(wC, Tx, stride_x)
        starts_y = _make_starts_with_tail_snap(hC, Ty, stride_y)

        W2D = _build_weight_window(Tx, Ty, int(g_tile))

    inference_time = 0.0
    for sy in starts_y:
        for sx in starts_x:
            # Tile size is constrained by context region C (wC, hC), not full image bounds
            tile_w = min(Tx, wC - sx)
            tile_h = min(Ty, hC - sy)
            if tile_w <= 0 or tile_h <= 0:
                continue
            x0 = xC + sx
            y0 = yC + sy
            x1 = x0 + tile_w
            y1 = y0 + tile_h

            ref_tile = ref_img[y0:y1, x0:x1, :]
            def_tile = def_img[y0:y1, x0:x1, :]

            t0 = time.time()
            try:
                _, flow_up = inference(model, ref_tile, def_tile, device, test_mode=True, iters=iters)
            except torch.cuda.OutOfMemoryError:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                _, flow_up = inference(model, ref_tile, def_tile, device, test_mode=True, iters=max(4, iters // 2))
            flow_up = flow_up.squeeze(0)
            inference_time += (time.time() - t0)

            u = flow_up[0].cpu().numpy()
            v = flow_up[1].cpu().numpy()
            flow = np.stack((u, v), axis=-1)

            # Match window to actual tile size (edge tiles may be smaller)
            Wtile = W2D[:tile_h, :tile_w]

            # Apply ROI mask within C for this tile region
            roi_local = roiC[sy:sy+tile_h, sx:sx+tile_w]
            weight_local = Wtile * roi_local

            accum[sy:sy+tile_h, sx:sx+tile_w, :] += flow * weight_local[..., None]
            wsum[sy:sy+tile_h, sx:sx+tile_w] += weight_local

    # Normalize within C
    with np.errstate(divide='ignore', invalid='ignore'):
        result_C = np.where(wsum[..., None] > 0, accum / wsum[..., None], np.nan)

    # Optional smoothing after fusion, only inside ROI
    if use_smooth:
        # Mask outside ROI to NaN before smoothing; then smooth and keep NaNs where no data
        masked = result_C.copy()
        invalid = ~roiC.astype(bool)
        masked[invalid] = np.nan
        result_C = smooth_displacement_field(masked, sigma=sigma)

    # Paste back to full image coordinates
    disp_full = np.full((H, W, 2), np.nan, dtype=np.float64)
    disp_full[yC:yC+hC, xC:xC+wC, :] = result_C

    return disp_full, (xC, yC, wC, hC)


def save_displacement_results(displacement_field: np.ndarray, output_dir: str, index: int,
                              roi_rect=None, roi_mask=None):
    npy_dir = os.path.join(output_dir, "displacement_results_npy")
    mat_dir = os.path.join(output_dir, "displacement_results_mat")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(mat_dir, exist_ok=True)

    npy_file = os.path.join(npy_dir, f"displacement_field_{index}.npy")
    np.save(npy_file, displacement_field)

    h, w = displacement_field.shape[:2]
    if roi_rect is not None:
        xmin, ymin, xmax, ymax = roi_rect
        y_coords, x_coords = np.mgrid[ymin:ymin+h, xmin:xmin+w]
        X_ref = x_coords.astype(np.float64)
        Y_ref = y_coords.astype(np.float64)
        u = displacement_field[:, :, 0]
        v = displacement_field[:, :, 1]
        X_current = X_ref + u
        Y_current = Y_ref + v
        if roi_mask is not None:
            roi_mask_crop = roi_mask[ymin:ymax, xmin:xmax]
            invalid_mask = ~roi_mask_crop
            X_ref[invalid_mask] = np.nan
            Y_ref[invalid_mask] = np.nan
            X_current[invalid_mask] = np.nan
            Y_current[invalid_mask] = np.nan
    else:
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        X_ref = x_coords.astype(np.float64)
        Y_ref = y_coords.astype(np.float64)
        u = displacement_field[:, :, 0]
        v = displacement_field[:, :, 1]
        X_current = X_ref + u
        Y_current = Y_ref + v

    mat_file = os.path.join(mat_dir, f"displacement_field_{index}.mat")
    u = displacement_field[:, :, 0]
    v = displacement_field[:, :, 1]
    sio.savemat(mat_file, {
        'U': u,
        'V': v,
        'X_ref': X_ref,
        'Y_ref': Y_ref,
        'X_current': X_current,
        'Y_current': Y_current,
    })


def save_displacement_sequence(displacements, output_dir: str, roi_rect=None, roi_mask=None,
                               save_numpy: bool = True, save_matlab: bool = True,
                               numpy_filename: str = "displacement_sequence.npz",
                               matlab_filename: str = "displacement_sequence.mat"):
    """
    Save an entire sequence of displacement fields into single Python (.npz) and/or MATLAB (.mat) files.

    Args:
        displacements: list of np.ndarray, each H x W x 2 (u,v), NaNs for invalid
        output_dir: destination directory (typically project root)
        roi_rect: (xmin, ymin, xmax, ymax) used to compute reference/current coordinates
        roi_mask: full-image boolean mask; cropped based on roi_rect
        save_numpy: whether to write consolidated .npz
        save_matlab: whether to write consolidated .mat with cell arrays
        numpy_filename: filename for npz
        matlab_filename: filename for mat
    """
    if not displacements:
        return

    os.makedirs(output_dir, exist_ok=True)

    h, w = displacements[0].shape[:2]

    # Compute reference grid and cropped mask once
    if roi_rect is not None:
        xmin, ymin, xmax, ymax = roi_rect
        y_coords, x_coords = np.mgrid[ymin:ymin+h, xmin:xmin+w]
    else:
        y_coords, x_coords = np.mgrid[0:h, 0:w]

    X_ref = x_coords.astype(np.float64)
    Y_ref = y_coords.astype(np.float64)

    roi_mask_crop = None
    if roi_mask is not None:
        if roi_rect is not None:
            xmin, ymin, xmax, ymax = roi_rect
            roi_mask_crop = roi_mask[ymin:ymax, xmin:xmax]
        else:
            roi_mask_crop = roi_mask

    # Prepare Python-friendly consolidated NPZ (stack frames)
    if save_numpy:
        # Stack displacements into (T, H, W, 2)
        disp_stack = np.stack(displacements, axis=0)
        out_npz = os.path.join(output_dir, numpy_filename)
        np.savez_compressed(
            out_npz,
            displacements=disp_stack,
            X_ref=X_ref,
            Y_ref=Y_ref,
            roi_rect=np.array(roi_rect if roi_rect is not None else [0, 0, w, h], dtype=np.int64),
            roi_mask=roi_mask_crop if roi_mask_crop is not None else np.ones((h, w), dtype=bool)
        )

    # Prepare MATLAB-friendly consolidated MAT with cell arrays
    if save_matlab:
        T = len(displacements)
        U_cells = np.empty((1, T), dtype=object)
        V_cells = np.empty((1, T), dtype=object)
        Xref_cells = np.empty((1, T), dtype=object)
        Yref_cells = np.empty((1, T), dtype=object)
        Xcur_cells = np.empty((1, T), dtype=object)
        Ycur_cells = np.empty((1, T), dtype=object)

        for i, disp in enumerate(displacements):
            u = disp[:, :, 0]
            v = disp[:, :, 1]

            Xc = X_ref + u
            Yc = Y_ref + v

            if roi_mask_crop is not None:
                invalid = ~roi_mask_crop
                Xr_i = X_ref.copy()
                Yr_i = Y_ref.copy()
                Xc_i = Xc.copy()
                Yc_i = Yc.copy()
                Xr_i[invalid] = np.nan
                Yr_i[invalid] = np.nan
                Xc_i[invalid] = np.nan
                Yc_i[invalid] = np.nan
            else:
                Xr_i, Yr_i, Xc_i, Yc_i = X_ref, Y_ref, Xc, Yc

            U_cells[0, i] = u
            V_cells[0, i] = v
            Xref_cells[0, i] = Xr_i
            Yref_cells[0, i] = Yr_i
            Xcur_cells[0, i] = Xc_i
            Ycur_cells[0, i] = Yc_i

        out_mat = os.path.join(output_dir, matlab_filename)
        sio.savemat(out_mat, {
            'U': U_cells,
            'V': V_cells,
            'X_ref': Xref_cells,
            'Y_ref': Yref_cells,
            'X_current': Xcur_cells,
            'Y_current': Ycur_cells,
        })
