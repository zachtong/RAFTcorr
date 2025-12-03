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
import io
from PIL import Image
from scipy.interpolate import griddata, Rbf
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
import matplotlib.cm as cm

from .model import inference


def load_and_convert_image(img_path: str) -> np.ndarray:
    """Load image and convert to 8-bit RGB with sensible handling for TIFF/float/uint16.

    Robustness improvements:
    - If a file has .tif/.tiff extension but is actually PNG (or another format),
      gracefully fall back to OpenCV instead of raising a hard error from tifffile.
    """
    ext = os.path.splitext(img_path)[1].lower()

    frame = None
    if ext in ['.tif', '.tiff']:
        try:
            frame = tifffile.imread(img_path)
        except Exception as e:
            # Fallback if mislabeled TIFF or unsupported TIFF variant; try generic reader
            print(f"Warning: TIFF reader failed for '{img_path}' ({e}); falling back to cv2.")
            frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
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
    _emit_status("\nTime statistics:")
    _emit_status(f"Total processing time: {total_time:.2f} seconds")
    _emit_status(f"RAFT inference time: {inference_time:.2f} seconds")
    if total_time > 0:
        _emit_status(f"RAFT inference percentage: {(inference_time/total_time*100):.1f}%")
        _emit_status(f"Other operations time: {(total_time-inference_time):.2f} seconds")

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
    _emit_status("\nTime statistics:")
    _emit_status(f"Total processing time: {total_time:.2f} seconds")
    _emit_status(f"RAFT inference time: {inference_time:.2f} seconds")
    if total_time > 0:
        _emit_status(f"RAFT inference percentage: {(inference_time/total_time*100):.1f}%")
        _emit_status(f"Other operations time: {(total_time-inference_time):.2f} seconds")

    return displacement_field, None


# ------------------------- New tiled ROI processing -------------------------
import torch

_status_logger = None

def set_status_logger(callback):
    global _status_logger
    _status_logger = callback

def _emit_status(message: str):
    print(message, flush=True)
    if _status_logger:
        try:
            _status_logger(message)
        except Exception:
            pass



_status_logger = None

def set_status_logger(callback):
    global _status_logger
    _status_logger = callback

def _emit_status(message: str):
    print(message, flush=True)
    if _status_logger:
        try:
            _status_logger(message)
        except Exception:
            pass



def _compute_min_bounding_box(roi_mask_full: np.ndarray):
    ys, xs = np.where(roi_mask_full)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    m = int(xs.max() - x0 + 1)
    n = int(ys.max() - y0 + 1)
    return x0, y0, m, n


def _expand_bbox(bbox, pad, width, height):
    x0, y0, m, n = bbox
    xC = max(0, x0 - pad)
    yC = max(0, y0 - pad)
    xR = min(width, x0 + m + pad)
    yB = min(height, y0 + n + pad)
    wC = int(xR - xC)
    hC = int(yB - yC)
    return xC, yC, wC, hC


def _generate_tile_starts(total_len, tile_len, overlap):
    """Generate start positions for sliding window with fixed overlap."""
    if total_len <= tile_len:
        return [0]
    
    stride = tile_len - overlap
    if stride <= 0:
        stride = 1 # Fallback to avoid infinite loop
        
    starts = []
    pos = 0
    while pos + tile_len < total_len:
        starts.append(pos)
        pos += stride
        
    # Ensure the last tile covers the end
    if starts[-1] + tile_len < total_len:
        starts.append(total_len - tile_len)
        
    # If the last step was too small (huge overlap), we might want to merge?
    # But for simplicity, just keeping it is fine, fusion handles it.
    return sorted(list(set(starts)))


def dic_over_roi_with_tiling(ref_img: np.ndarray,
                             def_img: np.ndarray,
                             roi_mask_full: np.ndarray,
                             model,
                             device: str,
                             context_padding: int = 32,
                             tile_overlap: int = 32,
                             p_max_pixels: int = 1100*1100,
                             use_smooth: bool = True,
                             sigma: float = 2.0,
                             iters: int = 12):
    """
    Perform DIC over the ROI using an adaptive tiling strategy with weighted fusion.
    
    Args:
        context_padding: Extra pixels around the ROI bounding box to include (default 32).
        tile_overlap: Overlap between tiles in pixels (default 32).
        p_max_pixels: Maximum pixels allowed per inference pass (VRAM limit).
    """
    H, W, _ = ref_img.shape
    if roi_mask_full is None or roi_mask_full.shape[:2] != (H, W):
        raise ValueError("roi_mask_full must be provided and match image size")

    bbox = _compute_min_bounding_box(roi_mask_full)
    if bbox is None:
        raise ValueError("ROI mask is empty")

    # 1. Define Work Area (ROI + Padding)
    xC, yC, wC, hC = _expand_bbox(bbox, int(context_padding), W, H)

    # 2. Prepare Accumulators
    accum = np.zeros((hC, wC, 2), dtype=np.float64)
    wsum = np.zeros((hC, wC), dtype=np.float64)
    
    # ROI mask cropped to Work Area
    roiC = roi_mask_full[yC:yC+hC, xC:xC+wC]

    # 3. Determine Tiling Strategy
    # Max safe square tile dimension
    max_T = int(np.sqrt(p_max_pixels))
    
    # Check if single shot is possible
    if wC * hC <= p_max_pixels:
        tiles = [(0, 0, wC, hC)]
        is_tiled = False
    else:
        is_tiled = True
        # Ensure tile size is at least larger than overlap
        T = max(max_T, tile_overlap * 2 + 16)
        # But must not exceed budget (if possible, otherwise we clamp T to max_T)
        T = min(T, max_T)
        
        # If T is too small relative to overlap, reduce overlap
        if T <= tile_overlap:
             tile_overlap = T // 4
        
        starts_x = _generate_tile_starts(wC, T, int(tile_overlap))
        starts_y = _generate_tile_starts(hC, T, int(tile_overlap))
        
        tiles = []
        for sy in starts_y:
            for sx in starts_x:
                # Clip tile size if at edges (though _generate_tile_starts usually handles this by shifting start)
                # But here we use fixed size T unless total size is smaller
                tw = min(T, wC - sx)
                th = min(T, hC - sy)
                tiles.append((sx, sy, tw, th))

    # 4. Execute Inference
    inference_time = 0.0
    
    # Pre-compute Hanning window cache
    window_cache = {}

    for (sx, sy, tw, th) in tiles:
        x0, y0 = xC + sx, yC + sy
        x1, y1 = x0 + tw, y0 + th
        
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
            # Simple retry with fewer iterations
            _, flow_up = inference(model, ref_tile, def_tile, device, test_mode=True, iters=max(4, iters // 2))
        
        flow_up = flow_up.squeeze(0)
        inference_time += (time.time() - t0)
        
        u = flow_up[0].cpu().numpy()
        v = flow_up[1].cpu().numpy()
        flow = np.stack((u, v), axis=-1)
        
        # 5. Weighted Fusion
        # Generate or retrieve window
        k = (tw, th)
        if k not in window_cache:
            # Hanning window for smooth blending
            # If single shot, window is all 1s (or we can still use Hanning to suppress edge artifacts if we wanted, but usually 1s)
            if not is_tiled:
                window_cache[k] = np.ones((th, tw), dtype=np.float64)
            else:
                wy = np.hanning(max(th, 4))
                wx = np.hanning(max(tw, 4))
                window_cache[k] = np.outer(wy, wx).astype(np.float64)
        
        weight = window_cache[k]
        
        accum[sy:sy+th, sx:sx+tw, :] += flow * weight[..., None]
        wsum[sy:sy+th, sx:sx+tw] += weight

    # 6. Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        result_C = np.where(wsum[..., None] > 0, accum / wsum[..., None], np.nan)

    # 7. Post-processing (Smoothing & Masking)
    # Mask out non-ROI areas in the work rectangle
    invalid_mask = ~roiC.astype(bool)
    result_C[invalid_mask] = np.nan
    
    if use_smooth:
        result_C = smooth_displacement_field(result_C, sigma=sigma)

    # 8. Paste back to full image
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


def fig2img(fig):
    """Convert matplotlib image to PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    # Set uniform display size
    display_size = (400, 400)
    img = img.resize(display_size, Image.LANCZOS)
    plt.close(fig)
    return img


def create_reference_displacement_visualization(displacement, background_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v, roi_rect):
    """Create displacement field visualization on reference image background"""
    xmin, ymin, xmax, ymax = roi_rect
    
    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=False)
    
    u = displacement[:, :, 0]
    v = displacement[:, :, 1]
    
    h, w = background_image.shape[:2]
    u_full = np.full((h, w), np.nan)
    v_full = np.full((h, w), np.nan)
    
    u_full[ymin:ymax, xmin:xmax] = u
    v_full[ymin:ymax, xmin:xmax] = v
    
    ax_u.imshow(background_image, cmap='gray')
    mask_u = ~np.isnan(u_full)
    u_masked = np.ma.array(u_full, mask=~mask_u)
    
    im_u = ax_u.imshow(u_masked, cmap=current_colormap, alpha=alpha * mask_u, vmin=vmin_u, vmax=vmax_u)
    ax_u.set_title("U Component on Reference Image")
    fig.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)
    
    ax_v.imshow(background_image, cmap='gray')
    mask_v = ~np.isnan(v_full)
    v_masked = np.ma.array(v_full, mask=~mask_v)
    
    im_v = ax_v.imshow(v_masked, cmap=current_colormap, alpha=alpha * mask_v, vmin=vmin_v, vmax=vmax_v)
    ax_v.set_title("V Component on Reference Image")
    fig.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)
    
    ax_u.set_axis_off()
    ax_v.set_axis_off()
    ax_u.set_aspect('equal')
    ax_v.set_aspect('equal')
    
    return fig


def _create_deformed_displacement_image_fast(img_crop, u_full, v_full, deformed_mask, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v):
    import matplotlib.colors as mcolors
    bg = img_crop
    if bg.ndim == 2:
        bg_rgb = np.stack([bg, bg, bg], axis=-1)
    else:
        bg_rgb = bg[:, :, :3]

    def blend_component(comp, vmin, vmax):
        norm = (comp - vmin) / max(1e-8, (vmax - vmin))
        norm = np.clip(norm, 0.0, 1.0)
        cmap = cm.get_cmap(current_colormap)
        rgba = cmap(norm)  # float 0-1
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        out = bg_rgb.copy()
        m = deformed_mask
        out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + rgb[m].astype(np.float32) * alpha).astype(np.uint8)
        return out

    img_u = blend_component(u_full, vmin_u, vmax_u)
    img_v = blend_component(v_full, vmin_v, vmax_v)
    combined = np.concatenate([img_u, img_v], axis=1)
    return Image.fromarray(combined)


def prepare_visualization_data(displacement, reference_image, deformed_image, roi_rect, roi_mask,
                               preview_scale, interp_sample_step, background_mode,
                               deform_display_mode, deform_interp, show_deformed_boundary, quiver_step):
    """Prepare data for visualization without creating a figure"""
    print(f"[DEBUG] prepare_visualization_data: Mode={background_mode}, ROI={roi_rect}")
    data = {
        'background_mode': background_mode,
        'img_crop': None,
        'u_masked': None,
        'v_masked': None,
        'boundary_points': None,
        'quiver_data': None,
        'error': None
    }

    if background_mode == 'reference':
        # Reference mode logic
        try:
            xmin, ymin, xmax, ymax = roi_rect
            print(f"[DEBUG] Reference mode ROI: {xmin},{ymin},{xmax},{ymax}")
            u = displacement[:, :, 0]
            v = displacement[:, :, 1]
            h, w = reference_image.shape[:2]
            u_full = np.full((h, w), np.nan)
            v_full = np.full((h, w), np.nan)
            u_full[ymin:ymax, xmin:xmax] = u
            v_full[ymin:ymax, xmin:xmax] = v
            
            data['img_crop'] = reference_image
            data['u_masked'] = np.ma.array(u_full, mask=np.isnan(u_full))
            data['v_masked'] = np.ma.array(v_full, mask=np.isnan(v_full))
            print(f"[DEBUG] Reference mode prepared. Masked U valid count: {data['u_masked'].count()}")
            return data
        except Exception as e:
            print(f"[DEBUG] Error in reference mode: {e}")
            data['error'] = str(e)
            return data

    try:
        xmin, ymin, xmax, ymax = roi_rect
        u = displacement[:, :, 0]
        v = displacement[:, :, 1]
        
        roi_mask_crop = roi_mask[ymin:ymax, xmin:xmax] if roi_mask is not None else None
        if roi_mask_crop is None:
             # Fallback to reference
             return prepare_visualization_data(displacement, reference_image, deformed_image, roi_rect, roi_mask,
                               preview_scale, interp_sample_step, 'reference',
                               deform_display_mode, deform_interp, show_deformed_boundary, quiver_step)

        roi_h, roi_w = u.shape
        Yg, Xg = np.mgrid[ymin:ymin+roi_h, xmin:xmin+roi_w]
        
        valid = roi_mask_crop & ~np.isnan(u) & ~np.isnan(v)
        sstep = max(1, int(interp_sample_step))
        if sstep > 1:
            stride_mask = np.zeros_like(valid, dtype=bool)
            stride_mask[::sstep, ::sstep] = True
            valid = valid & stride_mask
        if not np.any(valid):
             return prepare_visualization_data(displacement, reference_image, deformed_image, roi_rect, roi_mask,
                               preview_scale, interp_sample_step, 'reference',
                               deform_display_mode, deform_interp, show_deformed_boundary, quiver_step)

        x_def = (Xg[valid] + u[valid]).astype(np.float64)
        y_def = (Yg[valid] + v[valid]).astype(np.float64)
        pts = np.column_stack((x_def, y_def))

        h, w = deformed_image.shape[:2]
        x0 = int(max(0, np.floor(x_def.min())))
        x1 = int(min(w-1, np.ceil(x_def.max())))
        y0 = int(max(0, np.floor(y_def.min())))
        y1 = int(min(h-1, np.ceil(y_def.max())))
        if x1 <= x0 or y1 <= y0:
             return prepare_visualization_data(displacement, reference_image, deformed_image, roi_rect, roi_mask,
                               preview_scale, interp_sample_step, 'reference',
                               deform_display_mode, deform_interp, show_deformed_boundary, quiver_step)

        Xq, Yq = np.meshgrid(np.arange(x0, x1+1), np.arange(y0, y1+1))

        method = deform_interp
        # Interpolate U
        if method == 'rbf' and pts.shape[0] >= 10:
            try:
                rbf_u = Rbf(x_def, y_def, u[valid], function='multiquadric')
                u_grid = rbf_u(Xq, Yq)
            except Exception:
                u_grid = griddata(pts, u[valid], (Xq, Yq), method='linear')
        else:
            u_grid = griddata(pts, u[valid], (Xq, Yq), method='linear')
            if np.isnan(u_grid).all():
                u_grid = griddata(pts, u[valid], (Xq, Yq), method='nearest')

        # Interpolate V
        if method == 'rbf' and pts.shape[0] >= 10:
            try:
                rbf_v = Rbf(x_def, y_def, v[valid], function='multiquadric')
                v_grid = rbf_v(Xq, Yq)
            except Exception:
                v_grid = griddata(pts, v[valid], (Xq, Yq), method='linear')
        else:
            v_grid = griddata(pts, v[valid], (Xq, Yq), method='linear')
            if np.isnan(v_grid).all():
                v_grid = griddata(pts, v[valid], (Xq, Yq), method='nearest')

        # Build deformed mask
        mask_grid = np.zeros_like(u_grid, dtype=np.uint8)
        xi = np.clip(np.round(x_def).astype(int) - x0, 0, (x1 - x0))
        yi = np.clip(np.round(y_def).astype(int) - y0, 0, (y1 - y0))
        mask_grid[yi, xi] = 1
        try:
            kernel = np.ones((3, 3), np.uint8)
            mask_grid = cv2.morphologyEx(mask_grid, cv2.MORPH_CLOSE, kernel, iterations=1)
        except Exception:
            pass
        
        # Prepare final data
        img_crop = deformed_image[y0:y1+1, x0:x1+1]
        u_full = np.full(img_crop.shape[:2], np.nan)
        v_full = np.full(img_crop.shape[:2], np.nan)
        
        valid_mask = (mask_grid > 0)
        u_full[valid_mask] = u_grid[valid_mask]
        v_full[valid_mask] = v_grid[valid_mask]
        
        data['img_crop'] = img_crop
        data['u_masked'] = np.ma.array(u_full, mask=~valid_mask)
        data['v_masked'] = np.ma.array(v_full, mask=~valid_mask)
        
        if show_deformed_boundary:
             # Find contours
             try:
                 contours, _ = cv2.findContours(mask_grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                 if contours:
                     # Just take the largest one or all?
                     # Flatten points
                     all_pts = []
                     for cnt in contours:
                         pts = cnt.reshape(-1, 2)
                         # Add offset
                         pts += np.array([x0, y0])
                         all_pts.append(pts)
                     data['boundary_points'] = all_pts
             except Exception:
                 pass
                 
        if deform_display_mode == 'quiver':
             # Generate quiver data on grid
             qstep = max(1, int(quiver_step))
             # Create a grid of points
             h_c, w_c = img_crop.shape[:2]
             y_q, x_q = np.mgrid[0:h_c:qstep, 0:w_c:qstep]
             
             # Filter by valid mask
             valid_q = valid_mask[y_q, x_q]
             
             if np.any(valid_q):
                 x_q = x_q[valid_q] + x0
                 y_q = y_q[valid_q] + y0
                 u_q = u_grid[y_q-y0, x_q-x0]
                 v_q = v_grid[y_q-y0, x_q-x0]
                 data['quiver_data'] = (x_q, y_q, u_q, v_q)

        return data
        
    except Exception as e:
        data['error'] = str(e)
        return data


def calculate_strain_field(displacement_field: np.ndarray, method: str = 'green_lagrange', sigma: float = 0.0):
    """
    Calculate strain field from displacement.
    
    Args:
        displacement_field: (H, W, 2) array with (u, v) displacements.
        method: 'green_lagrange' (default) or 'engineering'.
        sigma: Gaussian smoothing sigma for displacement before differentiation.
        
    Returns:
        strain_dict: Dictionary containing strain components:
                     'exx', 'eyy', 'exy' (Green-Lagrange or Engineering tensors).
                     'e1', 'e2' (Principal strains), 'max_shear', 'von_mises'.
    """
    u = displacement_field[..., 0]
    v = displacement_field[..., 1]
    
    # Mask invalid regions
    mask = ~np.isnan(u) & ~np.isnan(v)
    if not np.any(mask):
        return None
        
    # Optional: Smooth displacement before differentiation
    if sigma > 0:
        # Fill NaNs for smoothing
        u_filled = u.copy()
        v_filled = v.copy()
        u_filled[~mask] = 0
        v_filled[~mask] = 0
        
        # Smooth
        u_smooth = gaussian_filter(u_filled, sigma)
        v_smooth = gaussian_filter(v_filled, sigma)
        
        # Re-apply mask (approximate, edges might be affected)
        # Ideally we use normalized convolution like in smooth_displacement_field
        # But for simplicity here:
        u = np.where(mask, u_smooth, np.nan)
        v = np.where(mask, v_smooth, np.nan)

    # Compute Gradients (Central Difference)
    # np.gradient returns [d/dy, d/dx]
    # du/dy, du/dx
    grad_u = np.gradient(u, edge_order=1)
    du_dy = grad_u[0]
    du_dx = grad_u[1]
    
    # dv/dy, dv/dx
    grad_v = np.gradient(v, edge_order=1)
    dv_dy = grad_v[0]
    dv_dx = grad_v[1]
    
    # Strain Calculation
    if method == 'green_lagrange':
        # E_xx = du/dx + 0.5 * ((du/dx)^2 + (dv/dx)^2)
        exx = du_dx + 0.5 * (du_dx**2 + dv_dx**2)
        
        # E_yy = dv/dy + 0.5 * ((du/dy)^2 + (dv/dy)^2)
        eyy = dv_dy + 0.5 * (du_dy**2 + dv_dy**2)
        
        # E_xy = 0.5 * (du/dy + dv/dx + (du/dx)*(du/dy) + (dv/dx)*(dv/dy))
        exy = 0.5 * (du_dy + dv_dx + du_dx*du_dy + dv_dx*dv_dy)
        
    elif method == 'engineering':
        # Small strain assumption
        exx = du_dx
        eyy = dv_dy
        exy = 0.5 * (du_dy + dv_dx) # Tensor shear strain (epsilon_xy)
        # Note: Engineering shear gamma_xy = 2 * epsilon_xy
        
    else:
        raise ValueError(f"Unknown strain method: {method}")
        
    # Principal Strains
    # E1,2 = (exx + eyy)/2 +/- sqrt(((exx - eyy)/2)^2 + exy^2)
    center = (exx + eyy) / 2.0
    radius = np.sqrt(((exx - eyy) / 2.0)**2 + exy**2)
    e1 = center + radius
    e2 = center - radius
    
    # Max Shear Strain (Radius of Mohr's circle)
    max_shear = radius # This is max tensor shear strain. Engineering max shear is 2*radius.
    
    # Von Mises Equivalent Strain
    # e_vm = 2/3 * sqrt( (e1-e2)^2 + (e2-e3)^2 + (e3-e1)^2 ) / sqrt(2) ?
    # For 2D Plane Strain (e3=0):
    # e_vm = sqrt( e1^2 - e1*e2 + e2^2 ) ?
    # Let's use the effective strain definition often used in DIC:
    # e_eff = sqrt( 2/3 * (e1^2 + e2^2 + (e1+e2)^2) ) ? No.
    # Standard Von Mises for 2D (Plane Stress):
    # sigma_vm = sqrt( s1^2 - s1*s2 + s2^2 )
    # Equivalent strain e_eq = 2/3 * sqrt( 3/2 * (e_dev : e_dev) )
    # Simplified for 2D principal strains:
    von_mises = np.sqrt(e1**2 - e1*e2 + e2**2) # Approximation for visualization
    
    return {
        'exx': exx,
        'eyy': eyy,
        'exy': exy,
        'e1': e1,
        'e2': e2,
        'max_shear': max_shear,
        'von_mises': von_mises
    }




def update_displacement_plot(ax_u, ax_v, data, colormap, alpha, vmin_u, vmax_u, vmin_v, vmax_v):
    """Update existing axes with new data"""
    if data.get('error'):
        return

    # Clear axes but keep them alive
    ax_u.clear()
    ax_v.clear()
    
    img_crop = data['img_crop']
    if img_crop is not None:
        ax_u.imshow(img_crop, cmap='gray')
        ax_v.imshow(img_crop, cmap='gray')

    u_masked = data['u_masked']
    v_masked = data['v_masked']

    if u_masked is not None:
        im_u = ax_u.imshow(u_masked, cmap=colormap, alpha=alpha, vmin=vmin_u, vmax=vmax_u)
        # Note: Colorbar is tricky to update if it's already there. 
        # For robustness, we assume the caller handles colorbar or we just don't update it if it's fixed.
        # But to be safe, we can let the caller handle colorbar creation/update or just re-create it.
        # However, re-creating colorbar every time might shrink the axis.
        # Ideally, we should update the mappable of the existing colorbar.
        pass

    if v_masked is not None:
        im_v = ax_v.imshow(v_masked, cmap=colormap, alpha=alpha, vmin=vmin_v, vmax=vmax_v)

    ax_u.set_title("U Component")
    ax_v.set_title("V Component")

    boundary = data.get('boundary_points')
    if boundary:
        xb, yb = boundary
        ax_u.plot(xb, yb, color='white', linewidth=1.0, alpha=0.8)
        ax_v.plot(xb, yb, color='white', linewidth=1.0, alpha=0.8)

    quiver = data.get('quiver_data')
    if quiver:
        xq, yq, uq, vq = quiver
        ax_u.quiver(xq, yq, uq, vq, color='white', angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.9)
        ax_v.quiver(xq, yq, uq, vq, color='white', angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.9)

    ax_u.set_axis_off()
    ax_v.set_axis_off()
    ax_u.set_aspect('equal')
    ax_v.set_aspect('equal')
    
    return im_u, im_v

