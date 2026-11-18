"""
Processing helpers for RAFT-DIC
- Robust image loading/normalization
- Windowed and full-image processing
- Smoothing and result saving
"""

import os
import time
from typing import Optional, Callable
import numpy as np
import cv2
import tifffile
import scipy.io as sio
import io
from PIL import Image
from scipy.interpolate import griddata, Rbf
from raft_dic_gui.smoothing import smooth_displacement_field  # noqa: E402
from raft_dic_gui.velocity import calculate_displacement_magnitude, calculate_velocity_field  # noqa: E402

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections

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
        raise ValueError(
            f"Tile overlap ({overlap}) must be smaller than tile size ({tile_len}). "
            f"Reduce overlap or increase memory budget."
        )
        
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
                             context_padding: int = 64,
                             tile_overlap: int = 64,
                             p_max_pixels: int = 1100*1100,
                             use_smooth: bool = True,
                             sigma: float = 2.0,
                             iters: int = 12,
                             tile_callback: Optional[Callable] = None):
    """
    Perform DIC over the ROI using an adaptive tiling strategy with weighted fusion.

    Args:
        context_padding: Extra pixels around the ROI bounding box to include (default 64).
        tile_overlap: Overlap between tiles in pixels (default 64).
        p_max_pixels: Maximum pixels allowed per inference pass (VRAM limit).
        tile_callback: Optional callable(tile_idx, tile_total) called before each tile inference.
    """
    H, W, _ = ref_img.shape
    start_total = time.time()
    if roi_mask_full is None or roi_mask_full.shape[:2] != (H, W):
        raise ValueError("roi_mask_full must be provided and match image size")

    bbox = _compute_min_bounding_box(roi_mask_full)
    if bbox is None:
        raise ValueError("ROI mask is empty")

    # 1. Define Work Area (ROI + Padding)
    xC, yC, wC, hC = _expand_bbox(bbox, int(context_padding), W, H)

    # ROI mask cropped to Work Area
    roiC = roi_mask_full[yC:yC+hC, xC:xC+wC]

    # --- Retry loop for OOM recovery ---
    max_retries = 1
    retry_count = 0
    current_p_max = p_max_pixels

    while True:
        try:
            # 2. Prepare Accumulators
            accum = np.zeros((hC, wC, 2), dtype=np.float64)
            wsum = np.zeros((hC, wC), dtype=np.float64)

            # 3. Determine Tiling Strategy
            max_T = int(np.sqrt(current_p_max))
            _tile_overlap = tile_overlap

            if wC * hC <= current_p_max:
                tiles = [(0, 0, wC, hC)]
                is_tiled = False
            else:
                is_tiled = True
                T = max_T
                # If tile is too small for meaningful overlap blending, reduce overlap
                if T < _tile_overlap * 2 + 16:
                    _tile_overlap = max(4, T // 4)

                starts_x = _generate_tile_starts(wC, T, int(_tile_overlap))
                starts_y = _generate_tile_starts(hC, T, int(_tile_overlap))

                tiles = []
                for sy in starts_y:
                    for sx in starts_x:
                        tw = min(T, wC - sx)
                        th = min(T, hC - sy)
                        tiles.append((sx, sy, tw, th))

            # 4. Execute Inference
            inference_time = 0.0
            window_cache = {}

            for tile_idx, (sx, sy, tw, th) in enumerate(tiles):
                if tile_callback:
                    tile_callback(tile_idx, len(tiles))

                x0, y0 = xC + sx, yC + sy
                x1, y1 = x0 + tw, y0 + th

                ref_tile = ref_img[y0:y1, x0:x1, :]
                def_tile = def_img[y0:y1, x0:x1, :]

                t0 = time.time()
                _, flow_up = inference(model, ref_tile, def_tile, device, test_mode=True, iters=iters)
                flow_up = flow_up.squeeze(0)
                inference_time += (time.time() - t0)

                u = flow_up[0].cpu().numpy()
                v = flow_up[1].cpu().numpy()
                flow = np.stack((u, v), axis=-1)

                # 5. Weighted Fusion
                k = (tw, th)
                if k not in window_cache:
                    if not is_tiled:
                        window_cache[k] = np.ones((th, tw), dtype=np.float64)
                    else:
                        wy = np.hanning(max(th, 4))
                        wx = np.hanning(max(tw, 4))
                        window_cache[k] = np.outer(wy, wx).astype(np.float64)

                weight = window_cache[k]
                accum[sy:sy+th, sx:sx+tw, :] += flow * weight[..., None]
                wsum[sy:sy+th, sx:sx+tw] += weight

            # Success — exit retry loop
            break

        except torch.cuda.OutOfMemoryError:
            if retry_count >= max_retries:
                raise RuntimeError(
                    f"CUDA out of memory even after reducing tile size. "
                    f"Current p_max_pixels={current_p_max}. Try reducing Memory Budget."
                )
            retry_count += 1
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            current_p_max = current_p_max // 2
            _emit_status(
                f"OOM: retrying with p_max_pixels={current_p_max} "
                f"(tile ~{int(np.sqrt(current_p_max))}x{int(np.sqrt(current_p_max))})"
            )

    # 6. Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        result_C = np.where(wsum[..., None] > 0, accum / wsum[..., None], np.nan)

    # 7. Post-processing (Smoothing & Masking)
    invalid_mask = ~roiC.astype(bool)
    result_C[invalid_mask] = np.nan

    if use_smooth:
        result_C = smooth_displacement_field(result_C, sigma=sigma)

    # 8. Paste back to full image
    disp_full = np.full((H, W, 2), np.nan, dtype=np.float64)
    disp_full[yC:yC+hC, xC:xC+wC, :] = result_C

    total_time = time.time() - start_total
    _emit_status("\nTime statistics:")
    _emit_status(f"Total processing time: {total_time:.2f} seconds")
    _emit_status(f"RAFT inference time: {inference_time:.2f} seconds")
    if total_time > 0:
        _emit_status(f"RAFT inference percentage: {(inference_time/total_time*100):.1f}%")
        _emit_status(f"Other operations time: {(total_time-inference_time):.2f} seconds")

    return disp_full, (xC, yC, wC, hC)


def save_displacement_results(displacement_field: np.ndarray, output_dir: str, index: int,
                               roi_rect=None, roi_mask=None):
    npy_dir = os.path.join(output_dir, "displacement_results_temp_npy")
    os.makedirs(npy_dir, exist_ok=True)

    npy_file = os.path.join(npy_dir, f"displacement_field_{index}.npy")
    np.save(npy_file, displacement_field)


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
        cmap = matplotlib.colormaps[current_colormap]
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
                               deform_display_mode, deform_interp, show_deformed_boundary, quiver_step,
                               deformed_view_cache=None, frame_idx=None):
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

    # Deformed mode: Show displacement WARPED to deformed frame coordinates
    # Uses forward scatter: for each reference pixel, place its value at the deformed position
    try:
        h, w = deformed_image.shape[:2]
        
        # Check cache first
        if deformed_view_cache is not None and frame_idx is not None:
            cached = deformed_view_cache.get_warped_displacement(frame_idx)
            if cached is not None:
                print(f"[DeformedView] Using cached warped displacement for frame {frame_idx}")
                data['img_crop'] = deformed_image
                data['u_masked'] = np.ma.array(cached['u_warped'], mask=np.isnan(cached['u_warped']))
                data['v_masked'] = np.ma.array(cached['v_warped'], mask=np.isnan(cached['v_warped']))
                return data
        
        print(f"[DeformedView] Computing inverse warp for frame {frame_idx}...")
        xmin, ymin, xmax, ymax = roi_rect
        u_crop = displacement[:, :, 0]
        v_crop = displacement[:, :, 1]
        roi_h, roi_w = u_crop.shape
        
        print(f"[DeformedView] ROI: ({xmin},{ymin}) to ({xmax},{ymax}), shape=({roi_h},{roi_w})")
        
        # Create coordinate grids for the ROI region
        y_ref, x_ref = np.mgrid[ymin:ymax, xmin:xmax]
        
        # Compute deformed coordinates for each pixel in ROI
        # x_def = x_ref + u, y_def = y_ref + v
        x_def = (x_ref + u_crop).astype(np.float32)
        y_def = (y_ref + v_crop).astype(np.float32)
        
        # Create output arrays (full image size) - initialize to 0 for accumulation
        # (np.add.at cannot accumulate on NaN values)
        u_warped = np.zeros((h, w), dtype=np.float32)
        v_warped = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # Valid mask: not NaN
        valid_mask = ~np.isnan(u_crop) & ~np.isnan(v_crop)
        
        # Get valid positions
        x_def_valid = x_def[valid_mask]
        y_def_valid = y_def[valid_mask]
        u_valid = u_crop[valid_mask]
        v_valid = v_crop[valid_mask]
        
        print(f"[DeformedView] Valid pixels: {len(u_valid)}")
        
        # Round to nearest integer coordinates
        x_idx = np.round(x_def_valid).astype(np.int32)
        y_idx = np.round(y_def_valid).astype(np.int32)
        
        # Filter to within bounds
        in_bounds = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)
        x_idx = x_idx[in_bounds]
        y_idx = y_idx[in_bounds]
        u_valid = u_valid[in_bounds]
        v_valid = v_valid[in_bounds]
        
        print(f"[DeformedView] In bounds: {len(u_valid)}")
        
        # Forward scatter: place values at deformed positions
        # Use accumulation for averaging when multiple points land on same pixel
        np.add.at(u_warped, (y_idx, x_idx), u_valid)
        np.add.at(v_warped, (y_idx, x_idx), v_valid)
        np.add.at(count_map, (y_idx, x_idx), 1.0)
        
        # Average where multiple values accumulated
        nonzero = count_map > 0
        u_warped[nonzero] /= count_map[nonzero]
        v_warped[nonzero] /= count_map[nonzero]
        u_warped[~nonzero] = np.nan
        v_warped[~nonzero] = np.nan
        
        # Close small holes using morphological dilation on the mask
        valid_warped = ~np.isnan(u_warped)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(valid_warped.astype(np.uint8), kernel, iterations=1)
        
        # For newly-filled pixels, interpolate from neighbors (simple approach: just keep as is)
        # More sophisticated: use cv2.inpaint or scipy interpolation
        
        valid_count = np.sum(~np.isnan(u_warped))
        print(f"[DeformedView] Warped valid pixels: {valid_count}")
        
        # Store in cache for future use
        if deformed_view_cache is not None and frame_idx is not None:
            deformed_view_cache.set_warped_displacement(frame_idx, u_warped, v_warped)
        
        data['img_crop'] = deformed_image
        data['u_masked'] = np.ma.array(u_warped, mask=np.isnan(u_warped))
        data['v_masked'] = np.ma.array(v_warped, mask=np.isnan(v_warped))
        
        print(f"[DeformedView] Deformed mode with inverse warp prepared.")
        return data
        
    except Exception as e:
        print(f"[DeformedView] Error in deformed mode: {e}")
        import traceback
        traceback.print_exc()
        data['error'] = str(e)
        return data




# Strain calculation moved to raft_dic_gui.strain — re-export for backward compatibility
from raft_dic_gui.strain import calculate_strain_field, _compute_rotation_field_numba  # noqa: F401




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


import scipy.io
import os

def save_scientific_results(displacement_results, strain_results, roi_mask, roi_rect, metadata, file_path, image_files=None, upsample_strain=True, warped_masks=None, variables=None):
    """Save full-field results to .npz / .mat (v7.3) / .h5 via streaming writes.

    Replaces the legacy in-memory implementation (12 simultaneous (T,H,W) float32
    stacks, which OOM'd at ~2000 frames). Now streams one frame at a time with
    peak RAM = O(H_roi * W_roi).

    Args:
        displacement_results: List of (H, W, 2) arrays or paths to .npy files.
        strain_results: List of dicts containing strain components.
        roi_mask: (H, W) boolean mask.
        roi_rect: (xmin, ymin, xmax, ymax).
        metadata: Dict containing analysis parameters.
        file_path: Output path with extension .npz / .mat / .h5 / .hdf5.
        image_files: List of absolute paths to the images corresponding to each frame.
        upsample_strain: If True, upsample strain to ROI resolution.
        warped_masks: Optional list of warped masks from incremental processing.
        variables: Optional list of variable names to export (e.g. ["U", "V",
            "exx"]). None or empty = export all. See raft_dic_gui.scientific_export
            for the full set of recognized names.
    """
    from raft_dic_gui.scientific_export import save_scientific_results_streaming
    try:
        return save_scientific_results_streaming(
            displacement_results=displacement_results,
            strain_results=strain_results,
            roi_mask=roi_mask,
            roi_rect=roi_rect,
            metadata=metadata or {},
            file_path=file_path,
            image_files=image_files,
            upsample_strain=upsample_strain,
            warped_masks=warped_masks,
            variables=variables,
        )
    except Exception as e:
        print(f"Error saving scientific results: {e}")
        import traceback
        traceback.print_exc()
        raise


