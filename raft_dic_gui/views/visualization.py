import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os
from scipy.interpolate import griddata, Rbf
from scipy.ndimage import gaussian_filter
from PIL import Image
import raft_dic_gui.processing as proc

def create_reference_displacement_visualization(displacement, background_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v, roi_rect):
    """
    Create displacement field visualization on reference image background
    """
    # Get ROI coordinates
    if roi_rect:
        xmin, ymin, xmax, ymax = roi_rect
    else:
        h, w = background_image.shape[:2]
        xmin, ymin, xmax, ymax = 0, 0, w, h
    
    # Create figure with two subplots and fixed layout for stability
    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=False)
    
    # Get displacement components
    u = displacement[:, :, 0]
    v = displacement[:, :, 1]
    
    # Create full-size displacement fields initialized with NaN
    h, w = background_image.shape[:2]
    u_full = np.full((h, w), np.nan)
    v_full = np.full((h, w), np.nan)
    
    # Place displacement fields in their correct position
    u_full[ymin:ymax, xmin:xmax] = u
    v_full[ymin:ymax, xmin:xmax] = v
    
    # Display U component
    ax_u.imshow(background_image, cmap='gray')
    mask_u = ~np.isnan(u_full)
    u_masked = np.ma.array(u_full, mask=~mask_u)
    
    im_u = ax_u.imshow(u_masked, cmap=current_colormap, 
                     alpha=alpha * mask_u, vmin=vmin_u, vmax=vmax_u)
    ax_u.set_title("U Component on Reference Image")
    fig.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)
    
    # Display V component
    ax_v.imshow(background_image, cmap='gray')
    mask_v = ~np.isnan(v_full)
    v_masked = np.ma.array(v_full, mask=~mask_v)
    
    im_v = ax_v.imshow(v_masked, cmap=current_colormap, 
                     alpha=alpha * mask_v, vmin=vmin_v, vmax=vmax_v)
    ax_v.set_title("V Component on Reference Image")
    fig.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)
    
    # Remove axes and set aspect ratio
    ax_u.set_axis_off()
    ax_v.set_axis_off()
    ax_u.set_aspect('equal')
    ax_v.set_aspect('equal')
    
    return fig

def create_deformed_displacement_visualization(displacement, current_frame, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v, 
                                             roi_rect, roi_mask, image_files, input_path, current_image, 
                                             interp_sample_step=1, deform_interp='linear', preview_scale=0.5,
                                             fast_preview=True, show_colorbars=False, show_deformed_boundary=True,
                                             deform_display_mode='heatmap', quiver_step=20):
    """
    Create displacement field visualization on deformed image background
    """
    try:
        # Load current deformed image
        if image_files and len(image_files) > current_frame + 1:
            deformed_img_path = os.path.join(input_path, image_files[current_frame + 1])
            deformed_image = proc.load_and_convert_image(deformed_img_path)
            if deformed_image is None:
                deformed_image = current_image
        else:
            deformed_image = current_image
            
        # Get ROI bounding box coordinates
        if roi_rect:
            xmin, ymin, xmax, ymax = roi_rect
        else:
            h, w = current_image.shape[:2]
            xmin, ymin, xmax, ymax = 0, 0, w, h

        # Get displacement components
        u = displacement[:, :, 0]
        v = displacement[:, :, 1]
        
        roi_mask_crop = roi_mask[ymin:ymax, xmin:xmax] if roi_mask is not None else None
        if roi_mask_crop is None:
            return create_reference_displacement_visualization(displacement, deformed_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v, roi_rect)

        roi_h, roi_w = u.shape
        Xg = np.mgrid[0:roi_h, 0:roi_w][1] + xmin
        Yg = np.mgrid[0:roi_h, 0:roi_w][0] + ymin

        valid = roi_mask_crop & ~np.isnan(u) & ~np.isnan(v)
        
        # Subsample valid points for interpolation if requested
        sstep = max(1, int(interp_sample_step))
        if sstep > 1:
            stride_mask = np.zeros_like(valid, dtype=bool)
            stride_mask[::sstep, ::sstep] = True
            valid = valid & stride_mask
            
        if not np.any(valid):
            return create_reference_displacement_visualization(displacement, deformed_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v, roi_rect)

        x_def = (Xg[valid] + u[valid]).astype(np.float64)
        y_def = (Yg[valid] + v[valid]).astype(np.float64)
        pts = np.column_stack((x_def, y_def))

        h, w = deformed_image.shape[:2]
        # Limit interpolation grid to bounding box of deformed points
        x0 = int(max(0, np.floor(x_def.min())))
        x1 = int(min(w-1, np.ceil(x_def.max())))
        y0 = int(max(0, np.floor(y_def.min())))
        y1 = int(min(h-1, np.ceil(y_def.max())))
        if x1 <= x0 or y1 <= y0:
            return create_reference_displacement_visualization(displacement, deformed_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v, roi_rect)

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

        # Build deformed mask by warping ROI mask forward (preserves holes)
        mask_grid = np.zeros_like(u_grid, dtype=np.uint8)
        xi = np.clip(np.round(x_def).astype(int) - x0, 0, (x1 - x0))
        yi = np.clip(np.round(y_def).astype(int) - y0, 0, (y1 - y0))
        mask_grid[yi, xi] = 1
        try:
            kernel = np.ones((3, 3), np.uint8)
            mask_grid = cv2.morphologyEx(mask_grid, cv2.MORPH_CLOSE, kernel, iterations=1)
        except Exception:
            pass

        # Mask interpolated fields outside deformed ROI (holes removed)
        u_grid = np.where(mask_grid.astype(bool), u_grid, np.nan)
        v_grid = np.where(mask_grid.astype(bool), v_grid, np.nan)

        # For faster display, use cropped region only
        img_crop = deformed_image[y0:y1+1, x0:x1+1]
        u_full = u_grid
        v_full = v_grid
        deformed_mask = ~np.isnan(u_full) & ~np.isnan(v_full)
        
        # Optional downsampling for display speed
        try:
            pscale = float(preview_scale)
        except Exception:
            pscale = 0.5
        base_w, base_h = 800, 400
        max_w = int(base_w * max(0.25, min(1.0, pscale)))
        max_h = int(base_h * max(0.25, min(1.0, pscale)))
        target_w = max_w // 2
        target_h = max_h
        scale = min(target_w / (x1 - x0 + 1), target_h / (y1 - y0 + 1), 1.0)
        if scale < 1.0:
            try:
                new_size = (max(1, int((x1 - x0 + 1) * scale)), max(1, int((y1 - y0 + 1) * scale)))
                img_crop = cv2.resize(img_crop, new_size, interpolation=cv2.INTER_AREA)
                u_full = cv2.resize(u_full.astype(np.float32), new_size, interpolation=cv2.INTER_AREA)
                v_full = cv2.resize(v_full.astype(np.float32), new_size, interpolation=cv2.INTER_AREA)
                deformed_mask = cv2.resize(deformed_mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST).astype(bool)
            except Exception:
                pass

        # If fast preview without colorbars, return a PIL image directly for speed
        if fast_preview and not show_colorbars:
            return _create_deformed_displacement_image_fast(
                img_crop, u_full, v_full, deformed_mask,
                alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v
            )

        # Create figure with two subplots
        fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=False)
        
        # Display U component on deformed image
        ax_u.imshow(img_crop, cmap='gray')

        u_masked = np.ma.array(u_full, mask=~deformed_mask)
        im_u = ax_u.imshow(u_masked, cmap=current_colormap, alpha=alpha, vmin=vmin_u, vmax=vmax_u)
        ax_u.set_title("U Component on Deformed Image (Polygon ROI)")
        fig.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04)

        # Display V component on deformed image
        ax_v.imshow(img_crop, cmap='gray')

        v_masked = np.ma.array(v_full, mask=~deformed_mask)
        im_v = ax_v.imshow(v_masked, cmap=current_colormap, alpha=alpha, vmin=vmin_v, vmax=vmax_v)
        ax_v.set_title("V Component on Deformed Image (Polygon ROI)")
        fig.colorbar(im_v, ax=ax_v, fraction=0.046, pad=0.04)

        # Optional: show deformed ROI boundary
        if show_deformed_boundary:
            try:
                # Extract boundary from roi_mask_crop
                cnts, _ = cv2.findContours(roi_mask_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(cnts) > 0:
                    cnt = max(cnts, key=cv2.contourArea)  # main contour
                    pts_local = cnt[:, 0, :]  # (N,2) as (x,y) in ROI-local
                    xs = pts_local[:, 0]
                    ys = pts_local[:, 1]
                    xs_g = xs + xmin
                    ys_g = ys + ymin
                    # Sample u,v along boundary with nearest neighbor
                    xs_r = np.clip((xs).astype(int), 0, roi_w-1)
                    ys_r = np.clip((ys).astype(int), 0, roi_h-1)
                    u_b = u[ys_r, xs_r]
                    v_b = v[ys_r, xs_r]
                    x_def_b = xs_g + u_b
                    y_def_b = ys_g + v_b
                    # Shift into crop coordinates and apply display downscale
                    x_def_b = (x_def_b - x0) * scale
                    y_def_b = (y_def_b - y0) * scale
                    ax_u.plot(x_def_b, y_def_b, color='white', linewidth=1.0, alpha=0.8)
                    ax_v.plot(x_def_b, y_def_b, color='white', linewidth=1.0, alpha=0.8)
            except Exception:
                pass

        # Quiver mode: optionally overlay sparse arrows
        if deform_display_mode == 'quiver':
            try:
                step = max(2, int(quiver_step))
            except Exception:
                step = 20
            # Sample on ROI local grid, then map to crop coordinates and downscale
            xsq = ((Xg[::step, ::step] - x0) * scale)
            ysq = ((Yg[::step, ::step] - y0) * scale)
            uq = u[::step, ::step]
            vq = v[::step, ::step]
            # Filter by ROI mask and finite values
            maskq = roi_mask_crop[::step, ::step] & np.isfinite(uq) & np.isfinite(vq)
            xsq = xsq[maskq]
            ysq = ysq[maskq]
            uq = uq[maskq]
            vq = vq[maskq]
            # Plot quivers on both axes
            ax_u.quiver(xsq, ysq, uq, vq, color='white', angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.9)
            ax_v.quiver(xsq, ysq, uq, vq, color='white', angles='xy', scale_units='xy', scale=1, width=0.002, alpha=0.9)
        
        # Remove axes and set aspect ratio
        ax_u.set_axis_off()
        ax_v.set_axis_off()
        ax_u.set_aspect('equal')
        ax_v.set_aspect('equal')
        
        return fig
        
    except Exception as e:
        print(f"Error in deformed visualization: {str(e)}, falling back to reference mode")
        import traceback
        traceback.print_exc()
        # Fallback to reference image visualization
        return create_reference_displacement_visualization(displacement, current_image, alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v, roi_rect)

def _create_deformed_displacement_image_fast(img_crop, u_full, v_full, deformed_mask,
                                             alpha, current_colormap, vmin_u, vmax_u, vmin_v, vmax_v):
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
        # Alpha composite only on valid mask
        out = bg_rgb.copy()
        m = deformed_mask
        out[m] = (out[m].astype(np.float32) * (1.0 - alpha) + rgb[m].astype(np.float32) * alpha).astype(np.uint8)
        return out

    img_u = blend_component(u_full, vmin_u, vmax_u)
    img_v = blend_component(v_full, vmin_v, vmax_v)
    combined = np.concatenate([img_u, img_v], axis=1)
    return Image.fromarray(combined)
