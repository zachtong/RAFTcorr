"""
Batch Image Export Module

Provides functions for exporting visualization images of displacement and strain fields.
This module contains pure rendering logic with no UI dependencies.
"""

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
import cv2


def _log(msg: str):
    """Print debug message with standard prefix."""
    print(f"[ImageExport] {msg}")


def render_single_frame(
    data: np.ndarray,
    background_image: Optional[np.ndarray],
    roi_rect: Optional[Tuple[int, int, int, int]],
    colormap: str,
    alpha: float,
    vmin: Optional[float],
    vmax: Optional[float],
    include_colorbar: bool,
    include_title: bool,
    title: str,
    colorbar_label: str,
    output_path: str,
    format: str,
    use_log_norm: bool = False,
    velocity_vectors: Optional[Dict] = None,
    dpi: int = 100,
    colorbar_settings: Optional[Dict] = None,
) -> bool:
    """
    Render a single visualization frame and save to file.
    
    Args:
        data: 2D array of values to visualize (can be full image size or ROI-warped)
        background_image: Optional background image (RGB)
        roi_rect: (xmin, ymin, xmax, ymax) for positioning data on background
        colormap: Matplotlib colormap name
        alpha: Overlay transparency (0-1)
        vmin, vmax: Color range (None for auto)
        include_colorbar: Whether to add colorbar
        include_title: Whether to add title
        title: Title text
        colorbar_label: Label for colorbar
        output_path: Full path to save file
        format: 'png', 'jpg', 'pdf', 'svg'
        use_log_norm: Whether to use logarithmic color scale
        velocity_vectors: Optional dict with 'du', 'dv', 'show_quiver', 'show_streamlines', etc.
        
    Returns:
        True if successful, False otherwise
    """
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatterMathtext
    
    try:
        # Create figure
        if background_image is not None:
            h, w = background_image.shape[:2]
            fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
            ax.imshow(background_image)
        else:
            h, w = data.shape
            fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
        
        # Create masked array for overlay
        masked_data = np.ma.array(data, mask=np.isnan(data))
        
        # Determine extent based on roi_rect or data shape
        if roi_rect is not None and background_image is not None:
            xmin, ymin, xmax, ymax = roi_rect
            extent = [xmin, xmax, ymax, ymin]  # Note: ymax, ymin for correct orientation
        else:
            extent = None
        
        # Determine color range
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
        
        # Setup normalization
        norm = None
        cbs = colorbar_settings or {}
        discrete_levels = cbs.get("discrete_levels", 0)

        if use_log_norm:
            log_vmin = vmin if vmin is not None and vmin > 0 else 1e-10
            log_vmax = vmax if vmax is not None and vmax > 0 else 1.0
            if log_vmin >= log_vmax:
                log_vmin = log_vmax / 1000
            norm = LogNorm(vmin=log_vmin, vmax=log_vmax)
            vmin, vmax = None, None  # Don't pass separately with norm
        elif discrete_levels and discrete_levels > 0 and vmin is not None and vmax is not None:
            from matplotlib.colors import BoundaryNorm
            boundaries = np.linspace(vmin, vmax, discrete_levels + 1)
            norm = BoundaryNorm(boundaries, discrete_levels)
            vmin, vmax = None, None

        # Plot data overlay
        if discrete_levels and discrete_levels > 0 and not use_log_norm:
            cmap = plt.get_cmap(colormap, discrete_levels)
        else:
            cmap = plt.get_cmap(colormap)
        cmap.set_bad(alpha=0)  # Transparent for NaN

        im = ax.imshow(masked_data, cmap=cmap, alpha=alpha,
                       vmin=vmin, vmax=vmax, extent=extent, norm=norm)
        
        # Plot velocity arrows if provided
        if velocity_vectors is not None:
            du = velocity_vectors.get('du')
            dv = velocity_vectors.get('dv')
            
            if du is not None and dv is not None:
                vh, vw = du.shape
                scale = velocity_vectors.get('scale', 100)
                color = velocity_vectors.get('color', 'white')
                line_width = velocity_vectors.get('line_width', 1.0)
                
                # Get spacing - use original_spacing for sampling to match GUI density
                sample_spacing = int(velocity_vectors.get('original_spacing', 30))
                sample_spacing = max(1, sample_spacing)
                
                # Determine coordinate offset for proper positioning
                if 'roi_offset' in velocity_vectors:
                    x_offset, y_offset = velocity_vectors['roi_offset']
                elif roi_rect is not None:
                    x_offset, y_offset = roi_rect[0], roi_rect[1]
                else:
                    x_offset, y_offset = 0, 0
                
                # DEBUG: Log positioning info
                _log(f"[Quiver Debug] du.shape={du.shape}, x_offset={x_offset}, y_offset={y_offset}")
                _log(f"[Quiver Debug] roi_rect={roi_rect}, extent={extent}")
                
                # Create coordinate grids and sample data using same spacing
                y_idx = np.arange(0, vh, sample_spacing)
                x_idx = np.arange(0, vw, sample_spacing)
                
                # Coordinates in image space
                X, Y = np.meshgrid(x_idx + x_offset, y_idx + y_offset)
                
                # Sample velocity data
                U = du[np.ix_(y_idx.astype(int), x_idx.astype(int))]
                V = dv[np.ix_(y_idx.astype(int), x_idx.astype(int))]
                
                # Handle NaN values
                valid = ~(np.isnan(U) | np.isnan(V))
                
                # Quiver plot
                if velocity_vectors.get('show_quiver', False) and np.any(valid):
                    # Normalize U, V to unit vectors (uniform arrow length)
                    # Color already indicates magnitude, so arrow length should be constant
                    magnitude = np.sqrt(U**2 + V**2)
                    magnitude[magnitude == 0] = 1  # Avoid division by zero
                    U_norm = U / magnitude
                    V_norm = V / magnitude
                    
                    # Use original_scale (raw user value) to match GUI appearance exactly
                    # GUI uses: quiver_scale = 1.0 / (scale + 0.001) where scale is the raw slider value
                    # We must use the same raw value, not the compensated one
                    original_scale = velocity_vectors.get('original_scale', scale)
                    original_line_width = velocity_vectors.get('original_line_width', line_width)
                    quiver_scale = 1.0 / (original_scale + 0.001)
                    quiver_width = 0.003 * original_line_width
                    
                    # DEBUG: Log quiver parameters for comparison with GUI
                    _log(f"[Export Quiver] original_scale={original_scale}, quiver_scale={quiver_scale:.6f}, quiver_width={quiver_width:.6f}, original_line_width={original_line_width}")
                    _log(f"[Export Quiver] data shape: vh={vh}, vw={vw}, sample_spacing={sample_spacing}, arrow_count={len(X[valid])}")
                    
                    ax.quiver(X[valid], Y[valid], U_norm[valid], V_norm[valid],
                             color=color, scale=quiver_scale, scale_units='xy',
                             width=quiver_width, headwidth=4, headlength=5, alpha=0.9)
                
                # Streamlines
                if velocity_vectors.get('show_streamlines', False):
                    try:
                        x_stream = np.arange(vw) + x_offset
                        y_stream = np.arange(vh) + y_offset
                        du_filled = np.nan_to_num(du, nan=0.0)
                        dv_filled = -np.nan_to_num(dv, nan=0.0)  # Negate for image coords
                        
                        # --- DENSITY COMPENSATION ---
                        # To ensure visual consistency (lines per image width) between modes
                        # (Ref: small ROI, Def: large Full Image), we scale density based on 
                        # the ratio of [Data Domain Width] to [Original Spacing].
                        # A baseline Preview window is approx 500px.
                        # Formula: density = (Data_Width_px / Original_Spacing_px) / 30.0
                        
                        original_spacing = velocity_vectors.get('original_spacing', 30)
                        # vw is the width of the data domain in pixels (whether ROI or Full Image)
                        domain_width = vw 
                        
                        # Calculate required density to match visual spacing
                        density = (domain_width / original_spacing) / 30.0
                        density = max(0.5, density)

                        # Scale arrowsize relative to image dimensions
                        base_arrowsize = max(vh, vw) / 500.0
                        arrowsize = base_arrowsize * (scale / 50.0)
                        arrowsize = max(0.5, min(arrowsize, 5.0))
                        
                        ax.streamplot(x_stream, y_stream, du_filled, dv_filled,
                                     color=color, density=density, linewidth=line_width,
                                     arrowsize=arrowsize, arrowstyle='->')
                    except Exception as e:
                        _log(f"Streamplot error: {e}")
        
        # Add colorbar
        if include_colorbar:
            cbs = colorbar_settings or {}
            # bar_thickness controls width: 1.0=default, >1=fatter, <1=thinner
            bt = cbs.get("bar_thickness", 1.0)
            cbar = fig.colorbar(
                im, ax=ax,
                shrink=0.8,
                pad=cbs.get("pad", 0.02),
                aspect=max(5, 20 / bt),
            )
            if use_log_norm:
                cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())

            cb_fontsize = cbs.get("font_size", 10)
            cbar.ax.tick_params(labelsize=cb_fontsize)

            # Number of ticks
            num_ticks = cbs.get("num_ticks")
            if num_ticks and not use_log_norm:
                from matplotlib.ticker import MaxNLocator
                cbar.locator = MaxNLocator(nbins=num_ticks)
                cbar.update_ticks()

            # Outline
            if cbs.get("hide_outline", False):
                cbar.outline.set_visible(False)

            # label_text override from colorbar settings
            display_label = cbs.get("label_text") or colorbar_label
            if display_label:
                label_fontsize = cbs.get("label_font_size", cb_fontsize)
                cbar.set_label(display_label, fontsize=label_fontsize)

        # Add title
        if include_title and title:
            ax.set_title(title)
        
        # Clean up axes
        ax.axis('off')
        
        # Adjust layout
        plt.tight_layout(pad=0.5)
        
        # Ensure axis limits cover the full image (prevents cropping to data extent only)
        if background_image is not None:
            bg_h, bg_w = background_image.shape[:2]
            ax.set_xlim(0, bg_w)
            ax.set_ylim(bg_h, 0)  # Inverted for image coordinates
        
        # Adjust layout to minimize whitespace
        plt.tight_layout(pad=0.5)
        
        # Save with tight bounding box to include colorbar/title
        fig.savefig(output_path, format=format, 
                   bbox_inches='tight',
                   pad_inches=0.1,
                   facecolor='white' if format in ['jpg', 'jpeg'] else 'auto',
                   dpi=dpi)
        plt.close(fig)
        
        return True
        
    except Exception as e:
        _log(f"ERROR rendering frame: {e}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        return False


def export_batch_images(
    output_dir: str,
    components: Dict[str, Dict],  # {'u': {'vmin': -100, 'vmax': 100}, ...}
    frame_range: Tuple[int, int],  # (start, end) 1-indexed inclusive
    displacement_results: List,
    strain_results: List,
    roi_rect: Tuple[int, int, int, int],
    roi_mask: np.ndarray,
    image_loader: Callable[[int], np.ndarray],  # fn(frame_idx) -> image
    settings: Dict,  # colormap, alpha, display_mode, format, include_colorbar, include_title
    deformed_view_cache,  # For deformed mode warp
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_event=None
) -> str:
    """
    Export visualization images for selected components and frames.
    
    Args:
        output_dir: Base output directory (will create timestamped subfolder)
        components: Dict of component settings {name: {'vmin': float, 'vmax': float}}
        frame_range: (start, end) 1-indexed inclusive
        displacement_results: List of displacement arrays or paths
        strain_results: List of strain dicts
        roi_rect: (xmin, ymin, xmax, ymax)
        roi_mask: ROI mask array
        image_loader: Function to load background image by frame index
        settings: General export settings
        deformed_view_cache: Cache for deformed mode operations
        progress_callback: Optional callback(current, total, message)
        
    Returns:
        Path to the export directory
    """
    start_time = time.time()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = os.path.join(output_dir, f"export_{timestamp}")
    os.makedirs(export_dir, exist_ok=True)
    
    _log(f"Export started: {export_dir}")
    _log(f"Settings: {settings}")
    _log(f"Components: {list(components.keys())}")
    _log(f"Frame range: {frame_range[0]} to {frame_range[1]}")
    
    # Extract settings
    colormap = settings.get('colormap', 'turbo')
    alpha = settings.get('alpha', 0.7)
    display_mode = settings.get('display_mode', 'reference')
    format = settings.get('format', 'png')
    include_colorbar = settings.get('include_colorbar', True)
    include_title = settings.get('include_title', False)
    
    # Physical unit settings
    use_physical_units = settings.get('use_physical_units', False)
    physical_ratio = settings.get('physical_ratio', 1.0)
    physical_unit = settings.get('physical_unit', 'mm')
    fps = settings.get('fps', 1.0)
    
    # Arrow settings
    show_quiver = settings.get('show_quiver', False)
    show_streamlines = settings.get('show_streamlines', False)
    arrow_spacing = settings.get('arrow_spacing', 100)
    arrow_scale = settings.get('arrow_scale', 100)
    arrow_color = settings.get('arrow_color', 'white')
    arrow_width = settings.get('arrow_width', 1.0)
    
    # Log scale
    use_log_scale = settings.get('use_log_scale', False)

    # DPI setting
    export_dpi = settings.get('dpi', 100)

    # Colorbar customization
    colorbar_settings = settings.get('colorbar_settings')
    
    # Calculate total operations for progress
    enabled_components = [name for name, cfg in components.items() if cfg.get('enabled', True)]
    frame_start, frame_end = frame_range
    total_frames = frame_end - frame_start + 1
    total_operations = len(enabled_components) * total_frames
    current_op = 0
    saved_count = 0
    
    _log(f"Total operations: {total_operations} ({len(enabled_components)} components x {total_frames} frames)")
    _log(f"Physical units: {use_physical_units}, ratio={physical_ratio}, unit={physical_unit}, fps={fps}")
    _log(f"Arrows: quiver={show_quiver}, streamlines={show_streamlines}, width={arrow_width}")
    
    # Process each component
    for comp_name in enabled_components:
        comp_config = components[comp_name]
        vmin = comp_config.get('vmin')
        vmax = comp_config.get('vmax')
        
        _log(f"Processing component: {comp_name}")
        _log(f"  Color range: vmin={vmin}, vmax={vmax}")
        
        # Create component subdirectory
        comp_dir = os.path.join(export_dir, comp_name)
        os.makedirs(comp_dir, exist_ok=True)
        
        # Process each frame
        for frame_idx in range(frame_start - 1, frame_end):  # Convert to 0-indexed
            current_op += 1

            # Check for cancellation
            if cancel_event and cancel_event.is_set():
                _log("Export cancelled by user")
                if progress_callback:
                    progress_callback(current_op, total_operations, "Export cancelled")
                return export_dir

            if progress_callback:
                progress_callback(current_op, total_operations, 
                                f"Exporting {comp_name} frame {frame_idx + 1}...")
            
            try:
                # Get data for this component and frame
                data = _get_component_data(comp_name, frame_idx, 
                                          displacement_results, strain_results, fps=fps)
                if data is None:
                    _log(f"  Frame {frame_idx + 1}: No data available, skipping")
                    continue
                
                _log(f"  Frame {frame_idx + 1}: data shape = {data.shape}")
                
                # Get background image
                if display_mode == 'deformed':
                    bg_frame_idx = frame_idx + 1  # Deformed uses next frame
                    bg_img = image_loader(bg_frame_idx)
                else:
                    bg_img = image_loader(0)  # Reference frame
                
                # Handle deformed mode warping
                display_data = data
                display_roi = roi_rect
                
                if display_mode == 'deformed' and deformed_view_cache is not None:
                    # Need to warp data to deformed coordinates
                    display_data, display_roi = _warp_for_deformed(
                        data, comp_name, frame_idx,
                        displacement_results, roi_rect, 
                        bg_img.shape[:2] if bg_img is not None else None,
                        deformed_view_cache
                    )
                
                # Apply physical units scaling
                if use_physical_units and comp_name in ['u', 'v', 'magnitude', 'velocity']:
                    display_data = display_data * physical_ratio
                
                # Determine colorbar label
                if comp_name in ['u', 'v', 'magnitude']:
                    if use_physical_units:
                        colorbar_label = f"{comp_name} [{physical_unit}]"
                    else:
                        colorbar_label = f"{comp_name} [px]"
                elif comp_name == 'velocity':
                    if use_physical_units:
                        colorbar_label = f"velocity [{physical_unit}/s]" if fps != 1.0 else f"velocity [{physical_unit}/frame]"
                    else:
                        colorbar_label = "velocity [px/s]" if fps != 1.0 else "velocity [px/frame]"
                else:
                    colorbar_label = comp_name
                
                # Prepare velocity vectors for arrows (only for velocity component)
                velocity_vectors = None
                if comp_name == 'velocity' and (show_quiver or show_streamlines):
                    # Always use original roi_rect for velocity vector positioning (not display_roi)
                    # This ensures arrows are at correct location even in deformed mode
                    velocity_vectors = _get_velocity_vectors(
                        frame_idx, displacement_results, fps,
                        roi_rect,  # Always use original roi_rect, not display_roi
                        physical_ratio if use_physical_units else 1.0
                    )
                    if velocity_vectors is not None:
                        # In deformed mode, velocity data needs to be warped too
                        if display_mode == 'deformed' and deformed_view_cache is not None:
                            try:
                                du_warped, _ = _warp_for_deformed(
                                    velocity_vectors['du'], 'velocity_du', frame_idx,
                                    displacement_results, roi_rect,
                                    bg_img.shape[:2] if bg_img is not None else None,
                                    deformed_view_cache
                                )
                                dv_warped, _ = _warp_for_deformed(
                                    velocity_vectors['dv'], 'velocity_dv', frame_idx,
                                    displacement_results, roi_rect,
                                    bg_img.shape[:2] if bg_img is not None else None,
                                    deformed_view_cache
                                )
                                velocity_vectors['du'] = du_warped
                                velocity_vectors['dv'] = dv_warped
                                # In deformed mode, arrows should have no offset (full image coords)
                                velocity_vectors['roi_offset'] = (0, 0)
                            except Exception as e:
                                _log(f"  Error warping velocity vectors: {e}")
                                velocity_vectors['roi_offset'] = (roi_rect[0], roi_rect[1]) if roi_rect else (0, 0)
                        else:
                            # Reference mode: use roi_rect offset
                            velocity_vectors['roi_offset'] = (roi_rect[0], roi_rect[1]) if roi_rect else (0, 0)
                        
                        # Calculate size compensation for export
                        # Preview figure has figsize=(5,4) inches @ 100 DPI = 500x400 pixels
                        # Export renders at full image resolution (e.g., 5472x3648)
                        # Scale all pixel-based parameters proportionally
                        preview_fig_width = 500  # Approximate preview figure width in pixels
                        export_size = max(bg_img.shape[:2]) if bg_img is not None else 1000
                        size_compensation = export_size / preview_fig_width
                        
                        # Apply compensation:
                        # - spacing: needs to scale up proportionally (larger image = more spacing)
                        # - scale: needs to scale up (larger image = larger arrows)
                        # - line_width: needs to scale up (larger image = thicker lines)
                        compensated_spacing = int(arrow_spacing * size_compensation)
                        compensated_scale = arrow_scale * size_compensation
                        compensated_line_width = arrow_width * size_compensation
                        
                        _log(f"  Size compensation: {size_compensation:.2f}x (preview={preview_fig_width}, export={export_size})")
                        _log(f"  Compensated: spacing={compensated_spacing}, scale={compensated_scale:.1f}, width={compensated_line_width:.2f}")
                        
                        velocity_vectors.update({
                            'show_quiver': show_quiver,
                            'show_streamlines': show_streamlines,
                            'spacing': compensated_spacing,  # For quiver grid sampling
                            'original_spacing': arrow_spacing,  # For streamline density calculation
                            'size_compensation': size_compensation,  # For density scaling
                            'scale': compensated_scale,  # Compensated scale (for streamlines)
                            'original_scale': arrow_scale,  # Original scale (for quiver)
                            'color': arrow_color,
                            'line_width': compensated_line_width,  # Compensated (for streamlines)
                            'original_line_width': arrow_width  # Original (for quiver)
                        })
                
                # Build output filename
                filename = f"{comp_name}_frame_{frame_idx + 1:03d}.{format}"
                output_path = os.path.join(comp_dir, filename)
                
                # Build title
                title = f"{comp_name.upper()} - Frame {frame_idx + 1}" if include_title else ""
                
                # Render and save
                success = render_single_frame(
                    data=display_data,
                    background_image=bg_img,
                    roi_rect=display_roi,
                    colormap=colormap,
                    alpha=alpha,
                    vmin=vmin,
                    vmax=vmax,
                    include_colorbar=include_colorbar,
                    include_title=include_title,
                    title=title,
                    colorbar_label=colorbar_label,
                    output_path=output_path,
                    format=format,
                    use_log_norm=use_log_scale,
                    velocity_vectors=velocity_vectors,
                    dpi=export_dpi,
                    colorbar_settings=colorbar_settings,
                )
                
                if success:
                    saved_count += 1
                    _log(f"  Saved: {filename}")
                    
            except Exception as e:
                _log(f"  ERROR on frame {frame_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save export settings
    settings_path = os.path.join(export_dir, "export_settings.json")
    _save_export_settings(settings_path, components, frame_range, settings)
    
    elapsed = time.time() - start_time
    _log(f"Export complete: {saved_count} images in {elapsed:.1f}s")
    _log(f"Output directory: {export_dir}")
    
    if progress_callback:
        progress_callback(total_operations, total_operations, "Export complete!")
    
    return export_dir


def _get_component_data(
    comp_name: str,
    frame_idx: int,
    displacement_results: List,
    strain_results: List,
    fps: float = 1.0
) -> Optional[np.ndarray]:
    """Get data array for a specific component and frame."""
    try:
        if comp_name == 'u':
            # Displacement U component
            if frame_idx >= len(displacement_results):
                return None
            d = displacement_results[frame_idx]
            if isinstance(d, str):
                d = np.load(d)
            return d[..., 0]
        
        elif comp_name == 'v':
            # Displacement V component
            if frame_idx >= len(displacement_results):
                return None
            d = displacement_results[frame_idx]
            if isinstance(d, str):
                d = np.load(d)
            return d[..., 1]
        
        elif comp_name == 'magnitude':
            # Displacement magnitude = sqrt(u^2 + v^2)
            if frame_idx >= len(displacement_results):
                return None
            d = displacement_results[frame_idx]
            if isinstance(d, str):
                d = np.load(d)
            return np.sqrt(d[..., 0]**2 + d[..., 1]**2)
        
        elif comp_name == 'velocity':
            # Velocity = |delta(U,V)| * fps
            if frame_idx >= len(displacement_results):
                return None
            d_curr = displacement_results[frame_idx]
            if isinstance(d_curr, str):
                d_curr = np.load(d_curr)
            
            if frame_idx > 0:
                d_prev = displacement_results[frame_idx - 1]
                if isinstance(d_prev, str):
                    d_prev = np.load(d_prev)
                du = d_curr[..., 0] - d_prev[..., 0]
                dv = d_curr[..., 1] - d_prev[..., 1]
                return np.sqrt(du**2 + dv**2) * fps
            else:
                # First frame: velocity = 0
                return np.zeros_like(d_curr[..., 0])
        
        else:
            # Strain component
            if frame_idx >= len(strain_results):
                return None
            s = strain_results[frame_idx]
            if s is None or comp_name not in s:
                return None
            return s[comp_name]
    except Exception as e:
        _log(f"ERROR getting component data: {e}")
        return None


def _get_velocity_vectors(
    frame_idx: int,
    displacement_results: List,
    fps: float,
    roi_rect: Optional[Tuple],
    physical_ratio: float = 1.0
) -> Optional[Dict]:
    """
    Get velocity vector components (du, dv) for arrow visualization.
    
    Returns:
        Dict with 'du', 'dv' arrays, or None if unavailable
    """
    try:
        if frame_idx >= len(displacement_results):
            return None
        
        d_curr = displacement_results[frame_idx]
        if isinstance(d_curr, str):
            d_curr = np.load(d_curr)
        
        if frame_idx > 0:
            d_prev = displacement_results[frame_idx - 1]
            if isinstance(d_prev, str):
                d_prev = np.load(d_prev)
            du = (d_curr[..., 0] - d_prev[..., 0]) * fps * physical_ratio
            dv = (d_curr[..., 1] - d_prev[..., 1]) * fps * physical_ratio
        else:
            # First frame: zero velocity
            du = np.zeros_like(d_curr[..., 0])
            dv = np.zeros_like(d_curr[..., 0])
        
        return {'du': du, 'dv': dv}
        
    except Exception as e:
        _log(f"ERROR getting velocity vectors: {e}")
        return None


def _warp_for_deformed(
    data: np.ndarray,
    comp_name: str,
    frame_idx: int,
    displacement_results: List,
    roi_rect: Tuple,
    output_shape: Tuple,
    deformed_view_cache
) -> Tuple[np.ndarray, Optional[Tuple]]:
    """
    Warp data to deformed coordinates if needed.
    
    Returns:
        (warped_data, display_roi) - roi is None for full-image warped data
    """
    try:
        if frame_idx >= len(displacement_results):
            return data, roi_rect
        
        # Get displacement for this frame
        disp = displacement_results[frame_idx]
        if isinstance(disp, str):
            disp = np.load(disp)
        u_crop = disp[..., 0]
        v_crop = disp[..., 1]
        
        # Check if data shape matches displacement
        data_to_warp = data
        if data.shape != u_crop.shape:
            # Upsample (strain may be downsampled)
            target_h, target_w = u_crop.shape
            _log(f"  Upsampling {comp_name}: {data.shape} -> ({target_h}, {target_w})")
            
            nan_mask = np.isnan(data)
            data_filled = np.nan_to_num(data, nan=0.0)
            data_upsampled = cv2.resize(data_filled.astype(np.float32), 
                                       (target_w, target_h),
                                       interpolation=cv2.INTER_LINEAR)
            nan_mask_upsampled = cv2.resize(nan_mask.astype(np.uint8), 
                                           (target_w, target_h),
                                           interpolation=cv2.INTER_NEAREST) > 0
            data_upsampled[nan_mask_upsampled] = np.nan
            data_to_warp = data_upsampled
        
        # Warp using cache
        if output_shape is None:
            output_shape = (3648, 5472)  # Fallback
        
        _log(f"  Warping to deformed coordinates...")
        warped = deformed_view_cache.warp_data_forward(
            data_to_warp, u_crop, v_crop, roi_rect, output_shape
        )
        
        return warped, None  # None roi means full image
        
    except Exception as e:
        _log(f"  ERROR in warp: {e}")
        return data, roi_rect


def _save_export_settings(
    path: str,
    components: Dict,
    frame_range: Tuple,
    settings: Dict
):
    """Save export settings to JSON for reproducibility."""
    export_info = {
        'timestamp': datetime.now().isoformat(),
        'frame_range': list(frame_range),
        'components': components,
        'settings': settings
    }
    
    try:
        with open(path, 'w') as f:
            json.dump(export_info, f, indent=2)
        _log(f"Settings saved: {path}")
    except Exception as e:
        _log(f"ERROR saving settings: {e}")
