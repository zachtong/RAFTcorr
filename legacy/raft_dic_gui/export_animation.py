"""
Animation export (GIF / MP4) for RAFT-DIC GUI.

Renders each frame via matplotlib, then encodes to GIF (Pillow) or MP4 (OpenCV).
Memory-efficient: frames are written incrementally, not all held in RAM.
"""

import io
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from PIL import Image as PILImage

from raft_dic_gui.export_images import _get_component_data, _get_velocity_vectors

logger = logging.getLogger(__name__)

# Human-readable colorbar labels for each component
COMPONENT_LABELS = {
    "u": "U displacement [px]",
    "v": "V displacement [px]",
    "magnitude": "Magnitude [px]",
    "velocity": "Velocity [px/frame]",
    "exx": "εxx",
    "eyy": "εyy",
    "exy": "εxy",
    "e1": "ε1",
    "e2": "ε2",
    "max_shear": "Max Shear",
    "von_mises": "Von Mises",
    "rotation": "Rotation [rad]",
    "dexx_dt": "dεxx/dt",
    "deyy_dt": "dεyy/dt",
    "dexy_dt": "dεxy/dt",
}


def export_animation(
    output_path: str,
    fmt: str,  # "gif" or "mp4"
    component: str,
    frame_range: Tuple[int, int],  # (start, end) 0-indexed inclusive
    displacement_results: List,
    strain_results: List,
    roi_rect: Optional[Tuple[int, int, int, int]],
    roi_mask: Optional[np.ndarray],
    image_loader: Optional[Callable[[int], Optional[np.ndarray]]],
    settings: Dict,
    deformed_view_cache=None,
    fps: int = 10,
    loop: bool = True,
    quality: int = 85,
    timestamp_overlay: bool = False,
    include_colorbar: bool = True,
    resize_factor: float = 1.0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    cancel_event=None,
    per_frame_rois: Optional[Dict[int, np.ndarray]] = None,
    inverse_map_cache=None,
) -> str:
    """
    Export an animation of a single component across frames.

    Args:
        output_path: Destination file path
        fmt: "gif" or "mp4"
        component: e.g. "u", "v", "magnitude", "exx"
        frame_range: (start_frame, end_frame) 0-indexed inclusive
        displacement_results: session.displacement_results
        strain_results: session.strain_results
        roi_rect: (xmin, ymin, xmax, ymax)
        roi_mask: boolean mask
        image_loader: callable(frame_idx) -> BGR image or None
        settings: dict with colormap, alpha, background, vmin, vmax, etc.
        deformed_view_cache: for deformed background mode
        fps: frames per second
        loop: whether GIF loops
        quality: JPEG quality for MP4 (not used currently)
        timestamp_overlay: add frame index text
        include_colorbar: add colorbar
        resize_factor: scale output (0.25-1.0)
        progress_callback: fn(current, total, message)
        cancel_event: threading.Event to signal cancellation

    Returns:
        Absolute path of saved file.
    """
    start, end = frame_range
    total_frames = end - start + 1
    abs_path = os.path.abspath(output_path)

    # Extract settings
    colormap = settings.get("colormap", "turbo")
    alpha = settings.get("alpha", 0.7)
    background = settings.get("background", "reference")
    vmin = settings.get("vmin")
    vmax = settings.get("vmax")
    use_log_norm = settings.get("log_scale", False)
    physical_ratio = settings.get("physical_ratio", 1.0)
    data_fps = settings.get("fps", 1.0)

    # Colorbar customization
    colorbar_settings = settings.get("colorbar_settings")
    colorbar_label = COMPONENT_LABELS.get(component, component)

    # Pre-compute global vmin/vmax if not specified (scan all frames)
    if vmin is None or vmax is None:
        global_min, global_max = _compute_global_range(
            component, start, end, displacement_results, strain_results, data_fps
        )
        if vmin is None:
            vmin = global_min
        if vmax is None:
            vmax = global_max

    # Share one inverse-map cache across frames to avoid recomputing the
    # Delaunay triangulation each frame in deformed mode.
    if inverse_map_cache is None:
        from server.deformed_warp import InverseMapCache
        inverse_map_cache = InverseMapCache(max_size=10)

    if fmt == "mp4":
        _export_mp4(
            abs_path, component, start, end,
            displacement_results, strain_results,
            roi_rect, image_loader, colormap, alpha, background,
            vmin, vmax, use_log_norm, physical_ratio, data_fps,
            deformed_view_cache,
            fps, timestamp_overlay, include_colorbar, resize_factor,
            colorbar_label, colorbar_settings,
            progress_callback, cancel_event,
            per_frame_rois, inverse_map_cache,
        )
    else:
        _export_gif(
            abs_path, component, start, end,
            displacement_results, strain_results,
            roi_rect, image_loader, colormap, alpha, background,
            vmin, vmax, use_log_norm, physical_ratio, data_fps,
            deformed_view_cache,
            fps, loop, timestamp_overlay, include_colorbar, resize_factor,
            colorbar_label, colorbar_settings,
            progress_callback, cancel_event,
            per_frame_rois, inverse_map_cache,
        )

    logger.info("Animation saved to %s", abs_path)
    return abs_path


# ---------------------------------------------------------------------------
# Internal rendering
# ---------------------------------------------------------------------------

def _render_frame_to_array(
    component: str,
    frame_idx: int,
    displacement_results: List,
    strain_results: List,
    roi_rect: Optional[Tuple],
    bg_image: Optional[np.ndarray],
    colormap: str,
    alpha: float,
    vmin: float,
    vmax: float,
    use_log_norm: bool,
    physical_ratio: float,
    data_fps: float,
    include_colorbar: bool,
    timestamp_overlay: bool,
    resize_factor: float,
    colorbar_label: str = "",
    colorbar_settings: Optional[Dict] = None,
    dpi: int = 80,
    background: str = "reference",
    per_frame_rois: Optional[Dict[int, np.ndarray]] = None,
    inverse_map_cache=None,
) -> Optional[np.ndarray]:
    """Render one frame to an RGB uint8 numpy array."""
    data = _get_component_data(component, frame_idx, displacement_results, strain_results, data_fps)
    if data is None:
        return None

    # Apply physical ratio
    if physical_ratio != 1.0:
        data = data * physical_ratio

    # Determine figure size from background or data
    if bg_image is not None:
        h, w = bg_image.shape[:2]
    else:
        h, w = data.shape

    # Deformed-mode warp: project ref-space data into deformed coords so the
    # overlay ROI follows per-frame masks (matches the interactive /render
    # endpoint).  Without this step, ``data`` is placed at a fixed roi_rect
    # extent and visually stays on the frame-0 ROI regardless of background.
    warped_full = False
    if (
        background == "deformed"
        and bg_image is not None
        and roi_rect is not None
        and frame_idx < len(displacement_results)
    ):
        try:
            from server.deformed_warp import get_warped_full_data
            disp = displacement_results[frame_idx]
            if isinstance(disp, str):
                disp = np.load(disp)
            U = disp[..., 0]
            V = disp[..., 1]
            x0, y0, x1, y1 = roi_rect
            roi_h_val, roi_w_val = y1 - y0, x1 - x0
            sh, sw = data.shape
            deformed_mask = None
            if per_frame_rois is not None:
                deformed_mask = per_frame_rois.get(frame_idx + 1)
            data = get_warped_full_data(
                data=data,
                frame_idx=frame_idx,
                U=U,
                V=V,
                roi_rect=roi_rect,
                image_shape=(h, w),
                cache=inverse_map_cache,
                needs_upsample=(sh != roi_h_val or sw != roi_w_val),
                roi_h=roi_h_val,
                roi_w=roi_w_val,
                deformed_mask=deformed_mask,
            )
            warped_full = True
        except Exception as exc:
            logger.warning("Deformed-mode warp failed at frame %d: %s", frame_idx, exc)

    # Apply resize
    fig_w = w * resize_factor / dpi
    fig_h = h * resize_factor / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    # Background
    if bg_image is not None:
        if resize_factor < 1.0:
            rh, rw = int(h * resize_factor), int(w * resize_factor)
            bg_resized = cv2.resize(bg_image, (rw, rh), interpolation=cv2.INTER_AREA)
            ax.imshow(bg_resized)
        else:
            ax.imshow(bg_image)

    # Data overlay
    masked_data = np.ma.array(data, mask=np.isnan(data))

    # Extent for positioning data on background.  After deformed warp the
    # data is already placed in full-image coordinates, so no extent offset
    # is needed.  In reference mode (or when warp was skipped) the data is
    # ROI-sized and must be anchored to roi_rect.
    extent = None
    if not warped_full and roi_rect is not None and bg_image is not None:
        xmin, ymin, xmax, ymax = roi_rect
        if resize_factor < 1.0:
            extent = [xmin * resize_factor, xmax * resize_factor,
                      ymax * resize_factor, ymin * resize_factor]
        else:
            extent = [xmin, xmax, ymax, ymin]
    elif warped_full and resize_factor < 1.0:
        extent = [0, int(w * resize_factor),
                  int(h * resize_factor), 0]

    # Resize data if needed
    display_data = masked_data
    if resize_factor < 1.0 and bg_image is not None:
        dh, dw = data.shape
        rdh, rdw = max(1, int(dh * resize_factor)), max(1, int(dw * resize_factor))
        resized = cv2.resize(np.nan_to_num(data, nan=0.0).astype(np.float32), (rdw, rdh), interpolation=cv2.INTER_AREA)
        nan_mask = cv2.resize(np.isnan(data).astype(np.uint8), (rdw, rdh), interpolation=cv2.INTER_NEAREST) > 0
        resized[nan_mask] = np.nan
        display_data = np.ma.array(resized, mask=np.isnan(resized))

    # Normalization
    norm = None
    plot_vmin, plot_vmax = vmin, vmax
    cbs = colorbar_settings or {}
    discrete_levels = cbs.get("discrete_levels", 0)

    if use_log_norm:
        log_vmin = max(vmin, 1e-10) if vmin > 0 else 1e-10
        log_vmax = max(vmax, 1.0) if vmax > 0 else 1.0
        if log_vmin >= log_vmax:
            log_vmin = log_vmax / 1000
        norm = LogNorm(vmin=log_vmin, vmax=log_vmax)
        plot_vmin, plot_vmax = None, None
    elif discrete_levels and discrete_levels > 0 and vmin is not None and vmax is not None:
        from matplotlib.colors import BoundaryNorm
        boundaries = np.linspace(vmin, vmax, discrete_levels + 1)
        norm = BoundaryNorm(boundaries, discrete_levels)
        plot_vmin, plot_vmax = None, None

    if discrete_levels and discrete_levels > 0 and not use_log_norm:
        cmap = plt.get_cmap(colormap, discrete_levels)
    else:
        cmap = plt.get_cmap(colormap)
    cmap.set_bad(alpha=0)
    im = ax.imshow(display_data, cmap=cmap, alpha=alpha,
                   vmin=plot_vmin, vmax=plot_vmax, extent=extent, norm=norm)

    if include_colorbar:
        # label_text override from colorbar settings
        display_label = cbs.get("label_text") or colorbar_label
        # bar_thickness controls width: 1.0=default, >1=fatter, <1=thinner
        bt = cbs.get("bar_thickness", 1.0)
        cbar = fig.colorbar(
            im, ax=ax,
            shrink=0.8,
            pad=cbs.get("pad", 0.02),
            aspect=max(5, 20 / bt),
        )
        cb_fontsize = cbs.get("font_size", 10)
        cbar.ax.tick_params(labelsize=cb_fontsize)

        num_ticks = cbs.get("num_ticks")
        if num_ticks and not use_log_norm:
            from matplotlib.ticker import MaxNLocator
            cbar.locator = MaxNLocator(nbins=num_ticks)
            cbar.update_ticks()

        if cbs.get("hide_outline", False):
            cbar.outline.set_visible(False)

        if display_label:
            label_fontsize = cbs.get("label_font_size", cb_fontsize)
            cbar.set_label(display_label, fontsize=label_fontsize)

    if timestamp_overlay:
        label = f"Frame {frame_idx + 1}"
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                fontsize=10, color="white", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6))

    ax.axis("off")

    if bg_image is not None:
        if resize_factor < 1.0:
            ax.set_xlim(0, int(w * resize_factor))
            ax.set_ylim(int(h * resize_factor), 0)
        else:
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

    plt.tight_layout(pad=0.3)

    # Render to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)
    rgb = arr[:, :, :3].copy()  # Drop alpha channel
    plt.close(fig)

    return rgb


def _compute_global_range(
    component: str, start: int, end: int,
    displacement_results: List, strain_results: List,
    fps: float,
) -> Tuple[float, float]:
    """Scan frames to find global min/max for consistent color range."""
    global_min = np.inf
    global_max = -np.inf
    for i in range(start, end + 1):
        data = _get_component_data(component, i, displacement_results, strain_results, fps)
        if data is not None:
            fmin = np.nanmin(data)
            fmax = np.nanmax(data)
            if fmin < global_min:
                global_min = fmin
            if fmax > global_max:
                global_max = fmax
    if np.isinf(global_min):
        global_min, global_max = 0.0, 1.0
    return float(global_min), float(global_max)


def _get_background_image(
    frame_idx: int, background: str,
    image_loader: Optional[Callable], roi_rect: Optional[Tuple],
) -> Optional[np.ndarray]:
    """Get background image in RGB format."""
    if image_loader is None:
        return None
    if background == "reference":
        img = image_loader(0)
    else:
        img = image_loader(frame_idx)
    if img is None:
        return None
    # Convert BGR to RGB for matplotlib
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ---------------------------------------------------------------------------
# GIF writer
# ---------------------------------------------------------------------------

def _export_gif(
    path, component, start, end,
    displacement_results, strain_results,
    roi_rect, image_loader, colormap, alpha, background,
    vmin, vmax, use_log_norm, physical_ratio, data_fps,
    deformed_view_cache,
    fps, loop, timestamp_overlay, include_colorbar, resize_factor,
    colorbar_label, colorbar_settings,
    progress_callback, cancel_event,
    per_frame_rois=None, inverse_map_cache=None,
):
    total = end - start + 1
    duration_ms = int(1000 / fps)
    pil_frames = []

    for i, frame_idx in enumerate(range(start, end + 1)):
        if cancel_event and cancel_event.is_set():
            logger.info("Animation export cancelled at frame %d", frame_idx)
            return

        # Deformed frame N on screen corresponds to image_files[frame_idx + 1]
        # (frame 0 is the reference image).  Use the same offset as the
        # interactive /render endpoint so the background matches the warped
        # data.
        bg_idx = (frame_idx + 1) if background == "deformed" else frame_idx
        bg = _get_background_image(bg_idx, background, image_loader, roi_rect)
        rgb = _render_frame_to_array(
            component, frame_idx,
            displacement_results, strain_results,
            roi_rect, bg, colormap, alpha,
            vmin, vmax, use_log_norm, physical_ratio, data_fps,
            include_colorbar, timestamp_overlay, resize_factor,
            colorbar_label, colorbar_settings,
            background=background,
            per_frame_rois=per_frame_rois,
            inverse_map_cache=inverse_map_cache,
        )
        if rgb is not None:
            pil_frames.append(PILImage.fromarray(rgb))

        if progress_callback:
            progress_callback(i + 1, total, f"Rendering frame {i + 1}/{total}")

    if not pil_frames:
        raise ValueError("No frames rendered")

    # Save GIF
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0 if loop else 1,
        optimize=True,
    )


# ---------------------------------------------------------------------------
# MP4 writer
# ---------------------------------------------------------------------------

def _export_mp4(
    path, component, start, end,
    displacement_results, strain_results,
    roi_rect, image_loader, colormap, alpha, background,
    vmin, vmax, use_log_norm, physical_ratio, data_fps,
    deformed_view_cache,
    fps, timestamp_overlay, include_colorbar, resize_factor,
    colorbar_label, colorbar_settings,
    progress_callback, cancel_event,
    per_frame_rois=None, inverse_map_cache=None,
):
    total = end - start + 1
    writer = None

    try:
        for i, frame_idx in enumerate(range(start, end + 1)):
            if cancel_event and cancel_event.is_set():
                logger.info("Animation export cancelled at frame %d", frame_idx)
                return

            bg_idx = (frame_idx + 1) if background == "deformed" else frame_idx
            bg = _get_background_image(bg_idx, background, image_loader, roi_rect)
            rgb = _render_frame_to_array(
                component, frame_idx,
                displacement_results, strain_results,
                roi_rect, bg, colormap, alpha,
                vmin, vmax, use_log_norm, physical_ratio, data_fps,
                include_colorbar, timestamp_overlay, resize_factor,
                colorbar_label, colorbar_settings,
                background=background,
                per_frame_rois=per_frame_rois,
                inverse_map_cache=inverse_map_cache,
            )
            if rgb is None:
                continue

            # Initialize writer on first frame (need dimensions)
            if writer is None:
                h, w = rgb.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {path}")

            # Convert RGB to BGR for OpenCV
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

            if progress_callback:
                progress_callback(i + 1, total, f"Encoding frame {i + 1}/{total}")
    finally:
        if writer is not None:
            writer.release()
