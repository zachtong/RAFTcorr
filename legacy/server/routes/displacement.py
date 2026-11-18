"""Displacement frame data and rendered visualization endpoints."""

import io
import os

import numpy as np
from flask import Blueprint, jsonify, request

from raft_dic_gui.processing import load_and_convert_image
from raft_dic_gui.velocity import (
    calculate_displacement_magnitude,
    calculate_velocity_central,
    calculate_velocity_field,
)
from server.render_cache import RenderCache, auto_cache_size
from server.render_utils import (
    fill_mask_nan_gaps,
    get_colormap_lut,
    render_composited_png,
    render_data_texture_png,
    render_overlay_png,
)
from server.serializers import data_texture_response, frame_data_to_json, png_response
from server.session import session
from server.validation import validate_range

displacement_bp = Blueprint("displacement", __name__)
_render_cache = RenderCache(auto_cache_size())


def _active_rect():
    """Return envelope_rect if available, else roi_rect."""
    return session.envelope_rect or session.roi_rect


def _get_displacement_component(
    frame_idx: int, component: str, ref_frame: int = 0
) -> np.ndarray:
    """Extract a displacement component for a given frame.

    When *ref_frame* > 0 the returned data is the relative displacement
    ``disp[frame_idx] - disp[ref_frame]``.  The subtraction is applied to the
    raw (H, W, 2) array **before** extracting the requested component so that
    magnitude / velocity calculations see the re-referenced field.
    """
    disp = session.displacement_results[frame_idx]  # shape (H, W, 2)

    # Subtract reference frame displacement when requested
    if ref_frame > 0 and ref_frame < len(session.displacement_results):
        ref_disp = session.displacement_results[ref_frame]
        disp = disp - ref_disp  # creates a copy — does NOT mutate session data

    if component == "u":
        return disp[:, :, 0]
    elif component == "v":
        return disp[:, :, 1]
    elif component == "magnitude":
        return calculate_displacement_magnitude(disp[:, :, 0], disp[:, :, 1])
    elif component == "velocity":
        frames_u = [d[:, :, 0] for d in session.displacement_results]
        frames_v = [d[:, :, 1] for d in session.displacement_results]
        return calculate_velocity_central(frames_u, frames_v, frame_idx)
    else:
        return disp[:, :, 0]


@displacement_bp.route("/info", methods=["GET"])
def get_info():
    """Return displacement result metadata."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 404

    sample = session.displacement_results[0]
    return jsonify({
        "num_frames": len(session.displacement_results),
        "roi_rect": session.roi_rect,
        "roi_shape": list(sample.shape[:2]),
    })


@displacement_bp.route("/frame/<int:idx>", methods=["GET"])
def get_frame_data(idx: int):
    """Return raw displacement data for a frame as JSON."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 404

    if idx < 0 or idx >= len(session.displacement_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "u")
    ref_frame = request.args.get("ref_frame", 0, type=int)
    data = _get_displacement_component(idx, component, ref_frame=ref_frame)

    vmin = request.args.get("vmin", type=float)
    vmax = request.args.get("vmax", type=float)

    return jsonify(frame_data_to_json(data, vmin=vmin, vmax=vmax))


@displacement_bp.route("/range/<int:idx>", methods=["GET"])
def get_range(idx: int):
    """Return auto vmin/vmax for a displacement component (lightweight)."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 404
    if idx < 0 or idx >= len(session.displacement_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "u")
    ref_frame = request.args.get("ref_frame", 0, type=int)
    data = _get_displacement_component(idx, component, ref_frame=ref_frame)
    finite = data[np.isfinite(data)]
    vmin = float(finite.min()) if finite.size > 0 else 0.0
    vmax = float(finite.max()) if finite.size > 0 else 1.0
    return jsonify({"vmin": vmin, "vmax": vmax})


@displacement_bp.route("/render/<int:idx>", methods=["GET"])
def render_frame(idx: int):
    """Render a displacement overlay as PNG."""
    # Snapshot to avoid mid-request replacement by another thread
    disp_results = session.displacement_results
    if not disp_results:
        return jsonify({"error": "No displacement results"}), 404

    if idx < 0 or idx >= len(disp_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "u")
    colormap = request.args.get("colormap", "turbo")
    alpha = request.args.get("alpha", 0.7, type=float)
    vmin = request.args.get("vmin", type=float)
    vmax = request.args.get("vmax", type=float)
    background = request.args.get("background", "reference")
    log_scale = request.args.get("log_scale", "false").lower() in ("true", "1", "yes")
    ref_frame = request.args.get("ref_frame", 0, type=int)
    vw = request.args.get("vw", 0, type=int)
    vh = request.args.get("vh", 0, type=int)
    overlay_only = request.args.get("overlay_only", "false").lower() in ("true", "1")

    # Check cache — exclude alpha when overlay_only (applied client-side)
    cache_params = {k: v for k, v in request.args.items() if k != "_t"}
    if overlay_only:
        cache_params.pop("alpha", None)
    cache_key = (session.result_version, idx, tuple(sorted(cache_params.items())))
    cached = _render_cache.get(cache_key)
    if cached is not None:
        if isinstance(cached, tuple):
            # Data texture: (png_bytes, data_min, data_max)
            return data_texture_response(*cached)
        return png_response(cached)

    disp_data = _get_displacement_component(idx, component, ref_frame=ref_frame)

    # Place displacement data in full image coordinates
    h, w = session.image_height, session.image_width
    if h == 0 or w == 0:
        if session.reference_image is not None:
            h, w = session.reference_image.shape[:2]
        else:
            return jsonify({"error": "No image dimensions"}), 500

    rect = _active_rect()
    if background == "deformed" and rect:
        from server.deformed_warp import get_warped_full_data
        disp = disp_results[idx]
        U, V = disp[:, :, 0], disp[:, :, 1]
        # User mask for the deformed frame (disp idx 0 = frame 2 = index 1)
        deformed_mask = session.per_frame_rois.get(idx + 1)
        full_data = get_warped_full_data(
            data=disp_data, frame_idx=idx,
            U=U, V=V,
            roi_rect=rect,
            image_shape=(h, w),
            cache=session.inverse_map_cache,
            vw=vw, vh=vh,
            deformed_mask=deformed_mask,
        )
    else:
        full_data = np.full((h, w), np.nan)
        if rect:
            x0, y0, x1, y1 = rect
            dh, dw = disp_data.shape
            sh = min(dh, y1 - y0)
            sw = min(dw, x1 - x0)
            full_data[y0:y0 + sh, x0:x0 + sw] = disp_data[:sh, :sw]

        # Reference mode: fix ROI shape to frame 0's mask.
        # _apply_current_frame_mask bakes coordinate-rounding NaN into stored
        # results; fill those gaps with nearest valid values so the visible
        # shape exactly matches frame 0's mask boundary.
        ref_mask = session.per_frame_rois.get(0)
        if ref_mask is not None and ref_mask.shape == full_data.shape:
            full_data[~ref_mask] = np.nan
            full_data = fill_mask_nan_gaps(full_data, ref_mask)

    if overlay_only:
        render_mode = request.args.get("render_mode", "colored")
        if render_mode == "data":
            png_bytes, data_min, data_max = render_data_texture_png(
                full_data, vw=vw, vh=vh,
            )
            _render_cache.put(cache_key, (png_bytes, data_min, data_max))
            return data_texture_response(png_bytes, data_min, data_max)

        png_bytes = render_overlay_png(
            full_data, colormap=colormap, vmin=vmin, vmax=vmax,
            log_scale=log_scale, vw=vw, vh=vh,
        )
        _render_cache.put(cache_key, png_bytes)
        return png_response(png_bytes)

    # --- Composited mode (legacy) ---
    if background == "deformed" and idx + 1 < len(session.image_files):
        bg_img = session.deformed_view_cache.get_deformed_image(idx + 1)
        if bg_img is None:
            bg_path = os.path.join(session.image_dir, session.image_files[idx + 1])
            bg_img = load_and_convert_image(bg_path)
    else:
        bg_img = session.reference_image

    if bg_img is None:
        return jsonify({"error": "No reference image"}), 500

    png_bytes = render_composited_png(
        full_data, bg_img, colormap=colormap, alpha=alpha,
        vmin=vmin, vmax=vmax, log_scale=log_scale, vw=vw, vh=vh,
    )
    _render_cache.put(cache_key, png_bytes)
    return png_response(png_bytes)


@displacement_bp.route("/download/<int:idx>", methods=["GET"])
def download_frame(idx):
    """Download a single rendered displacement frame as a high-res PNG."""
    import os
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from flask import send_file

    disp_results = session.displacement_results
    if not disp_results:
        return jsonify({"error": "No displacement results"}), 404
    if idx < 0 or idx >= len(disp_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "u")
    colormap = request.args.get("colormap", "turbo")
    alpha = request.args.get("alpha", 0.7, type=float)
    vmin = request.args.get("vmin", type=float)
    vmax = request.args.get("vmax", type=float)
    background = request.args.get("background", "reference")
    log_scale = request.args.get("log_scale", "false").lower() in ("true", "1")
    ref_frame = request.args.get("ref_frame", 0, type=int)
    try:
        dpi = int(validate_range(request.args.get("dpi", 150), "dpi", 50, 600))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    disp_data = _get_displacement_component(idx, component, ref_frame=ref_frame)

    # Background image
    if background == "deformed" and idx + 1 < len(session.image_files):
        bg_path = os.path.join(session.image_dir, session.image_files[idx + 1])
        bg_img = load_and_convert_image(bg_path)
    else:
        bg_img = session.reference_image

    if bg_img is None:
        return jsonify({"error": "No background image"}), 500

    h, w = bg_img.shape[:2]

    # Place data in full image coordinates (apply deformed warp when needed)
    rect = _active_rect()
    if background == "deformed" and rect:
        from server.deformed_warp import get_warped_full_data
        disp = disp_results[idx]
        U, V = disp[:, :, 0], disp[:, :, 1]
        deformed_mask = session.per_frame_rois.get(idx + 1)
        full_data = get_warped_full_data(
            data=disp_data, frame_idx=idx,
            U=U, V=V,
            roi_rect=rect,
            image_shape=(h, w),
            cache=session.inverse_map_cache,
            deformed_mask=deformed_mask,
        )
    else:
        full_data = np.full((h, w), np.nan)
        if rect:
            x0, y0, x1, y1 = rect
            dh, dw = disp_data.shape
            sh = min(dh, y1 - y0)
            sw = min(dw, x1 - x0)
            full_data[y0:y0 + sh, x0:x0 + sw] = disp_data[:sh, :sw]
        else:
            full_data[:disp_data.shape[0], :disp_data.shape[1]] = disp_data

        # Reference mode: fix ROI shape to frame 0's mask
        ref_mask = session.per_frame_rois.get(0)
        if ref_mask is not None and ref_mask.shape == full_data.shape:
            full_data[~ref_mask] = np.nan
            full_data = fill_mask_nan_gaps(full_data, ref_mask)

    # Render with matplotlib
    fig_w = w / dpi
    fig_h = h / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    ax.imshow(bg_img, cmap="gray" if bg_img.ndim == 2 else None)

    masked = np.ma.array(full_data, mask=np.isnan(full_data))

    if vmin is None:
        valid = full_data[~np.isnan(full_data)]
        vmin = float(valid.min()) if valid.size > 0 else 0.0
    if vmax is None:
        valid = full_data[~np.isnan(full_data)]
        vmax = float(valid.max()) if valid.size > 0 else 1.0

    norm = None
    if log_scale and vmin > 0 and vmax > 0:
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
        im = ax.imshow(masked, cmap=colormap, alpha=alpha, norm=norm)
    else:
        im = ax.imshow(masked, cmap=colormap, alpha=alpha, vmin=vmin, vmax=vmax)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"{component.upper()} — Frame {idx + 1}", fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.5)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return send_file(
        buf,
        mimetype="image/png",
        as_attachment=True,
        download_name=f"{component}_frame_{idx + 1:04d}.png",
    )


# Global colormap LUT cache (colormaps are static, cache indefinitely)
_colormap_lut_cache: dict = {}


@displacement_bp.route("/colormap/<name>", methods=["GET"])
def colormap_lut(name: str):
    """Return a 256×1 RGBA PNG encoding a matplotlib colormap LUT."""
    if name in _colormap_lut_cache:
        png_bytes = _colormap_lut_cache[name]
    else:
        try:
            png_bytes = get_colormap_lut(name)
        except ValueError:
            return jsonify({"error": f"Unknown colormap: {name}"}), 400
        _colormap_lut_cache[name] = png_bytes

    from server.serializers import image_response
    resp = image_response(png_bytes, jpeg=False)
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp
