"""Strain calculation and data endpoints."""

import io
import os
import threading

import numpy as np
from flask import Blueprint, jsonify, request

from raft_dic_gui.strain import calculate_strain_field
from server.app import socketio
from server.render_cache import RenderCache, auto_cache_size
from server.render_utils import fill_mask_nan_gaps, render_composited_png, render_data_texture_png, render_overlay_png
from server.serializers import data_texture_response, frame_data_to_json, png_response
from server.session import session
from server.validation import validate_choice, validate_non_negative, validate_odd_positive_int, validate_positive, validate_positive_int, validate_range

strain_bp = Blueprint("strain", __name__)
_render_cache = RenderCache(auto_cache_size())

STRAIN_COMPONENTS = [
    "exx", "eyy", "exy", "e1", "e2", "max_shear", "von_mises", "rotation",
    "dexx_dt", "deyy_dt", "dexy_dt",
]


def _active_rect():
    """Return envelope_rect if available, else roi_rect."""
    return session.envelope_rect or session.roi_rect


@strain_bp.route("/calculate", methods=["POST"])
def calculate():
    """Calculate strain fields for all displacement frames (async)."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results available"}), 400

    if session.strain_computing:
        return jsonify({"error": "Strain calculation already in progress"}), 409

    data = request.get_json(force=True)

    try:
        method = validate_choice(
            data.get("method", "green_lagrange"),
            "method",
            ("green_lagrange", "engineering"),
        )
        vsg_size = validate_odd_positive_int(
            data.get("vsg_size", 31), "vsg_size", minimum=9
        )
        poly_order = validate_positive_int(data.get("poly_order", 1), "poly_order")
        weighting = validate_choice(
            data.get("weighting", "Gaussian"), "weighting", ("Gaussian", "gaussian", "uniform", "Uniform")
        )
        step = validate_positive_int(data.get("step", 1), "step")
        temporal_sigma = validate_non_negative(data.get("temporal_sigma", 0), "temporal_sigma")
        strain_fps = validate_positive(data.get("fps", 1.0), "fps")
        gaussian_sigma_raw = data.get("gaussian_sigma", None)
        gaussian_sigma = float(gaussian_sigma_raw) if gaussian_sigma_raw is not None else None
        spatial_sigma = validate_non_negative(data.get("spatial_sigma", 0), "spatial_sigma")
        cond_threshold = validate_positive(data.get("cond_threshold", 1000), "cond_threshold")
        boundary_erosion = int(data.get("boundary_erosion", -1))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    def compute():
        try:
            session.strain_computing = True
            results = []
            total = len(session.displacement_results)

            for i, disp in enumerate(session.displacement_results):
                # Spatial smoothing of displacement before strain calculation
                if spatial_sigma > 0:
                    from scipy.ndimage import gaussian_filter
                    smoothed = np.empty_like(disp)
                    for ch in range(disp.shape[2]):
                        plane = disp[:, :, ch]
                        valid = np.isfinite(plane)
                        filled = np.nan_to_num(plane, nan=0.0)
                        s_data = gaussian_filter(filled, sigma=spatial_sigma)
                        s_weight = gaussian_filter(valid.astype(float), sigma=spatial_sigma)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            smoothed[:, :, ch] = np.where(
                                s_weight > 0.01, s_data / s_weight, np.nan
                            )
                    disp = smoothed

                strain_dict = calculate_strain_field(
                    disp,
                    method=method,
                    vsg_size=vsg_size,
                    poly_order=poly_order,
                    weighting=weighting,
                    step=step,
                    gaussian_sigma=gaussian_sigma,
                    cond_threshold=cond_threshold,
                    boundary_erosion=boundary_erosion,
                )
                results.append(strain_dict)

                socketio.emit("strain:progress", {
                    "percent": round((i + 1) / total * 100, 1),
                    "current": i + 1,
                    "total": total,
                })

            if temporal_sigma > 0:
                from raft_dic_gui.strain import smooth_strain_temporal
                results = smooth_strain_temporal(results, sigma_t=temporal_sigma)

            # Compute strain rate (central difference)
            from raft_dic_gui.strain import calculate_strain_rate
            results = calculate_strain_rate(results, fps=strain_fps)

            # Compute max strain for large-deformation warning
            max_strain = 0.0
            for r in results:
                if r is not None:
                    for comp in ['exx', 'eyy', 'exy']:
                        if comp in r and r[comp] is not None:
                            finite = r[comp][np.isfinite(r[comp])]
                            if finite.size > 0:
                                max_strain = max(max_strain, float(np.max(np.abs(finite))))

            with session._lock:
                session.strain_results = results
                session.result_version += 1
                session.strain_components = STRAIN_COMPONENTS
                session.strain_computing = False
                _render_cache.clear()

            socketio.emit("strain:complete", {
                "num_frames": len(results),
                "components": STRAIN_COMPONENTS,
                "max_strain": round(max_strain, 6),
                "method": method,
            })

        except Exception as e:
            session.strain_computing = False
            socketio.emit("strain:error", {"error": str(e)})

    thread = threading.Thread(target=compute, daemon=True)
    thread.start()

    return jsonify({"ok": True})


@strain_bp.route("/status", methods=["GET"])
def get_status():
    """Return strain calculation status."""
    return jsonify({
        "computed": len(session.strain_results) > 0,
        "computing": session.strain_computing,
        "components": session.strain_components,
        "num_frames": len(session.strain_results),
    })


@strain_bp.route("/frame/<int:idx>", methods=["GET"])
def get_frame_data(idx: int):
    """Return raw strain data for a frame as JSON."""
    if not session.strain_results:
        return jsonify({"error": "No strain results"}), 404

    if idx < 0 or idx >= len(session.strain_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "exx")
    strain_dict = session.strain_results[idx]

    if strain_dict is None or component not in strain_dict:
        return jsonify({"error": f"Component '{component}' not available"}), 400

    data = strain_dict[component]
    vmin = request.args.get("vmin", type=float)
    vmax = request.args.get("vmax", type=float)

    return jsonify(frame_data_to_json(data, vmin=vmin, vmax=vmax))


@strain_bp.route("/range/<int:idx>", methods=["GET"])
def get_range(idx: int):
    """Return auto vmin/vmax for a strain component (lightweight)."""
    if not session.strain_results:
        return jsonify({"error": "No strain results"}), 404
    if idx < 0 or idx >= len(session.strain_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "exx")
    strain_dict = session.strain_results[idx]
    if strain_dict is None or component not in strain_dict:
        return jsonify({"error": f"Component '{component}' not available"}), 400

    data = strain_dict[component]
    finite = data[np.isfinite(data)]
    vmin = float(finite.min()) if finite.size > 0 else 0.0
    vmax = float(finite.max()) if finite.size > 0 else 1.0
    return jsonify({"vmin": vmin, "vmax": vmax})


@strain_bp.route("/render/<int:idx>", methods=["GET"])
def render_frame(idx: int):
    """Render a strain field overlay as PNG."""
    # Snapshot to avoid mid-request replacement by another thread
    strain_results = session.strain_results
    if not strain_results:
        return jsonify({"error": "No strain results"}), 404

    if idx < 0 or idx >= len(strain_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "exx")
    colormap = request.args.get("colormap", "turbo")
    alpha = request.args.get("alpha", 0.7, type=float)
    vmin = request.args.get("vmin", type=float)
    vmax = request.args.get("vmax", type=float)
    background = request.args.get("background", "reference")
    log_scale = request.args.get("log_scale", "false").lower() in ("true", "1", "yes")
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
            return data_texture_response(*cached)
        return png_response(cached)

    strain_dict = strain_results[idx]
    if strain_dict is None or component not in strain_dict:
        return jsonify({"error": f"Component '{component}' not available"}), 400

    strain_data = strain_dict[component]

    # Get full image dimensions
    h, w = session.image_height, session.image_width
    if h == 0 or w == 0:
        if session.reference_image is not None:
            h, w = session.reference_image.shape[:2]
        else:
            return jsonify({"error": "No image dimensions"}), 500

    full_data = _place_strain_in_full_image(strain_data, idx, h, w, background,
                                            vw=vw, vh=vh)

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
        from raft_dic_gui.processing import load_and_convert_image
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


def _place_strain_in_full_image(
    strain_data: np.ndarray,
    idx: int,
    h: int,
    w: int,
    background: str = "reference",
    vw: int = 0,
    vh: int = 0,
) -> np.ndarray:
    """Place strain data into full image coordinates, applying deformed warp if needed."""
    import cv2

    rect = _active_rect()
    if background == "deformed" and rect:
        from server.deformed_warp import get_warped_full_data
        x0, y0, x1, y1 = rect
        roi_h, roi_w = y1 - y0, x1 - x0
        sh, sw = strain_data.shape

        disp = session.displacement_results[idx]
        U, V = disp[:, :, 0], disp[:, :, 1]
        deformed_mask = session.per_frame_rois.get(idx + 1)
        return get_warped_full_data(
            data=strain_data, frame_idx=idx,
            U=U, V=V,
            roi_rect=rect,
            image_shape=(h, w),
            needs_upsample=(sh != roi_h or sw != roi_w),
            roi_h=roi_h, roi_w=roi_w,
            cache=session.inverse_map_cache,
            vw=vw, vh=vh,
            deformed_mask=deformed_mask,
        )

    full_data = np.full((h, w), np.nan)
    if rect:
        x0, y0, x1, y1 = rect
        roi_h, roi_w = y1 - y0, x1 - x0
        sh, sw = strain_data.shape

        if sh != roi_h or sw != roi_w:
            mask_valid = ~np.isnan(strain_data)
            data_clean = np.nan_to_num(strain_data, nan=0.0)
            data_resized = cv2.resize(data_clean, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(mask_valid.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            data_resized[mask_resized == 0] = np.nan
            full_data[y0:y1, x0:x1] = data_resized
        else:
            full_data[y0:y1, x0:x1] = strain_data

    # Reference mode: fix ROI shape to frame 0's mask
    ref_mask = session.per_frame_rois.get(0)
    if ref_mask is not None and ref_mask.shape == full_data.shape:
        full_data[~ref_mask] = np.nan
        full_data = fill_mask_nan_gaps(full_data, ref_mask)

    return full_data


@strain_bp.route("/download/<int:idx>", methods=["GET"])
def download_frame(idx):
    """Download a single rendered strain frame as a high-res PNG."""
    from io import BytesIO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from flask import send_file
    from raft_dic_gui.processing import load_and_convert_image

    strain_results = session.strain_results
    if not strain_results:
        return jsonify({"error": "No strain results"}), 404
    if idx < 0 or idx >= len(strain_results):
        return jsonify({"error": "Frame index out of range"}), 400

    component = request.args.get("component", "exx")
    colormap = request.args.get("colormap", "turbo")
    alpha = request.args.get("alpha", 0.7, type=float)
    vmin = request.args.get("vmin", type=float)
    vmax = request.args.get("vmax", type=float)
    background = request.args.get("background", "reference")
    log_scale = request.args.get("log_scale", "false").lower() in ("true", "1")
    try:
        dpi = int(validate_range(request.args.get("dpi", 150), "dpi", 50, 600))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    strain_dict = strain_results[idx]
    if strain_dict is None or component not in strain_dict:
        return jsonify({"error": f"Component '{component}' not available"}), 400

    strain_data = strain_dict[component]

    # Background image
    if background == "deformed" and idx + 1 < len(session.image_files):
        bg_path = os.path.join(session.image_dir, session.image_files[idx + 1])
        bg_img = load_and_convert_image(bg_path)
    else:
        bg_img = session.reference_image

    if bg_img is None:
        return jsonify({"error": "No background image"}), 500

    h, w = bg_img.shape[:2]

    # Place strain data in full image (applies deformed warp when needed)
    full_data = _place_strain_in_full_image(strain_data, idx, h, w, background)

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
    ax.set_title(f"{component} — Frame {idx + 1}", fontsize=10)
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
