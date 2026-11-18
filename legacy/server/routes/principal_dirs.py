"""Principal strain direction overlay as transparent PNG with cross marks."""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, jsonify, request

from server.render_cache import RenderCache, auto_cache_size
from server.serializers import png_response
from server.session import session

principal_bp = Blueprint("principal", __name__)
_render_cache = RenderCache(auto_cache_size())


@principal_bp.route("/render/<int:idx>", methods=["GET"])
def render_principal(idx: int):
    """Render principal strain direction crosses as transparent PNG overlay."""
    if not session.strain_results:
        return jsonify({"error": "No strain results"}), 404
    if idx < 0 or idx >= len(session.strain_results):
        return jsonify({"error": "Frame index out of range"}), 400

    spacing = request.args.get("spacing", 30, type=int)
    scale = request.args.get("scale", 15, type=float)
    line_width = request.args.get("line_width", 1.0, type=float)
    vw = request.args.get("vw", 0, type=int)
    vh = request.args.get("vh", 0, type=int)

    # Check cache
    cache_params = {k: v for k, v in request.args.items() if k != "_t"}
    key = (session.result_version, idx, "principal", tuple(sorted(cache_params.items())))
    cached = _render_cache.get(key)
    if cached is not None:
        return png_response(cached)

    strain = session.strain_results[idx]
    if strain is None:
        return jsonify({"error": "No strain for this frame"}), 400

    exx = strain.get("exx")
    eyy = strain.get("eyy")
    exy = strain.get("exy")
    e1 = strain.get("e1")
    e2 = strain.get("e2")

    if exx is None or eyy is None or exy is None:
        return jsonify({"error": "Missing strain components"}), 400

    sh, sw = exx.shape

    # ROI offset
    if session.roi_rect:
        x0, y0 = session.roi_rect[0], session.roi_rect[1]
        roi_h = session.roi_rect[3] - session.roi_rect[1]
        roi_w = session.roi_rect[2] - session.roi_rect[0]
    else:
        x0, y0 = 0, 0
        roi_h, roi_w = sh, sw

    # Full image size
    if session.reference_image is not None:
        img_h, img_w = session.reference_image.shape[:2]
    else:
        img_h, img_w = roi_h + y0, roi_w + x0

    # Scale factor if strain is downsampled
    sy = roi_h / sh if sh != roi_h else 1.0
    sx = roi_w / sw if sw != roi_w else 1.0

    # Grid points
    y_idx = np.arange(0, sh, max(1, spacing // max(1, int(sy))))
    x_idx = np.arange(0, sw, max(1, spacing // max(1, int(sx))))

    # Principal direction angle: theta = 0.5 * atan2(2*exy, exx - eyy)
    theta = 0.5 * np.arctan2(2 * exy, exx - eyy)  # radians

    dpi = 100
    if vw > 0 and vh > 0:
        render_scale = min(vw / img_w, vh / img_h, 1.0)
    else:
        render_scale = 1.0
    render_w = max(1, int(img_w * render_scale))
    render_h = max(1, int(img_h * render_scale))

    fig = None
    try:
        fig, ax = plt.subplots(figsize=(render_w / dpi, render_h / dpi), dpi=dpi)
        fig.patch.set_alpha(0)
        ax.set_position([0, 0, 1, 1])
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.axis("off")
        ax.patch.set_alpha(0)

        for yi in y_idx:
            for xi in x_idx:
                if np.isnan(theta[yi, xi]):
                    continue

                cx = xi * sx + x0
                cy = yi * sy + y0
                ang = theta[yi, xi]

                # Major principal direction (e1) -- red
                mag1 = abs(e1[yi, xi]) if e1 is not None and np.isfinite(e1[yi, xi]) else 0
                dx1 = scale * np.cos(ang) * min(mag1 * 100, 1)
                dy1 = scale * np.sin(ang) * min(mag1 * 100, 1)
                ax.plot([cx - dx1, cx + dx1], [cy - dy1, cy + dy1],
                        color='red', linewidth=line_width, alpha=0.8)

                # Minor principal direction (e2) -- blue, perpendicular
                mag2 = abs(e2[yi, xi]) if e2 is not None and np.isfinite(e2[yi, xi]) else 0
                dx2 = scale * np.cos(ang + np.pi / 2) * min(mag2 * 100, 1)
                dy2 = scale * np.sin(ang + np.pi / 2) * min(mag2 * 100, 1)
                ax.plot([cx - dx2, cx + dx2], [cy - dy2, cy + dy2],
                        color='blue', linewidth=line_width, alpha=0.8)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, dpi=dpi)
        buf.seek(0)
        png_bytes = buf.read()
    finally:
        if fig is not None:
            plt.close(fig)

    _render_cache.put(key, png_bytes)
    return png_response(png_bytes)
