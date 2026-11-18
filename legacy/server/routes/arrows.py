"""Vector field (quiver + streamline) overlay rendering as transparent PNG.

Supports both displacement and velocity vector fields via the ``vector_source``
query parameter.
"""

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Blueprint, jsonify, request

from server.render_cache import RenderCache, auto_cache_size
from server.serializers import png_response
from server.session import session
from server.validation import validate_positive, validate_positive_int

arrows_bp = Blueprint("arrows", __name__)
_render_cache = RenderCache(auto_cache_size())


def _parse_bool(val: str) -> bool:
    return val.lower() in ("true", "1", "yes")


@arrows_bp.route("/render/<int:idx>", methods=["GET"])
def render_arrows(idx: int):
    """Render vector field arrows / streamlines as a transparent PNG overlay."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 404
    if idx < 0 or idx >= len(session.displacement_results):
        return jsonify({"error": "Frame index out of range"}), 400

    # Parse query parameters with validation
    show_quiver = _parse_bool(request.args.get("show_quiver", "false"))
    show_streamlines = _parse_bool(request.args.get("show_streamlines", "false"))
    try:
        spacing = int(validate_positive_int(request.args.get("spacing", 30), "spacing"))
        scale = float(validate_positive(request.args.get("scale", 20), "scale"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    color = request.args.get("color", "white")
    line_width = request.args.get("line_width", 1.5, type=float)
    background = request.args.get("background", "reference")
    vector_source = request.args.get("vector_source", "velocity")  # "displacement" or "velocity"
    stream_ds = request.args.get("stream_ds", 4, type=int)  # downsample factor for streamlines
    vw = request.args.get("vw", 0, type=int)
    vh = request.args.get("vh", 0, type=int)

    if not show_quiver and not show_streamlines:
        return jsonify({"error": "Nothing to render"}), 400

    # Check cache — return immediately if already rendered
    cache_params = {k: v for k, v in request.args.items() if k != "_t"}
    key = (session.result_version, idx, tuple(sorted(cache_params.items())))
    cached = _render_cache.get(key)
    if cached is not None:
        return png_response(cached)

    # Compute vector field (du, dv)
    disp = session.displacement_results[idx]
    if vector_source == "displacement":
        # Raw cumulative displacement field
        du = disp[:, :, 0].copy()
        dv = disp[:, :, 1].copy()
    else:
        # Velocity: difference between consecutive frames
        if idx == 0:
            du = disp[:, :, 0].copy()
            dv = disp[:, :, 1].copy()
        else:
            prev = session.displacement_results[idx - 1]
            du = disp[:, :, 0] - prev[:, :, 0]
            dv = disp[:, :, 1] - prev[:, :, 1]
    h, w = du.shape

    # ROI offset for coordinate grids
    if session.roi_rect:
        x_offset, y_offset = session.roi_rect[0], session.roi_rect[1]
    else:
        x_offset, y_offset = 0, 0

    # Full image dimensions
    if session.reference_image is not None:
        img_h, img_w = session.reference_image.shape[:2]
    else:
        img_h, img_w = h + y_offset, w + x_offset

    # Deformed mode: cumulative displacement for coordinate shifting
    deformed = background == "deformed"
    if deformed:
        U_disp = np.nan_to_num(disp[:, :, 0], nan=0.0)
        V_disp = np.nan_to_num(disp[:, :, 1], nan=0.0)

    # Create transparent figure — viewport-sized for performance
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
        ax.set_ylim(img_h, 0)  # Image coordinates: Y-axis inverted
        ax.axis("off")
        ax.patch.set_alpha(0)

        # --- Quiver rendering ---
        if show_quiver:
            y_idx = np.arange(0, h, max(1, spacing))
            x_idx = np.arange(0, w, max(1, spacing))
            X, Y = np.meshgrid(x_idx + x_offset, y_idx + y_offset)

            U = du[np.ix_(y_idx, x_idx)]
            V = dv[np.ix_(y_idx, x_idx)]

            # In deformed mode, forward-map arrow positions by cumulative displacement
            if deformed:
                X = X + U_disp[np.ix_(y_idx, x_idx)]
                Y = Y + V_disp[np.ix_(y_idx, x_idx)]

            valid = ~(np.isnan(U) | np.isnan(V))
            if np.any(valid):
                magnitude = np.sqrt(U ** 2 + V ** 2)
                magnitude[magnitude == 0] = 1
                U_norm = U / magnitude
                V_norm = V / magnitude

                quiver_scale = 1.0 / (scale + 0.001)
                quiver_width = 0.003 * line_width

                ax.quiver(
                    X[valid], Y[valid], U_norm[valid], V_norm[valid],
                    angles="xy", color=color, scale=quiver_scale,
                    scale_units="xy", width=quiver_width,
                    headwidth=4, headlength=5, alpha=0.9,
                )

        # --- Streamlines rendering (with optional downsampling for speed) ---
        if show_streamlines:
            ds = max(1, min(stream_ds, 16))  # clamp to [1, 16]

            if ds > 1:
                # Downsample velocity field — streamlines are smooth curves,
                # visual quality barely changes even at 4-8x downsample.
                h_ds = max(2, h // ds)
                w_ds = max(2, w // ds)
                # Evenly-spaced coordinates (streamplot requires uniform grid)
                x_stream = np.linspace(x_offset, x_offset + w - 1, w_ds)
                y_stream = np.linspace(y_offset, y_offset + h - 1, h_ds)
                # Nearest-pixel indices for velocity sampling
                y_idx_s = np.round(np.linspace(0, h - 1, h_ds)).astype(int)
                x_idx_s = np.round(np.linspace(0, w - 1, w_ds)).astype(int)
                du_raw = du[np.ix_(y_idx_s, x_idx_s)]
                dv_raw = dv[np.ix_(y_idx_s, x_idx_s)]
            else:
                du_raw = du
                dv_raw = dv
                x_stream = np.arange(w, dtype=float) + x_offset
                y_stream = np.arange(h, dtype=float) + y_offset

            if deformed:
                # Forward-map velocity samples to their deformed positions,
                # then interpolate onto the regular stream grid via Delaunay.
                # (The old inverse-mapping approach had unit-mismatch bugs
                # and first-order approximation errors that caused the valid
                # region to shrink progressively with large displacements.)
                from scipy.interpolate import griddata as _griddata

                if ds > 1:
                    U_ds = U_disp[np.ix_(y_idx_s, x_idx_s)]
                    V_ds = V_disp[np.ix_(y_idx_s, x_idx_s)]
                    ref_r, ref_c = np.meshgrid(y_idx_s, x_idx_s, indexing="ij")
                else:
                    U_ds = U_disp
                    V_ds = V_disp
                    ref_r, ref_c = np.mgrid[0:h, 0:w]

                # Deformed positions (full-image coordinates)
                def_x = ref_c.astype(np.float64) + x_offset + U_ds
                def_y = ref_r.astype(np.float64) + y_offset + V_ds

                valid_vel = ~(np.isnan(du_raw) | np.isnan(dv_raw))

                # Cap scatter points for griddata performance
                n_valid = int(valid_vel.sum())
                if n_valid > 20000:
                    step = int(np.ceil(np.sqrt(n_valid / 20000)))
                    sub = np.zeros_like(valid_vel)
                    sub[::step, ::step] = True
                    valid_vel = valid_vel & sub

                if valid_vel.sum() > 10:
                    pts = np.column_stack([def_x[valid_vel], def_y[valid_vel]])
                    xx, yy = np.meshgrid(x_stream, y_stream)
                    du_filled = _griddata(
                        pts, du_raw[valid_vel].astype(np.float64),
                        (xx, yy), method="linear", fill_value=0.0,
                    )
                    dv_filled = _griddata(
                        pts, dv_raw[valid_vel].astype(np.float64),
                        (xx, yy), method="linear", fill_value=0.0,
                    )
                else:
                    du_filled = np.zeros((len(y_stream), len(x_stream)))
                    dv_filled = np.zeros((len(y_stream), len(x_stream)))
            else:
                du_filled = np.nan_to_num(du_raw, nan=0.0)
                dv_filled = np.nan_to_num(dv_raw, nan=0.0)

            density = max(0.5, 30.0 / max(1, spacing))
            base_arrowsize = max(h, w) / 500.0
            arrowsize = base_arrowsize * (scale / 50.0)
            arrowsize = max(0.5, min(arrowsize, 5.0))

            try:
                ax.streamplot(
                    x_stream, y_stream, du_filled, dv_filled,
                    color=color, density=density, linewidth=line_width,
                    arrowsize=arrowsize, arrowstyle="->",
                )
            except Exception:
                pass  # streamplot can fail on degenerate fields

        # Render to PNG buffer and cache
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True, dpi=dpi)
        buf.seek(0)
        png_bytes = buf.read()
    finally:
        if fig is not None:
            plt.close(fig)

    _render_cache.put(key, png_bytes)
    return png_response(png_bytes)
