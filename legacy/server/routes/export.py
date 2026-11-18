"""Scientific and batch image export endpoints."""

import os
import threading

from flask import Blueprint, jsonify, request

from raft_dic_gui.processing import load_and_convert_image, save_scientific_results
from raft_dic_gui.export_images import export_batch_images
from raft_dic_gui.export_animation import export_animation
from server.app import socketio
from server.session import session
from server.validation import validate_choice, validate_positive, validate_range

export_bp = Blueprint("export", __name__)


def _active_rect():
    """Return envelope_rect if available, else roi_rect."""
    return session.envelope_rect or session.roi_rect


_SCIENTIFIC_EXTS = (".npz", ".mat", ".h5", ".hdf5")


@export_bp.route("/scientific", methods=["POST"])
def export_scientific():
    """Export displacement + strain data via streaming writes.

    Supported extensions:
      .npz                  numpy-native, Python-friendly
      .mat                  MATLAB v7.3 (HDF5-based), MATLAB R2011a+ required
      .h5 / .hdf5           plain HDF5, language-agnostic
    """
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 400

    data = request.get_json(force=True)
    file_path = data.get("file_path", "").strip()
    upsample_strain = data.get("upsample_strain", True)
    metadata = data.get("metadata", {})
    variables = data.get("variables")  # None or list[str]; None = export all

    if not file_path:
        return jsonify({"error": "Missing file_path"}), 400

    if variables is not None and not isinstance(variables, list):
        return jsonify({"error": "variables must be a list of strings"}), 400

    ext = os.path.splitext(file_path)[1].lower()
    if ext and ext not in _SCIENTIFIC_EXTS:
        return jsonify({
            "error": f"Unsupported extension '{ext}'. "
                     f"Use one of: {', '.join(_SCIENTIFIC_EXTS)}",
        }), 400

    # Validate directory exists
    export_dir = os.path.dirname(file_path)
    if export_dir and not os.path.isdir(export_dir):
        return jsonify({"error": f"Directory does not exist: {export_dir}"}), 400

    # Check for overwrite
    overwrite = data.get("overwrite", False)
    if os.path.exists(file_path) and not overwrite:
        return jsonify({
            "error": "File already exists",
            "exists": True,
            "path": file_path,
        }), 409

    # Build metadata dict
    export_metadata = {
        "model_path": session.config.model_path,
        "model_label": session.config.model_label,
        "mode": session.config.mode,
        "img_dir": session.image_dir,
        "von_mises_assumption": "plane_stress_2D",
        "strain_method": "VWLS",
        "strain_components": session.strain_components,
    }
    export_metadata.update(metadata)

    # Collect image paths
    image_files = None
    if session.image_files and session.image_dir:
        image_files = [
            os.path.join(session.image_dir, f) for f in session.image_files
        ]

    try:
        save_scientific_results(
            displacement_results=session.displacement_results,
            strain_results=session.strain_results,
            roi_mask=session.roi_mask,
            roi_rect=_active_rect(),
            metadata=export_metadata,
            file_path=file_path,
            image_files=image_files,
            upsample_strain=upsample_strain,
            variables=variables,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"ok": True, "path": file_path})


@export_bp.route("/images", methods=["POST"])
def export_images():
    """Start batch image export (async)."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 400

    if session.export_active:
        return jsonify({"error": "Export already in progress"}), 409

    data = request.get_json(force=True)
    output_dir = data.get("output_dir", "").strip()
    components = data.get("components", {})
    frame_range = tuple(data.get("frame_range", [1, len(session.displacement_results)]))
    settings = data.get("settings", {})

    if not output_dir:
        return jsonify({"error": "Missing output_dir"}), 400

    # Validate parent directory
    parent_dir = os.path.dirname(output_dir.rstrip("/\\"))
    if parent_dir and not os.path.isdir(parent_dir):
        return jsonify({"error": f"Parent directory does not exist: {parent_dir}"}), 400

    def image_loader(frame_idx):
        if frame_idx < len(session.image_files):
            img_path = os.path.join(session.image_dir, session.image_files[frame_idx])
            return load_and_convert_image(img_path)
        return session.reference_image

    def progress_callback(current, total, message):
        session.export_progress = current
        session.export_total = total
        socketio.emit("export:progress", {
            "current": current,
            "total": total,
            "percent": round(current / max(1, total) * 100, 1),
            "message": message,
        })

    def run_export():
        try:
            session.export_active = True
            session.export_progress = 0
            session.export_cancel.clear()

            result_dir = export_batch_images(
                output_dir=output_dir,
                components=components,
                frame_range=frame_range,
                displacement_results=session.displacement_results,
                strain_results=session.strain_results,
                roi_rect=_active_rect(),
                roi_mask=session.roi_mask,
                image_loader=image_loader,
                settings=settings,
                deformed_view_cache=session.deformed_view_cache,
                progress_callback=progress_callback,
                cancel_event=session.export_cancel,
            )

            session.export_active = False
            socketio.emit("export:complete", {"path": result_dir})

        except Exception as e:
            session.export_active = False
            socketio.emit("export:error", {"error": str(e)})

    thread = threading.Thread(target=run_export, daemon=True)
    thread.start()

    return jsonify({"ok": True})


@export_bp.route("/images/status", methods=["GET"])
def export_status():
    """Return batch image export progress."""
    return jsonify({
        "active": session.export_active,
        "progress": session.export_progress,
        "total": session.export_total,
        "percent": round(
            session.export_progress / max(1, session.export_total) * 100, 1
        ) if session.export_total > 0 else 0,
    })


@export_bp.route("/images/cancel", methods=["POST"])
def cancel_export():
    """Cancel an in-progress batch image export."""
    if not session.export_active:
        return jsonify({"error": "No export in progress"}), 400
    session.export_cancel.set()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Animation export
# ---------------------------------------------------------------------------

@export_bp.route("/animation", methods=["POST"])
def export_animation_endpoint():
    """Start animation export (GIF/MP4) — async with progress."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 400

    if session.export_active:
        return jsonify({"error": "Export already in progress"}), 409

    data = request.get_json(force=True)
    output_path = data.get("output_path", "").strip()

    try:
        fmt = validate_choice(data.get("format", "gif"), "format", ("gif", "mp4")).lower()
        fps = validate_positive(data.get("fps", 10), "fps")
        resize_factor = validate_range(data.get("resize_factor", 1.0), "resize_factor", 0.1, 4.0)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    component = data.get("component", "u")
    frame_range = data.get("frame_range", [0, len(session.displacement_results) - 1])
    loop = data.get("loop", True)
    timestamp_overlay = data.get("timestamp_overlay", False)
    include_colorbar = data.get("include_colorbar", True)
    anim_settings = data.get("settings", {})

    if not output_path:
        return jsonify({"error": "Missing output_path"}), 400

    # Validate extension
    expected_ext = ".gif" if fmt == "gif" else ".mp4"
    if not output_path.lower().endswith(expected_ext):
        output_path += expected_ext

    # Validate directory
    export_dir = os.path.dirname(output_path)
    if export_dir and not os.path.isdir(export_dir):
        return jsonify({"error": f"Directory does not exist: {export_dir}"}), 400

    def image_loader(frame_idx):
        if frame_idx < len(session.image_files):
            img_path = os.path.join(session.image_dir, session.image_files[frame_idx])
            return load_and_convert_image(img_path)
        return session.reference_image

    def progress_callback(current, total, message):
        session.export_progress = current
        session.export_total = total
        socketio.emit("export:progress", {
            "current": current,
            "total": total,
            "percent": round(current / max(1, total) * 100, 1),
            "message": message,
        })

    def run_anim_export():
        try:
            session.export_active = True
            session.export_progress = 0
            session.export_cancel.clear()

            export_animation(
                output_path=output_path,
                fmt=fmt,
                component=component,
                frame_range=tuple(frame_range),
                displacement_results=session.displacement_results,
                strain_results=session.strain_results,
                roi_rect=_active_rect(),
                roi_mask=session.roi_mask,
                image_loader=image_loader,
                settings=anim_settings,
                deformed_view_cache=session.deformed_view_cache,
                fps=fps,
                loop=loop,
                timestamp_overlay=timestamp_overlay,
                include_colorbar=include_colorbar,
                resize_factor=resize_factor,
                progress_callback=progress_callback,
                cancel_event=session.export_cancel,
                per_frame_rois=session.per_frame_rois,
                inverse_map_cache=session.inverse_map_cache,
            )

            session.export_active = False
            socketio.emit("export:complete", {"path": output_path})
        except Exception as e:
            session.export_active = False
            socketio.emit("export:error", {"error": str(e)})

    thread = threading.Thread(target=run_anim_export, daemon=True)
    thread.start()

    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

@export_bp.route("/report", methods=["POST"])
def export_report():
    """Generate analysis report (synchronous, typically <15s)."""
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 400

    data = request.get_json(force=True)
    output_path = data.get("output_path", "").strip()
    sections = data.get("sections")  # None = all sections

    try:
        format_ = validate_choice(data.get("format", "html"), "format", ("html", "pdf", "both"))
        theme = validate_choice(data.get("theme", "light"), "theme", ("light", "dark"))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    custom_title = data.get("custom_title", "")
    author = data.get("author", "")
    notes = data.get("notes", "")
    key_frame_components = data.get("key_frame_components")  # None = ["magnitude"]

    if not output_path:
        return jsonify({"error": "Missing output_path"}), 400

    # Auto-append appropriate extension
    if format_ == "pdf":
        if not output_path.lower().endswith(".pdf"):
            output_path += ".pdf"
        # For PDF-only, we still generate HTML first internally;
        # convert output_path to .html for the generator (it derives .pdf from it)
        html_path = os.path.splitext(output_path)[0] + ".html"
    else:
        if not output_path.lower().endswith(".html"):
            output_path += ".html"
        html_path = output_path

    export_dir = os.path.dirname(html_path)
    if export_dir and not os.path.isdir(export_dir):
        return jsonify({"error": f"Directory does not exist: {export_dir}"}), 400

    try:
        from raft_dic_gui.report_generator import generate_report
        result = generate_report(
            session,
            html_path,
            sections=sections,
            format=format_,
            theme=theme,
            custom_title=custom_title,
            author=author,
            notes=notes,
            key_frame_components=key_frame_components,
            vis_settings=session.vis_settings,
        )

        response = {"ok": True}
        if result.get("html_path"):
            response["html_path"] = result["html_path"]
        if result.get("pdf_path"):
            response["pdf_path"] = result["pdf_path"]
        # Provide a single "path" key for backward compatibility
        response["path"] = result.get("pdf_path") or result.get("html_path")
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
