"""Session save/load endpoints."""

import os

from flask import Blueprint, jsonify, request

from raft_dic_gui.session_io import save_session, load_session, get_session_info
from server.session import session

session_bp = Blueprint("session", __name__)


@session_bp.route("/save", methods=["POST"])
def save():
    """Save current session to a .raftproj file."""
    if not session.displacement_results:
        return jsonify({"error": "No results to save"}), 400

    data = request.get_json(force=True)
    file_path = data.get("file_path", "").strip()

    if not file_path:
        return jsonify({"error": "Missing file_path"}), 400

    # Ensure .raftproj extension
    if not file_path.lower().endswith(".raftproj"):
        file_path += ".raftproj"

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

    try:
        saved_path = save_session(session, file_path)
        size_mb = os.path.getsize(saved_path) / (1024 * 1024)
        return jsonify({
            "ok": True,
            "path": saved_path,
            "size_mb": round(size_mb, 1),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@session_bp.route("/load", methods=["POST"])
def load():
    """Load a session from a .raftproj file."""
    data = request.get_json(force=True)
    file_path = data.get("file_path", "").strip()

    if not file_path:
        return jsonify({"error": "Missing file_path"}), 400

    if not os.path.isfile(file_path):
        return jsonify({"error": f"File not found: {file_path}"}), 404

    try:
        # Reset session before loading
        session.reset()

        result = load_session(file_path, session)

        # If image directory exists, reload reference image
        image_dir = session.image_dir
        if image_dir and os.path.isdir(image_dir) and session.image_files:
            _reload_reference_image(image_dir, session.image_files[0])

        return jsonify({
            "ok": True,
            "warnings": result["warnings"],
            "summary": result["summary"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@session_bp.route("/info", methods=["GET"])
def info():
    """Get current session state summary."""
    return jsonify(get_session_info(session))


def _reload_reference_image(image_dir: str, first_file: str):
    """Reload the reference image after session load."""
    import cv2

    ref_path = os.path.join(image_dir, first_file)
    if os.path.isfile(ref_path):
        img = cv2.imread(ref_path)
        if img is not None:
            session.reference_image = img
