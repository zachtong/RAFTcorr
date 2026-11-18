"""Model discovery and description endpoints."""

import dataclasses

from flask import Blueprint, jsonify, request

from raft_dic_gui.model import (
    describe_checkpoint,
    discover_models,
    estimate_safe_pmax,
)
from server.session import session

models_bp = Blueprint("models", __name__)


@models_bp.route("", methods=["GET"])
def list_models():
    """Return available model checkpoints."""
    entries = discover_models()
    return jsonify({
        "models": [{"path": e.path, "label": e.label} for e in entries],
    })


@models_bp.route("/describe", methods=["POST"])
def describe_model():
    """Describe a model checkpoint's architecture."""
    data = request.get_json(force=True)
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "Missing model path"}), 400

    try:
        metadata = describe_checkpoint(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    safe_pmax = estimate_safe_pmax(metadata, device=session.config.device)

    result = dataclasses.asdict(metadata)
    result["summary"] = metadata.summary()
    result["safe_pmax"] = safe_pmax
    return jsonify({"metadata": result})


@models_bp.route("/select", methods=["POST"])
def select_model():
    """Select a model for processing."""
    data = request.get_json(force=True)
    path = data.get("path", "").strip()
    if not path:
        return jsonify({"error": "Missing model path"}), 400

    try:
        metadata = describe_checkpoint(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    safe_pmax = estimate_safe_pmax(metadata, device=session.config.device)

    with session._lock:
        session.config.model_path = metadata.path
        session.config.model_label = metadata.label
        session.config.model_metadata = metadata
        session.config.p_max_pixels = safe_pmax

    return jsonify({
        "ok": True,
        "label": metadata.label,
        "summary": metadata.summary(),
        "safe_pmax": safe_pmax,
    })
