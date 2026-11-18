"""DIC processing run/stop/pause/resume endpoints with WebSocket progress."""

import threading

import numpy as np
from flask import Blueprint, jsonify, request

from raft_dic_gui.config import DICConfig
from raft_dic_gui.controller import DICProcessor
from server.app import socketio
from server.session import session
from server.validation import validate_positive, validate_non_negative, validate_positive_int, validate_choice

processing_bp = Blueprint("processing", __name__)


@processing_bp.route("/configure", methods=["POST"])
def configure():
    """Update DIC processing configuration."""
    data = request.get_json(force=True)

    try:
        with session._lock:
            cfg = session.config
            # Apply provided fields with validation
            if "mode" in data:
                cfg.mode = validate_choice(
                    data["mode"], "mode", ("accumulative", "incremental")
                )
            if "context_padding" in data:
                cfg.context_padding = validate_positive_int(
                    data["context_padding"], "context_padding"
                )
            if "tile_overlap" in data:
                cfg.tile_overlap = validate_positive_int(
                    data["tile_overlap"], "tile_overlap"
                )
            if "use_smooth" in data:
                cfg.use_smooth = bool(data["use_smooth"])
            if "sigma" in data:
                cfg.sigma = validate_non_negative(data["sigma"], "sigma")
            if "p_max_pixels" in data:
                cfg.p_max_pixels = validate_positive_int(
                    data["p_max_pixels"], "p_max_pixels"
                )
            if "device" in data:
                cfg.device = validate_choice(
                    data["device"], "device", ("cuda", "cpu")
                )
            if "project_root" in data:
                cfg.project_root = data["project_root"]

            # Incremental mode settings
            if "key_frames" in data:
                val = data["key_frames"]
                cfg.key_frames = list(val) if val else None
            if "key_frame_interval" in data:
                val = data["key_frame_interval"]
                cfg.key_frame_interval = int(val) if val else None
            if "mask_dir" in data:
                val = data["mask_dir"]
                cfg.mask_dir = val if val else None
            if "use_median_filter" in data:
                cfg.use_median_filter = bool(data["use_median_filter"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"ok": True})


@processing_bp.route("/validate-masks", methods=["POST"])
def validate_masks():
    """Validate a mask folder against loaded images."""
    data = request.get_json(force=True)
    mask_dir = data.get("mask_dir", "")

    if not session.image_files:
        return jsonify({"error": "No images loaded"}), 400

    if not mask_dir:
        return jsonify({"error": "No mask directory provided"}), 400

    from raft_dic_gui.mask_loader import discover_masks

    masks = discover_masks(
        mask_dir,
        session.image_files,
        image_shape=(session.image_height, session.image_width),
    )

    matched_frames = sorted([idx + 1 for idx in masks.keys()])  # 1-indexed for UI
    has_frame_1 = 0 in masks  # 0-based index: frame 1 is index 0
    total = len(session.image_files)

    return jsonify({
        "matched_count": len(masks),
        "total_frames": total,
        "masks_needed": total,  # one mask per image (ref ROI + current-frame boundary)
        "matched_frames": matched_frames,
        "has_frame_1": has_frame_1,
    })


@processing_bp.route("/apply-frame1-mask", methods=["POST"])
def apply_frame1_mask():
    """Load the Frame 1 mask from a mask folder and set it as the ROI."""
    data = request.get_json(force=True)
    mask_dir = data.get("mask_dir", "").strip()

    if not mask_dir or not session.image_files:
        return jsonify({"error": "Missing mask_dir or no images loaded"}), 400

    from raft_dic_gui.mask_loader import discover_masks
    image_shape = (session.image_height, session.image_width)
    matched = discover_masks(mask_dir, session.image_files, image_shape)

    if 0 not in matched:
        return jsonify({"error": "No Frame 1 mask found in folder"}), 404

    with session._lock:
        session.roi_mask = matched[0]
        y_idx, x_idx = np.where(session.roi_mask)
        if len(x_idx) > 0:
            session.roi_rect = (
                int(x_idx.min()), int(y_idx.min()),
                int(x_idx.max()) + 1, int(y_idx.max()) + 1,
            )
        session.roi_confirmed = True

    area = int(session.roi_mask.sum())
    return jsonify({"rect": session.roi_rect, "area_px": area})


@processing_bp.route("/run", methods=["POST"])
def run_processing():
    """Start DIC processing in a background thread."""
    if session.processing_active:
        return jsonify({"error": "Processing already active"}), 409

    # Validate prerequisites
    cfg = session.config
    valid, err = cfg.validate()
    if not valid:
        return jsonify({"error": err}), 400

    if session.roi_mask is None or not session.roi_confirmed:
        return jsonify({"error": "ROI must be confirmed before processing"}), 400

    # Set project_root to a temp output dir if not already set
    if not cfg.project_root:
        import os
        import tempfile
        cfg.project_root = os.path.join(
            tempfile.gettempdir(), "raft_dic_output"
        )

    # Track temp output dir for cleanup on session reset
    session._temp_output_dir = cfg.project_root

    def progress_callback(percent, current, total, **extra):
        payload = {
            "percent": round(percent, 1),
            "current": current,
            "total": total,
        }
        # Forward tile-level progress, timing, VRAM info from controller
        for key in ("tile_current", "tile_total", "vram_used_mb", "vram_total_mb",
                    "elapsed_seconds", "est_remaining_seconds"):
            if key in extra:
                payload[key] = extra[key]
        socketio.emit("processing:progress", payload)

    def check_stop():
        """Check stop/pause. Blocks while paused."""
        if session.stop_requested:
            return True
        if session.pause_requested:
            socketio.emit("processing:paused", {})
            # Block until resumed or cancelled
            session.pause_event.wait()
            session.pause_requested = False
            return session.stop_requested
        return False

    def run_in_thread():
        try:
            session.processing_active = True
            session.stop_requested = False
            session.pause_requested = False
            session.pause_event.set()  # Start unblocked

            # Free memory from previous run before starting new one
            with session._lock:
                session.displacement_results = []
                session.strain_results = []

            processor = DICProcessor(
                update_progress_callback=progress_callback,
                check_stop_callback=check_stop,
            )

            # Pass per-frame ROIs (if any) so controller uses them
            # instead of re-discovering from mask_dir
            per_frame = dict(session.per_frame_rois) if session.per_frame_rois else None
            results = processor.run(
                session.config,
                session.roi_mask,
                session.roi_rect,
                image_files=session.image_files,
                user_masks=per_frame,
            )

            with session._lock:
                session.displacement_results = results
                session.result_version += 1
                session.processor = processor
                session.envelope_rect = getattr(processor, 'envelope_rect', session.roi_rect)
                session.processing_active = False
                # New displacement results invalidate cached inverse maps
                session.inverse_map_cache.clear()

                # Clear all render caches (stale after new results)
                from server.routes.displacement import _render_cache as disp_cache
                from server.routes.strain import _render_cache as strain_cache
                from server.routes.arrows import _render_cache as arrow_cache
                disp_cache.clear()
                strain_cache.clear()
                arrow_cache.clear()

                # Set up deformed view cache with results
                if results:
                    import os
                    image_paths = [
                        os.path.join(session.image_dir, f)
                        for f in session.image_files
                    ]
                    session.deformed_view_cache.set_image_paths(image_paths)

            if session.stop_requested:
                socketio.emit("processing:complete", {
                    "stopped": True,
                    "num_frames": len(results),
                })
            else:
                socketio.emit("processing:complete", {
                    "stopped": False,
                    "num_frames": len(results),
                })

        except Exception as e:
            session.processing_active = False
            socketio.emit("processing:error", {"error": str(e)})

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    return jsonify({"ok": True})


@processing_bp.route("/pause", methods=["POST"])
def pause_processing():
    """Pause processing — blocks the worker thread between frames."""
    if not session.processing_active:
        return jsonify({"error": "No processing active"}), 400

    session.pause_requested = True
    session.pause_event.clear()  # Block the worker
    return jsonify({"ok": True})


@processing_bp.route("/resume", methods=["POST"])
def resume_processing():
    """Resume paused processing."""
    if not session.processing_active:
        return jsonify({"error": "No processing active"}), 400

    session.pause_requested = False
    session.pause_event.set()  # Unblock the worker
    socketio.emit("processing:resumed", {})
    return jsonify({"ok": True})


@processing_bp.route("/stop", methods=["POST"])
def stop_processing():
    """Cancel processing — stop and keep partial results."""
    if not session.processing_active:
        return jsonify({"error": "No processing active"}), 400

    session.stop_requested = True
    session.pause_event.set()  # Unblock if paused, so thread can exit
    return jsonify({"ok": True})


@processing_bp.route("/status", methods=["GET"])
def get_status():
    """Return current processing status."""
    return jsonify({
        "active": session.processing_active,
        "paused": session.pause_requested,
        "has_results": len(session.displacement_results) > 0,
        "num_frames": len(session.displacement_results),
    })
