"""ROI mask creation endpoints.

Supports per-frame ROI: every endpoint accepts an optional ``frame_idx``
(0-based) parameter.  When omitted or 0, the endpoint operates on the
primary ROI (``session.roi_mask``) — fully backward compatible.
When > 0, it operates on ``session.per_frame_rois[frame_idx]``.
"""

import io
import os
import re

import cv2
import numpy as np
from flask import Blueprint, jsonify, request

from server.serializers import png_response
from server.session import session

roi_bp = Blueprint("roi", __name__)


# ---------------------------------------------------------------------------
# Internal helpers — frame-0 (primary ROI)
# ---------------------------------------------------------------------------

def _image_hw():
    """Return (h, w) from session state, or (0, 0) if unavailable."""
    if session.image_height > 0 and session.image_width > 0:
        return session.image_height, session.image_width
    if session.reference_image is not None:
        return session.reference_image.shape[:2]
    return 0, 0


def _ensure_mask():
    """Ensure roi_mask is initialized to the reference image size."""
    h, w = _image_hw()
    if h == 0:
        return False
    if session.roi_mask is None or session.roi_mask.shape != (h, w):
        session.roi_mask = np.zeros((h, w), dtype=bool)
    return True


def _update_rect():
    """Recompute bounding rect from current mask."""
    if session.roi_mask is None:
        session.roi_rect = None
        return
    y_idx, x_idx = np.where(session.roi_mask)
    if len(x_idx) > 0:
        session.roi_rect = (
            int(x_idx.min()),
            int(y_idx.min()),
            int(x_idx.max()) + 1,
            int(y_idx.max()) + 1,
        )
    else:
        h, w = session.roi_mask.shape
        session.roi_rect = (0, 0, w, h)


def _apply_shape(new_mask_bool: np.ndarray, mode: str):
    """Apply a shape mask using add or cut mode (frame 0 only)."""
    with session._lock:
        if not _ensure_mask():
            return jsonify({"error": "No reference image loaded"}), 400
        if mode == "cut":
            session.roi_mask = session.roi_mask & ~new_mask_bool
        else:
            session.roi_mask = session.roi_mask | new_mask_bool
        _update_rect()
        # Keep per_frame_rois[0] in sync
        session.per_frame_rois[0] = session.roi_mask

    area = int(session.roi_mask.sum())
    return jsonify({"rect": session.roi_rect, "area_px": area})


# ---------------------------------------------------------------------------
# Internal helpers — per-frame ROI
# ---------------------------------------------------------------------------

def _set_frame_mask(frame_idx: int, mask: np.ndarray):
    """Set the ROI mask for a specific frame and sync frame-0 with roi_mask."""
    session.per_frame_rois[frame_idx] = mask
    if frame_idx == 0:
        session.roi_mask = mask
        _update_rect()


def _get_frame_mask(frame_idx: int):
    """Get the ROI mask for a frame, or None."""
    if frame_idx == 0:
        return session.roi_mask
    return session.per_frame_rois.get(frame_idx)


def _ensure_frame_mask(frame_idx: int) -> bool:
    """Ensure a writable mask buffer exists for *frame_idx*."""
    h, w = _image_hw()
    if h == 0:
        return False
    if frame_idx == 0:
        return _ensure_mask()
    if frame_idx not in session.per_frame_rois:
        session.per_frame_rois[frame_idx] = np.zeros((h, w), dtype=bool)
    return True


def _apply_shape_for_frame(new_mask_bool: np.ndarray, mode: str, frame_idx: int):
    """Apply a shape to a specific frame's mask."""
    if frame_idx == 0:
        return _apply_shape(new_mask_bool, mode)

    with session._lock:
        if not _ensure_frame_mask(frame_idx):
            return jsonify({"error": "No reference image loaded"}), 400
        mask = session.per_frame_rois[frame_idx]
        if mode == "cut":
            result = mask & ~new_mask_bool
        else:
            result = mask | new_mask_bool
        _set_frame_mask(frame_idx, result)

    area = int(result.sum())
    return jsonify({"area_px": area, "frame_idx": frame_idx})


# ---------------------------------------------------------------------------
# Shape endpoints (polygon / rectangle / circle)
# ---------------------------------------------------------------------------

@roi_bp.route("/polygon", methods=["POST"])
def add_polygon():
    """Add or cut a polygon from the ROI mask."""
    data = request.get_json(force=True)
    points = data.get("points", [])
    mode = data.get("mode", "add")
    frame_idx = int(data.get("frame_idx", 0))

    if not points or len(points) < 3:
        return jsonify({"error": "Need at least 3 points"}), 400

    h, w = _image_hw()
    if h == 0:
        return jsonify({"error": "No images loaded"}), 400

    pts = np.array(points, np.int32)
    new_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(new_mask, [pts], 1)

    return _apply_shape_for_frame(new_mask.astype(bool), mode, frame_idx)


@roi_bp.route("/rectangle", methods=["POST"])
def add_rectangle():
    """Add or cut a rectangle from the ROI mask."""
    data = request.get_json(force=True)
    mode = data.get("mode", "add")
    frame_idx = int(data.get("frame_idx", 0))

    h, w = _image_hw()
    if h == 0:
        return jsonify({"error": "No images loaded"}), 400

    x0 = int(data.get("x0", 0))
    y0 = int(data.get("y0", 0))
    x1 = int(data.get("x1", w))
    y1 = int(data.get("y1", h))

    new_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(new_mask, (x0, y0), (x1, y1), 1, -1)

    return _apply_shape_for_frame(new_mask.astype(bool), mode, frame_idx)


@roi_bp.route("/circle", methods=["POST"])
def add_circle():
    """Add or cut a circle from the ROI mask."""
    data = request.get_json(force=True)
    mode = data.get("mode", "add")
    frame_idx = int(data.get("frame_idx", 0))

    h, w = _image_hw()
    if h == 0:
        return jsonify({"error": "No images loaded"}), 400

    cx = int(data.get("cx", 0))
    cy = int(data.get("cy", 0))
    r = int(data.get("r", 0))

    if r <= 0:
        return jsonify({"error": "Radius must be positive"}), 400

    new_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(new_mask, (cx, cy), r, 1, -1)

    return _apply_shape_for_frame(new_mask.astype(bool), mode, frame_idx)


# ---------------------------------------------------------------------------
# Import / export / invert
# ---------------------------------------------------------------------------

@roi_bp.route("/import", methods=["POST"])
def import_mask():
    """Import an ROI mask from an image file.

    Optional parameters:
    - min_area (int, default 0): remove connected components smaller than this.
    - smooth_radius (int, default 0): Gaussian blur radius before binarization.
      Kernel size = smooth_radius * 2 + 1.
    - frame_idx (int, default 0): target frame (0-based).
    """
    data = request.get_json(force=True)
    path = data.get("path", "").strip()
    min_area = int(data.get("min_area", 0))
    smooth_radius = int(data.get("smooth_radius", 0))
    frame_idx = int(data.get("frame_idx", 0))

    if not path:
        return jsonify({"error": "Missing path"}), 400

    h, w = _image_hw()
    if h == 0:
        return jsonify({"error": "No images loaded"}), 400

    mask_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return jsonify({"error": "Failed to read mask image"}), 400

    # 1. Resize to match image dimensions
    mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

    # 2. Optional smoothing (Gaussian blur before binarization)
    if smooth_radius > 0:
        ksize = smooth_radius * 2 + 1
        mask_img = cv2.GaussianBlur(mask_img, (ksize, ksize), 0)

    # 3. Binarize
    binary = (mask_img > 127).astype(np.uint8)

    # 4. Optional small-component removal
    if min_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] < min_area:
                binary[labels == label_id] = 0

    mask_bool = binary.astype(bool)
    with session._lock:
        _set_frame_mask(frame_idx, mask_bool)
        if frame_idx == 0:
            session.roi_confirmed = True

    area = int(mask_bool.sum())
    return jsonify({"rect": session.roi_rect, "area_px": area, "frame_idx": frame_idx})


@roi_bp.route("/mask/upload", methods=["POST"])
def upload_mask():
    """Replace a frame's ROI mask with an uploaded binary PNG.

    Used by the frontend undo/redo stack: it snapshots the current mask
    locally, then uploads the previous snapshot on Ctrl+Z.

    Multipart fields:
    - mask: binary PNG file (any size — nearest-resized to image shape)
    - frame_idx: target frame (default 0)
    - empty: if "1", clear the mask for this frame instead of reading mask
    """
    frame_idx = int(request.form.get("frame_idx", 0))
    empty = request.form.get("empty", "0") == "1"

    h, w = _image_hw()
    if h == 0:
        return jsonify({"error": "No images loaded"}), 400

    if empty:
        mask_bool = np.zeros((h, w), dtype=bool)
    else:
        file = request.files.get("mask")
        if file is None:
            return jsonify({"error": "Missing mask file"}), 400
        buf = np.frombuffer(file.read(), np.uint8)
        mask_img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            return jsonify({"error": "Failed to decode uploaded mask"}), 400
        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = (mask_img > 127)

    with session._lock:
        _set_frame_mask(frame_idx, mask_bool)
        if frame_idx == 0:
            # Frame 0 is auto-confirmed when any ROI exists; empty mask
            # un-confirms so downstream gating behaves correctly.
            session.roi_confirmed = bool(mask_bool.any())

    area = int(mask_bool.sum())
    return jsonify({
        "rect": session.roi_rect,
        "area_px": area,
        "frame_idx": frame_idx,
    })


@roi_bp.route("/invert", methods=["POST"])
def invert_mask():
    """Invert the ROI mask for a given frame (default frame 0)."""
    data = request.get_json(silent=True) or {}
    frame_idx = int(data.get("frame_idx", 0))

    if frame_idx == 0:
        if session.roi_mask is None:
            return jsonify({"error": "No ROI mask exists"}), 400
        with session._lock:
            session.roi_mask = ~session.roi_mask
            _update_rect()
            session.per_frame_rois[0] = session.roi_mask
        area = int(session.roi_mask.sum())
    else:
        mask = session.per_frame_rois.get(frame_idx)
        if mask is None:
            return jsonify({"error": f"No ROI mask for frame {frame_idx}"}), 400
        with session._lock:
            session.per_frame_rois[frame_idx] = ~mask
        area = int((~mask).sum())

    return jsonify({"rect": session.roi_rect, "area_px": area, "frame_idx": frame_idx})


@roi_bp.route("/mask/binary", methods=["GET"])
def export_mask_binary():
    """Export the current ROI mask as a binary (white-on-black) PNG for saving."""
    from PIL import Image

    frame_idx = request.args.get("frame_idx", 0, type=int)
    mask = _get_frame_mask(frame_idx)
    if mask is None:
        return jsonify({"error": "No ROI mask exists"}), 404

    mask_uint8 = mask.astype(np.uint8) * 255
    pil_img = Image.fromarray(mask_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    resp = png_response(buf.read())
    resp.headers["Content-Disposition"] = (
        f'attachment; filename="roi_mask_frame{frame_idx}.png"'
    )
    return resp


@roi_bp.route("/mask", methods=["GET"])
def get_mask():
    """Return an ROI mask as a colored RGBA PNG overlay.

    Query params:
    - frame_idx (int, default 0): which frame's mask to render.
    """
    from PIL import Image

    frame_idx = request.args.get("frame_idx", 0, type=int)
    mask = _get_frame_mask(frame_idx)
    if mask is None:
        return jsonify({"error": "No ROI mask exists"}), 404

    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Blue highlight where mask is True
    rgba[mask, 0] = 74   # R
    rgba[mask, 1] = 108  # G
    rgba[mask, 2] = 247  # B (matches --primary)
    rgba[mask, 3] = 80   # Alpha

    pil_img = Image.fromarray(rgba)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return png_response(buf.read())


# ---------------------------------------------------------------------------
# Confirm / clear
# ---------------------------------------------------------------------------

@roi_bp.route("/confirm", methods=["POST"])
def confirm_roi():
    """Finalize the ROI for processing."""
    if session.roi_mask is None or session.roi_mask.sum() == 0:
        return jsonify({"error": "No ROI mask to confirm"}), 400

    with session._lock:
        _update_rect()
        session.roi_confirmed = True
        session.per_frame_rois[0] = session.roi_mask

    area = int(session.roi_mask.sum())
    return jsonify({
        "ok": True,
        "rect": session.roi_rect,
        "area_px": area,
    })


@roi_bp.route("", methods=["DELETE"])
def clear_roi():
    """Clear the ROI mask (frame 0 only — keeps other per-frame masks)."""
    with session._lock:
        session.roi_mask = None
        session.roi_rect = None
        session.roi_confirmed = False
        session.per_frame_rois.pop(0, None)

    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Per-frame endpoints
# ---------------------------------------------------------------------------

@roi_bp.route("/frame/<int:frame_idx>", methods=["DELETE"])
def clear_frame_roi(frame_idx: int):
    """Clear ROI for a specific frame."""
    with session._lock:
        if frame_idx == 0:
            session.roi_mask = None
            session.roi_rect = None
            session.roi_confirmed = False
        session.per_frame_rois.pop(frame_idx, None)
    return jsonify({"ok": True})


@roi_bp.route("/frames/clear-all", methods=["DELETE"])
def clear_all_frame_rois():
    """Clear ROI masks for ALL frames (including frame 0)."""
    with session._lock:
        session.roi_mask = None
        session.roi_rect = None
        session.roi_confirmed = False
        session.per_frame_rois.clear()
    return jsonify({"ok": True})


@roi_bp.route("/frames/status", methods=["GET"])
def frames_roi_status():
    """Return which frames have ROI masks."""
    return jsonify({
        "total_frames": len(session.image_files),
        "frames_with_roi": sorted(session.per_frame_rois.keys()),
        "frame_0_confirmed": session.roi_confirmed,
    })


@roi_bp.route("/frames/batch-import", methods=["POST"])
def batch_import_rois():
    """Batch import ROI masks from a folder.

    JSON body:
    - folder: str — path to folder containing mask images
    - strategy: "auto_match" | "sequential"
      auto_match: extract last number from filename → frame index (1-based → 0-based)
      sequential: 1st file → frame 0, 2nd → frame 1, …

    Returns: {imported, total_files, assignments}
    """
    data = request.get_json(force=True)
    folder = data.get("folder", "").strip()
    strategy = data.get("strategy", "sequential")

    if not folder or not os.path.isdir(folder):
        return jsonify({"error": "Invalid folder path"}), 400

    h, w = _image_hw()
    if h == 0:
        return jsonify({"error": "No images loaded"}), 400

    from pathlib import Path
    from raft_dic_gui.mask_loader import MASK_EXTENSIONS, _natural_sort_key

    mask_path = Path(folder)
    mask_files = sorted(
        (f for f in mask_path.iterdir()
         if f.is_file() and f.suffix.lower() in MASK_EXTENSIONS),
        key=lambda f: _natural_sort_key(f.name),
    )

    num_images = len(session.image_files)
    assignments = {}
    imported = 0

    for file_idx, mask_file in enumerate(mask_files):
        # Determine target frame index
        if strategy == "auto_match":
            numbers = re.findall(r"\d+", mask_file.stem)
            if not numbers:
                continue
            # Use last number in filename as 0-based frame index directly
            target_idx = int(numbers[-1])
        else:  # sequential
            target_idx = file_idx

        if target_idx >= num_images:
            continue

        # Load mask
        mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            continue
        mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = (mask_img > 127).astype(bool)

        with session._lock:
            _set_frame_mask(target_idx, mask_bool)

        assignments[target_idx] = mask_file.name
        imported += 1

    # If frame 0 was imported, auto-confirm ROI
    if 0 in assignments:
        with session._lock:
            session.roi_confirmed = True

    return jsonify({
        "imported": imported,
        "total_files": len(mask_files),
        "assignments": {str(k): v for k, v in sorted(assignments.items())},
    })
