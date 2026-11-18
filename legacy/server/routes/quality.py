"""Data quality metrics -- NCC between warped reference and deformed image."""

import os

import cv2
import numpy as np
from flask import Blueprint, jsonify

from raft_dic_gui.processing import load_and_convert_image
from server.session import session

quality_bp = Blueprint("quality", __name__)


@quality_bp.route("/ncc/<int:idx>", methods=["GET"])
def get_ncc(idx: int):
    """Compute NCC between warped reference and deformed frame.

    High NCC (~1.0) means the displacement field correctly maps
    reference to deformed. Low NCC indicates tracking errors.
    """
    if not session.displacement_results:
        return jsonify({"error": "No displacement results"}), 404
    if idx < 0 or idx >= len(session.displacement_results):
        return jsonify({"error": "Frame index out of range"}), 400

    ref_img = session.reference_image
    if ref_img is None:
        return jsonify({"error": "No reference image"}), 400

    if idx + 1 >= len(session.image_files):
        return jsonify({"error": "No deformed image for this frame"}), 400

    # Load deformed image
    def_path = os.path.join(session.image_dir, session.image_files[idx + 1])
    def_img = load_and_convert_image(def_path)

    # Convert to grayscale float64
    if ref_img.ndim == 3:
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        ref_gray = ref_img.astype(np.float64)
    if def_img.ndim == 3:
        def_gray = cv2.cvtColor(def_img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    else:
        def_gray = def_img.astype(np.float64)

    disp = session.displacement_results[idx]
    u, v = disp[:, :, 0], disp[:, :, 1]
    h, w = u.shape

    # ROI region
    if session.roi_rect:
        x0, y0, x1, y1 = session.roi_rect
    else:
        x0, y0 = 0, 0

    # Warp reference using displacement field
    yy, xx = np.mgrid[y0:y0 + h, x0:x0 + w]
    map_x = (xx + u).astype(np.float32)
    map_y = (yy + v).astype(np.float32)
    warped_ref = cv2.remap(
        ref_gray.astype(np.float32), map_x, map_y,
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Crop deformed to same region
    def_crop = def_gray[y0:y0 + h, x0:x0 + w].astype(np.float32)

    # Valid region mask
    valid = np.isfinite(u) & np.isfinite(v) & (warped_ref > 0)
    if not np.any(valid):
        return jsonify({"ncc": 0.0, "frame": idx})

    a = warped_ref[valid] - warped_ref[valid].mean()
    b = def_crop[valid] - def_crop[valid].mean()
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    ncc = float(np.sum(a * b) / (denom + 1e-20))

    return jsonify({"ncc": round(ncc, 6), "frame": idx})
