"""Tiling preview and GPU info endpoint."""

import numpy as np
from flask import Blueprint, jsonify

from server.session import session

tiling_bp = Blueprint("tiling", __name__)


def _generate_tile_starts(total_len: int, tile_len: int, overlap: int) -> list:
    """Generate start positions for sliding window with fixed overlap.

    Mirrors the logic in raft_dic_gui/processing.py.
    """
    if total_len <= tile_len:
        return [0]
    stride = tile_len - overlap
    if stride <= 0:
        stride = max(1, tile_len // 4)
    starts = []
    pos = 0
    while pos + tile_len < total_len:
        starts.append(pos)
        pos += stride
    if starts[-1] + tile_len < total_len:
        starts.append(total_len - tile_len)
    return sorted(list(set(starts)))


def _compute_tiling_preview(image_w: int, image_h: int,
                            roi_rect: tuple, roi_mask,
                            p_max_pixels: int,
                            tile_overlap: int,
                            context_padding: int) -> dict:
    """Compute tiling grid preview from current session state."""

    # ROI bounding box
    xmin, ymin, xmax, ymax = roi_rect
    roi_w = xmax - xmin
    roi_h = ymax - ymin

    # If we have a mask, compute tight bbox
    if roi_mask is not None:
        ys, xs = np.where(roi_mask)
        if len(xs) > 0:
            x0, y0 = int(xs.min()), int(ys.min())
            m, n = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
        else:
            x0, y0, m, n = xmin, ymin, roi_w, roi_h
    else:
        x0, y0, m, n = xmin, ymin, roi_w, roi_h

    # Expand bbox with context padding
    pad = context_padding
    xC = max(0, x0 - pad)
    yC = max(0, y0 - pad)
    xR = min(image_w, x0 + m + pad)
    yB = min(image_h, y0 + n + pad)
    wC = xR - xC
    hC = yB - yC

    # Tiling decision
    max_T = int(np.sqrt(p_max_pixels))

    if wC * hC <= p_max_pixels:
        # Single shot
        tiles = [[0, 0, wC, hC]]
        tile_size = max(wC, hC)
        is_single_shot = True
    else:
        is_single_shot = False
        T = max_T
        eff_overlap = tile_overlap
        if T < eff_overlap * 2 + 16:
            eff_overlap = max(4, T // 4)

        starts_x = _generate_tile_starts(wC, T, eff_overlap)
        starts_y = _generate_tile_starts(hC, T, eff_overlap)

        tiles = []
        for sy in starts_y:
            for sx in starts_x:
                tw = min(T, wC - sx)
                th = min(T, hC - sy)
                tiles.append([sx, sy, tw, th])
        tile_size = T

    # Overlap ratio
    if len(tiles) > 1 and tile_size > 0:
        overlap_ratio = tile_overlap / tile_size
    else:
        overlap_ratio = 0.0

    # GPU info
    gpu_info = _get_gpu_info()

    # Estimate memory per tile (rough)
    est_memory_per_tile_mb = _estimate_tile_memory(tile_size, session.config.model_metadata)

    # Estimate time
    num_frames = max(1, len(session.image_files) - 1)
    est_time = _estimate_time(tile_size, len(tiles), num_frames)

    return {
        "image_size": [image_w, image_h],
        "roi_bbox": [xmin, ymin, roi_w, roi_h],
        "work_area": {"x": xC, "y": yC, "w": wC, "h": hC},
        "tile_size": tile_size,
        "tiles": tiles,
        "tile_count": len(tiles),
        "overlap_ratio": round(overlap_ratio, 3),
        "gpu": gpu_info,
        "est_memory_per_tile_mb": est_memory_per_tile_mb,
        "est_time_seconds": est_time,
        "is_single_shot": is_single_shot,
    }


def _get_gpu_info() -> dict:
    """Get GPU name and memory info."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            return {
                "name": name,
                "total_mb": round(total_mem / 1024 / 1024),
                "free_mb": round(free_mem / 1024 / 1024),
            }
    except Exception:
        pass
    return {"name": "CPU", "total_mb": 0, "free_mb": 0}


def _estimate_tile_memory(tile_size: int, model_metadata) -> int:
    """Rough estimate of memory per tile in MB.

    Alpha is bytes per (total_pixels)^2, where total_pixels = tile_size^2.
    So memory = alpha * (tile_size^2)^2 = alpha * tile_size^4.
    """
    if model_metadata and hasattr(model_metadata, 'full_resolution') and model_metadata.full_resolution:
        # RAFT-Fine: ~5 bytes per pixel^2
        alpha = 5.0
    else:
        # RAFT-Large: ~0.0013 bytes per pixel^2
        alpha = 0.0013

    total_pixels = tile_size * tile_size
    mem_bytes = alpha * (total_pixels ** 2)
    return round(mem_bytes / 1024 / 1024)


def _estimate_time(tile_size: int, num_tiles: int, num_frames: int) -> float:
    """Rough time estimate in seconds.

    Uses empirical per-pixel-pair time constants.
    These are rough estimates and will vary by GPU.
    """
    # Rough constant: ~0.5 microseconds per pixel for RAFT inference
    per_pixel_us = 0.5
    pixels_per_tile = tile_size * tile_size
    time_per_tile = pixels_per_tile * per_pixel_us / 1e6  # seconds
    total = time_per_tile * num_tiles * num_frames
    return round(total, 1)


@tiling_bp.route("/preview", methods=["GET"])
def tiling_preview():
    """Return tiling grid preview based on current session state.

    Requires: images loaded, ROI confirmed, model selected.
    """
    if not session.image_files:
        return jsonify({"error": "No images loaded"}), 400

    if session.roi_mask is None or not session.roi_confirmed:
        return jsonify({"error": "ROI not confirmed"}), 400

    if not session.config.model_path:
        return jsonify({"error": "No model selected"}), 400

    if session.roi_rect is None:
        return jsonify({"error": "ROI rect not set"}), 400

    result = _compute_tiling_preview(
        session.image_width,
        session.image_height,
        session.roi_rect,
        session.roi_mask,
        session.config.p_max_pixels,
        session.config.tile_overlap,
        session.config.context_padding,
    )

    return jsonify(result)
