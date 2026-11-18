"""Image loading and serving endpoints."""

import os
import re

from flask import Blueprint, jsonify, request

from raft_dic_gui.processing import load_and_convert_image
from server.serializers import image_to_png_bytes, png_response
from server.session import session

images_bp = Blueprint("images", __name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@images_bp.route("/browse", methods=["POST"])
def browse_directory():
    """List contents of a directory for the file browser."""
    data = request.get_json(force=True)
    path = data.get("path", "").strip()
    # Optional: extra file extensions to include (e.g. [".raftproj"])
    extra_extensions = data.get("file_extensions", [])
    extra_ext_tuple = tuple(ext.lower() for ext in extra_extensions) if extra_extensions else ()

    # Default: return filesystem roots / home
    if not path:
        # On Colab, files live under /content/ (not /root/)
        if os.path.isdir("/content"):
            path = "/content"
        else:
            path = os.path.expanduser("~")

    if not os.path.isdir(path):
        return jsonify({"error": f"Not a directory: {path}"}), 400

    entries = []
    try:
        for name in sorted(os.listdir(path), key=_natural_sort_key):
            full = os.path.join(path, name)
            if name.startswith("."):
                continue
            if os.path.isdir(full):
                entries.append({"name": name, "type": "dir"})
            elif name.lower().endswith(IMAGE_EXTENSIONS):
                entries.append({"name": name, "type": "image"})
            elif extra_ext_tuple and name.lower().endswith(extra_ext_tuple):
                entries.append({"name": name, "type": "file"})
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403

    # Count images in this directory
    image_count = sum(1 for e in entries if e["type"] == "image")

    return jsonify({
        "path": os.path.abspath(path),
        "parent": os.path.dirname(os.path.abspath(path)),
        "entries": entries,
        "image_count": image_count,
    })


def _natural_sort_key(s: str):
    """Split filename into text/number parts for natural ordering."""
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", s)]


@images_bp.route("/load", methods=["POST"])
def load_images():
    """Load images from a directory path."""
    data = request.get_json(force=True)
    directory = data.get("dir", "").strip()
    natural_sort = data.get("natural_sort", False)

    if not directory or not os.path.isdir(directory):
        return jsonify({"error": "Invalid or missing directory"}), 400

    image_files = [
        f for f in os.listdir(directory)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    files = sorted(image_files, key=_natural_sort_key if natural_sort else str.lower)
    if not files:
        return jsonify({"error": "No supported image files found in directory"}), 400

    # Load reference (first) image
    ref_path = os.path.join(directory, files[0])
    ref_img = load_and_convert_image(ref_path)

    with session._lock:
        # Reset previous session state (ROI, probes, results, caches)
        session.reset()

        # Clear render caches so stale PNG bytes from previous dataset are freed
        from server.routes.displacement import _render_cache as disp_cache
        from server.routes.strain import _render_cache as strain_cache
        from server.routes.arrows import _render_cache as arrow_cache
        disp_cache.clear()
        strain_cache.clear()
        arrow_cache.clear()

        session.image_dir = directory
        session.image_files = files
        session.reference_image = ref_img
        session.image_height, session.image_width = ref_img.shape[:2]
        session.config.img_dir = directory

        # Set up deformed view cache
        image_paths = [os.path.join(directory, f) for f in files]
        session.deformed_view_cache.set_image_paths(image_paths)
        session.deformed_view_cache.set_load_function(load_and_convert_image)

    # Detect if filenames contain numbers (natural sort recommended)
    has_numeric = any(re.search(r"\d", f) for f in files)
    sort_suggestion = "natural" if (has_numeric and not natural_sort) else None

    resp = {
        "files": files,
        "count": len(files),
        "width": session.image_width,
        "height": session.image_height,
    }
    if sort_suggestion:
        resp["sort_suggestion"] = sort_suggestion
    return jsonify(resp)


@images_bp.route("/reference", methods=["GET"])
def get_reference():
    """Return the reference (first) image as PNG."""
    if session.reference_image is None:
        return jsonify({"error": "No images loaded"}), 404

    png_bytes = image_to_png_bytes(session.reference_image)
    return png_response(png_bytes)


@images_bp.route("/frame/<int:index>", methods=["GET"])
def get_frame(index: int):
    """Return a specific frame image as PNG."""
    if not session.image_files:
        return jsonify({"error": "No images loaded"}), 404

    if index < 0 or index >= len(session.image_files):
        return jsonify({"error": f"Frame index {index} out of range [0, {len(session.image_files) - 1})"}), 400

    # Check in-memory cache first
    cached = session.get_cached_image(index)
    if cached is not None:
        return png_response(image_to_png_bytes(cached))

    # Load from disk and cache
    img_path = os.path.join(session.image_dir, session.image_files[index])
    img = load_and_convert_image(img_path)
    session.cache_image(index, img)
    png_bytes = image_to_png_bytes(img)
    return png_response(png_bytes)
