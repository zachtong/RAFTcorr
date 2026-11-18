"""Helpers for converting numpy arrays to JSON-safe and PNG formats."""

import io
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from flask import Response


def ndarray_to_png_bytes(
    arr: np.ndarray,
    colormap: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    jpeg: bool = False,
) -> bytes:
    """Convert a 2D numpy array to image bytes using matplotlib colormap + PIL."""
    from PIL import Image
    import matplotlib
    import matplotlib.colors as mcolors

    finite = arr[np.isfinite(arr)]
    if vmin is None:
        vmin = float(finite.min()) if finite.size > 0 else 0.0
    if vmax is None:
        vmax = float(finite.max()) if finite.size > 0 else 1.0
    if vmin >= vmax:
        vmax = vmin + 1e-10

    cmap = matplotlib.colormaps[colormap]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colored = cmap(norm(np.nan_to_num(arr, nan=0.0)))  # (H, W, 4) float [0,1]
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)

    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    if jpeg:
        pil_img.save(buf, format="JPEG", quality=80)
    else:
        pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def image_to_png_bytes(img: np.ndarray, jpeg: bool = False) -> bytes:
    """Convert an image array (grayscale or RGB) to image bytes using PIL."""
    from PIL import Image

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    if img.ndim == 2:
        pil_img = Image.fromarray(img)
    else:
        pil_img = Image.fromarray(img)

    buf = io.BytesIO()
    if jpeg:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img.save(buf, format="JPEG", quality=80)
    else:
        pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def image_response(img_bytes: bytes, jpeg: bool = False) -> Response:
    """Wrap image bytes in a Flask Response with cache headers."""
    mimetype = "image/jpeg" if jpeg else "image/png"
    resp = Response(img_bytes, mimetype=mimetype)
    resp.headers["Cache-Control"] = "private, max-age=30"
    return resp


def png_response(png_bytes: bytes) -> Response:
    """Wrap image bytes in a Flask Response. Auto-converts to JPEG when
    RAFTCORR_COMPRESS=1 is set (e.g. on Colab) to reduce transfer size."""
    if os.environ.get("RAFTCORR_COMPRESS") == "1":
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        return image_response(buf.read(), jpeg=True)
    return image_response(png_bytes, jpeg=False)


def data_texture_response(
    png_bytes: bytes, data_min: float, data_max: float,
) -> Response:
    """Wrap data texture PNG bytes with min/max metadata headers."""
    resp = Response(png_bytes, mimetype="image/png")
    resp.headers["Cache-Control"] = "private, max-age=30"
    resp.headers["X-Data-Min"] = str(data_min)
    resp.headers["X-Data-Max"] = str(data_max)
    resp.headers["Access-Control-Expose-Headers"] = "X-Data-Min, X-Data-Max"
    return resp


def ndarray_to_json(arr: np.ndarray) -> Dict[str, Any]:
    """Convert a numpy array to a JSON-serializable dict."""
    return {
        "data": arr.tolist(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def frame_data_to_json(
    arr: np.ndarray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Dict[str, Any]:
    """Convert frame displacement/strain data to JSON with stats."""
    finite = arr[np.isfinite(arr)]
    return {
        "data": arr.tolist(),
        "shape": list(arr.shape),
        "vmin": float(vmin if vmin is not None else (finite.min() if finite.size > 0 else 0)),
        "vmax": float(vmax if vmax is not None else (finite.max() if finite.size > 0 else 0)),
        "mean": float(finite.mean()) if finite.size > 0 else 0,
    }
