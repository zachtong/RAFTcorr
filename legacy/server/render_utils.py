"""Shared rendering utilities for displacement and strain overlay endpoints."""

import io
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def fill_mask_nan_gaps(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill NaN gaps inside *mask* using nearest valid neighbor value.

    In reference-background mode, displacement/strain data may have NaN pixels
    inside frame 0's ROI mask due to coordinate rounding in
    ``_apply_current_frame_mask``.  This function fills those gaps so the
    visible ROI shape exactly matches the mask boundary.

    Pixels outside the mask remain NaN (transparent).
    """
    has_value = np.isfinite(data) & mask
    gap = mask & ~has_value

    if not gap.any() or not has_value.any():
        return data

    from scipy.ndimage import distance_transform_edt

    # distance_transform_edt(~has_value) gives, for every pixel, the distance
    # to the nearest pixel where has_value is True.  return_indices gives the
    # coordinates of that nearest pixel.
    _, nearest_idx = distance_transform_edt(~has_value, return_indices=True)

    result = data.copy()
    gap_rows, gap_cols = np.where(gap)
    src_rows = nearest_idx[0][gap_rows, gap_cols]
    src_cols = nearest_idx[1][gap_rows, gap_cols]
    result[gap_rows, gap_cols] = data[src_rows, src_cols]

    return result


def _auto_range(
    valid_vals: np.ndarray,
    vmin: Optional[float],
    vmax: Optional[float],
) -> Tuple[float, float]:
    """Compute vmin/vmax from data if not provided."""
    if vmin is None:
        vmin = float(valid_vals.min()) if valid_vals.size > 0 else 0.0
    if vmax is None:
        vmax = float(valid_vals.max()) if valid_vals.size > 0 else 1.0
    if vmin >= vmax:
        vmax = vmin + 1e-10
    return vmin, vmax


def _build_norm(
    vmin: float, vmax: float, log_scale: bool,
):
    """Build a matplotlib Normalize (linear or log)."""
    import matplotlib.colors as mcolors

    if log_scale:
        log_vmin = vmin if vmin > 0 else 1e-10
        log_vmax = vmax if vmax > 0 else 1.0
        if log_vmin >= log_vmax:
            log_vmin = log_vmax / 1000
        return mcolors.LogNorm(vmin=log_vmin, vmax=log_vmax)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


def _viewport_downsample_data(
    full_data: np.ndarray, h: int, w: int, vw: int, vh: int,
) -> Tuple[np.ndarray, int, int]:
    """NaN-safe downsample scalar data to viewport size. Returns (data, h, w)."""
    if vw <= 0 or vh <= 0:
        return full_data, h, w
    scale = min(vw / w, vh / h, 1.0)
    if scale >= 1.0:
        return full_data, h, w
    out_w, out_h = max(1, int(w * scale)), max(1, int(h * scale))
    mask_f = np.isfinite(full_data).astype(np.float32)
    filled = np.nan_to_num(full_data, nan=0.0).astype(np.float32)
    data_ds = cv2.resize(filled, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float64)
    mask_ds = cv2.resize(mask_f, (out_w, out_h), interpolation=cv2.INTER_AREA)
    data_ds[mask_ds < 0.5] = np.nan
    return data_ds, out_h, out_w


def render_overlay_png(
    full_data: np.ndarray,
    colormap: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    log_scale: bool = False,
    vw: int = 0,
    vh: int = 0,
) -> bytes:
    """Render a scalar field as an RGBA overlay PNG (alpha = valid mask).

    Returns PNG bytes ready for HTTP response.
    """
    import matplotlib

    h, w = full_data.shape[:2]
    full_data, h, w = _viewport_downsample_data(full_data, h, w, vw, vh)

    mask = ~np.isnan(full_data)
    valid_vals = full_data[mask]
    vmin, vmax = _auto_range(valid_vals, vmin, vmax)

    cmap = matplotlib.colormaps[colormap]
    norm = _build_norm(vmin, vmax, log_scale)
    normalized = norm(np.nan_to_num(full_data, nan=0.0))
    colored = cmap(normalized)
    colored[:, :, 3] = mask.astype(np.float64)
    overlay_rgba = (colored * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay_rgba).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def render_data_texture_png(
    full_data: np.ndarray,
    vw: int = 0,
    vh: int = 0,
) -> Tuple[bytes, float, float]:
    """Encode scalar data as a 16-bit-precision RGBA data texture PNG.

    Encoding per pixel:
      R = high byte of 16-bit normalized value
      G = low byte of 16-bit normalized value
      B = 0
      A = 255 (valid) or 0 (NaN)

    Returns (png_bytes, data_min, data_max).
    The client reconstructs: value = data_min + ((R*256+G)/65535) * (data_max - data_min)
    """
    h, w = full_data.shape[:2]
    full_data, h, w = _viewport_downsample_data(full_data, h, w, vw, vh)

    mask = np.isfinite(full_data)
    valid = full_data[mask]
    data_min = float(valid.min()) if valid.size > 0 else 0.0
    data_max = float(valid.max()) if valid.size > 0 else 1.0

    data_range = data_max - data_min
    if data_range < 1e-15:
        data_range = 1.0
        data_max = data_min + data_range  # ensure client sees non-zero range

    normalized = np.clip(
        (np.nan_to_num(full_data, nan=0.0) - data_min) / data_range,
        0.0, 1.0,
    )
    val_16 = (normalized * 65535 + 0.5).astype(np.uint16)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = (val_16 >> 8).astype(np.uint8)
    rgba[:, :, 1] = (val_16 & 0xFF).astype(np.uint8)
    rgba[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG", compress_level=1)
    buf.seek(0)
    return buf.read(), data_min, data_max


def get_colormap_lut(name: str) -> bytes:
    """Return a 256×1 RGBA PNG encoding the colormap LUT."""
    import matplotlib

    cmap = matplotlib.colormaps[name]
    lut = np.array([cmap(i / 255) for i in range(256)])
    lut_u8 = (lut * 255).astype(np.uint8).reshape(256, 1, 4)

    buf = io.BytesIO()
    Image.fromarray(lut_u8).save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def render_composited_png(
    full_data: np.ndarray,
    bg_img: np.ndarray,
    colormap: str = "turbo",
    alpha: float = 0.7,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    log_scale: bool = False,
    vw: int = 0,
    vh: int = 0,
) -> bytes:
    """Render a scalar field composited on a background image as PNG.

    Returns PNG bytes ready for HTTP response.
    """
    import matplotlib
    from server.viewport import downsample_for_viewport

    h, w = bg_img.shape[:2]
    full_data, bg_img, h, w = downsample_for_viewport(full_data, bg_img, vw, vh)

    mask = ~np.isnan(full_data)
    valid_vals = full_data[mask]
    vmin, vmax = _auto_range(valid_vals, vmin, vmax)

    # Background → RGB PIL
    if bg_img.ndim == 2:
        bg_pil = Image.fromarray(bg_img).convert("RGB")
    elif bg_img.shape[2] == 4:
        bg_pil = Image.fromarray(bg_img[:, :, :3])
    else:
        bg_pil = Image.fromarray(bg_img)

    # Colormap overlay
    cmap = matplotlib.colormaps[colormap]
    norm = _build_norm(vmin, vmax, log_scale)
    normalized = norm(np.nan_to_num(full_data, nan=0.0))
    colored = cmap(normalized)
    colored[:, :, 3] = alpha * mask.astype(np.float64)
    overlay_rgba = (colored * 255).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay_rgba)

    # Composite
    result = bg_pil.copy()
    result.paste(overlay_pil, (0, 0), overlay_pil)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
