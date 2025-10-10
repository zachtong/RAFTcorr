"""
Lightweight preview utilities for GUI overlays on Tk canvas.
Now mirrors processing tiling and shows valid-core margins.
"""

from PIL import Image, ImageTk, ImageDraw
import numpy as np
from .processing import calculate_window_positions


def create_preview_image(image: np.ndarray, crop_size=None, shift=None, max_displacement: int = 0) -> (Image.Image, dict):
    """Create a PIL preview image with crop grid and valid-core overlay.

    Returns: (PIL.Image, metrics dict) where metrics includes:
      tiles_x, tiles_y, tiles_total, core_w, core_h, overlap_x, overlap_y
    """
    preview_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(preview_pil)
    metrics = None

    h, w = image.shape[:2]
    if crop_size and shift:
        crop_h, crop_w = crop_size
        # Mirror actual tiling logic
        xs = calculate_window_positions(w, crop_w, shift)
        ys = calculate_window_positions(h, crop_h, shift)
        tiles_x, tiles_y = len(xs), len(ys)

        # Valid core sizes (non-negative)
        core_w = max(0, crop_w - 2 * max(0, max_displacement))
        core_h = max(0, crop_h - 2 * max(0, max_displacement))
        overlap_x = max(0, crop_w - 2 * max(0, max_displacement) - shift)
        overlap_y = max(0, crop_h - 2 * max(0, max_displacement) - shift)
        metrics = {
            'tiles_x': tiles_x,
            'tiles_y': tiles_y,
            'tiles_total': tiles_x * tiles_y,
            'core_w': core_w,
            'core_h': core_h,
            'overlap_x': overlap_x,
            'overlap_y': overlap_y,
        }

        # Draw tiles and their valid cores
        for y in ys:
            for x in xs:
                x_end = min(x + crop_w, w)
                y_end = min(y + crop_h, h)
                # Outer tile
                draw.rectangle([(x, y), (x_end, y_end)], outline='lightblue', width=1)
                # Valid core
                x0 = x + max_displacement
                y0 = y + max_displacement
                x1 = min(x + crop_w - max_displacement, w)
                y1 = min(y + crop_h - max_displacement, h)
                if x1 > x0 and y1 > y0:
                    draw.rectangle([(x0, y0), (x1, y1)], outline='orange', width=1)

        # Highlight first tile and offset samples for clarity
        if tiles_x > 0 and tiles_y > 0:
            x0, y0 = xs[0], ys[0]
            draw.rectangle([(x0, y0), (min(x0 + crop_w, w), min(y0 + crop_h, h))], outline='blue', width=2)
            if len(xs) > 1:
                x1 = xs[1]
                draw.rectangle([(x1, y0), (min(x1 + crop_w, w), min(y0 + crop_h, h))], outline='yellow', width=2)
            if len(ys) > 1:
                y1 = ys[1]
                draw.rectangle([(x0, y1), (min(x0 + crop_w, w), min(y1 + crop_h, h))], outline='green', width=2)

    return preview_pil, metrics


def update_preview(canvas, image: np.ndarray, crop_size=None, shift=None, max_displacement: int = 0):
    """Render preview image (with optional crop grid) to a Tk canvas.
    Returns a metrics dict (or None) about tiling/overlap.
    """
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    if canvas_width <= 1 or canvas_height <= 1:
        return None

    preview, metrics = create_preview_image(image, crop_size, shift, max_displacement)

    w, h = preview.size
    scale = min(canvas_width / w, canvas_height / h)
    display_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    x_offset = (canvas_width - display_size[0]) // 2
    y_offset = (canvas_height - display_size[1]) // 2

    preview = preview.resize(display_size, Image.LANCZOS)
    photo = ImageTk.PhotoImage(preview)
    canvas.delete("all")
    canvas.create_image(x_offset, y_offset, anchor='nw', image=photo)
    canvas.image = photo  # keep reference

    return metrics
