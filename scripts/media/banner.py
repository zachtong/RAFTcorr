"""The README banner (1280 x 400 px): reference image, vertical displacement and strain of the demo
specimen around its two holes, then the icon and the name. Called by ``screenshots.py`` with the
arrays of its run; the pictures are measurements of RAFTcorr, not illustrations.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

BG = "#0b0f1a"
ACCENT = "#6366f1"
ACCENT_LIGHT = "#818cf8"
TEXT = "#e2e8f0"
TEXT_MUTED = "#94a3b8"
CROP = (120, 1890, 410, 2276)            # x0, y0, x1, y1 in image pixels: both holes, 3:4


def _on_image(field: np.ndarray, grid_x0: int, grid_y0: int, step: int, shape: tuple[int, int]) -> np.ndarray:
    """``field`` sampled every ``step`` px from (grid_x0, grid_y0), spread onto the image pixels."""
    out = np.full(shape, np.nan, np.float32)
    h, w = field.shape
    up = np.repeat(np.repeat(field, step, axis=0), step, axis=1)
    y1, x1 = min(shape[0], grid_y0 + up.shape[0]), min(shape[1], grid_x0 + up.shape[1])
    out[grid_y0:y1, grid_x0:x1] = up[: y1 - grid_y0, : x1 - grid_x0]
    return out


def render_banner(reference: np.ndarray, v: np.ndarray, eyy: np.ndarray, icon_png: Path, out: Path) -> None:
    """``reference`` (H, W) grey image; ``v`` and ``eyy`` (H, W) float with NaN outside the analysis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from matplotlib.patches import FancyArrowPatch, Rectangle

    x0, y0, x1, y1 = CROP
    ref = reference[y0:y1, x0:x1].astype(np.float32)
    ref = (ref - np.nanpercentile(ref, 1)) / max(np.nanpercentile(ref, 99) - np.nanpercentile(ref, 1), 1e-6)
    panels = [("Reference image", None), ("Displacement v", v[y0:y1, x0:x1]), ("Strain εyy", eyy[y0:y1, x0:x1])]

    fig = plt.figure(figsize=(12.8, 4.0), dpi=100, facecolor=BG)
    bg = fig.add_axes([0, 0, 1, 1])
    bg.set_xlim(0, 1)
    bg.set_ylim(0, 1)
    bg.axis("off")
    yy, xx = np.mgrid[0:200, 0:640] / 200.0
    r2 = (xx - 1.1) ** 2 + (yy - 0.5) ** 2
    glow = np.zeros((200, 640, 4))
    base, tint = np.array([0.043, 0.059, 0.102]), np.array([0.388, 0.400, 0.945])
    for i in range(3):
        glow[..., i] = base[i] + (tint[i] - base[i]) * np.exp(-r2 / 0.35) * 0.13
    glow[..., 3] = 1.0
    bg.imshow(glow, extent=[0, 1, 0, 1], aspect="auto", interpolation="bicubic")

    ph = 0.68                                # panel height (figure fraction)
    pw = ph * (4.0 / 12.8) * (x1 - x0) / (y1 - y0)
    left, gap = 0.035, 0.035
    for k, (title, field) in enumerate(panels):
        ax = fig.add_axes([left + k * (pw + gap), 0.12, pw, ph])
        ax.imshow(ref, cmap="gray", vmin=0, vmax=1, interpolation="antialiased")
        if field is not None:
            lo, hi = np.nanpercentile(field, 1), np.nanpercentile(field, 99)
            ax.imshow(np.ma.masked_invalid(field), cmap="turbo", vmin=lo, vmax=hi, alpha=0.92,
                      interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(ACCENT)
            s.set_linewidth(1.4)
        ax.set_title(title, color=TEXT_MUTED, fontsize=12, pad=6, family="DejaVu Sans")
        if k < 2:
            xa = left + k * (pw + gap) + pw
            bg.add_patch(FancyArrowPatch((xa + 0.006, 0.12 + ph / 2), (xa + gap - 0.006, 0.12 + ph / 2),
                                         arrowstyle="-|>", mutation_scale=14, color=ACCENT_LIGHT, lw=1.6))

    text_x = left + 3 * pw + 2 * gap + 0.045
    icon = imread(str(icon_png))
    icon_h = 0.34                            # square: width = height x 4.0 / 12.8 in figure fractions
    ia = fig.add_axes([text_x, 0.33, icon_h * 4.0 / 12.8, icon_h])
    ia.imshow(icon)
    ia.axis("off")
    tx = text_x + icon_h * 4.0 / 12.8 + 0.022
    bg.text(tx, 0.62, "RAFTcorr", color=TEXT, fontsize=40, fontweight="bold", family="DejaVu Sans", va="center")
    for y, line in ((0.47, "Learned Digital"), (0.38, "Image Correlation")):
        bg.text(tx, y, line, color=TEXT_MUTED, fontsize=17, family="DejaVu Sans", va="center")
    bg.text(text_x, 0.20, "desktop application  ·  command line  ·  Python library", color=TEXT_MUTED,
            fontsize=11.5, family="DejaVu Sans", va="center")
    bg.add_patch(Rectangle((0.035, 0.045), 0.93, 0.0025, color=ACCENT, alpha=0.55, transform=bg.transAxes))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor=BG)
    plt.close(fig)
