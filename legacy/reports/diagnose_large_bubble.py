"""Diagnose deformed-view ROI rendering for LARGE bubble expansion (R: 20 -> 100).

Reproduces the user-reported symptom: "When I give a per-frame ROI that excludes
a large bubble, the deformed view still does not match the ROI I supplied."

Scenario A: Pure expansion 20 -> 100 (nine frames).
Scenario B: Expand 20 -> 100 then shrink back to 20.

Metrics per frame:
  - IoU(rendered_valid, user_mask)
  - Deficit   = user_mask  & ~rendered  (user supplied, renderer dropped)
  - Overflow  = rendered   & ~user_mask (renderer painted outside user mask)
  - Boundary band deficit (within 5 px of user-mask boundary)

Outputs PNG diagnostics and a numeric summary to stdout.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scipy.ndimage import map_coordinates

from server.deformed_warp import (
    InverseMapCache,
    compute_inverse_map,
    get_warped_full_data,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "diagnose_output")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic speckle + bubble scenario
# ---------------------------------------------------------------------------

def _speckle(h: int, w: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.random((h, w)).astype(np.float32)
    n_dots = h * w // 20
    ys = rng.integers(0, h, n_dots)
    xs = rng.integers(0, w, n_dots)
    for y, x in zip(ys, xs):
        cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 3)), float(rng.random()), -1)
    img = cv2.GaussianBlur(img, (5, 5), 1.2)
    return (img * 255).clip(0, 255).astype(np.uint8)


@dataclass
class BubbleScenario:
    h: int
    w: int
    cx: int
    cy: int
    radii: List[float]
    images: List[np.ndarray]
    displacements: List[np.ndarray]
    per_frame_rois: Dict[int, np.ndarray]
    roi_rect: tuple

    @property
    def n_frames(self) -> int:
        return len(self.displacements)


def make_scenario(
    h: int = 400,
    w: int = 400,
    cx: int = 200,
    cy: int = 200,
    radii=None,
    apply_roundtrip_mask: bool = True,
    legacy_oob_nan: bool = False,
) -> BubbleScenario:
    """Incompressible 2D plane-strain bubble. Default radii: 20 -> 100 expansion."""
    if radii is None:
        radii = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    R0 = radii[0]
    yy, xx = np.mgrid[:h, :w]
    dy = yy - cy
    dx = xx - cx
    dist = np.sqrt(dx ** 2 + dy ** 2)
    r_safe = np.maximum(dist, 1e-10)

    base = _speckle(h, w, seed=42)
    images = []
    for r in radii:
        img = base.copy()
        img[dist <= r] = 0
        images.append(img)

    per_frame_rois: Dict[int, np.ndarray] = {}
    for i, r in enumerate(radii):
        mask = np.ones((h, w), dtype=bool)
        mask[dist <= r] = False
        per_frame_rois[i] = mask

    roi_rect = (0, 0, w, h)
    displacements: List[np.ndarray] = []
    for i in range(1, len(radii)):
        R = radii[i]
        arg = r_safe ** 2 + R ** 2 - R0 ** 2
        r_new = np.where(arg > 0, np.sqrt(arg), 0.0)
        u_r = r_new - r_safe
        u = u_r * dx / r_safe
        v = u_r * dy / r_safe

        disp_full = np.stack([u, v], axis=-1).astype(np.float64)
        disp_full[dist <= R0, :] = np.nan

        if apply_roundtrip_mask:
            # Reproduce what _apply_current_frame_mask does in the real pipeline.
            # legacy_oob_nan=True reproduces the pre-fix bug where OOB pixels
            # were NaN-ed; False reproduces the fixed behaviour that keeps OOB.
            valid = ~np.isnan(disp_full[..., 0])
            vyy, vxx = np.where(valid)
            U_vals = disp_full[vyy, vxx, 0]
            V_vals = disp_full[vyy, vxx, 1]
            def_x = np.round(vxx + U_vals).astype(np.intp)
            def_y = np.round(vyy + V_vals).astype(np.intp)
            in_bounds = (def_x >= 0) & (def_x < w) & (def_y >= 0) & (def_y < h)
            cur_mask = per_frame_rois[i]
            default_kept = False if legacy_oob_nan else True
            in_specimen = np.full(len(vyy), default_kept, dtype=bool)
            in_specimen[in_bounds] = cur_mask[def_y[in_bounds], def_x[in_bounds]]
            outside = ~in_specimen
            if outside.any():
                disp_full[vyy[outside], vxx[outside], :] = np.nan

        displacements.append(disp_full)

    return BubbleScenario(
        h=h, w=w, cx=cx, cy=cy,
        radii=radii, images=images,
        displacements=displacements,
        per_frame_rois=per_frame_rois,
        roi_rect=roi_rect,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _warp_data_inverse_legacy(
    data: np.ndarray, inv_map, image_shape
) -> np.ndarray:
    """Reproduce the pre-fix warp_data_inverse: no nearest-fill on NaN,
    final_valid = user_mask & has_data.  Used only for before/after plotting.
    """
    if inv_map.ref_row_coords.size == 0:
        return np.full(image_shape, np.nan)
    out_h = inv_map.out_y1 - inv_map.out_y0
    out_w = inv_map.out_x1 - inv_map.out_x0
    valid_src = ~np.isnan(data)
    data_clean = np.nan_to_num(data, nan=0.0).astype(np.float64)
    src_valid = valid_src.astype(np.float64)
    coords = np.array([
        inv_map.ref_row_coords.ravel(),
        inv_map.ref_col_coords.ravel(),
    ])
    warped = map_coordinates(
        data_clean, coords, order=1, mode="constant", cval=0.0
    ).reshape(out_h, out_w)
    valid_warped = map_coordinates(
        src_valid, coords, order=1, mode="constant", cval=0.0
    ).reshape(out_h, out_w)
    has_data = valid_warped > 0.5
    if inv_map.has_user_mask:
        final_valid = inv_map.validity_mask & has_data
    else:
        final_valid = has_data & inv_map.validity_mask
    full = np.full(image_shape, np.nan)
    patch = np.where(final_valid, warped, np.nan)
    full[inv_map.out_y0:inv_map.out_y1, inv_map.out_x0:inv_map.out_x1] = patch
    return full


def render_deformed(
    sc: BubbleScenario, idx: int, cache: InverseMapCache,
    legacy_warp: bool = False,
) -> np.ndarray:
    disp = sc.displacements[idx]
    U, V = disp[:, :, 0], disp[:, :, 1]
    mask = sc.per_frame_rois.get(idx + 1)
    if legacy_warp:
        inv_map = compute_inverse_map(
            U, V, sc.roi_rect, (sc.h, sc.w), deformed_frame_mask=mask,
        )
        inv_map.frame_idx = idx
        return _warp_data_inverse_legacy(U, inv_map, (sc.h, sc.w))
    return get_warped_full_data(
        data=U, frame_idx=idx, U=U, V=V,
        roi_rect=sc.roi_rect, image_shape=(sc.h, sc.w),
        cache=cache, deformed_mask=mask,
    )


@dataclass
class FrameMetrics:
    frame_idx: int
    radius: float
    mask_area: int
    rendered_area: int
    intersection: int
    deficit: int
    overflow: int
    iou: float
    boundary_deficit: int
    boundary_px_total: int


def compute_metrics(rendered: np.ndarray, mask: np.ndarray, boundary_width: int = 5) -> Dict:
    has_value = np.isfinite(rendered)
    both = has_value & mask
    deficit = mask & ~has_value
    overflow = has_value & ~mask
    union = mask | has_value
    iou = float(both.sum()) / float(max(1, union.sum()))

    mask_u8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * boundary_width + 1,) * 2)
    boundary = (cv2.dilate(mask_u8, kernel) - cv2.erode(mask_u8, kernel)).astype(bool)
    boundary_deficit = int((deficit & boundary).sum())
    boundary_total = int(boundary.sum())
    return dict(
        mask_area=int(mask.sum()),
        rendered_area=int(has_value.sum()),
        intersection=int(both.sum()),
        deficit=int(deficit.sum()),
        overflow=int(overflow.sum()),
        iou=iou,
        boundary_deficit=boundary_deficit,
        boundary_total=boundary_total,
    )


# ---------------------------------------------------------------------------
# Visualization — diagnostic grid
# ---------------------------------------------------------------------------

def save_grid(sc: BubbleScenario, rendered: List[np.ndarray], filename: str, title: str):
    n = sc.n_frames
    fig, axes = plt.subplots(3, n, figsize=(2.3 * n, 7), dpi=110)
    if n == 1:
        axes = axes[:, np.newaxis]

    all_u = np.concatenate([d[:, :, 0].ravel() for d in sc.displacements])
    finite = all_u[np.isfinite(all_u)]
    vmin, vmax = float(finite.min()), float(finite.max())

    for i in range(n):
        mask = sc.per_frame_rois[i + 1]
        rendered_i = rendered[i]

        # Row 0: deformed view overlay
        ax = axes[0, i]
        ax.imshow(sc.images[i + 1], cmap="gray", alpha=0.35)
        masked = np.ma.array(rendered_i, mask=~np.isfinite(rendered_i))
        ax.imshow(masked, cmap="turbo", vmin=vmin, vmax=vmax, alpha=0.85)
        ax.set_title(f"Def F{i}  R={sc.radii[i+1]}", fontsize=8)
        ax.axis("off")

        # Row 1: user mask (ground truth ROI for this frame)
        ax = axes[1, i]
        vis = np.zeros((sc.h, sc.w, 3), dtype=np.uint8)
        vis[mask] = [40, 180, 40]
        vis[~mask] = [40, 40, 40]
        ax.imshow(vis)
        ax.set_title(f"User mask F{i+1}\n{mask.sum()} px", fontsize=8)
        ax.axis("off")

        # Row 2: error map (green = ok, red = deficit, yellow = overflow)
        ax = axes[2, i]
        err = np.zeros((sc.h, sc.w, 3), dtype=np.uint8)
        has = np.isfinite(rendered_i)
        err[has & mask] = [30, 150, 30]        # correct
        err[mask & ~has] = [220, 30, 30]       # deficit (user asked, renderer dropped)
        err[has & ~mask] = [230, 200, 20]      # overflow (renderer painted outside)
        ax.imshow(err)
        m = compute_metrics(rendered_i, mask)
        ax.set_title(
            f"IoU={m['iou']:.3f}\n"
            f"def={m['deficit']}  ov={m['overflow']}",
            fontsize=8,
        )
        ax.axis("off")

    axes[0, 0].set_ylabel("Deformed\nOverlay", fontsize=9, rotation=0, labelpad=45, va="center")
    axes[1, 0].set_ylabel("User\nMask", fontsize=9, rotation=0, labelpad=45, va="center")
    axes[2, 0].set_ylabel("Error\n(R=miss, Y=extra)", fontsize=9, rotation=0, labelpad=45, va="center")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0.06, 0, 1, 0.94])
    out_path = os.path.join(OUT_DIR, filename)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_scenario(
    name: str, radii: List[float], legacy_oob_nan: bool = False,
    legacy_warp: bool = False,
) -> List[Dict]:
    tag = "LEGACY" if (legacy_oob_nan and legacy_warp) else "FIXED"
    print(f"\n=== Scenario: {name} [{tag}] ===")
    print(f"Radii: {radii}")
    sc = make_scenario(radii=radii, legacy_oob_nan=legacy_oob_nan)
    cache = InverseMapCache(max_size=20)
    rendered = [
        render_deformed(sc, i, cache, legacy_warp=legacy_warp)
        for i in range(sc.n_frames)
    ]

    metrics_list: List[Dict] = []
    print(f"{'Frame':>5} {'R':>5} {'Mask':>8} {'Rend':>8} {'IoU':>7} {'Deficit':>8} "
          f"{'Overflow':>9} {'BndDef':>7} {'BndTot':>7}")
    for i in range(sc.n_frames):
        m = compute_metrics(rendered[i], sc.per_frame_rois[i + 1])
        m["frame_idx"] = i
        m["radius"] = sc.radii[i + 1]
        metrics_list.append(m)
        print(
            f"{i:>5d} {sc.radii[i+1]:>5.0f} "
            f"{m['mask_area']:>8d} {m['rendered_area']:>8d} "
            f"{m['iou']:>7.4f} {m['deficit']:>8d} {m['overflow']:>9d} "
            f"{m['boundary_deficit']:>7d} {m['boundary_total']:>7d}"
        )
    suffix = "_legacy" if legacy_oob_nan else "_fixed"
    out = save_grid(sc, rendered, f"{name}{suffix}_diag.png",
                    f"{name} [{tag}] (large bubble)")
    print(f"Saved: {out}")
    return metrics_list


def main():
    results = {}
    scenarios = [
        ("expand_20_to_100", [20, 30, 40, 50, 60, 70, 80, 90, 100]),
        ("expand_shrink_20_100_20", [20, 40, 60, 80, 100, 80, 60, 40, 20]),
        ("fine_step_60_to_100", [60, 70, 80, 85, 90, 95, 100]),
        # Shrink below R_initial: frame N bubble is smaller than reference
        # bubble. Newly-exposed ring pixels correspond to NaN ref data.
        ("shrink_below_ref_30_80_15",
         [30, 50, 70, 80, 70, 50, 30, 20, 15]),
    ]
    for name, radii in scenarios:
        results[f"{name}_legacy"] = run_scenario(
            name, radii, legacy_oob_nan=True, legacy_warp=True,
        )
        results[f"{name}_fixed"] = run_scenario(
            name, radii, legacy_oob_nan=False, legacy_warp=False,
        )
    return results


if __name__ == "__main__":
    main()
