"""
Incremental Reference Processing Utilities

This module provides functions for incremental DIC processing where the
reference image is updated at user-specified key frames to handle large
deformations.

Key functions:
- warp_mask_with_holes: Warp ROI mask preserving internal holes
- accumulate_displacement: Combine segment displacements in original coordinates
- validate_key_frames: Validate key frame selection
- key_frames_every_n: Generate key frame list at regular intervals
- build_ref_map: Build frame→reference mapping from key frames
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


def _smooth_contour(contour: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Gaussian-smooth contour coordinates to remove residual jitter.

    Operates on a closed contour using wrap mode so endpoints blend
    seamlessly.  Very mild (sigma=2) to avoid distorting the shape.
    """
    if len(contour) < 5:
        return contour
    from scipy.ndimage import gaussian_filter1d
    pts = contour.reshape(-1, 2).astype(np.float64)
    pts[:, 0] = gaussian_filter1d(pts[:, 0], sigma=sigma, mode='wrap')
    pts[:, 1] = gaussian_filter1d(pts[:, 1], sigma=sigma, mode='wrap')
    return pts.reshape(-1, 1, 2).astype(np.float32)


def warp_mask_with_holes(mask: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """
    Warp a binary mask according to a displacement field, preserving holes.
    
    Uses contour-based warping to maintain the topology of complex masks
    with internal holes (e.g., masks with cutouts or complex geometry).
    
    Args:
        mask: (H, W) boolean or uint8 mask, True/1 = valid region
        displacement: (H, W, 2) displacement field (u, v)
        
    Returns:
        warped_mask: (H, W) boolean mask after applying displacement
    """
    H, W = mask.shape
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    
    # Find contours with hierarchy (RETR_TREE captures holes)
    # CHAIN_APPROX_NONE keeps every contour point so that straight edges
    # warp correctly under non-uniform displacement fields.
    contours, hierarchy = cv2.findContours(
        mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )
    
    if len(contours) == 0:
        return np.zeros((H, W), dtype=bool)
    
    # Warp each contour point using spatial-median displacement
    def warp_contour(contour: np.ndarray) -> np.ndarray:
        """Warp contour points using median displacement from a spatial patch.

        For each contour point, samples displacement from a (2R+1)x(2R+1)
        neighborhood including interior pixels and takes the median.
        This is robust against DIC errors at ROI boundaries where the
        correlation quality is lowest.
        """
        R = 10  # patch half-size in pixels
        warped_pts = []
        for pt in contour:
            x, y = int(pt[0][0]), int(pt[0][1])

            # Spatial patch bounds (clipped to image)
            y0p = max(0, y - R)
            y1p = min(H, y + R + 1)
            x0p = max(0, x - R)
            x1p = min(W, x + R + 1)

            patch_dx = displacement[y0p:y1p, x0p:x1p, 0]
            patch_dy = displacement[y0p:y1p, x0p:x1p, 1]
            valid = ~(np.isnan(patch_dx) | np.isnan(patch_dy))

            if valid.any():
                dx = float(np.median(patch_dx[valid]))
                dy = float(np.median(patch_dy[valid]))
                new_x = x + dx
                new_y = y + dy
            else:
                new_x, new_y = float(x), float(y)

            warped_pts.append([[new_x, new_y]])

        return np.array(warped_pts, dtype=np.float32)
    
    # Separate outer contours and holes
    outer_contours = []
    hole_contours = []
    
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            warped_cnt = warp_contour(cnt)
            warped_cnt = _smooth_contour(warped_cnt, sigma=2.0)
            # Clip to image bounds
            warped_cnt = np.clip(warped_cnt, 0, [[W-1, H-1]])
            warped_cnt = np.rint(warped_cnt).astype(np.int32)
            
            # hierarchy: [Next, Previous, First_Child, Parent]
            parent = hierarchy[0][i][3]
            if parent == -1:
                # This is an outer contour
                outer_contours.append(warped_cnt)
            else:
                # This is a hole (has parent)
                hole_contours.append(warped_cnt)
    
    # Draw result
    new_mask = np.zeros((H, W), dtype=np.uint8)
    
    # Fill outer contours
    for cnt in outer_contours:
        if len(cnt) >= 3:  # Need at least 3 points for polygon
            cv2.drawContours(new_mask, [cnt], 0, 255, -1)
    
    # Subtract holes
    for cnt in hole_contours:
        if len(cnt) >= 3:
            cv2.drawContours(new_mask, [cnt], 0, 0, -1)
    
    # Morphological cleanup to fill small gaps from warping
    kernel = np.ones((3, 3), np.uint8)
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)
    
    return new_mask > 0


def warp_displacement_field(delta_u: np.ndarray,
                            accumulated_u: np.ndarray) -> np.ndarray:
    """
    Sample incremental displacement at deformed coordinates.

    Uses NaN-aware bilinear interpolation so that NaN regions do not
    contaminate valid boundary pixels.

    Args:
        delta_u: (H, W, 2) incremental displacement (in deformed coordinates)
        accumulated_u: (H, W, 2) accumulated displacement so far (in original coords)

    Returns:
        delta_u_sampled: (H, W, 2) delta_u sampled at deformed positions
    """
    from raft_dic_gui.nan_interp import bilinear_sample_nan_aware

    H, W = delta_u.shape[:2]

    # Create coordinate grids
    x, y = np.meshgrid(np.arange(W, dtype=np.float64),
                        np.arange(H, dtype=np.float64))

    # Deformed coordinates
    u_prev_x = accumulated_u[..., 0].astype(np.float64)
    u_prev_y = accumulated_u[..., 1].astype(np.float64)

    x_def = x + u_prev_x
    y_def = y + u_prev_y

    # Mask: accumulated_u itself is NaN → result is NaN
    u_prev_nan = np.isnan(u_prev_x)

    # Flatten for vectorized sampling.
    # Clamp to valid bounds (equivalent to BORDER_REPLICATE) so that
    # edge pixels outside the ROI sample from the nearest edge value
    # instead of returning NaN.  This prevents progressive NaN erosion
    # at ROI boundaries during incremental accumulation.
    x_flat = x_def.ravel()
    y_flat = y_def.ravel()

    nan_coord = np.isnan(x_flat) | np.isnan(y_flat)
    x_flat = np.where(nan_coord, 0.0, np.clip(x_flat, 0, W - 1))
    y_flat = np.where(nan_coord, 0.0, np.clip(y_flat, 0, H - 1))

    # Sample each channel with NaN-aware interpolation
    delta_u_sampled = np.empty_like(delta_u)
    for ch in range(2):
        sampled = bilinear_sample_nan_aware(delta_u[..., ch], x_flat, y_flat)
        delta_u_sampled[..., ch] = sampled.reshape(H, W)

    # Also mark NaN where accumulated_u was NaN
    delta_u_sampled[u_prev_nan, 0] = np.nan
    delta_u_sampled[u_prev_nan, 1] = np.nan

    return delta_u_sampled


def accumulate_displacement(u_prev: np.ndarray, 
                           delta_u: np.ndarray,
                           debug: bool = True) -> np.ndarray:
    """
    Accumulate incremental displacement onto previous accumulated displacement.
    
    Handles coordinate transformation: delta_u is sampled at deformed positions
    before being added to the accumulated displacement.
    
    Args:
        u_prev: (H, W, 2) previous accumulated displacement (in original coords)
        delta_u: (H, W, 2) incremental displacement from new reference
        debug: Whether to print debug info
        
    Returns:
        u_total: (H, W, 2) total displacement in original coordinates
    """
    if debug:
        print(f"[Accumulate] Input shapes: u_prev={u_prev.shape}, delta_u={delta_u.shape}")
        
        # Stats for u_prev
        valid_prev = ~np.isnan(u_prev[..., 0])
        if np.any(valid_prev):
            print(f"[Accumulate] u_prev stats: valid_pixels={np.sum(valid_prev)}, "
                  f"u_range=[{np.nanmin(u_prev[...,0]):.2f}, {np.nanmax(u_prev[...,0]):.2f}], "
                  f"v_range=[{np.nanmin(u_prev[...,1]):.2f}, {np.nanmax(u_prev[...,1]):.2f}]")
        else:
            print(f"[Accumulate] u_prev: all NaN!")
        
        # Stats for delta_u
        valid_delta = ~np.isnan(delta_u[..., 0])
        if np.any(valid_delta):
            print(f"[Accumulate] delta_u stats: valid_pixels={np.sum(valid_delta)}, "
                  f"u_range=[{np.nanmin(delta_u[...,0]):.2f}, {np.nanmax(delta_u[...,0]):.2f}], "
                  f"v_range=[{np.nanmin(delta_u[...,1]):.2f}, {np.nanmax(delta_u[...,1]):.2f}]")
        else:
            print(f"[Accumulate] delta_u: all NaN!")
    
    # Sample delta_u at deformed positions
    delta_u_sampled = warp_displacement_field(delta_u, u_prev)
    
    if debug:
        valid_sampled = ~np.isnan(delta_u_sampled[..., 0])
        if np.any(valid_sampled):
            print(f"[Accumulate] After warp: valid_pixels={np.sum(valid_sampled)} (lost {np.sum(valid_delta)-np.sum(valid_sampled)} pixels), "
                  f"u_range=[{np.nanmin(delta_u_sampled[...,0]):.2f}, {np.nanmax(delta_u_sampled[...,0]):.2f}], "
                  f"v_range=[{np.nanmin(delta_u_sampled[...,1]):.2f}, {np.nanmax(delta_u_sampled[...,1]):.2f}]")
        else:
            print(f"[Accumulate] After warp: all NaN! (all pixels lost)")
    
    # Accumulate
    u_total = u_prev + delta_u_sampled
    
    if debug:
        valid_total = ~np.isnan(u_total[..., 0])
        if np.any(valid_total):
            print(f"[Accumulate] Result: valid_pixels={np.sum(valid_total)}, "
                  f"u_range=[{np.nanmin(u_total[...,0]):.2f}, {np.nanmax(u_total[...,0]):.2f}], "
                  f"v_range=[{np.nanmin(u_total[...,1]):.2f}, {np.nanmax(u_total[...,1]):.2f}]")
        else:
            print(f"[Accumulate] Result: all NaN!")
    
    return u_total


def median_filter_displacement(disp: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filter to accumulated displacement to suppress outlier pixels.

    This is a post-accumulation smoothing step that reduces localized errors
    introduced by repeated interpolation in incremental DIC.  It modifies the
    displacement data (not a view-only effect).

    NaN-safe: NaN regions are preserved and do not bleed into valid data.

    Args:
        disp: (H, W, 2) accumulated displacement field (may contain NaN)
        kernel_size: median filter window size (must be odd, default 5)

    Returns:
        Filtered displacement field, same shape as input.
    """
    from scipy.ndimage import median_filter

    if kernel_size < 3:
        return disp

    result = disp.copy()
    valid = ~np.isnan(disp[..., 0])

    for ch in range(2):
        channel = disp[..., ch].copy()
        # Fill NaN with 0 for median filter, then restore
        channel_filled = np.where(valid, channel, 0.0)
        filtered = median_filter(channel_filled, size=kernel_size)
        result[..., ch] = np.where(valid, filtered, np.nan)

    return result


def validate_key_frames(key_frames: List[int],
                        total_frames: int) -> Tuple[bool, str]:
    """
    Validate key frame selection for incremental processing.
    
    Requirements:
    - Must include frame 1 (or 0 if 0-indexed)
    - All frames must be within valid range
    - Must be sorted in ascending order
    
    Args:
        key_frames: List of frame indices (1-indexed, frame 1 = reference)
        total_frames: Total number of frames in sequence
        
    Returns:
        (is_valid, error_message): Tuple of validation result and message
    """
    if not key_frames:
        return False, "No key frames selected. At least frame 1 must be selected."
    
    # Sort key frames
    key_frames_sorted = sorted(key_frames)
    
    # Check frame 1 is included
    if key_frames_sorted[0] != 1:
        return False, "Frame 1 must be selected as the first key frame (initial reference)."
    
    # Check all frames are in valid range
    for f in key_frames_sorted:
        if f < 1 or f > total_frames:
            return False, f"Frame {f} is out of range (1-{total_frames})."
    
    # Check for duplicates
    if len(key_frames_sorted) != len(set(key_frames_sorted)):
        return False, "Duplicate key frames detected."
    
    return True, "Key frames valid."


def get_segment_ranges(key_frames: List[int], 
                       total_frames: int) -> List[Tuple[int, int, int]]:
    """
    Get processing segment ranges from key frames.
    
    Args:
        key_frames: List of key frame indices (1-indexed, sorted)
        total_frames: Total number of frames
        
    Returns:
        List of (ref_frame, start_frame, end_frame) tuples for each segment
    """
    key_frames_sorted = sorted(key_frames)
    segments = []
    
    for i, ref_frame in enumerate(key_frames_sorted):
        start_frame = ref_frame + 1  # First frame to process in this segment

        if i < len(key_frames_sorted) - 1:
            end_frame = key_frames_sorted[i + 1]  # Up to (and including) next key frame
        else:
            end_frame = total_frames  # Last segment goes to end

        if start_frame <= end_frame:
            segments.append((ref_frame, start_frame, end_frame))

    return segments


def key_frames_every_n(total_frames: int, n: int) -> List[int]:
    """Generate key frame list at every N frames (1-indexed).

    Frame 1 is always included as the first key frame.

    Args:
        total_frames: Total number of frames (1-indexed, so frames are 1..total_frames)
        n: Interval between key frames. Values < 1 are treated as 1.

    Returns:
        Sorted list of key frame numbers.

    Examples:
        >>> key_frames_every_n(200, 50)
        [1, 51, 101, 151]
        >>> key_frames_every_n(5, 1)
        [1, 2, 3, 4, 5]
        >>> key_frames_every_n(10, 100)
        [1]
    """
    if n < 1:
        n = 1
    return [i for i in range(1, total_frames + 1, n)]


def build_ref_map(
    total_frames: int,
    key_frames: Optional[List[int]] = None,
    ref_map: Optional[Dict[int, int]] = None,
) -> Dict[int, int]:
    """Build a frame->reference mapping.

    If *ref_map* is provided, return it directly (passthrough for future
    adaptive strategies).  Otherwise, build from *key_frames* using the
    linear segment strategy (Strategy A): within each segment
    ``[KF_i, KF_{i+1})``, **all** frames correlate against ``KF_i``.

    Args:
        total_frames: Total number of frames (1-indexed, so frames are
            1..total_frames).
        key_frames: List of key frame numbers (1-indexed).  Frame 1 is
            always included (auto-added if missing).
        ref_map: Pre-built mapping (passthrough).  Takes priority over
            *key_frames*.

    Returns:
        Dict mapping ``frame_number -> reference_frame_number``.
        Frame 1 is never in the dict (it IS the origin reference).
    """
    if ref_map is not None:
        return dict(ref_map)

    if key_frames is None:
        key_frames = [1]

    kf_sorted = sorted(set(key_frames))
    if kf_sorted[0] != 1:
        kf_sorted.insert(0, 1)

    result: Dict[int, int] = {}
    for idx in range(len(kf_sorted)):
        kf = kf_sorted[idx]
        # End of this segment (inclusive): up to next key frame, or total_frames
        if idx + 1 < len(kf_sorted):
            seg_end = kf_sorted[idx + 1]
        else:
            seg_end = total_frames
        for frame in range(kf + 1, seg_end + 1):
            result[frame] = kf

    return result
