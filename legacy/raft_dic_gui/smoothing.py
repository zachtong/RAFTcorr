"""Displacement field smoothing for RAFT-DIC."""

import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_displacement_field(displacement_field: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    smoothed = np.zeros_like(displacement_field)
    for i in range(2):
        comp = displacement_field[..., i]
        valid_mask = ~np.isnan(comp)
        if not np.any(valid_mask):
            smoothed[..., i] = comp
            continue
        filled = np.where(valid_mask, comp, 0)
        smoothed_data = gaussian_filter(filled, sigma)
        weight = gaussian_filter(valid_mask.astype(float), sigma)
        with np.errstate(divide='ignore', invalid='ignore'):
            smoothed[..., i] = np.where(weight > 0.01, smoothed_data / weight, np.nan)
    return smoothed
