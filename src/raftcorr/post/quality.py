"""Run-level quality hints from the intensity residual.

The residual (:func:`raftcorr.engine.mapping.intensity_residual`) grows slowly with deformation
and illumination changes, but jumps when the tracking fails, typically when the displacement
from the reference exceeds the model's reach. :func:`suspect_frames` flags frames after such a
jump. It is a hint for the user, not a test: it never changes a result.
"""

from __future__ import annotations

import numpy as np

JUMP_RATIO = 1.6      # residual of a frame / residual of the previous computed frame
BASELINE_RATIO = 2.0  # residual of a frame / residual of the first computed frame


def frame_residuals(residual: np.ndarray, done: np.ndarray) -> np.ndarray:
    """Median residual of every computed frame (NaN elsewhere; frame 0 is the reference)."""
    out = np.full(residual.shape[0], np.nan)
    for k in range(1, residual.shape[0]):
        if done[k]:
            vals = np.asarray(residual[k])
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out[k] = float(np.median(vals))
    return out


def suspect_frames(medians: np.ndarray) -> list[int]:
    """Frames at or after a residual jump whose residual stays at the jumped level."""
    ks = [k for k in range(1, len(medians)) if np.isfinite(medians[k])]
    if len(ks) < 2:
        return []
    base = medians[ks[0]]
    if base <= 0:
        return []
    flagged: list[int] = []
    level = None
    for prev, k in zip(ks[:-1], ks[1:]):
        m = medians[k]
        jumped = m > JUMP_RATIO * medians[prev] and m > BASELINE_RATIO * base
        if level is None and jumped:
            level = medians[prev]
        if level is not None and m > JUMP_RATIO * level:
            flagged.append(k)
    return flagged


def describe(flagged: list[int], medians: np.ndarray, accumulative: bool) -> str | None:
    if not flagged:
        return None
    first = flagged[0]
    before = medians[[k for k in range(1, first) if np.isfinite(medians[k])]]
    ref = float(before.max()) if before.size else float("nan")
    advice = ("the displacement from frame 0 may exceed the model's reach: try incremental tracking "
              "or a model with a larger s" if accumulative else
              "the step between reference frames may exceed the model's reach: use more reference "
              "frames or a model with a larger s")
    frames = ", ".join(str(k) for k in flagged[:12]) + (" ..." if len(flagged) > 12 else "")
    return (f"tracking quality drops from frame {first} on (intensity residual {medians[first]:.1f} vs "
            f"{ref:.1f} grey levels before; frames {frames}); {advice}")
