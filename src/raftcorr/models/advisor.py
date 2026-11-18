"""Which model to use: a quick s8 survey of the displacement, then the finest model that reaches it.

A smaller ``s`` resolves finer detail but reaches less far (nominal lookup radius ``32 s`` px).
The survey measures the displacement of the ROI between frame 0 and a few sampled frames with s8
(largest reach, fastest) and recommends the smallest ``s`` whose reach exceeds the largest
measured displacement (99th percentile) by a safety factor. In accumulative tracking every frame
is compared with frame 0, so the largest displacement of the sequence is what matters; with
incremental tracking only the displacement between consecutive references matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..callbacks import Callbacks, coerce
from .zoo import MODELS

SAFETY = 1.3


@dataclass(frozen=True)
class Advice:
    key: str                               # recommended model
    max_displacement_px: float             # 99th percentile over the sampled frames
    per_frame: tuple[tuple[int, float], ...]   # (0-based frame index, 99th percentile |d|)
    reason: str
    beyond_reach: bool = False             # even s8 is too short: incremental tracking is needed


def sample_frames(n_frames: int, k: int = 3) -> list[int]:
    if n_frames < 2:
        return []
    picks = {n_frames - 1, max(1, (n_frames - 1) // 2), max(1, (n_frames - 1) // 4)}
    return sorted(picks)[: max(1, k)]


def recommend(max_disp_px: float, safety: float = SAFETY) -> str:
    """Smallest released ``s`` whose reach (``32 s`` px) exceeds ``safety * max_disp_px``."""
    for key in ("s1", "s2", "s4", "s8"):
        spec = MODELS[key]
        if spec.lookup_radius_px >= safety * max_disp_px:
            return key
    return "s8"


def survey(pipeline_s8: Any, images: Sequence[np.ndarray], roi: np.ndarray, pairs: list[tuple[int, int]],
           callbacks: Callbacks | None = None) -> Advice:
    """Measure the displacement on ``pairs`` with an s8 :class:`~raftcorr.engine.pipeline.Pipeline`."""
    cb = coerce(callbacks)
    per: list[tuple[int, float]] = []
    for i, (r, k) in enumerate(pairs):
        cb.progress(i / max(1, len(pairs)), f"survey: frame {k} against frame {r}")
        field, x0, y0 = pipeline_s8.solve_pair(np.asarray(images[r]), np.asarray(images[k]), roi)
        h, w = field.shape[1:]
        sub = roi[y0 : y0 + h, x0 : x0 + w]
        mag = np.hypot(field[0], field[1])[sub]
        mag = mag[np.isfinite(mag)]
        per.append((k, float(np.percentile(mag, 99)) if mag.size else 0.0))
    cb.progress(1.0, "survey done")
    dmax = max((d for _, d in per), default=0.0)
    key = recommend(dmax)
    reach = MODELS[key].lookup_radius_px
    reason = (f"largest displacement of the ROI about {dmax:.1f} px (99th percentile over frames "
              f"{', '.join(str(k) for k, _ in per)}); RAFTcorr-{key} reaches {reach} px")
    beyond = key == "s8" and dmax * SAFETY > reach
    if beyond:
        reason += "; this exceeds even s8's reach: use incremental tracking"
    return Advice(key, dmax, tuple(per), reason, beyond)
