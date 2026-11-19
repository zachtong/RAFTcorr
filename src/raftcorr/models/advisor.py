"""Which model to use: a quick s8 survey of the displacement, then the finest model that reaches it.

A smaller ``s`` resolves finer detail but reaches less far (nominal lookup radius ``32 s`` px).
The survey measures the displacement of the ROI on a few sampled (reference, frame) pairs with s8
(largest reach, fastest) and recommends the smallest ``s`` whose reach exceeds the largest
measured displacement (99th percentile) by a safety factor. In accumulative tracking every frame
is compared with frame 0, so the largest displacement of the sequence is what matters; with
incremental tracking only the displacement between a frame and its reference matters
(:func:`survey_pairs` picks the pairs of the run's schedule). Frame indices are 0-based.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..callbacks import Callbacks, coerce
from ..schedule import FrameSchedule
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


def survey_pairs(schedule: FrameSchedule, k: int = 3) -> list[tuple[int, int]]:
    """The (reference, frame) pairs worth surveying for a run with ``schedule``.

    Accumulative: frame 0 against the frames of :func:`sample_frames`. Otherwise only the step
    from a frame to its reference matters, and within the span of one reference it is largest at
    the span's last frame, the farthest from it: those last frames, sampled the same way.
    """
    if schedule.is_accumulative:
        return [(0, f) for f in sample_frames(schedule.n_frames, k)]
    span_ends: dict[int, int] = {}
    for ref, frame in schedule.pairs():
        span_ends[ref] = frame                       # pairs come in frame order: the last one wins
    ends = sorted((ref, frame) for ref, frame in span_ends.items())
    m = len(ends)
    if m <= k:
        return ends
    # The last span, the middle one and the one a quarter in, as sample_frames picks frames.
    return [ends[i] for i in sorted({m - 1, (m - 1) // 2, (m - 1) // 4})[: max(1, k)]]


def recommend(max_disp_px: float, safety: float = SAFETY) -> str:
    """Smallest released ``s`` whose reach (``32 s`` px) exceeds ``safety * max_disp_px``."""
    for key in ("s1", "s2", "s4", "s8"):
        spec = MODELS[key]
        if spec.lookup_radius_px >= safety * max_disp_px:
            return key
    return "s8"


def survey(pipeline_s8: Any, images: Sequence[np.ndarray], roi: np.ndarray, pairs: list[tuple[int, int]],
           callbacks: Callbacks | None = None, label: Callable[[int, int], str] | None = None) -> Advice:
    """Measure the displacement on ``pairs`` with an s8 :class:`~raftcorr.engine.pipeline.Pipeline`.

    ``label(reference, frame)`` (0-based indices) gives the progress text of a pair, e.g. a
    translated one; by default it is English and counts frames from 1, as the image lists do.
    """
    cb = coerce(callbacks)
    describe = label or (lambda r, k: f"survey: frame {k + 1} against frame {r + 1}")
    per: list[tuple[int, float]] = []
    for i, (r, k) in enumerate(pairs):
        cb.progress(i / max(1, len(pairs)), describe(r, k))
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
        incremental = any(r != 0 for r, _k in pairs)
        reason += ("; this exceeds even s8's reach: " +
                   ("update the reference more often" if incremental else "use incremental tracking"))
    return Advice(key, dmax, tuple(per), reason, beyond)
