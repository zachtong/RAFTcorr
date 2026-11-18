"""RAFTcorr: learned digital image correlation (desktop application, command line, Python library).

Three lines from an image folder to displacement and strain::

    import raftcorr
    series = raftcorr.track("images/", roi="roi.png", model="s4")
    strain = raftcorr.compute_strain(series)

Conventions (:mod:`raftcorr.conventions`): ``(u, v)`` = reference -> deformed displacement, +u right,
+v down, in pixels, stored in frame-0 (Lagrangian) coordinates; frame ``k`` of a result is image
``k`` of the sequence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._version import __version__
from .callbacks import Callbacks, ConsoleCallbacks, FunctionCallbacks, NullCallbacks, StopRequested
from .config import PAPER_STRAIN, RunConfig, StrainConfig, TilingConfig, TrackingConfig
from .results import DisplacementSeries, RunInfo, StrainSeries
from .schedule import FrameSchedule


def track(
    images: str | Path | Sequence[np.ndarray] | Any,
    roi: str | Path | np.ndarray | None = None,
    model: str = "s4",
    config: RunConfig | None = None,
    frame_masks: dict[int, np.ndarray] | None = None,
    callbacks: Callbacks | None = None,
) -> DisplacementSeries:
    """Displacement of every frame relative to frame 0.

    ``images``: a folder, a list of image files, an :class:`~raftcorr.io.ImageSequence` or a list
    of 2-D arrays (grey levels 0-255). ``roi``: a mask image path, a boolean array, or None (whole
    image). ``model``: ``s1``, ``s2``, ``s4``, ``s8`` or a checkpoint path (ignored when ``config``
    is given).
    """
    from .engine.pipeline import Pipeline
    from .io.images import ImageSequence
    from .roi.masks import load_mask

    if isinstance(images, (str, Path)):
        seq: Any = ImageSequence.from_folder(images)
    elif isinstance(images, (list, tuple)) and images and isinstance(images[0], (str, Path)):
        seq = ImageSequence(list(images))
    else:
        seq = images
    shape = np.asarray(seq[0]).shape
    mask = load_mask(roi, shape) if isinstance(roi, (str, Path)) else roi
    cfg = config or RunConfig(model=model)
    return Pipeline(cfg, callbacks).run(seq, mask, frame_masks=frame_masks)


def compute_strain(series: DisplacementSeries, config: StrainConfig | None = None,
                   callbacks: Callbacks | None = None) -> StrainSeries:
    """Strain of every computed frame (paper settings by default)."""
    from .post.strain import compute_strain as _cs

    return _cs(series, config or PAPER_STRAIN, callbacks=callbacks)


def load_model(key_or_path: str = "s4", device: str = "auto"):
    """A released model (``s1`` ... ``s8``) or a checkpoint, ready for inference."""
    from .models import load_model as _lm
    from .models import resolve_weights

    return _lm(resolve_weights(key_or_path), device)


__all__ = [
    "Callbacks",
    "ConsoleCallbacks",
    "DisplacementSeries",
    "FrameSchedule",
    "FunctionCallbacks",
    "NullCallbacks",
    "PAPER_STRAIN",
    "RunConfig",
    "RunInfo",
    "StopRequested",
    "StrainConfig",
    "StrainSeries",
    "TilingConfig",
    "TrackingConfig",
    "__version__",
    "compute_strain",
    "load_model",
    "track",
]
