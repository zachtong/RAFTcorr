"""A probe's curve, and the reason behind every frame that is not plain.

A ``TimeSeries`` carries its own gaps and the reason for each one. Charts and
CSV both read ``status``, so what a user sees on screen and what they get in a
file cannot disagree about which frames mean anything.

``status`` says *why* a frame is special; whether a value survived is a
separate fact (``values[i]`` finite or NaN). A line that crosses a region the
run lost on some frame (a crack or hole cut out of the deformed frame's ROI,
material that left the image) keeps a value from the material either side and
is marked ``UNTRACKED``; a region whose reliable points collapse is
``BELOW_THRESHOLD`` and blank.

Values are in image axes, as every RAFTcorr result: ``v`` positive down.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


class FrameStatus(enum.Enum):
    """Why a frame is not plain OK. The value's presence is separate."""

    #: Everything the probe covers was measured.
    OK = "ok"
    #: Part of the probe's frame-0 material has no displacement on this frame
    #: (excluded by the frame's ROI -- a crack, a hole -- or out of view). For
    #: a line or region the value comes from the material that remains; for a
    #: gauge it is the reading across the lost part (elongation including a
    #: crack's opening).
    UNTRACKED = "untracked"
    #: Fewer valid points than the threshold asks for; left blank.
    BELOW_THRESHOLD = "below_threshold"
    #: Only unreliable points: tracked material without a strain value (the
    #: strain gauge found too few points or an ill-conditioned fit there).
    UNRELIABLE = "unreliable"
    #: A gauge endpoint is not on tracked material on this frame.
    ENDPOINT_LOST = "endpoint_lost"
    #: The field does not exist for this frame -- the frame was not computed,
    #: or strain was not computed.
    NOT_COMPUTED = "not_computed"
    #: The probe covers no material at all.
    NO_DATA = "no_data"


@dataclass(frozen=True)
class TimeSeries:
    """One probe, one quantity, across the sequence."""

    frames: NDArray[np.int64]
    values: NDArray[np.float64]
    valid_fraction: NDArray[np.float64]
    status: list[FrameStatus]
    unit: str

    def first_frame(self, status: FrameStatus) -> int | None:
        """The first frame with *status*, if any -- e.g. where tracking was lost."""
        for i, st in enumerate(self.status):
            if st is status:
                return int(self.frames[i])
        return None

    def contiguous_runs(self) -> Iterator[tuple[int, int]]:
        """Half-open index ranges of consecutive finite values.

        Charts draw one polyline per run, so a gap stays a gap.
        """
        finite = np.isfinite(self.values)
        start: int | None = None
        for i, ok in enumerate(finite):
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                yield (start, i)
                start = None
        if start is not None:
            yield (start, len(finite))


__all__ = ["FrameStatus", "TimeSeries"]
