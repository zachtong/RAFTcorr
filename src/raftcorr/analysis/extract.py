"""Turn a probe and a finished run into a curve -- the scripting entry point.

    from raftcorr.analysis import LineGeom, Probe, extract_series

    probe = Probe(id=1, kind="line", geometry=LineGeom(100, 200, 400, 200),
                  label="E1", color="#ef4444")
    ts = extract_series(series, probe, None, "strain")
    ts = extract_series(series, probe, "strain_exx", "mean", strain=strain)

A thin front for ``AnalysisEngine``: each call builds one engine. Code reading
several probes or fields from the same run should hold an engine and call it
directly, so the sample plans are reused.

Frame indices follow ``raftcorr.results``: 0 is the reference image, frame
``k`` is image ``k``.

Physical units are parameters, never read from application state. Lengths --
displacement fields, elongation, crack opening -- are multiplied by
``pixel_size``; strains are ratios and never are.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from raftcorr.analysis.engine import DEFAULT_MIN_VALID_FRACTION, AnalysisEngine
from raftcorr.analysis.probes import Probe
from raftcorr.analysis.series import TimeSeries
from raftcorr.gui.support.fields import is_strain_field, validate_field
from raftcorr.results import DisplacementSeries, StrainSeries


def field_values(
    series: DisplacementSeries,
    field: str,
    frame: int,
    *,
    strain: StrainSeries | None = None,
    pixel_size: float = 1.0,
) -> NDArray[np.float64] | None:
    """*field* on *frame* on its grid, NaN where not valid.

    The displacement grid is ``series.grid``; strain is on the same box
    subsampled by ``strain.step``. None when the field does not exist for
    this run (strain not computed, or a frame not computed or past the end).
    """
    validate_field(field)
    got = AnalysisEngine(series, strain).node_field(field, frame)
    if got is None:
        return None
    vals, valid = got
    out = np.where(valid, vals, np.nan)
    return out if is_strain_field(field) else out * float(pixel_size)


def frame_count(series: DisplacementSeries) -> int:
    """Number of frames a probe can report on, reference frame included."""
    return series.n_frames


def extract_series(
    series: DisplacementSeries,
    probe: Probe,
    field: str | None,
    reduction: str,
    *,
    strain: StrainSeries | None = None,
    pixel_size: float = 1.0,
    length_unit: str = "px",
    min_valid_fraction: float = DEFAULT_MIN_VALID_FRACTION,
    frames: Sequence[int] | None = None,
) -> TimeSeries:
    """Read *probe* across the sequence and reduce each frame to one number.

    Parameters
    ----------
    reduction:
        A statistic (``value``, ``mean``, ``median``, ``max``, ``min``,
        ``std``, ``valid_fraction``) or, for a line probe, a gauge quantity
        (``strain``, ``true_strain``, ``elongation``, ``cod``,
        ``cod_sliding``, ``cod_magnitude``), for which *field* is ignored.
    strain:
        The strain series, needed for the strain fields.
    min_valid_fraction:
        A line's or region's frame is left blank when fewer of its samples
        than this are valid.
    """
    engine = AnalysisEngine(series, strain)
    return engine.series(
        probe, field, reduction,
        pixel_size=pixel_size,
        length_unit=length_unit,
        min_valid_fraction=min_valid_fraction,
        frames=frames,
    )


__all__ = [
    "DEFAULT_MIN_VALID_FRACTION", "extract_series", "field_values",
    "frame_count",
]
