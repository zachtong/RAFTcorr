"""Parameters of a run as a JSON file (pyALDIC's ``export_params``, RAFTcorr results).

The file records everything needed to reproduce and read the results without the GUI session:
software version, frames (names, which were computed, reference of each frame), the grid and the
ROI size, the run provenance (model, checkpoint hash, iterations, correlation mode, device, the
run configuration *as run*), the strain settings when a strain series is given, the calibration
(pixel size, length unit, frame rate) and the sign / indexing conventions.

The data files themselves stay in pixels; this file carries the calibration next to them.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import numpy as np

from .._version import __version__
from ..post.strain import COMPONENTS
from ..results import DisplacementSeries, StrainSeries
from .scientific import CONVENTIONS, PhysicalUnits, atomic_output, to_json

__all__ = ["export_params", "run_parameters"]


def run_parameters(series: DisplacementSeries, strain: StrainSeries | None = None,
                   units: PhysicalUnits | None = None) -> dict[str, Any]:
    """Every recorded parameter of ``series`` (and ``strain``) as a JSON-ready dict."""
    return {
        "software": "RAFTcorr",
        "software_version": __version__,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "n_frames": series.n_frames,
        "frame_names": list(series.frame_names),
        "computed_frames": series.completed_frames,
        "schedule": series.schedule.to_dict(),
        "grid": series.grid.to_dict(),
        "roi_points": int(np.count_nonzero(series.mask)),
        "has_residual": series.residual is not None,
        "run_info": series.info.to_dict(),
        "strain": None if strain is None else _strain_parameters(strain),
        "physical_units": (units or PhysicalUnits()).to_dict(),
        "conventions": CONVENTIONS,
    }


def export_params(path: str | Path, series: DisplacementSeries, strain: StrainSeries | None = None,
                  units: PhysicalUnits | None = None, overwrite: bool = False) -> Path:
    """Write :func:`run_parameters` to ``path`` (indented JSON, UTF-8), atomically.

    Raises FileExistsError when ``path`` exists and ``overwrite`` is False.
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (pass overwrite=True to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = to_json(run_parameters(series, strain, units), indent=2)
    with atomic_output(path) as tmp:
        tmp.write_text(text + "\n", encoding="utf-8")
    return path


def _strain_parameters(strain: StrainSeries) -> dict[str, Any]:
    return {
        "settings": strain.settings.to_dict(),
        "step": int(strain.step),
        "components": [name for name in COMPONENTS if name in strain.components],
        "computed_frames": [int(k) for k in np.flatnonzero(strain.done)],
    }
