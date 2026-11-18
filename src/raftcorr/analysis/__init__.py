"""Probe-based post-processing analysis of RAFTcorr's dense results.

Core layer: no Qt, no ``tr()``. The GUI in ``raftcorr.gui.panels.analysis``
is one consumer; a script holding a ``DisplacementSeries`` is another.
"""

from raftcorr.analysis.engine import (
    AnalysisEngine,
    Kymograph,
    Profile,
    SampleStatus,
)
from raftcorr.analysis.extract import extract_series, field_values, frame_count
from raftcorr.analysis.load_data import (
    LoadData,
    LoadSync,
    LoadTable,
    parse_load_table,
    read_load_table,
)
from raftcorr.analysis.probes import (
    AreaGeom,
    LineGeom,
    PointGeom,
    Probe,
    ProbeSet,
    allowed_reductions,
)
from raftcorr.analysis.series import FrameStatus, TimeSeries

__all__ = [
    "AnalysisEngine",
    "AreaGeom",
    "FrameStatus",
    "Kymograph",
    "LineGeom",
    "LoadData",
    "LoadSync",
    "LoadTable",
    "PointGeom",
    "Probe",
    "ProbeSet",
    "Profile",
    "SampleStatus",
    "TimeSeries",
    "allowed_reductions",
    "extract_series",
    "field_values",
    "frame_count",
    "parse_load_table",
    "read_load_table",
]
