"""Probe time-series and line CSV (port of pyALDIC's ``test_analysis/test_export_probes.py``
and ``test_export_line.py``).

RAFTcorr writes every value in image axes (v positive down), as all its results, so pyALDIC's
world-axes columns are gone; the header says which convention the file uses.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from raftcorr.analysis.engine import Kymograph
from raftcorr.analysis.probes import LineGeom, PointGeom, Probe
from raftcorr.analysis.series import FrameStatus, TimeSeries
from raftcorr.export.probes_csv import ProbeSeries, export_line_csv, export_probe_csv, run_parameters

from .test_analysis_engine import build_series, build_strain, stretch


def _series(values, unit="px", status_at=None) -> TimeSeries:
    v = np.asarray(values, dtype=float)
    status = [FrameStatus.OK if np.isfinite(x) else FrameStatus.NO_DATA for x in v]
    for i, st in (status_at or {}).items():
        status[i] = st
    return TimeSeries(frames=np.arange(len(v), dtype=np.int64), values=v,
                      valid_fraction=np.where(np.isfinite(v), 1.0, 0.0), status=status, unit=unit)


def _entry(label="p1", field="disp_u", reduction="value", values=(0.0, 1.0, 2.0), probe_id=1,
           **kw) -> ProbeSeries:
    return ProbeSeries(
        probe=Probe(id=probe_id, kind="point", geometry=PointGeom(10.0, 20.0), label=label, color="#FF0000"),
        field=field, reduction=reduction, series=_series(values, **kw),
    )


def _gauge(label="E1") -> ProbeSeries:
    return ProbeSeries(
        probe=Probe(id=2, kind="line", geometry=LineGeom(0.0, 0.0, 30.0, 40.0), label=label, color="#00FF00"),
        field=None, reduction="strain", series=_series((0.0, 0.01), unit=""),
    )


def _read(path):
    with open(path, encoding="utf-8-sig") as fh:
        lines = fh.read().splitlines()
    comments = [ln[1:].strip() for ln in lines if ln.startswith("#")]
    rows = list(csv.reader(ln for ln in lines if not ln.startswith("#")))
    return comments, rows[0], rows[1:]


# --- probe table -----------------------------------------------------------------


def test_one_row_per_frame_and_frames_are_one_based(tmp_path):
    _, header, rows = _read(export_probe_csv(tmp_path / "probes.csv", [_entry()]))
    assert header[0] == "frame"
    assert [r[0] for r in rows] == ["1", "2", "3"]


def test_the_image_column_names_each_frame(tmp_path):
    p = export_probe_csv(tmp_path / "probes.csv", [_entry()], frame_names=["a.tif", "b.tif", "c.tif"])
    _, header, rows = _read(p)
    assert header[:2] == ["frame", "image"]
    assert [r[1] for r in rows] == ["a.tif", "b.tif", "c.tif"]


def test_value_column_is_named_for_probe_field_and_reduction(tmp_path):
    _, header, _ = _read(export_probe_csv(tmp_path / "probes.csv", [_entry()]))
    assert "p1_disp_u_value" in header


def test_a_gauge_column_is_not_named_after_a_field_it_does_not_read(tmp_path):
    comments, header, _ = _read(export_probe_csv(tmp_path / "probes.csv", [_gauge()]))
    assert "E1_strain" in header
    assert "gauge strain" in "\n".join(comments)


def test_quality_columns_travel_with_the_value_and_can_be_turned_off(tmp_path):
    _, header, _ = _read(export_probe_csv(tmp_path / "a.csv", [_entry()]))
    assert {"p1_disp_u_value_valid_fraction", "p1_disp_u_value_flag"} <= set(header)
    _, header, _ = _read(export_probe_csv(tmp_path / "b.csv", [_entry()], include_quality=False))
    assert header == ["frame", "p1_disp_u_value"]


def test_missing_values_are_empty_cells_not_the_word_nan(tmp_path):
    _, header, rows = _read(export_probe_csv(tmp_path / "probes.csv", [_entry(values=(0.0, np.nan, 2.0))]))
    assert rows[1][header.index("p1_disp_u_value")] == ""
    assert all("nan" not in cell.lower() for row in rows for cell in row)


def test_flag_column_says_why(tmp_path):
    entry = _entry(values=(0.0, np.nan, 2.0),
                   status_at={1: FrameStatus.BELOW_THRESHOLD, 2: FrameStatus.UNTRACKED})
    _, header, rows = _read(export_probe_csv(tmp_path / "probes.csv", [entry]))
    flag = header.index("p1_disp_u_value_flag")
    assert [r[flag] for r in rows] == ["ok", "below_threshold", "untracked"]
    assert rows[2][header.index("p1_disp_u_value")] == "2", "a flagged frame can still carry a value"


def test_header_records_geometry_units_frames_axes_and_version(tmp_path):
    blob = "\n".join(_read(export_probe_csv(tmp_path / "probes.csv", [_gauge("gauge")]))[0])
    assert "RAFTcorr" in blob
    assert "line (0.000, 0.000) -> (30.000, 40.000) px" in blob
    assert "length 50.000 px" in blob
    assert "dimensionless" in blob
    assert "index N-1 of the scientific export" in blob
    assert "image axes" in blob and "v is positive down" in blob


def test_run_parameters_come_from_the_results(tmp_path):
    s = build_series(stretch(), 2)
    params = run_parameters(s, build_strain(s, lambda x, y, f: 0.0 * x, step=3))
    assert params["model"].startswith("synthetic (sha256 abababab")
    assert params["iterations"] == "20"
    assert params["tracking"] == "accumulative"
    assert params["strain"] == ("green-lagrange, virtual strain gauge 41 px (uniform weighting, plane fit), "
                                "step 3 px")
    p = export_probe_csv(tmp_path / "probes.csv", [_entry()], parameters={**params, "extra": "x"})
    blob = "\n".join(_read(p)[0])
    assert "Good Practices" in blob and "iterations: 20" in blob and "extra: x" in blob
    assert run_parameters(None) == {"software": params["software"]}


def test_time_column_appears_only_with_a_frame_rate(tmp_path):
    assert "time_s" not in _read(export_probe_csv(tmp_path / "a.csv", [_entry()]))[1]
    _, header, rows = _read(export_probe_csv(tmp_path / "b.csv", [_entry()], frame_rate=2.0))
    col = header.index("time_s")
    assert [r[col] for r in rows] == ["0", "0.5", "1"]


def test_several_probes_share_one_file_and_duplicates_are_disambiguated(tmp_path):
    _, header, _ = _read(export_probe_csv(tmp_path / "a.csv", [_entry(label="a"), _entry(label="b", probe_id=2)]))
    assert "a_disp_u_value" in header and "b_disp_u_value" in header
    _, header, _ = _read(export_probe_csv(tmp_path / "b.csv", [_entry(label="x"), _entry(label="x", probe_id=7)]))
    assert "x_disp_u_value_id1" in header and "x_disp_u_value_id7" in header


def test_series_of_different_lengths_or_none_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="same frames"):
        export_probe_csv(tmp_path / "p.csv", [_entry(values=(0.0, 1.0)), _entry(values=(0.0,))])
    with pytest.raises(ValueError, match="Nothing to export"):
        export_probe_csv(tmp_path / "p.csv", [])


def test_a_newline_in_a_label_cannot_break_the_header(tmp_path):
    _, header, rows = _read(export_probe_csv(tmp_path / "p.csv", [_entry(label="left\nedge")]))
    assert "left edge_disp_u_value" in header
    assert len(rows) == 3


def test_the_file_opens_in_excel_with_cjk_labels(tmp_path):
    p = export_probe_csv(tmp_path / "p.csv", [_entry(label="试样中心")], include_quality=False)
    with open(p, "rb") as fh:
        assert fh.read(3) == b"\xef\xbb\xbf"
    assert _read(p)[1][1].startswith("试样中心")


def test_extra_per_frame_columns_follow_the_frame(tmp_path):
    p = export_probe_csv(tmp_path / "probes.csv", [_entry()],
                         extra_columns={"load_N": np.array([0.0, 5.0]), "stress_MPa": np.array([0.0, np.nan])},
                         notes=["load_N: from machine.csv"])
    comments, header, rows = _read(p)
    assert header[:3] == ["frame", "load_N", "stress_MPa"]
    assert [r[1] for r in rows] == ["0", "5", ""], "a missing frame is an empty cell"
    assert [r[2] for r in rows] == ["0", "", ""]
    assert "load_N: from machine.csv" in comments


# --- line table ------------------------------------------------------------------


def _probe(label="E1") -> Probe:
    return Probe(id=3, kind="line", geometry=LineGeom(10.0, 20.0, 40.0, 20.0), label=label, color="#00FF00")


def _kymo(values, unit="px", distance_unit="px", n_xy=None) -> Kymograph:
    v = np.asarray(values, dtype=float)                 # [frames, samples]
    n_frames, n = v.shape
    m = n if n_xy is None else n_xy
    return Kymograph(frames=np.arange(n_frames, dtype=np.int64), distance=np.linspace(0.0, 30.0, n),
                     xy=np.column_stack([np.linspace(10.0, 40.0, m), np.full(m, 20.0)]),
                     values=v, status=np.zeros(v.shape, dtype=np.uint8), unit=unit, distance_unit=distance_unit)


def test_rows_are_samples_and_columns_are_frames(tmp_path):
    p = export_line_csv(tmp_path / "l.csv", _probe(), _kymo([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]), field="disp_u")
    _, header, rows = _read(p)
    assert header == ["distance_px", "x_px", "y_px", "frame_1", "frame_2"]
    assert len(rows) == 3
    assert [float(c) for c in rows[2]] == pytest.approx([30.0, 40.0, 20.0, 0.0, 0.3])


def test_a_gap_is_an_empty_cell_not_a_zero(tmp_path):
    comments, _, rows = _read(export_line_csv(tmp_path / "l.csv", _probe(), _kymo([[0.0, 0.0], [np.nan, 0.5]]),
                                              field="disp_u"))
    assert rows[0][-1] == ""
    assert any("empty cell" in c for c in comments)


def test_the_distance_column_carries_the_length_unit(tmp_path):
    p = export_line_csv(tmp_path / "l.csv", _probe(), _kymo([[0.0, 0.0]], unit="mm", distance_unit="mm"),
                        field="disp_u")
    _, header, _ = _read(p)
    assert header[0] == "distance_mm"
    assert header[1:3] == ["x_px", "y_px"], "positions stay in image pixels"


def test_the_header_says_what_was_measured_and_where(tmp_path):
    p = export_line_csv(tmp_path / "l.csv", _probe("gauge, 1\nA"), _kymo([[0.0, 0.0]]), field="disp_v",
                        frame_rate=10.0, parameters={"model": "RAFTcorr-s4"})
    text = "\n".join(_read(p)[0])
    assert "gauge, 1 A" in text, "a newline in a label must not break the header"
    assert "(10.000, 20.000) -> (40.000, 20.000)" in text
    assert "disp_v" in text and "image axes (positive down)" in text
    assert "frame_N is image N" in text
    assert "(N - 1) / 10" in text
    assert "model: RAFTcorr-s4" in text


def test_strain_is_labelled_dimensionless(tmp_path):
    p = export_line_csv(tmp_path / "l.csv", _probe(), _kymo([[0.0, 0.0]], unit=""), field="strain_eyy")
    assert any("strain_eyy, unit dimensionless" in c for c in _read(p)[0])


def test_mismatched_positions_are_refused(tmp_path):
    with pytest.raises(ValueError, match="positions"):
        export_line_csv(tmp_path / "l.csv", _probe(), _kymo([[0.0, 0.0, 0.0]], n_xy=2), field="disp_u")
