"""Per-frame CSV tables, the parameters JSON and the HTML report (raftcorr.export.frame_csv /
params / report): content, units, frame selection, errors and cancellation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pytest

from raftcorr.callbacks import FunctionCallbacks, StopRequested
from raftcorr.export.frame_csv import csv_variables, export_frame_csv, frame_csv_paths
from raftcorr.export.params import export_params, run_parameters
from raftcorr.export.report import export_html_report, frame_statistics, sampled_frames
from raftcorr.export.scientific import CONVENTIONS, PhysicalUnits
from raftcorr.post.strain import COMPONENTS
from raftcorr.viz.fields import FieldProvider, Units

from .test_export_scientific import GRID, STEP, make_series, make_strain


def read_table(path: Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


# ------------------------------------------------------------------------------------ frame_csv
def test_tables_of_every_computed_frame_with_a_separate_strain_table(tmp_path):
    series, strain = make_series(), make_strain()          # strain step 2: its own rows
    paths = export_frame_csv(tmp_path, series, strain)
    assert [p.name for p in paths] == ["frame_0000.csv", "frame_0000_strain.csv",
                                       "frame_0001.csv", "frame_0001_strain.csv"]   # frame 2 not computed
    assert paths == frame_csv_paths(tmp_path, series, strain)

    rows = read_table(tmp_path / "frame_0001.csv")
    assert rows[0] == ["x [px]", "y [px]", "U [px]", "V [px]", "Magnitude [px]"]
    ii, jj = np.nonzero(series.mask)
    assert len(rows) - 1 == ii.size
    xs = np.array([float(r[0]) for r in rows[1:]])
    np.testing.assert_array_equal(xs, GRID.x0 + jj)
    u = np.array([float(r[2]) if r[2] else np.nan for r in rows[1:]], dtype=np.float32)
    np.testing.assert_array_equal(u, series.u[1][series.mask])       # 9 digits: float32 round trip
    assert np.isnan(u).sum() == 1                                     # the untracked point: empty field

    srows = read_table(tmp_path / "frame_0001_strain.csv")
    assert srows[0] == ["x [px]", "y [px]"] + [
        f"{c} [{'deg' if c in ('rotation', 'principal_angle') else '-'}]" for c in COMPONENTS]
    si, sj = np.nonzero(series.mask[::STEP, ::STEP])
    assert len(srows) - 1 == si.size
    assert float(srows[1][0]) == GRID.x0 + sj[0] * STEP and float(srows[1][1]) == GRID.y0 + si[0] * STEP
    exx = np.array([float(r[2]) if r[2] else np.nan for r in srows[1:]], dtype=np.float32)
    np.testing.assert_array_equal(exx, strain.components["exx"][1][::1][si, sj])


def test_table_variables_and_frames_are_selectable(tmp_path):
    series = make_series()
    paths = export_frame_csv(tmp_path, series, None, variables=["V"], frames=[1, 2])
    assert [p.name for p in paths] == ["frame_0001.csv"]                # frame 2 not computed
    assert read_table(paths[0])[0] == ["x [px]", "y [px]", "V [px]"]
    assert csv_variables(None, series) == ("U", "V", "Magnitude")       # no residual in this series


def test_table_errors_and_cancellation(tmp_path):
    series, strain = make_series(), make_strain()
    with pytest.raises(ValueError, match="strain variables need a strain series"):
        export_frame_csv(tmp_path, series, None, variables=["exx"])
    with pytest.raises(ValueError, match="no variables"):
        export_frame_csv(tmp_path, series, strain, variables=[])
    with pytest.raises(IndexError):
        export_frame_csv(tmp_path, series, strain, frames=[5])
    export_frame_csv(tmp_path, series, None, variables=["U"])
    with pytest.raises(FileExistsError):
        export_frame_csv(tmp_path, series, None, variables=["U"])
    export_frame_csv(tmp_path, series, None, variables=["U"], overwrite=True)
    out = tmp_path / "cancelled"
    with pytest.raises(StopRequested):
        export_frame_csv(out, series, strain, callbacks=FunctionCallbacks(should_stop=lambda: True))
    assert list(out.iterdir()) == []


# --------------------------------------------------------------------------------------- params
def test_parameters_json_records_the_run(tmp_path):
    series, strain = make_series(), make_strain()
    units = PhysicalUnits(0.25, "mm", 50.0)
    path = export_params(tmp_path / "p.json", series, strain, units)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == json.loads(json.dumps(run_parameters(series, strain, units) | {"created": data["created"]}))
    assert data["run_info"]["model_name"] == "RAFTcorr-s4" and data["run_info"]["iters"] == 20
    assert data["computed_frames"] == [0, 1] and data["grid"] == GRID.to_dict()
    assert data["roi_points"] == int(series.mask.sum())
    assert data["strain"]["step"] == STEP and data["strain"]["computed_frames"] == [0, 1]
    assert data["physical_units"] == {"scale": 0.25, "unit": "mm", "frame_rate": 50.0}
    assert data["conventions"] == CONVENTIONS
    assert run_parameters(series)["strain"] is None
    with pytest.raises(FileExistsError):
        export_params(path, series)
    export_params(path, series, overwrite=True)


# --------------------------------------------------------------------------------------- report
def test_report_statistics_pictures_and_parameters(tmp_path):
    series, strain = make_series(), make_strain()
    provider = FieldProvider(series, strain)
    units = Units(pixel_size=2.0, length_unit="mm")
    path = export_html_report(tmp_path / "r.html", provider, None, ["u", "exx", "residual"], units=units,
                              sample_every=1)
    page = path.read_text(encoding="utf-8")
    assert "u (horizontal displacement) [mm]" in page and "εxx [-]" in page
    assert "intensity residual" not in page                       # not in this run: skipped
    assert page.count("data:image/png;base64,") == 2 + 2           # computed frames 0, 1 of each
    assert "RAFTcorr-s4" in page and "physical_units.scale" in page

    stats = frame_statistics(provider, "u", 1, units)
    values = series.u[1][series.mask]
    assert stats["min"] == pytest.approx(2.0 * np.nanmin(values))
    assert stats["std"] == pytest.approx(2.0 * np.nanstd(values), rel=1e-5)
    assert f"{stats['max']:.4g}" in page
    assert frame_statistics(provider, "u", 2, units) is None     # not computed
    assert re.search(r"<tr><td>2</td><td class='num'>", page)    # 1-based frame numbers


def test_report_sections_are_optional(tmp_path):
    provider = FieldProvider(make_series())
    page = export_html_report(tmp_path / "r.html", provider, fields=["v"], include_parameters=False,
                              include_statistics=False, include_images=False).read_text(encoding="utf-8")
    assert "<table>" not in page and "base64" not in page and "Run parameters" not in page
    with pytest.raises(FileExistsError):
        export_html_report(tmp_path / "r.html", provider)
    with pytest.raises(StopRequested):
        export_html_report(tmp_path / "c.html", provider, callbacks=FunctionCallbacks(should_stop=lambda: True))
    assert not (tmp_path / "c.html").exists()


def test_sampled_frames_keep_the_last_one():
    assert sampled_frames([0, 1, 2, 3, 4, 5, 6], 3) == [0, 3, 6]
    assert sampled_frames([0, 1, 2, 3, 4], 3) == [0, 3, 4]
    assert sampled_frames([], 5) == []
