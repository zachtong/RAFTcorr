"""Deep tests for probe CRUD, time series extraction, and CSV export endpoints."""

import csv
import io

import numpy as np
import pytest

from server.session import session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strain_results(num_frames=3, h=64, w=64):
    """Build mock strain results for probe extraction tests."""
    results = []
    for _ in range(num_frames):
        strain_dict = {
            "exx": np.random.randn(h, w).astype(np.float32) * 0.01,
            "eyy": np.random.randn(h, w).astype(np.float32) * 0.01,
        }
        results.append(strain_dict)
    return results


# ===========================================================================
# 1. Point Probe CRUD
# ===========================================================================

class TestPointProbeCrud:
    """Add, list, delete point probes."""

    def test_add_point_basic(self, client):
        resp = client.post("/api/probes/point", json={"x": 32, "y": 16})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["type"] == "point"
        assert data["id"] == 1
        assert data["coords"] == [32.0, 16.0]

    def test_add_multiple_points(self, client):
        r1 = client.post("/api/probes/point", json={"x": 10, "y": 20})
        r2 = client.post("/api/probes/point", json={"x": 30, "y": 40})
        assert r1.get_json()["id"] == 1
        assert r2.get_json()["id"] == 2

        resp = client.get("/api/probes")
        probes = resp.get_json()["probes"]
        assert len(probes) == 2

    def test_add_point_at_origin(self, client):
        resp = client.post("/api/probes/point", json={"x": 0, "y": 0})
        assert resp.status_code == 200
        assert resp.get_json()["coords"] == [0.0, 0.0]

    def test_add_point_float_coords(self, client):
        resp = client.post("/api/probes/point", json={"x": 15.5, "y": 22.7})
        assert resp.status_code == 200
        coords = resp.get_json()["coords"]
        assert abs(coords[0] - 15.5) < 1e-6
        assert abs(coords[1] - 22.7) < 1e-6

    def test_point_has_color(self, client):
        resp = client.post("/api/probes/point", json={"x": 10, "y": 10})
        data = resp.get_json()
        assert "color" in data
        assert data["color"].startswith("#")

    def test_point_has_label(self, client):
        resp = client.post("/api/probes/point", json={"x": 10, "y": 10})
        data = resp.get_json()
        assert "label" in data
        assert len(data["label"]) > 0

    def test_delete_point_by_id(self, client):
        client.post("/api/probes/point", json={"x": 10, "y": 10})
        client.post("/api/probes/point", json={"x": 20, "y": 20})

        resp = client.delete("/api/probes/1?type=point")
        assert resp.status_code == 200

        probes = client.get("/api/probes").get_json()["probes"]
        assert len(probes) == 1
        assert probes[0]["id"] == 2

    def test_delete_nonexistent_probe(self, client):
        """Deleting a probe that doesn't exist should still return 200 (idempotent)."""
        resp = client.delete("/api/probes/999?type=point")
        assert resp.status_code == 200


# ===========================================================================
# 2. Line Probe CRUD
# ===========================================================================

class TestLineProbeCrud:

    def test_add_line(self, client):
        resp = client.post("/api/probes/line", json={
            "p1": [10, 10], "p2": [50, 50]
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["type"] == "line"
        assert data["id"] == 1

    def test_add_multiple_lines(self, client):
        client.post("/api/probes/line", json={"p1": [0, 0], "p2": [10, 10]})
        client.post("/api/probes/line", json={"p1": [5, 5], "p2": [15, 15]})
        probes = client.get("/api/probes").get_json()["probes"]
        line_probes = [p for p in probes if p["type"] == "line"]
        assert len(line_probes) == 2

    def test_delete_line_by_id(self, client):
        client.post("/api/probes/line", json={"p1": [0, 0], "p2": [10, 10]})
        resp = client.delete("/api/probes/1?type=line")
        assert resp.status_code == 200
        probes = client.get("/api/probes").get_json()["probes"]
        assert len(probes) == 0

    def test_line_coords_structure(self, client):
        resp = client.post("/api/probes/line", json={
            "p1": [5, 15], "p2": [55, 45]
        })
        coords = resp.get_json()["coords"]
        assert len(coords) == 2
        assert len(coords[0]) == 2
        assert len(coords[1]) == 2


# ===========================================================================
# 3. Area Probe CRUD
# ===========================================================================

class TestAreaProbeCrud:

    def test_add_rect_area(self, client):
        resp = client.post("/api/probes/area", json={
            "shape": "rect", "data": [10, 10, 30, 30]
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["type"] == "area"
        assert data["id"] == 1

    def test_add_circle_area(self, client):
        resp = client.post("/api/probes/area", json={
            "shape": "circle", "data": [32, 32, 10]
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["type"] == "area"

    def test_add_polygon_area(self, client):
        resp = client.post("/api/probes/area", json={
            "shape": "polygon",
            "data": [[10, 10], [50, 10], [50, 50], [10, 50]],
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["type"] == "area"

    def test_invalid_shape_returns_400(self, client):
        resp = client.post("/api/probes/area", json={
            "shape": "hexagon", "data": []
        })
        assert resp.status_code == 400
        assert "shape" in resp.get_json()["error"].lower()

    def test_area_has_coords_with_shape(self, client):
        resp = client.post("/api/probes/area", json={
            "shape": "rect", "data": [5, 5, 25, 25]
        })
        coords = resp.get_json()["coords"]
        assert coords["shape"] == "rect"
        assert coords["data"] == [5.0, 5.0, 25.0, 25.0]

    def test_delete_area(self, client):
        client.post("/api/probes/area", json={
            "shape": "rect", "data": [10, 10, 30, 30]
        })
        resp = client.delete("/api/probes/1?type=area")
        assert resp.status_code == 200
        probes = client.get("/api/probes").get_json()["probes"]
        assert len(probes) == 0


# ===========================================================================
# 4. List and Clear
# ===========================================================================

class TestListAndClear:

    def test_list_empty(self, client):
        resp = client.get("/api/probes")
        assert resp.status_code == 200
        assert resp.get_json()["probes"] == []

    def test_list_mixed_types(self, client):
        client.post("/api/probes/point", json={"x": 10, "y": 10})
        client.post("/api/probes/line", json={"p1": [0, 0], "p2": [10, 10]})
        client.post("/api/probes/area", json={"shape": "rect", "data": [0, 0, 5, 5]})

        probes = client.get("/api/probes").get_json()["probes"]
        assert len(probes) == 3
        types = {p["type"] for p in probes}
        assert types == {"point", "line", "area"}

    def test_clear_all(self, client):
        client.post("/api/probes/point", json={"x": 1, "y": 1})
        client.post("/api/probes/line", json={"p1": [0, 0], "p2": [10, 10]})
        client.post("/api/probes/area", json={"shape": "rect", "data": [0, 0, 5, 5]})

        resp = client.delete("/api/probes")
        assert resp.status_code == 200

        probes = client.get("/api/probes").get_json()["probes"]
        assert len(probes) == 0

    def test_clear_by_type_point(self, client):
        client.post("/api/probes/point", json={"x": 1, "y": 1})
        client.post("/api/probes/line", json={"p1": [0, 0], "p2": [10, 10]})

        resp = client.delete("/api/probes?type=point")
        assert resp.status_code == 200

        probes = client.get("/api/probes").get_json()["probes"]
        assert len(probes) == 1
        assert probes[0]["type"] == "line"

    def test_clear_by_type_area(self, client):
        client.post("/api/probes/area", json={"shape": "rect", "data": [0, 0, 5, 5]})
        client.post("/api/probes/point", json={"x": 1, "y": 1})

        resp = client.delete("/api/probes?type=area")
        assert resp.status_code == 200

        probes = client.get("/api/probes").get_json()["probes"]
        assert len(probes) == 1
        assert probes[0]["type"] == "point"

    def test_clear_empty_is_noop(self, client):
        resp = client.delete("/api/probes")
        assert resp.status_code == 200


# ===========================================================================
# 5. Point Probe Time Series
# ===========================================================================

class TestPointTimeSeries:
    """GET /api/probes/timeseries — point probe extraction."""

    def test_timeseries_u_component(self, client, mock_displacement_results):
        client.post("/api/probes/point", json={"x": 32, "y": 32})
        resp = client.get("/api/probes/timeseries?component=u&type=point")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "1" in data["series"]
        assert len(data["series"]["1"]) == 3

    def test_timeseries_v_component(self, client, mock_displacement_results):
        client.post("/api/probes/point", json={"x": 32, "y": 32})
        resp = client.get("/api/probes/timeseries?component=v&type=point")
        assert resp.status_code == 200
        assert len(resp.get_json()["series"]["1"]) == 3

    def test_timeseries_magnitude_component(self, client, mock_displacement_results):
        client.post("/api/probes/point", json={"x": 32, "y": 32})
        resp = client.get("/api/probes/timeseries?component=magnitude&type=point")
        assert resp.status_code == 200
        values = resp.get_json()["series"]["1"]
        # Magnitude is always non-negative
        for v in values:
            if v is not None:
                assert v >= 0

    def test_timeseries_velocity_component(self, client, mock_displacement_results):
        client.post("/api/probes/point", json={"x": 32, "y": 32})
        resp = client.get("/api/probes/timeseries?component=velocity&type=point")
        assert resp.status_code == 200
        values = resp.get_json()["series"]["1"]
        assert len(values) == 3
        # First frame velocity should be 0 (no previous frame)
        assert values[0] == 0.0 or values[0] is None

    def test_timeseries_multiple_probes(self, client, mock_displacement_results):
        client.post("/api/probes/point", json={"x": 10, "y": 10})
        client.post("/api/probes/point", json={"x": 50, "y": 50})
        resp = client.get("/api/probes/timeseries?component=u&type=point")
        data = resp.get_json()
        assert "1" in data["series"]
        assert "2" in data["series"]

    def test_timeseries_no_data_returns_404(self, client):
        """No displacement results should return 404."""
        client.post("/api/probes/point", json={"x": 10, "y": 10})
        resp = client.get("/api/probes/timeseries?component=u&type=point")
        assert resp.status_code == 404

    def test_timeseries_no_probes_returns_empty_series(self, client, mock_displacement_results):
        """No probes placed: series dict should be empty."""
        resp = client.get("/api/probes/timeseries?component=u&type=point")
        assert resp.status_code == 200
        assert resp.get_json()["series"] == {}

    def test_timeseries_strain_component(self, client, mock_displacement_results):
        """Strain component extraction should use strain_results."""
        results = _make_strain_results()
        session.strain_results = results
        client.post("/api/probes/point", json={"x": 32, "y": 32})
        resp = client.get("/api/probes/timeseries?component=exx&type=point")
        assert resp.status_code == 200
        assert len(resp.get_json()["series"]["1"]) == 3


# ===========================================================================
# 6. Line Probe Time Series
# ===========================================================================

class TestLineTimeSeries:
    """Line probe time series extraction."""

    def test_line_timeseries_avg(self, client, mock_displacement_results):
        client.post("/api/probes/line", json={"p1": [10, 10], "p2": [50, 50]})
        resp = client.get("/api/probes/timeseries?component=u&type=line&metric=avg")
        assert resp.status_code == 200
        series = resp.get_json()["series"]
        assert "1" in series
        assert len(series["1"]) == 3

    def test_line_timeseries_max(self, client, mock_displacement_results):
        client.post("/api/probes/line", json={"p1": [10, 10], "p2": [50, 50]})
        resp = client.get("/api/probes/timeseries?component=u&type=line&metric=max")
        assert resp.status_code == 200
        assert len(resp.get_json()["series"]["1"]) == 3

    def test_line_timeseries_min(self, client, mock_displacement_results):
        client.post("/api/probes/line", json={"p1": [10, 10], "p2": [50, 50]})
        resp = client.get("/api/probes/timeseries?component=u&type=line&metric=min")
        assert resp.status_code == 200

    def test_line_timeseries_no_lines(self, client, mock_displacement_results):
        """No line probes → empty series (point probes don't count)."""
        client.post("/api/probes/point", json={"x": 10, "y": 10})
        resp = client.get("/api/probes/timeseries?component=u&type=line")
        assert resp.status_code == 200
        assert resp.get_json()["series"] == {}


# ===========================================================================
# 7. Area Probe Time Series
# ===========================================================================

class TestAreaTimeSeries:
    """Area probe time series extraction."""

    def test_rect_area_timeseries(self, client, mock_displacement_results):
        client.post("/api/probes/area", json={
            "shape": "rect", "data": [10, 10, 50, 50]
        })
        resp = client.get("/api/probes/timeseries?component=u&type=area&metric=avg")
        assert resp.status_code == 200
        series = resp.get_json()["series"]
        assert "1" in series
        assert len(series["1"]) == 3

    def test_circle_area_timeseries(self, client, mock_displacement_results):
        client.post("/api/probes/area", json={
            "shape": "circle", "data": [32, 32, 15]
        })
        resp = client.get("/api/probes/timeseries?component=u&type=area&metric=avg")
        assert resp.status_code == 200
        series = resp.get_json()["series"]
        assert "1" in series
        assert len(series["1"]) == 3

    def test_polygon_area_timeseries(self, client, mock_displacement_results):
        client.post("/api/probes/area", json={
            "shape": "polygon",
            "data": [[10, 10], [50, 10], [50, 50], [10, 50]],
        })
        resp = client.get("/api/probes/timeseries?component=u&type=area&metric=avg")
        assert resp.status_code == 200
        series = resp.get_json()["series"]
        assert "1" in series

    def test_area_max_metric(self, client, mock_displacement_results):
        client.post("/api/probes/area", json={
            "shape": "rect", "data": [5, 5, 60, 60]
        })
        resp = client.get("/api/probes/timeseries?component=u&type=area&metric=max")
        assert resp.status_code == 200

    def test_area_min_metric(self, client, mock_displacement_results):
        client.post("/api/probes/area", json={
            "shape": "rect", "data": [5, 5, 60, 60]
        })
        resp = client.get("/api/probes/timeseries?component=u&type=area&metric=min")
        assert resp.status_code == 200

    def test_area_no_probes_returns_empty(self, client, mock_displacement_results):
        resp = client.get("/api/probes/timeseries?component=u&type=area")
        assert resp.status_code == 200
        assert resp.get_json()["series"] == {}


# ===========================================================================
# 8. Extensometer Endpoint
# ===========================================================================

class TestExtensometer:
    """GET /api/probes/extensometer/<line_id>."""

    def test_extensometer_basic(self, client, mock_displacement_results):
        client.post("/api/probes/line", json={"p1": [10, 10], "p2": [50, 50]})
        resp = client.get("/api/probes/extensometer/1")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "initial_length" in data
        assert "lengths" in data
        assert "strains" in data
        assert data["num_frames"] == 3
        assert data["initial_length"] > 0

    def test_extensometer_no_displacement(self, client):
        client.post("/api/probes/line", json={"p1": [10, 10], "p2": [50, 50]})
        resp = client.get("/api/probes/extensometer/1")
        assert resp.status_code == 404

    def test_extensometer_missing_line(self, client, mock_displacement_results):
        resp = client.get("/api/probes/extensometer/999")
        assert resp.status_code == 404

    def test_extensometer_lengths_match_frames(self, client, mock_displacement_results):
        client.post("/api/probes/line", json={"p1": [5, 5], "p2": [55, 55]})
        resp = client.get("/api/probes/extensometer/1")
        data = resp.get_json()
        assert len(data["lengths"]) == 3
        assert len(data["strains"]) == 3


# ===========================================================================
# 9. Kymograph Endpoint
# ===========================================================================

class TestKymograph:
    """GET /api/probes/kymograph/<line_id>."""

    def test_kymograph_basic(self, client, mock_displacement_results):
        client.post("/api/probes/line", json={"p1": [10, 10], "p2": [50, 50]})
        resp = client.get("/api/probes/kymograph/1?component=u")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "data" in data
        assert "dist_axis" in data
        assert "shape" in data
        # Default 100 sample points x 3 frames
        assert data["shape"][0] == 100
        assert data["shape"][1] == 3

    def test_kymograph_missing_line(self, client, mock_displacement_results):
        resp = client.get("/api/probes/kymograph/999?component=u")
        assert resp.status_code == 404

    def test_kymograph_no_data(self, client):
        client.post("/api/probes/line", json={"p1": [10, 10], "p2": [50, 50]})
        resp = client.get("/api/probes/kymograph/1?component=u")
        assert resp.status_code == 404

    def test_kymograph_v_component(self, client, mock_displacement_results):
        client.post("/api/probes/line", json={"p1": [10, 10], "p2": [50, 50]})
        resp = client.get("/api/probes/kymograph/1?component=v")
        assert resp.status_code == 200


# ===========================================================================
# 10. CSV Export
# ===========================================================================

class TestCsvExport:
    """GET /api/probes/export/csv."""

    def test_export_no_data_returns_404(self, client):
        """Export without any displacement data should return 404."""
        resp = client.get("/api/probes/export/csv?component=u&type=point")
        assert resp.status_code == 404

    def test_export_no_probes_returns_404(self, client, mock_displacement_results):
        """Export with displacement data but no probes should return 404."""
        resp = client.get("/api/probes/export/csv?component=u&type=point")
        assert resp.status_code == 404

    def test_export_point_csv(self, client, mock_displacement_results):
        client.post("/api/probes/point", json={"x": 32, "y": 32})
        resp = client.get("/api/probes/export/csv?component=u&type=point")
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/csv")

        # Parse CSV
        text = resp.data.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        # Header + 3 data rows
        assert len(rows) == 4
        header = rows[0]
        assert header[0] == "frame"
        assert "probe_1_u" in header

    def test_export_csv_has_attachment_header(self, client, mock_displacement_results):
        client.post("/api/probes/point", json={"x": 32, "y": 32})
        resp = client.get("/api/probes/export/csv?component=u&type=point")
        disp = resp.headers.get("Content-Disposition", "")
        assert "attachment" in disp
        assert ".csv" in disp

    def test_export_csv_filename_contains_type_and_component(self, client, mock_displacement_results):
        client.post("/api/probes/point", json={"x": 10, "y": 10})
        resp = client.get("/api/probes/export/csv?component=v&type=point")
        disp = resp.headers.get("Content-Disposition", "")
        assert "point" in disp
        assert "v" in disp

    def test_export_multiple_points(self, client, mock_displacement_results):
        client.post("/api/probes/point", json={"x": 10, "y": 10})
        client.post("/api/probes/point", json={"x": 50, "y": 50})
        resp = client.get("/api/probes/export/csv?component=u&type=point")
        assert resp.status_code == 200

        text = resp.data.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        header = rows[0]
        # frame + 2 probe columns
        assert len(header) == 3
        assert "probe_1_u" in header
        assert "probe_2_u" in header

    def test_export_line_csv(self, client, mock_displacement_results):
        client.post("/api/probes/line", json={"p1": [10, 10], "p2": [50, 50]})
        resp = client.get("/api/probes/export/csv?component=u&type=line")
        assert resp.status_code == 200
        text = resp.data.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        header = rows[0]
        assert "line_1_u" in header

    def test_export_area_csv(self, client, mock_displacement_results):
        client.post("/api/probes/area", json={
            "shape": "rect", "data": [10, 10, 50, 50]
        })
        resp = client.get("/api/probes/export/csv?component=u&type=area")
        assert resp.status_code == 200
        text = resp.data.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        header = rows[0]
        assert "area_1_u" in header

    def test_export_csv_frame_numbers(self, client, mock_displacement_results):
        """Frame numbers in CSV should be 1-based."""
        client.post("/api/probes/point", json={"x": 32, "y": 32})
        resp = client.get("/api/probes/export/csv?component=u&type=point")
        text = resp.data.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        # First data row, first column = frame number = 1
        assert rows[1][0] == "1"
        assert rows[2][0] == "2"
        assert rows[3][0] == "3"

    def test_export_csv_data_values_are_numeric(self, client, mock_displacement_results):
        """All data values should be parseable as float or empty."""
        client.post("/api/probes/point", json={"x": 32, "y": 32})
        resp = client.get("/api/probes/export/csv?component=u&type=point")
        text = resp.data.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        for row in rows[1:]:  # skip header
            for cell in row[1:]:  # skip frame number
                if cell != "":
                    float(cell)  # should not raise

    def test_export_line_with_extensometer(self, client, mock_displacement_results):
        """Extensometer columns should be included when requested."""
        client.post("/api/probes/line", json={"p1": [10, 10], "p2": [50, 50]})
        resp = client.get(
            "/api/probes/export/csv?component=u&type=line&extensometer=true"
        )
        assert resp.status_code == 200
        text = resp.data.decode("utf-8")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        header = rows[0]
        # Should have extensometer strain and deformed_length columns
        assert any("strain" in h for h in header)
        assert any("deformed_length" in h for h in header)

    def test_export_magnitude_component(self, client, mock_displacement_results):
        """Export with 'magnitude' component should work."""
        client.post("/api/probes/point", json={"x": 32, "y": 32})
        resp = client.get("/api/probes/export/csv?component=magnitude&type=point")
        assert resp.status_code == 200


# ===========================================================================
# 11. Mixed Probe Types: ID Independence
# ===========================================================================

class TestIdIndependence:
    """Point, line, and area probes have independent ID counters."""

    def test_independent_ids(self, client):
        r_pt = client.post("/api/probes/point", json={"x": 1, "y": 1})
        r_ln = client.post("/api/probes/line", json={"p1": [0, 0], "p2": [10, 10]})
        r_ar = client.post("/api/probes/area", json={
            "shape": "rect", "data": [0, 0, 5, 5]
        })
        # Each type starts at ID 1
        assert r_pt.get_json()["id"] == 1
        assert r_ln.get_json()["id"] == 1
        assert r_ar.get_json()["id"] == 1

    def test_delete_preserves_other_types(self, client):
        client.post("/api/probes/point", json={"x": 1, "y": 1})
        client.post("/api/probes/line", json={"p1": [0, 0], "p2": [10, 10]})
        client.post("/api/probes/area", json={"shape": "rect", "data": [0, 0, 5, 5]})

        # Delete the point probe
        client.delete("/api/probes/1?type=point")

        probes = client.get("/api/probes").get_json()["probes"]
        assert len(probes) == 2
        types = {p["type"] for p in probes}
        assert "line" in types
        assert "area" in types


# ===========================================================================
# 12. Edge Cases
# ===========================================================================

class TestProbeEdgeCases:

    def test_point_probe_at_image_edge(self, client, mock_displacement_results):
        """Probe at the edge of the displacement field."""
        client.post("/api/probes/point", json={"x": 63, "y": 63})
        resp = client.get("/api/probes/timeseries?component=u&type=point")
        assert resp.status_code == 200
        # Should return values (possibly NaN at boundary but no crash)
        assert len(resp.get_json()["series"]["1"]) == 3

    def test_area_probe_covering_full_roi(self, client, mock_displacement_results):
        """Area covering the entire displacement field."""
        client.post("/api/probes/area", json={
            "shape": "rect", "data": [0, 0, 64, 64]
        })
        resp = client.get("/api/probes/timeseries?component=u&type=area")
        assert resp.status_code == 200

    def test_circle_area_large_radius(self, client, mock_displacement_results):
        """Circle with radius bigger than image — should still work."""
        client.post("/api/probes/area", json={
            "shape": "circle", "data": [32, 32, 100]
        })
        resp = client.get("/api/probes/timeseries?component=u&type=area")
        assert resp.status_code == 200
        series = resp.get_json()["series"]
        assert "1" in series

    def test_polygon_triangle(self, client, mock_displacement_results):
        """Triangular polygon probe."""
        client.post("/api/probes/area", json={
            "shape": "polygon",
            "data": [[32, 5], [5, 55], [59, 55]],
        })
        resp = client.get("/api/probes/timeseries?component=u&type=area")
        assert resp.status_code == 200

    def test_timeseries_values_are_float_or_none(self, client, mock_displacement_results):
        """All values in the time series should be float or None (JSON-safe)."""
        client.post("/api/probes/point", json={"x": 32, "y": 32})
        resp = client.get("/api/probes/timeseries?component=u&type=point")
        values = resp.get_json()["series"]["1"]
        for v in values:
            assert v is None or isinstance(v, (int, float))
