"""Integration test covering the full workflow."""

import numpy as np
import pytest

from server.session import session


class TestFullWorkflow:
    """Test the complete API workflow in sequence."""

    def test_01_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_02_load_images(self, client, mock_images):
        resp = client.post("/api/images/load", json={"dir": mock_images})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 3

    def test_03_reference_image(self, client, loaded_images):
        resp = client.get("/api/images/reference")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_04_models(self, client):
        resp = client.get("/api/models")
        assert resp.status_code == 200
        assert "models" in resp.get_json()

    def test_05_roi_polygon(self, client, loaded_images):
        points = [[10, 10], [50, 10], [50, 50], [10, 50]]
        resp = client.post("/api/roi/polygon", json={"points": points, "mode": "add"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["area_px"] > 0
        assert data["rect"] is not None

    def test_06_roi_mask(self, client, loaded_images):
        # Add polygon first
        points = [[10, 10], [50, 10], [50, 50], [10, 50]]
        client.post("/api/roi/polygon", json={"points": points, "mode": "add"})

        resp = client.get("/api/roi/mask")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_07_roi_confirm(self, client, loaded_images):
        points = [[10, 10], [50, 10], [50, 50], [10, 50]]
        client.post("/api/roi/polygon", json={"points": points, "mode": "add"})
        resp = client.post("/api/roi/confirm")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_08_displacement_with_mock(self, client, mock_displacement_results):
        resp = client.get("/api/displacement/info")
        assert resp.status_code == 200
        assert resp.get_json()["num_frames"] == 3

        resp = client.get("/api/displacement/frame/0?component=u")
        assert resp.status_code == 200

        resp = client.get("/api/displacement/render/0")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_09_probes_full_cycle(self, client, mock_displacement_results):
        # Add probes
        resp = client.post("/api/probes/point", json={"x": 32, "y": 32})
        assert resp.status_code == 200

        resp = client.post("/api/probes/line", json={
            "p1": [10, 10], "p2": [50, 50]
        })
        assert resp.status_code == 200

        # List
        resp = client.get("/api/probes")
        assert len(resp.get_json()["probes"]) == 2

        # Time series
        resp = client.get("/api/probes/timeseries?component=u&type=point")
        assert resp.status_code == 200
        assert len(resp.get_json()["series"]) > 0

        # Kymograph
        resp = client.get("/api/probes/kymograph/1?component=u")
        assert resp.status_code == 200
        assert "data" in resp.get_json()

        # Clear
        resp = client.delete("/api/probes")
        assert resp.status_code == 200

    def test_10_strain_mock(self, client, mock_displacement_results):
        """Test strain endpoints with mock data."""
        strain_dict = {
            "exx": np.random.randn(64, 64).astype(np.float32) * 0.01,
            "eyy": np.random.randn(64, 64).astype(np.float32) * 0.01,
            "exy": np.random.randn(64, 64).astype(np.float32) * 0.005,
            "e1": np.random.randn(64, 64).astype(np.float32) * 0.01,
            "e2": np.random.randn(64, 64).astype(np.float32) * 0.01,
            "max_shear": np.abs(np.random.randn(64, 64)).astype(np.float32) * 0.005,
            "von_mises": np.abs(np.random.randn(64, 64)).astype(np.float32) * 0.01,
            "rotation": np.random.randn(64, 64).astype(np.float32) * 0.1,
        }
        session.strain_results = [strain_dict] * 3
        session.strain_components = list(strain_dict.keys())

        resp = client.get("/api/strain/status")
        assert resp.get_json()["computed"] is True

        resp = client.get("/api/strain/frame/0?component=exx")
        assert resp.status_code == 200

        resp = client.get("/api/strain/render/0?component=exx")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"
