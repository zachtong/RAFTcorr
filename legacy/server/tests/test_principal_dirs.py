"""Tests for principal strain direction overlay rendering route."""

import numpy as np
import pytest

from server.session import session


@pytest.fixture()
def mock_strain_results():
    """Create mock strain results in session with all required components."""
    strain_dict = {
        "exx": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "eyy": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "exy": np.random.randn(64, 64).astype(np.float32) * 0.005,
        "e1": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "e2": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "max_shear": np.abs(np.random.randn(64, 64).astype(np.float32) * 0.005),
        "von_mises": np.abs(np.random.randn(64, 64).astype(np.float32) * 0.01),
        "rotation": np.random.randn(64, 64).astype(np.float32) * 0.1,
    }
    session.strain_results = [strain_dict] * 3
    session.strain_components = list(strain_dict.keys())
    session.roi_rect = (0, 0, 64, 64)
    session.roi_mask = np.ones((64, 64), dtype=bool)
    session.image_width = 64
    session.image_height = 64
    session.reference_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return session.strain_results


class TestPrincipalNoResults:
    """Tests when no strain results exist."""

    def test_returns_404(self, client):
        resp = client.get("/api/principal/render/0")
        assert resp.status_code == 404
        assert "No strain results" in resp.get_json()["error"]


class TestPrincipalWithResults:
    """Tests with mock strain results loaded."""

    def test_frame_out_of_range(self, client, mock_strain_results):
        """Should return 400 for out-of-range frame index."""
        resp = client.get("/api/principal/render/9999")
        assert resp.status_code == 400
        assert "out of range" in resp.get_json()["error"]

    def test_render_returns_png(self, client, mock_strain_results):
        """Should return a PNG image for valid frame."""
        resp = client.get("/api/principal/render/0")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"
        assert len(resp.data) > 100

    def test_custom_spacing(self, client, mock_strain_results):
        """Should accept custom spacing parameter."""
        resp = client.get("/api/principal/render/0?spacing=10")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_custom_scale(self, client, mock_strain_results):
        """Should accept custom scale parameter."""
        resp = client.get("/api/principal/render/0?scale=30")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_custom_line_width(self, client, mock_strain_results):
        """Should accept custom line_width parameter."""
        resp = client.get("/api/principal/render/0?line_width=2.5")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_viewport_size_parameters(self, client, mock_strain_results):
        """Should accept viewport size (vw, vh) for scaled rendering."""
        resp = client.get("/api/principal/render/0?vw=320&vh=240")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_cache_hit_returns_same_image(self, client, mock_strain_results):
        """Second identical request should hit the render cache."""
        url = "/api/principal/render/0?spacing=30&scale=15"
        resp1 = client.get(url)
        resp2 = client.get(url)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.data == resp2.data

    def test_last_valid_frame(self, client, mock_strain_results):
        """Should render the last available frame (index 2)."""
        resp = client.get("/api/principal/render/2")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_none_strain_entry(self, client, mock_strain_results):
        """Should return 400 when a specific strain entry is None."""
        session.strain_results[1] = None
        resp = client.get("/api/principal/render/1")
        assert resp.status_code == 400
        assert "No strain for this frame" in resp.get_json()["error"]

    def test_missing_strain_components(self, client, mock_strain_results):
        """Should return 400 when required strain components are missing."""
        # Remove exx to trigger the missing component check
        session.strain_results[0] = {
            "e1": np.zeros((64, 64), dtype=np.float32),
            "e2": np.zeros((64, 64), dtype=np.float32),
        }
        resp = client.get("/api/principal/render/0")
        assert resp.status_code == 400
        assert "Missing strain components" in resp.get_json()["error"]
