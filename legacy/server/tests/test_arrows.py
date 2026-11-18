"""Tests for vector field (quiver + streamline) overlay rendering routes."""

import numpy as np
import pytest

from server.session import session


class TestRenderArrowsNoResults:
    """Tests when no displacement results exist."""

    def test_returns_404(self, client):
        resp = client.get("/api/arrows/render/0?show_quiver=true")
        assert resp.status_code == 404
        assert "No displacement results" in resp.get_json()["error"]


class TestRenderArrowsWithResults:
    """Tests with mock displacement results loaded."""

    def test_nothing_to_render(self, client, mock_displacement_results):
        """Should return 400 when neither quiver nor streamlines requested."""
        resp = client.get("/api/arrows/render/0")
        assert resp.status_code == 400
        assert "Nothing to render" in resp.get_json()["error"]

    def test_frame_out_of_range(self, client, mock_displacement_results):
        """Should return 400 for out-of-range frame index."""
        resp = client.get("/api/arrows/render/9999?show_quiver=true")
        assert resp.status_code == 400
        assert "out of range" in resp.get_json()["error"]

    def test_negative_frame_index(self, client, mock_displacement_results):
        """Should return 400 for negative frame index."""
        resp = client.get("/api/arrows/render/-1?show_quiver=true")
        # Flask may return 404 for negative int in URL, or 400 from our guard
        assert resp.status_code in (400, 404)

    def test_quiver_render_png(self, client, mock_displacement_results):
        """Should return a PNG image when quiver is enabled."""
        resp = client.get("/api/arrows/render/0?show_quiver=true")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"
        assert len(resp.data) > 100

    def test_streamlines_render_png(self, client, mock_displacement_results):
        """Should return a PNG image when streamlines are enabled."""
        resp = client.get("/api/arrows/render/0?show_streamlines=true")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"
        assert len(resp.data) > 100

    def test_both_quiver_and_streamlines(self, client, mock_displacement_results):
        """Should return PNG when both quiver and streamlines enabled."""
        resp = client.get(
            "/api/arrows/render/0?show_quiver=true&show_streamlines=true"
        )
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_custom_spacing_and_scale(self, client, mock_displacement_results):
        """Should accept custom spacing and scale parameters."""
        resp = client.get(
            "/api/arrows/render/0?show_quiver=true&spacing=10&scale=50"
        )
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_displacement_vector_source(self, client, mock_displacement_results):
        """Should accept vector_source=displacement."""
        resp = client.get(
            "/api/arrows/render/0?show_quiver=true&vector_source=displacement"
        )
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_velocity_with_previous_frame(self, client, mock_displacement_results):
        """Velocity source on frame > 0 uses frame differencing."""
        resp = client.get(
            "/api/arrows/render/1?show_quiver=true&vector_source=velocity"
        )
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_deformed_background_mode(self, client, mock_displacement_results):
        """Should handle deformed background with forward-mapped arrows."""
        resp = client.get(
            "/api/arrows/render/0?show_quiver=true&background=deformed"
        )
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_viewport_size_parameters(self, client, mock_displacement_results):
        """Should accept viewport size (vw, vh) for scaled rendering."""
        resp = client.get(
            "/api/arrows/render/0?show_quiver=true&vw=320&vh=240"
        )
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_cache_hit_returns_same_image(self, client, mock_displacement_results):
        """Second identical request should hit the render cache."""
        url = "/api/arrows/render/0?show_quiver=true&spacing=30&scale=20"
        resp1 = client.get(url)
        resp2 = client.get(url)
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        # Cached response should be identical
        assert resp1.data == resp2.data

    def test_line_width_parameter(self, client, mock_displacement_results):
        """Should accept custom line_width."""
        resp = client.get(
            "/api/arrows/render/0?show_quiver=true&line_width=3.0"
        )
        assert resp.status_code == 200
        assert resp.content_type == "image/png"

    def test_color_parameter(self, client, mock_displacement_results):
        """Should accept custom arrow color."""
        resp = client.get(
            "/api/arrows/render/0?show_quiver=true&color=red"
        )
        assert resp.status_code == 200
        assert resp.content_type == "image/png"
