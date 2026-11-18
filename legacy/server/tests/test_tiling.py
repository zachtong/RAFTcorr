"""Tests for tiling preview route."""

import numpy as np
import pytest

from server.session import session


class TestTilingPreviewValidation:
    """Tests for input validation in the tiling preview endpoint."""

    def test_no_images_returns_400(self, client):
        resp = client.get("/api/tiling/preview")
        assert resp.status_code == 400
        assert "No images loaded" in resp.get_json()["error"]

    def test_no_roi_returns_400(self, client, loaded_images):
        """Should return 400 when ROI is not confirmed."""
        resp = client.get("/api/tiling/preview")
        assert resp.status_code == 400
        assert "ROI not confirmed" in resp.get_json()["error"]

    def test_no_model_returns_400(self, client, loaded_images):
        """Should return 400 when no model is selected."""
        # Set ROI but explicitly clear model to ensure no leakage from prior tests
        session.roi_mask = np.ones((64, 64), dtype=bool)
        session.roi_confirmed = True
        session.roi_rect = (0, 0, 64, 64)
        session.config.model_path = ""
        resp = client.get("/api/tiling/preview")
        assert resp.status_code == 400
        assert "No model selected" in resp.get_json()["error"]

    def test_no_roi_rect_returns_400(self, client, loaded_images):
        """Should return 400 when roi_rect is None."""
        session.roi_mask = np.ones((64, 64), dtype=bool)
        session.roi_confirmed = True
        session.roi_rect = None
        session.config.model_path = "/fake/model.pth"
        resp = client.get("/api/tiling/preview")
        assert resp.status_code == 400
        assert "ROI rect not set" in resp.get_json()["error"]


class TestTilingPreviewSuccess:
    """Tests for successful tiling preview responses."""

    @pytest.fixture()
    def tiling_ready(self, loaded_images):
        """Set up session state for a valid tiling preview request."""
        session.roi_mask = np.ones((64, 64), dtype=bool)
        session.roi_confirmed = True
        session.roi_rect = (0, 0, 64, 64)
        session.config.model_path = "/fake/model.pth"

    def test_single_shot_small_image(self, client, tiling_ready):
        """Small ROI within p_max_pixels should produce a single tile."""
        resp = client.get("/api/tiling/preview")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["is_single_shot"] is True
        assert data["tile_count"] == 1
        assert len(data["tiles"]) == 1

    def test_response_structure(self, client, tiling_ready):
        """Should return all expected keys in the response."""
        resp = client.get("/api/tiling/preview")
        assert resp.status_code == 200
        data = resp.get_json()
        expected_keys = {
            "image_size", "roi_bbox", "work_area", "tile_size",
            "tiles", "tile_count", "overlap_ratio", "gpu",
            "est_memory_per_tile_mb", "est_time_seconds", "is_single_shot",
        }
        assert expected_keys.issubset(set(data.keys()))

    def test_gpu_info_present(self, client, tiling_ready):
        """GPU info dict should have name, total_mb, free_mb keys."""
        resp = client.get("/api/tiling/preview")
        data = resp.get_json()
        gpu = data["gpu"]
        assert "name" in gpu
        assert "total_mb" in gpu
        assert "free_mb" in gpu

    def test_multi_tile_large_roi(self, client, loaded_images):
        """Large ROI exceeding p_max_pixels should produce multiple tiles."""
        big_size = 2048
        session.roi_mask = np.ones((big_size, big_size), dtype=bool)
        session.roi_confirmed = True
        session.roi_rect = (0, 0, big_size, big_size)
        session.image_width = big_size
        session.image_height = big_size
        session.config.model_path = "/fake/model.pth"
        # p_max_pixels default is 1_210_000 (~1100x1100),
        # so 2048x2048 should require multiple tiles
        resp = client.get("/api/tiling/preview")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["is_single_shot"] is False
        assert data["tile_count"] > 1
        assert len(data["tiles"]) == data["tile_count"]

    def test_overlap_ratio_for_multi_tile(self, client, loaded_images):
        """Multi-tile result should have a positive overlap ratio."""
        big_size = 2048
        session.roi_mask = np.ones((big_size, big_size), dtype=bool)
        session.roi_confirmed = True
        session.roi_rect = (0, 0, big_size, big_size)
        session.image_width = big_size
        session.image_height = big_size
        session.config.model_path = "/fake/model.pth"
        resp = client.get("/api/tiling/preview")
        data = resp.get_json()
        assert data["overlap_ratio"] > 0
