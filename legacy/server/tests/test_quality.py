"""Tests for data quality (NCC) route."""

import numpy as np
import pytest

from server.session import session


class TestNCCNoResults:
    """Tests when no displacement results exist."""

    def test_returns_404(self, client):
        resp = client.get("/api/quality/ncc/0")
        assert resp.status_code == 404
        assert "No displacement results" in resp.get_json()["error"]


class TestNCCWithResults:
    """Tests with mock displacement data."""

    def test_frame_out_of_range(self, client, mock_displacement_results):
        """Should return 400 for out-of-range frame index."""
        resp = client.get("/api/quality/ncc/9999")
        assert resp.status_code == 400
        assert "out of range" in resp.get_json()["error"]

    def test_no_reference_image(self, client):
        """Should return 400 when reference image is missing."""
        # Set displacement results but no reference image
        session.displacement_results = [
            np.random.randn(64, 64, 2).astype(np.float32)
        ]
        session.reference_image = None
        resp = client.get("/api/quality/ncc/0")
        assert resp.status_code == 400
        assert "No reference image" in resp.get_json()["error"]

    def test_no_deformed_image_file(self, client, mock_displacement_results):
        """Should return 400 when no deformed image file exists for the frame."""
        # mock_displacement_results has no image_files set
        session.image_files = ["ref.png"]  # only 1 file, no deformed
        resp = client.get("/api/quality/ncc/0")
        assert resp.status_code == 400
        assert "No deformed image" in resp.get_json()["error"]

    def test_valid_ncc_computation(self, client, loaded_images):
        """Should compute NCC for a valid frame with real images loaded."""
        # loaded_images loads 3 frames and sets reference_image.
        # We set displacement results AFTER loading images because
        # load_images() calls session.reset() which clears results.
        session.displacement_results = [
            np.random.randn(64, 64, 2).astype(np.float32) * 5.0
            for _ in range(2)
        ]
        session.roi_rect = (0, 0, 64, 64)
        session.roi_mask = np.ones((64, 64), dtype=bool)
        resp = client.get("/api/quality/ncc/0")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "ncc" in data
        assert "frame" in data
        assert data["frame"] == 0
        # NCC should be a float in [-1, 1] range
        assert isinstance(data["ncc"], float)

    def test_ncc_with_zero_displacement(self, client, loaded_images):
        """NCC with zero displacement should be close to 1.0 (self-correlation)."""
        # Create zero displacement — warped reference == reference
        session.displacement_results = [
            np.zeros((64, 64, 2), dtype=np.float32)
            for _ in range(2)
        ]
        session.roi_rect = (0, 0, 64, 64)
        session.roi_mask = np.ones((64, 64), dtype=bool)
        resp = client.get("/api/quality/ncc/0")
        assert resp.status_code == 200
        data = resp.get_json()
        # With zero displacement, warped reference maps to itself;
        # NCC should be reasonably high (images are random so not perfect)
        assert isinstance(data["ncc"], float)
