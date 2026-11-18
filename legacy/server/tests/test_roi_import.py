"""Tests for enhanced ROI import with noise filtering."""
import numpy as np
import cv2
import os
import pytest


class TestRoiImportFiltering:

    def _create_mask_with_noise(self, tmp_path, H=64, W=64):
        """Create a mask image with a large region + small noise blobs."""
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[10:40, 10:40] = 255  # Main region: 30x30 = 900px
        mask[50, 50] = 255        # 1px noise
        mask[55:57, 55:57] = 255  # 2x2 = 4px noise
        path = str(tmp_path / "mask.png")
        cv2.imwrite(path, mask)
        return path

    def test_import_with_min_area_removes_noise(self, client, loaded_images, tmp_path):
        """min_area=10 removes the 1px and 4px blobs, keeps 900px block."""
        mask_path = self._create_mask_with_noise(tmp_path)
        resp = client.post("/api/roi/import", json={"path": mask_path, "min_area": 10})
        assert resp.status_code == 200
        assert resp.get_json()["area_px"] == 900

    def test_import_with_min_area_zero_keeps_all(self, client, loaded_images, tmp_path):
        """min_area=0 keeps everything."""
        mask_path = self._create_mask_with_noise(tmp_path)
        resp = client.post("/api/roi/import", json={"path": mask_path, "min_area": 0})
        assert resp.status_code == 200
        assert resp.get_json()["area_px"] == 900 + 1 + 4

    def test_import_with_smoothing(self, client, loaded_images, tmp_path):
        """Smoothing applies Gaussian blur before binarization."""
        mask_path = self._create_mask_with_noise(tmp_path)
        resp = client.post("/api/roi/import", json={"path": mask_path, "smooth_radius": 2, "min_area": 0})
        assert resp.status_code == 200
        assert resp.get_json()["area_px"] > 0

    def test_import_no_filtering_default(self, client, loaded_images, tmp_path):
        """Default import (no params) keeps all pixels."""
        mask_path = self._create_mask_with_noise(tmp_path)
        resp = client.post("/api/roi/import", json={"path": mask_path})
        assert resp.status_code == 200
        assert resp.get_json()["area_px"] == 900 + 1 + 4

    def test_import_min_area_removes_partial(self, client, loaded_images, tmp_path):
        """min_area=2 removes only the 1px blob, keeps 4px and 900px."""
        mask_path = self._create_mask_with_noise(tmp_path)
        resp = client.post("/api/roi/import", json={"path": mask_path, "min_area": 2})
        assert resp.status_code == 200
        assert resp.get_json()["area_px"] == 900 + 4

    def test_import_missing_path(self, client, loaded_images):
        """Missing path returns 400."""
        resp = client.post("/api/roi/import", json={})
        assert resp.status_code == 400

    def test_import_no_images_loaded(self, client, tmp_path):
        """Import without loaded images returns 400."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        path = str(tmp_path / "mask.png")
        cv2.imwrite(path, mask)
        resp = client.post("/api/roi/import", json={"path": path})
        assert resp.status_code == 400
