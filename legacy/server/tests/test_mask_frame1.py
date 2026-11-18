"""Tests for validate-masks has_frame_1 detection."""

import numpy as np
import cv2
import os
import pytest
from server.session import session


class TestValidateMasksFrame1:

    def test_reports_has_frame_1_true(self, client, loaded_images, tmp_path):
        """When mask folder contains frame 1 match, has_frame_1=True."""
        first_name = os.path.splitext(session.image_files[0])[0]
        mask = np.ones((64, 64), dtype=np.uint8) * 255
        cv2.imwrite(str(tmp_path / f"{first_name}.png"), mask)

        resp = client.post("/api/processing/validate-masks", json={
            "mask_dir": str(tmp_path),
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["has_frame_1"] is True
        assert data["matched_count"] >= 1

    def test_reports_has_frame_1_false(self, client, loaded_images, tmp_path):
        """When mask folder has a mask with wrong dimensions, has_frame_1=False."""
        # With index-based matching, any valid mask maps to idx 0 (frame 1).
        # To get has_frame_1=False, the mask must fail to load (e.g., wrong size).
        mask = np.ones((32, 32), dtype=np.uint8) * 255  # wrong dimensions
        cv2.imwrite(str(tmp_path / "frame_099.png"), mask)

        resp = client.post("/api/processing/validate-masks", json={
            "mask_dir": str(tmp_path),
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["has_frame_1"] is False

    def test_empty_folder_has_frame_1_false(self, client, loaded_images, tmp_path):
        """Empty folder: has_frame_1=False, matched_count=0."""
        resp = client.post("/api/processing/validate-masks", json={
            "mask_dir": str(tmp_path),
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["has_frame_1"] is False
        assert data["matched_count"] == 0

    def test_apply_frame1_mask_sets_roi(self, client, loaded_images, tmp_path):
        """POST /processing/apply-frame1-mask loads frame 1 mask as ROI."""
        first_name = os.path.splitext(session.image_files[0])[0]
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:50, 10:50] = 255
        cv2.imwrite(str(tmp_path / f"{first_name}.png"), mask)

        resp = client.post("/api/processing/apply-frame1-mask", json={
            "mask_dir": str(tmp_path),
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["area_px"] == 40 * 40
        assert session.roi_mask is not None
        assert session.roi_mask.sum() == 40 * 40

    def test_apply_frame1_mask_no_frame1(self, client, loaded_images, tmp_path):
        """Returns 404 when mask has wrong dimensions (no valid mask loaded)."""
        # With index-based matching, any valid mask maps to idx 0 (frame 1).
        # Use wrong dimensions so the mask is skipped during loading.
        mask = np.ones((32, 32), dtype=np.uint8) * 255  # wrong dimensions
        cv2.imwrite(str(tmp_path / "frame_099.png"), mask)

        resp = client.post("/api/processing/apply-frame1-mask", json={
            "mask_dir": str(tmp_path),
        })
        assert resp.status_code == 404

    def test_apply_frame1_mask_no_images(self, client, tmp_path):
        """Returns 400 when no images loaded."""
        resp = client.post("/api/processing/apply-frame1-mask", json={
            "mask_dir": str(tmp_path),
        })
        assert resp.status_code == 400
