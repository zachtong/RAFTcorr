"""Integration tests for per-frame ROI system.

Verifies that all ROI endpoints correctly handle frame_idx,
per-frame mask storage/retrieval, status reporting, and
backward compatibility with frame-0 defaults.
"""

import numpy as np
import pytest
from PIL import Image

from server.session import session


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _setup_images(client, tmp_path, n=5, size=64):
    """Create n test images and load them into the session."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)
    for i in range(n):
        img = Image.fromarray(
            np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        )
        img.save(img_dir / f"frame_{i:03d}.png")
    resp = client.post("/api/images/load", json={"dir": str(img_dir)})
    assert resp.status_code == 200
    return resp.get_json()


def _add_rect(client, frame_idx=0, mode="add",
              x0=5, y0=5, x1=55, y1=55):
    """Add a rectangle ROI on a specific frame."""
    return client.post("/api/roi/rectangle", json={
        "x0": x0, "y0": y0, "x1": x1, "y1": y1,
        "mode": mode, "frame_idx": frame_idx,
    })


# ──────────────────────────────────────────────
# Backward Compatibility — frame_idx defaults to 0
# ──────────────────────────────────────────────

class TestBackwardCompatibility:
    """Existing ROI calls without frame_idx still work on frame 0."""

    def test_rectangle_default_frame(self, client, tmp_path):
        _setup_images(client, tmp_path)
        resp = client.post("/api/roi/rectangle", json={
            "x0": 10, "y0": 10, "x1": 50, "y1": 50,
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["area_px"] > 0
        # Session roi_mask should be set
        assert session.roi_mask is not None
        assert session.roi_mask.sum() > 0

    def test_polygon_default_frame(self, client, tmp_path):
        _setup_images(client, tmp_path)
        resp = client.post("/api/roi/polygon", json={
            "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
        })
        assert resp.status_code == 200
        assert resp.get_json()["area_px"] > 0

    def test_circle_default_frame(self, client, tmp_path):
        _setup_images(client, tmp_path)
        resp = client.post("/api/roi/circle", json={
            "cx": 32, "cy": 32, "r": 20,
        })
        assert resp.status_code == 200
        assert resp.get_json()["area_px"] > 0

    def test_confirm_still_works(self, client, tmp_path):
        _setup_images(client, tmp_path)
        _add_rect(client, frame_idx=0)
        resp = client.post("/api/roi/confirm")
        assert resp.status_code == 200
        assert session.roi_confirmed is True

    def test_clear_still_works(self, client, tmp_path):
        _setup_images(client, tmp_path)
        _add_rect(client, frame_idx=0)
        assert session.roi_mask is not None
        resp = client.delete("/api/roi")
        assert resp.status_code == 200
        assert session.roi_mask is None

    def test_mask_url_default_frame(self, client, tmp_path):
        _setup_images(client, tmp_path)
        _add_rect(client, frame_idx=0)
        resp = client.get("/api/roi/mask")
        assert resp.status_code == 200
        assert resp.content_type.startswith("image/")


# ──────────────────────────────────────────────
# Per-Frame ROI Operations
# ──────────────────────────────────────────────

class TestPerFrameRoi:
    """ROI operations on non-zero frames."""

    def test_add_rect_on_frame_2(self, client, tmp_path):
        _setup_images(client, tmp_path, n=5)
        resp = _add_rect(client, frame_idx=2)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["area_px"] > 0
        assert data["frame_idx"] == 2
        # Stored in per_frame_rois
        assert 2 in session.per_frame_rois
        assert session.per_frame_rois[2].sum() > 0

    def test_frame_0_syncs_with_roi_mask(self, client, tmp_path):
        _setup_images(client, tmp_path)
        _add_rect(client, frame_idx=0)
        # per_frame_rois[0] should match session.roi_mask
        assert 0 in session.per_frame_rois
        np.testing.assert_array_equal(
            session.per_frame_rois[0], session.roi_mask
        )

    def test_multiple_frames_independent(self, client, tmp_path):
        _setup_images(client, tmp_path, n=5)
        # Different ROI on each frame
        _add_rect(client, frame_idx=0, x0=5, y0=5, x1=30, y1=30)
        _add_rect(client, frame_idx=1, x0=10, y0=10, x1=60, y1=60)
        _add_rect(client, frame_idx=3, x0=0, y0=0, x1=20, y1=20)

        # All stored independently
        assert 0 in session.per_frame_rois
        assert 1 in session.per_frame_rois
        assert 3 in session.per_frame_rois
        assert 2 not in session.per_frame_rois  # not set

        # Areas differ
        area_0 = session.per_frame_rois[0].sum()
        area_1 = session.per_frame_rois[1].sum()
        area_3 = session.per_frame_rois[3].sum()
        assert area_0 != area_1 or area_0 != area_3

    def test_cut_mode_on_frame(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3)
        # Add full rect, then cut a hole
        _add_rect(client, frame_idx=1, x0=0, y0=0, x1=64, y1=64)
        area_before = session.per_frame_rois[1].sum()

        resp = client.post("/api/roi/rectangle", json={
            "x0": 20, "y0": 20, "x1": 40, "y1": 40,
            "mode": "cut", "frame_idx": 1,
        })
        assert resp.status_code == 200
        area_after = session.per_frame_rois[1].sum()
        assert area_after < area_before

    def test_polygon_on_frame(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3)
        resp = client.post("/api/roi/polygon", json={
            "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
            "frame_idx": 2,
        })
        assert resp.status_code == 200
        assert 2 in session.per_frame_rois

    def test_circle_on_frame(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3)
        resp = client.post("/api/roi/circle", json={
            "cx": 32, "cy": 32, "r": 15, "frame_idx": 1,
        })
        assert resp.status_code == 200
        assert 1 in session.per_frame_rois


# ──────────────────────────────────────────────
# Mask Retrieval Per Frame
# ──────────────────────────────────────────────

class TestMaskRetrieval:
    """GET /roi/mask?frame_idx=N returns correct mask."""

    def test_mask_for_frame_with_roi(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3)
        _add_rect(client, frame_idx=1)
        resp = client.get("/api/roi/mask?frame_idx=1")
        assert resp.status_code == 200
        assert resp.content_type.startswith("image/")

    def test_mask_for_frame_without_roi(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3)
        # Frame 2 has no ROI → returns 404
        resp = client.get("/api/roi/mask?frame_idx=2")
        assert resp.status_code == 404


# ──────────────────────────────────────────────
# Clear Per-Frame ROI
# ──────────────────────────────────────────────

class TestClearFrameRoi:
    """DELETE /roi/frame/<idx> clears only that frame."""

    def test_clear_specific_frame(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3)
        _add_rect(client, frame_idx=1)
        _add_rect(client, frame_idx=2)
        assert 1 in session.per_frame_rois
        assert 2 in session.per_frame_rois

        resp = client.delete("/api/roi/frame/1")
        assert resp.status_code == 200
        assert 1 not in session.per_frame_rois
        assert 2 in session.per_frame_rois  # untouched

    def test_clear_frame_0_clears_roi_mask(self, client, tmp_path):
        _setup_images(client, tmp_path)
        _add_rect(client, frame_idx=0)
        assert session.roi_mask is not None

        resp = client.delete("/api/roi/frame/0")
        assert resp.status_code == 200
        assert session.roi_mask is None
        assert session.roi_confirmed is False


# ──────────────────────────────────────────────
# Invert Per-Frame
# ──────────────────────────────────────────────

class TestInvertPerFrame:
    """POST /roi/invert with frame_idx."""

    def test_invert_frame_0(self, client, tmp_path):
        _setup_images(client, tmp_path)
        _add_rect(client, frame_idx=0, x0=10, y0=10, x1=50, y1=50)
        area_before = session.roi_mask.sum()

        resp = client.post("/api/roi/invert", json={"frame_idx": 0})
        assert resp.status_code == 200
        area_after = session.roi_mask.sum()
        # Inverted area + original area = total pixels
        total = session.roi_mask.size
        assert area_before + area_after == total

    def test_invert_non_zero_frame(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3)
        _add_rect(client, frame_idx=2, x0=10, y0=10, x1=50, y1=50)
        area_before = session.per_frame_rois[2].sum()

        resp = client.post("/api/roi/invert", json={"frame_idx": 2})
        assert resp.status_code == 200
        area_after = session.per_frame_rois[2].sum()
        total = session.per_frame_rois[2].size
        assert area_before + area_after == total

    def test_invert_no_mask_returns_error(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3)
        resp = client.post("/api/roi/invert", json={"frame_idx": 2})
        assert resp.status_code == 400


# ──────────────────────────────────────────────
# Frames Status Endpoint
# ──────────────────────────────────────────────

class TestFramesStatus:
    """GET /roi/frames/status returns correct summary."""

    def test_status_no_rois(self, client, tmp_path):
        _setup_images(client, tmp_path, n=5)
        resp = client.get("/api/roi/frames/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total_frames"] == 5
        assert data["frames_with_roi"] == []
        assert data["frame_0_confirmed"] is False

    def test_status_with_rois(self, client, tmp_path):
        _setup_images(client, tmp_path, n=5)
        _add_rect(client, frame_idx=0)
        _add_rect(client, frame_idx=2)
        _add_rect(client, frame_idx=4)
        client.post("/api/roi/confirm")

        resp = client.get("/api/roi/frames/status")
        data = resp.get_json()
        assert set(data["frames_with_roi"]) == {0, 2, 4}
        assert data["frame_0_confirmed"] is True

    def test_status_after_clear(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3)
        _add_rect(client, frame_idx=1)
        client.delete("/api/roi/frame/1")

        resp = client.get("/api/roi/frames/status")
        data = resp.get_json()
        assert 1 not in data["frames_with_roi"]


# ──────────────────────────────────────────────
# Batch Import
# ──────────────────────────────────────────────

class TestBatchImport:
    """POST /roi/frames/batch-import imports masks from a folder."""

    def _create_mask_folder(self, tmp_path, n=3, size=64):
        """Create a folder with binary mask PNGs."""
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir(exist_ok=True)
        for i in range(n):
            # Create a simple binary mask (white rectangle on black)
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[10:50, 10:50] = 255
            Image.fromarray(mask).save(mask_dir / f"mask_{i:03d}.png")
        return str(mask_dir)

    def test_batch_import_sequential(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3)
        mask_dir = self._create_mask_folder(tmp_path, n=3)

        resp = client.post("/api/roi/frames/batch-import", json={
            "folder": mask_dir,
            "strategy": "sequential",
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["imported"] == 3

        # All three frames should have ROIs
        assert 0 in session.per_frame_rois
        assert 1 in session.per_frame_rois
        assert 2 in session.per_frame_rois

    def test_batch_import_confirms_frame_0(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3)
        mask_dir = self._create_mask_folder(tmp_path, n=3)

        client.post("/api/roi/frames/batch-import", json={
            "folder": mask_dir,
            "strategy": "sequential",
        })
        # Frame 0 mask should be confirmed
        assert session.roi_confirmed is True
        assert session.roi_mask is not None


# ──────────────────────────────────────────────
# Import Mask Per Frame
# ──────────────────────────────────────────────

class TestImportMaskPerFrame:
    """POST /roi/import with frame_idx."""

    def test_import_to_specific_frame(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3, size=64)
        # Create a mask file
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:55, 5:55] = 255
        mask_path = tmp_path / "test_mask.png"
        Image.fromarray(mask).save(mask_path)

        resp = client.post("/api/roi/import", json={
            "path": str(mask_path),
            "frame_idx": 2,
        })
        assert resp.status_code == 200
        assert 2 in session.per_frame_rois
        assert session.per_frame_rois[2].sum() > 0

    def test_import_to_frame_0_confirms(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3, size=64)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:55, 5:55] = 255
        mask_path = tmp_path / "test_mask.png"
        Image.fromarray(mask).save(mask_path)

        resp = client.post("/api/roi/import", json={
            "path": str(mask_path),
            "frame_idx": 0,
        })
        assert resp.status_code == 200
        assert session.roi_confirmed is True


# ──────────────────────────────────────────────
# Session Reset Clears Per-Frame ROIs
# ──────────────────────────────────────────────

class TestSessionReset:
    """Session reset clears all per-frame ROIs."""

    def test_reset_clears_per_frame_rois(self, client, tmp_path):
        _setup_images(client, tmp_path, n=3)
        _add_rect(client, frame_idx=0)
        _add_rect(client, frame_idx=1)
        _add_rect(client, frame_idx=2)
        assert len(session.per_frame_rois) == 3

        session.reset()
        assert len(session.per_frame_rois) == 0
        assert session.roi_mask is None
