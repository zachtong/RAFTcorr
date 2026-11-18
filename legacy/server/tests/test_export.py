"""Tests for scientific, batch image, animation, and report export endpoints."""

import os
from unittest.mock import patch

import numpy as np
import pytest

from server.session import session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_strain(n_frames: int = 3, size: int = 64):
    """Populate session with mock strain results."""
    strain_dict = {
        "exx": np.random.randn(size, size).astype(np.float32) * 0.01,
        "eyy": np.random.randn(size, size).astype(np.float32) * 0.01,
        "exy": np.random.randn(size, size).astype(np.float32) * 0.005,
    }
    session.strain_results = [strain_dict] * n_frames
    session.strain_components = list(strain_dict.keys())


# ===================================================================
# POST /api/export/scientific
# ===================================================================


class TestExportScientific:
    """Tests for the scientific export endpoint."""

    def test_no_results_returns_400(self, client):
        resp = client.post("/api/export/scientific", json={"file_path": "out.npz"})
        assert resp.status_code == 400
        assert "No displacement results" in resp.get_json()["error"]

    def test_missing_file_path_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/scientific", json={})
        assert resp.status_code == 400
        assert "Missing file_path" in resp.get_json()["error"]

    def test_empty_file_path_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/scientific", json={"file_path": "  "})
        assert resp.status_code == 400
        assert "Missing file_path" in resp.get_json()["error"]

    def test_nonexistent_directory_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/scientific", json={
            "file_path": "/nonexistent/dir/results.npz",
        })
        assert resp.status_code == 400
        assert "does not exist" in resp.get_json()["error"]

    def test_overwrite_false_returns_409(self, client, mock_displacement_results, tmp_path):
        existing_file = tmp_path / "existing.npz"
        existing_file.write_text("placeholder")

        resp = client.post("/api/export/scientific", json={
            "file_path": str(existing_file),
            "overwrite": False,
        })
        assert resp.status_code == 409
        data = resp.get_json()
        assert data["exists"] is True
        assert "already exists" in data["error"]

    def test_overwrite_true_proceeds(self, client, mock_displacement_results, tmp_path):
        existing_file = tmp_path / "existing.npz"
        existing_file.write_text("placeholder")

        with patch("server.routes.export.save_scientific_results") as mock_save:
            resp = client.post("/api/export/scientific", json={
                "file_path": str(existing_file),
                "overwrite": True,
            })
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True
            mock_save.assert_called_once()

    def test_successful_export(self, client, mock_displacement_results, tmp_path):
        out_path = str(tmp_path / "results.npz")

        with patch("server.routes.export.save_scientific_results") as mock_save:
            resp = client.post("/api/export/scientific", json={
                "file_path": out_path,
                "upsample_strain": False,
                "metadata": {"user_note": "test"},
            })
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            assert data["path"] == out_path
            mock_save.assert_called_once()
            # Verify upsample_strain was forwarded
            assert mock_save.call_args.kwargs["upsample_strain"] is False

    def test_export_exception_returns_500(self, client, mock_displacement_results, tmp_path):
        out_path = str(tmp_path / "results.npz")

        with patch(
            "server.routes.export.save_scientific_results",
            side_effect=RuntimeError("write failed"),
        ):
            resp = client.post("/api/export/scientific", json={
                "file_path": out_path,
            })
            assert resp.status_code == 500
            assert "write failed" in resp.get_json()["error"]


# ===================================================================
# POST /api/export/images (batch)
# ===================================================================


class TestExportImages:
    """Tests for the batch image export endpoint."""

    def test_no_results_returns_400(self, client):
        resp = client.post("/api/export/images", json={"output_dir": "/tmp/out"})
        assert resp.status_code == 400
        assert "No displacement results" in resp.get_json()["error"]

    def test_already_exporting_returns_409(self, client, mock_displacement_results, tmp_path):
        session.export_active = True
        resp = client.post("/api/export/images", json={
            "output_dir": str(tmp_path),
        })
        assert resp.status_code == 409
        assert "already in progress" in resp.get_json()["error"]

    def test_missing_output_dir_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/images", json={})
        assert resp.status_code == 400
        assert "Missing output_dir" in resp.get_json()["error"]

    def test_empty_output_dir_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/images", json={"output_dir": "   "})
        assert resp.status_code == 400
        assert "Missing output_dir" in resp.get_json()["error"]

    def test_nonexistent_parent_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/images", json={
            "output_dir": "/nonexistent/parent/subdir",
        })
        assert resp.status_code == 400
        assert "does not exist" in resp.get_json()["error"]

    def test_successful_start(self, client, mock_displacement_results, tmp_path):
        with patch("server.routes.export.export_batch_images"):
            resp = client.post("/api/export/images", json={
                "output_dir": str(tmp_path / "output"),
                "components": {"u": True, "v": True},
                "frame_range": [0, 2],
                "settings": {"dpi": 150},
            })
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True


# ===================================================================
# GET /api/export/images/status
# ===================================================================


class TestExportStatus:
    """Tests for the export status endpoint."""

    def test_idle_status(self, client):
        resp = client.get("/api/export/images/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["active"] is False
        assert data["progress"] == 0
        assert data["total"] == 0
        assert data["percent"] == 0

    def test_active_status(self, client):
        session.export_active = True
        session.export_progress = 5
        session.export_total = 10
        resp = client.get("/api/export/images/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["active"] is True
        assert data["progress"] == 5
        assert data["total"] == 10
        assert data["percent"] == 50.0


# ===================================================================
# POST /api/export/images/cancel
# ===================================================================


class TestExportCancel:
    """Tests for the export cancel endpoint."""

    def test_no_export_returns_400(self, client):
        resp = client.post("/api/export/images/cancel")
        assert resp.status_code == 400
        assert "No export in progress" in resp.get_json()["error"]

    def test_cancel_active_export(self, client):
        session.export_active = True
        resp = client.post("/api/export/images/cancel")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        assert session.export_cancel.is_set()


# ===================================================================
# POST /api/export/animation
# ===================================================================


class TestExportAnimation:
    """Tests for the animation export endpoint."""

    def test_no_results_returns_400(self, client):
        resp = client.post("/api/export/animation", json={
            "output_path": "out.gif",
        })
        assert resp.status_code == 400
        assert "No displacement results" in resp.get_json()["error"]

    def test_already_exporting_returns_409(self, client, mock_displacement_results):
        session.export_active = True
        resp = client.post("/api/export/animation", json={
            "output_path": "out.gif",
        })
        assert resp.status_code == 409
        assert "already in progress" in resp.get_json()["error"]

    def test_invalid_format_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/animation", json={
            "output_path": "out.gif",
            "format": "avi",
        })
        assert resp.status_code == 400
        assert "format" in resp.get_json()["error"].lower()

    def test_negative_fps_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/animation", json={
            "output_path": "out.gif",
            "fps": -5,
        })
        assert resp.status_code == 400
        assert "fps" in resp.get_json()["error"]

    def test_zero_fps_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/animation", json={
            "output_path": "out.gif",
            "fps": 0,
        })
        assert resp.status_code == 400
        assert "fps" in resp.get_json()["error"]

    def test_resize_factor_too_small_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/animation", json={
            "output_path": "out.gif",
            "resize_factor": 0.01,
        })
        assert resp.status_code == 400
        assert "resize_factor" in resp.get_json()["error"]

    def test_resize_factor_too_large_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/animation", json={
            "output_path": "out.gif",
            "resize_factor": 10.0,
        })
        assert resp.status_code == 400
        assert "resize_factor" in resp.get_json()["error"]

    def test_missing_output_path_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/animation", json={
            "format": "gif",
        })
        assert resp.status_code == 400
        assert "Missing output_path" in resp.get_json()["error"]

    def test_nonexistent_directory_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/animation", json={
            "output_path": "/nonexistent/dir/anim.gif",
            "format": "gif",
        })
        assert resp.status_code == 400
        assert "does not exist" in resp.get_json()["error"]

    def test_successful_gif_start(self, client, mock_displacement_results, tmp_path):
        out_path = str(tmp_path / "anim.gif")
        with patch("server.routes.export.export_animation"):
            resp = client.post("/api/export/animation", json={
                "output_path": out_path,
                "format": "gif",
                "fps": 15,
                "resize_factor": 1.5,
                "component": "magnitude",
            })
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True

    def test_successful_mp4_start(self, client, mock_displacement_results, tmp_path):
        out_path = str(tmp_path / "anim.mp4")
        with patch("server.routes.export.export_animation"):
            resp = client.post("/api/export/animation", json={
                "output_path": out_path,
                "format": "mp4",
                "fps": 30,
            })
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True

    def test_auto_appends_extension(self, client, mock_displacement_results, tmp_path):
        """If the output_path lacks the correct extension, it is appended."""
        out_path = str(tmp_path / "anim")
        with patch("server.routes.export.export_animation"):
            resp = client.post("/api/export/animation", json={
                "output_path": out_path,
                "format": "gif",
            })
            assert resp.status_code == 200


# ===================================================================
# POST /api/export/report
# ===================================================================


class TestExportReport:
    """Tests for the report generation endpoint."""

    def test_no_results_returns_400(self, client):
        resp = client.post("/api/export/report", json={
            "output_path": "report.html",
        })
        assert resp.status_code == 400
        assert "No displacement results" in resp.get_json()["error"]

    def test_invalid_format_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/report", json={
            "output_path": "report.html",
            "format": "docx",
        })
        assert resp.status_code == 400
        assert "format" in resp.get_json()["error"].lower()

    def test_invalid_theme_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/report", json={
            "output_path": "report.html",
            "theme": "neon",
        })
        assert resp.status_code == 400
        assert "theme" in resp.get_json()["error"].lower()

    def test_missing_output_path_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/report", json={
            "format": "html",
        })
        assert resp.status_code == 400
        assert "Missing output_path" in resp.get_json()["error"]

    def test_nonexistent_directory_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/export/report", json={
            "output_path": "/nonexistent/dir/report.html",
        })
        assert resp.status_code == 400
        assert "does not exist" in resp.get_json()["error"]

    def test_successful_html_report(self, client, mock_displacement_results, tmp_path):
        out_path = str(tmp_path / "report.html")
        mock_result = {"html_path": out_path}

        with patch(
            "raft_dic_gui.report_generator.generate_report",
            return_value=mock_result,
        ):
            resp = client.post("/api/export/report", json={
                "output_path": out_path,
                "format": "html",
                "theme": "dark",
                "custom_title": "My Report",
            })
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            assert data["html_path"] == out_path

    def test_successful_pdf_report(self, client, mock_displacement_results, tmp_path):
        out_path = str(tmp_path / "report.pdf")
        html_path = str(tmp_path / "report.html")
        mock_result = {"html_path": html_path, "pdf_path": out_path}

        with patch(
            "raft_dic_gui.report_generator.generate_report",
            return_value=mock_result,
        ):
            resp = client.post("/api/export/report", json={
                "output_path": out_path,
                "format": "pdf",
                "theme": "light",
            })
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            assert data["pdf_path"] == out_path

    def test_report_exception_returns_500(self, client, mock_displacement_results, tmp_path):
        out_path = str(tmp_path / "report.html")

        with patch(
            "raft_dic_gui.report_generator.generate_report",
            side_effect=RuntimeError("render failed"),
        ):
            resp = client.post("/api/export/report", json={
                "output_path": out_path,
                "format": "html",
            })
            assert resp.status_code == 500
            assert "render failed" in resp.get_json()["error"]
