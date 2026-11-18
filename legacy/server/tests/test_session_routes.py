"""Tests for session save/load/info endpoints."""

import os
from unittest.mock import patch

import numpy as np
import pytest

from server.session import session


# ===================================================================
# POST /api/session/save
# ===================================================================


class TestSessionSave:
    """Tests for the session save endpoint."""

    def test_no_results_returns_400(self, client):
        resp = client.post("/api/session/save", json={
            "file_path": "session.raftproj",
        })
        assert resp.status_code == 400
        assert "No results" in resp.get_json()["error"]

    def test_missing_file_path_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/session/save", json={})
        assert resp.status_code == 400
        assert "Missing file_path" in resp.get_json()["error"]

    def test_empty_file_path_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/session/save", json={"file_path": "   "})
        assert resp.status_code == 400
        assert "Missing file_path" in resp.get_json()["error"]

    def test_nonexistent_directory_returns_400(self, client, mock_displacement_results):
        resp = client.post("/api/session/save", json={
            "file_path": "/nonexistent/dir/session.raftproj",
        })
        assert resp.status_code == 400
        assert "does not exist" in resp.get_json()["error"]

    def test_overwrite_false_returns_409(self, client, mock_displacement_results, tmp_path):
        existing = tmp_path / "session.raftproj"
        existing.write_text("placeholder")

        resp = client.post("/api/session/save", json={
            "file_path": str(existing),
            "overwrite": False,
        })
        assert resp.status_code == 409
        data = resp.get_json()
        assert data["exists"] is True
        assert "already exists" in data["error"]

    def test_overwrite_true_proceeds(self, client, mock_displacement_results, tmp_path):
        existing = tmp_path / "session.raftproj"
        existing.write_text("placeholder")

        with patch(
            "server.routes.session_routes.save_session",
            return_value=str(existing),
        ):
            resp = client.post("/api/session/save", json={
                "file_path": str(existing),
                "overwrite": True,
            })
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True

    def test_successful_save(self, client, mock_displacement_results, tmp_path):
        out_path = tmp_path / "analysis.raftproj"

        def fake_save(sess, path):
            # Create the file so os.path.getsize succeeds after save
            with open(path, "wb") as f:
                f.write(b"\x00" * 2048)
            return path

        with patch(
            "server.routes.session_routes.save_session",
            side_effect=fake_save,
        ):
            resp = client.post("/api/session/save", json={
                "file_path": str(out_path),
            })
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            assert data["path"] == str(out_path)
            assert "size_mb" in data

    def test_auto_appends_extension(self, client, mock_displacement_results, tmp_path):
        """If file_path lacks .raftproj, the extension is added automatically."""
        base_path = tmp_path / "myproject"
        expected_path = str(base_path) + ".raftproj"

        def fake_save(sess, path):
            with open(path, "wb") as f:
                f.write(b"\x00" * 1024)
            return path

        with patch(
            "server.routes.session_routes.save_session",
            side_effect=fake_save,
        ):
            resp = client.post("/api/session/save", json={
                "file_path": str(base_path),
            })
            assert resp.status_code == 200
            assert resp.get_json()["ok"] is True

    def test_save_exception_returns_500(self, client, mock_displacement_results, tmp_path):
        out_path = str(tmp_path / "session.raftproj")

        with patch(
            "server.routes.session_routes.save_session",
            side_effect=RuntimeError("disk full"),
        ):
            resp = client.post("/api/session/save", json={
                "file_path": out_path,
            })
            assert resp.status_code == 500
            assert "disk full" in resp.get_json()["error"]


# ===================================================================
# POST /api/session/load
# ===================================================================


class TestSessionLoad:
    """Tests for the session load endpoint."""

    def test_missing_file_path_returns_400(self, client):
        resp = client.post("/api/session/load", json={})
        assert resp.status_code == 400
        assert "Missing file_path" in resp.get_json()["error"]

    def test_empty_file_path_returns_400(self, client):
        resp = client.post("/api/session/load", json={"file_path": "  "})
        assert resp.status_code == 400
        assert "Missing file_path" in resp.get_json()["error"]

    def test_file_not_found_returns_404(self, client, tmp_path):
        resp = client.post("/api/session/load", json={
            "file_path": str(tmp_path / "nonexistent.raftproj"),
        })
        assert resp.status_code == 404
        assert "not found" in resp.get_json()["error"].lower()

    def test_successful_load(self, client, tmp_path):
        proj_file = tmp_path / "session.raftproj"
        proj_file.write_text("placeholder")

        mock_result = {
            "warnings": [],
            "summary": {
                "n_frames": 10,
                "image_dir": "/data/images",
                "has_strain": True,
            },
        }

        with patch(
            "server.routes.session_routes.load_session",
            return_value=mock_result,
        ):
            resp = client.post("/api/session/load", json={
                "file_path": str(proj_file),
            })
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            assert data["warnings"] == []
            assert data["summary"]["n_frames"] == 10

    def test_load_with_warnings(self, client, tmp_path):
        proj_file = tmp_path / "old.raftproj"
        proj_file.write_text("placeholder")

        mock_result = {
            "warnings": ["Image directory not found, using cached data"],
            "summary": {"n_frames": 5, "image_dir": None, "has_strain": False},
        }

        with patch(
            "server.routes.session_routes.load_session",
            return_value=mock_result,
        ):
            resp = client.post("/api/session/load", json={
                "file_path": str(proj_file),
            })
            assert resp.status_code == 200
            data = resp.get_json()
            assert len(data["warnings"]) == 1

    def test_load_exception_returns_500(self, client, tmp_path):
        proj_file = tmp_path / "corrupt.raftproj"
        proj_file.write_text("corrupted content")

        with patch(
            "server.routes.session_routes.load_session",
            side_effect=RuntimeError("invalid file format"),
        ):
            resp = client.post("/api/session/load", json={
                "file_path": str(proj_file),
            })
            assert resp.status_code == 500
            assert "invalid file format" in resp.get_json()["error"]


# ===================================================================
# GET /api/session/info
# ===================================================================


class TestSessionInfo:
    """Tests for the session info endpoint."""

    def test_info_returns_dict(self, client):
        mock_info = {
            "has_images": False,
            "n_images": 0,
            "image_dir": "",
            "has_roi": False,
            "has_displacement": False,
            "n_displacement_frames": 0,
            "has_strain": False,
            "n_probes": 0,
        }

        with patch(
            "server.routes.session_routes.get_session_info",
            return_value=mock_info,
        ):
            resp = client.get("/api/session/info")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "has_images" in data
            assert "has_displacement" in data
            assert data["n_images"] == 0

    def test_info_with_loaded_data(self, client, mock_displacement_results):
        """After displacement results are loaded, info reflects that."""
        mock_info = {
            "has_images": True,
            "n_images": 3,
            "image_dir": "/data/images",
            "has_roi": True,
            "has_displacement": True,
            "n_displacement_frames": 3,
            "has_strain": False,
            "n_probes": 0,
        }

        with patch(
            "server.routes.session_routes.get_session_info",
            return_value=mock_info,
        ):
            resp = client.get("/api/session/info")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["has_displacement"] is True
            assert data["n_displacement_frames"] == 3
            assert data["has_images"] is True
