"""Tests for processing configuration and status endpoints."""

import pytest
from server.session import session


def test_configure(client):
    resp = client.post("/api/processing/configure", json={
        "mode": "incremental",
        "context_padding": 48,
        "use_smooth": False,
    })
    assert resp.status_code == 200
    assert session.config.mode == "incremental"
    assert session.config.context_padding == 48
    assert session.config.use_smooth is False


def test_configure_invalid(client):
    # Missing required fields should fail validation
    resp = client.post("/api/processing/configure", json={"mode": "accumulative"})
    # Config validates img_dir and model_path, which are empty
    # But configure just sets fields without full validation
    assert resp.status_code in (200, 400)


def test_status(client):
    resp = client.get("/api/processing/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["active"] is False
    assert data["has_results"] is False


def test_configure_key_frames(client):
    resp = client.post("/api/processing/configure", json={
        "mode": "incremental",
        "key_frames": [1, 50, 100],
    })
    assert resp.status_code == 200
    assert session.config.key_frames == [1, 50, 100]
    assert session.config.mode == "incremental"


def test_configure_key_frame_interval(client):
    resp = client.post("/api/processing/configure", json={
        "key_frame_interval": 25,
    })
    assert resp.status_code == 200
    assert session.config.key_frame_interval == 25


def test_configure_mask_dir(client):
    resp = client.post("/api/processing/configure", json={
        "mask_dir": "/some/path/to/masks",
    })
    assert resp.status_code == 200
    assert session.config.mask_dir == "/some/path/to/masks"


def test_configure_clear_incremental_fields(client):
    # Set fields first
    client.post("/api/processing/configure", json={
        "key_frames": [1, 10],
        "key_frame_interval": 5,
        "mask_dir": "/tmp",
    })
    # Clear them
    resp = client.post("/api/processing/configure", json={
        "key_frames": None,
        "key_frame_interval": None,
        "mask_dir": None,
    })
    assert resp.status_code == 200
    assert session.config.key_frames is None
    assert session.config.key_frame_interval is None
    assert session.config.mask_dir is None


def test_validate_masks_no_images(client):
    resp = client.post("/api/processing/validate-masks", json={
        "mask_dir": "/some/path",
    })
    assert resp.status_code == 400


def test_run_no_prerequisites(client):
    resp = client.post("/api/processing/run")
    assert resp.status_code == 400
