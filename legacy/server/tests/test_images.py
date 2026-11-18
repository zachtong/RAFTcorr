"""Tests for image loading and serving endpoints."""

import pytest


def test_load_images(client, mock_images):
    resp = client.post("/api/images/load", json={"dir": mock_images})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["count"] == 3
    assert data["width"] == 64
    assert data["height"] == 64
    assert len(data["files"]) == 3


def test_load_images_invalid_dir(client):
    resp = client.post("/api/images/load", json={"dir": "/nonexistent/path"})
    assert resp.status_code == 400


def test_load_images_empty_dir(client, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    resp = client.post("/api/images/load", json={"dir": str(empty)})
    assert resp.status_code == 400


def test_get_reference(client, loaded_images):
    resp = client.get("/api/images/reference")
    assert resp.status_code == 200
    assert resp.content_type == "image/png"
    assert len(resp.data) > 0


def test_get_reference_no_images(client):
    resp = client.get("/api/images/reference")
    assert resp.status_code == 404


def test_get_frame(client, loaded_images):
    resp = client.get("/api/images/frame/0")
    assert resp.status_code == 200
    assert resp.content_type == "image/png"


def test_get_frame_out_of_range(client, loaded_images):
    resp = client.get("/api/images/frame/99")
    assert resp.status_code == 400
