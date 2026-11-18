"""Tests for model discovery and description endpoints."""

import pytest


def test_list_models(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "models" in data
    assert isinstance(data["models"], list)
    # Each model should have path and label
    for m in data["models"]:
        assert "path" in m
        assert "label" in m


def test_describe_model(client):
    # First get a model path
    resp = client.get("/api/models")
    models = resp.get_json()["models"]
    if not models:
        pytest.skip("No models available")

    resp = client.post("/api/models/describe", json={"path": models[0]["path"]})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "metadata" in data
    meta = data["metadata"]
    assert "small" in meta
    assert "summary" in meta
    assert "safe_pmax" in meta


def test_describe_model_missing_path(client):
    resp = client.post("/api/models/describe", json={"path": ""})
    assert resp.status_code == 400


def test_select_model(client):
    resp = client.get("/api/models")
    models = resp.get_json()["models"]
    if not models:
        pytest.skip("No models available")

    resp = client.post("/api/models/select", json={"path": models[0]["path"]})
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True
