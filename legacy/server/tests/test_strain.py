"""Tests for strain calculation and data endpoints."""

import numpy as np
import pytest

from server.session import session


def test_status_no_results(client):
    resp = client.get("/api/strain/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["computed"] is False


def test_calculate_no_displacement(client):
    resp = client.post("/api/strain/calculate", json={})
    assert resp.status_code == 400


def test_strain_frame_no_results(client):
    resp = client.get("/api/strain/frame/0")
    assert resp.status_code == 404


def test_with_mock_strain(client, mock_displacement_results):
    """Test strain endpoints with pre-populated mock strain data."""
    # Manually set strain results
    strain_dict = {
        "exx": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "eyy": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "exy": np.random.randn(64, 64).astype(np.float32) * 0.005,
        "e1": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "e2": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "max_shear": np.abs(np.random.randn(64, 64).astype(np.float32) * 0.005),
        "von_mises": np.abs(np.random.randn(64, 64).astype(np.float32) * 0.01),
        "rotation": np.random.randn(64, 64).astype(np.float32) * 0.1,
    }
    session.strain_results = [strain_dict] * 3
    session.strain_components = list(strain_dict.keys())

    # Status
    resp = client.get("/api/strain/status")
    assert resp.status_code == 200
    assert resp.get_json()["computed"] is True

    # Frame data
    resp = client.get("/api/strain/frame/0?component=exx")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["shape"] == [64, 64]

    # Render
    resp = client.get("/api/strain/render/0?component=exx")
    assert resp.status_code == 200
    assert resp.content_type == "image/png"
