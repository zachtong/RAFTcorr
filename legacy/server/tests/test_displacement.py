"""Tests for displacement data and render endpoints."""

import pytest


def test_info_no_results(client):
    resp = client.get("/api/displacement/info")
    assert resp.status_code == 404


def test_info_with_results(client, mock_displacement_results):
    resp = client.get("/api/displacement/info")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["num_frames"] == 3
    assert data["roi_shape"] == [64, 64]


def test_frame_data(client, mock_displacement_results):
    resp = client.get("/api/displacement/frame/0?component=u")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["shape"] == [64, 64]
    assert "vmin" in data
    assert "vmax" in data


def test_frame_data_v_component(client, mock_displacement_results):
    resp = client.get("/api/displacement/frame/1?component=v")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["shape"] == [64, 64]


def test_frame_data_out_of_range(client, mock_displacement_results):
    resp = client.get("/api/displacement/frame/99")
    assert resp.status_code == 400


def test_render(client, mock_displacement_results):
    resp = client.get("/api/displacement/render/0?component=u&colormap=turbo")
    assert resp.status_code == 200
    assert resp.content_type == "image/png"
    assert len(resp.data) > 100
