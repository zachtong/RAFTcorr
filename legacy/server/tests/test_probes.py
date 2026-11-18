"""Tests for probe CRUD and time series extraction endpoints."""

import numpy as np
import pytest

from server.session import session


def test_list_empty(client):
    resp = client.get("/api/probes")
    assert resp.status_code == 200
    assert resp.get_json()["probes"] == []


def test_add_point(client):
    resp = client.post("/api/probes/point", json={"x": 32, "y": 16})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["type"] == "point"
    assert data["id"] == 1

    resp = client.get("/api/probes")
    assert len(resp.get_json()["probes"]) == 1


def test_add_line(client):
    resp = client.post("/api/probes/line", json={
        "p1": [10, 10], "p2": [50, 50]
    })
    assert resp.status_code == 200
    assert resp.get_json()["type"] == "line"


def test_add_area(client):
    resp = client.post("/api/probes/area", json={
        "shape": "rect", "data": [10, 10, 30, 30]
    })
    assert resp.status_code == 200
    assert resp.get_json()["type"] == "area"


def test_remove_probe(client):
    client.post("/api/probes/point", json={"x": 10, "y": 10})
    resp = client.delete("/api/probes/1?type=point")
    assert resp.status_code == 200
    assert len(client.get("/api/probes").get_json()["probes"]) == 0


def test_clear_all(client):
    client.post("/api/probes/point", json={"x": 10, "y": 10})
    client.post("/api/probes/line", json={"p1": [0, 0], "p2": [10, 10]})
    resp = client.delete("/api/probes")
    assert resp.status_code == 200
    assert len(client.get("/api/probes").get_json()["probes"]) == 0


def test_timeseries(client, mock_displacement_results):
    client.post("/api/probes/point", json={"x": 32, "y": 32})
    resp = client.get("/api/probes/timeseries?component=u&type=point")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "1" in data["series"]
    assert len(data["series"]["1"]) == 3
