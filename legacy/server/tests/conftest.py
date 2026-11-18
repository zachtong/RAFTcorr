"""Shared fixtures for backend tests."""

import sys
import os
import numpy as np
import pytest

# Ensure project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from server.app import create_app, socketio
from server.session import session


@pytest.fixture()
def app():
    """Create a test Flask application."""
    app = create_app({"TESTING": True})
    yield app


@pytest.fixture()
def client(app):
    """Create a test client."""
    return app.test_client()


@pytest.fixture(autouse=True)
def reset_session():
    """Reset session state before each test."""
    session.reset()
    yield
    session.reset()


@pytest.fixture()
def mock_images(tmp_path):
    """Create a temporary directory with test images."""
    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    for i in range(3):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(img_dir / f"frame_{i:03d}.png")

    return str(img_dir)


@pytest.fixture()
def loaded_images(client, mock_images):
    """Load images into the session and return the response."""
    resp = client.post("/api/images/load", json={"dir": mock_images})
    assert resp.status_code == 200
    return resp.get_json()


@pytest.fixture()
def mock_displacement_results():
    """Create mock displacement results in session."""
    results = []
    for _ in range(3):
        disp = np.random.randn(64, 64, 2).astype(np.float32) * 5.0
        results.append(disp)
    session.displacement_results = results
    session.roi_rect = (0, 0, 64, 64)
    session.roi_mask = np.ones((64, 64), dtype=bool)
    session.image_width = 64
    session.image_height = 64
    # Create a fake reference image
    session.reference_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return results
