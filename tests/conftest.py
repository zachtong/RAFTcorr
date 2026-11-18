"""Shared fixtures: model weights, CUDA, synthetic pairs; garbage collection in the main thread."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


HAS_CUDA = _cuda()


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if HAS_CUDA else "cpu"


@pytest.fixture(scope="session")
def require_cuda() -> None:
    if not HAS_CUDA:
        pytest.skip("needs a CUDA device")


def weights_for(key: str) -> Path | None:
    from raftcorr.models import available_models

    return available_models().get(key)


@pytest.fixture(scope="session")
def weights():
    """Callable ``weights(key) -> Path``; skips the test when the file is missing."""

    def _get(key: str) -> Path:
        p = weights_for(key)
        if p is None:
            pytest.skip(f"weights for RAFTcorr-{key} not installed")
        return p

    return _get


# -- garbage collection -------------------------------------------------------------------------
# Qt objects must be destroyed in the thread that owns them, but Python collects reference cycles
# in whichever thread allocates: an export or strain worker could destroy a widget an earlier test
# left behind. The application collects in its GUI thread only (raftcorr.gui.gc_guard); the tests
# collect between tests, in the main thread.
_THRESHOLD = gc.get_threshold()


@pytest.fixture(scope="session", autouse=True)
def _garbage_collection_in_the_main_thread():
    gc.disable()
    yield
    _release_qt_objects()
    gc.enable()


@pytest.fixture(autouse=True)
def _collect_between_tests():
    yield
    c0, c1, c2 = gc.get_count()
    if c0 > _THRESHOLD[0]:
        gc.collect(2 if c2 > _THRESHOLD[2] else 1 if c1 > _THRESHOLD[1] else 0)


def _release_qt_objects() -> None:
    """Close and delete the windows left over while the QApplication exists: a widget freed at
    interpreter exit, after the application, crashes some Python / PySide6 versions."""
    if "PySide6.QtWidgets" not in sys.modules:
        gc.collect()
        return
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is not None:
        for widget in QApplication.topLevelWidgets():
            widget.close()
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
    gc.collect()
