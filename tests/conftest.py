"""Shared fixtures: model weights, CUDA, synthetic pairs."""

from __future__ import annotations

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

