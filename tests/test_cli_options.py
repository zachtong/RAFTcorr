"""Command-line options checked before any computation (no model weights or GPU needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import raftcorr.engine.pipeline as pipeline_module
from raftcorr.cli.main import main


class _TrackingStarted(Exception):
    pass


class _NoTracking:
    def __init__(self, *_args, **_kwargs) -> None:
        raise _TrackingStarted


@pytest.fixture()
def folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(pipeline_module, "Pipeline", _NoTracking)
    rng = np.random.default_rng(0)
    d = tmp_path / "imgs"
    d.mkdir()
    for i in range(2):
        Image.fromarray(rng.integers(0, 255, (96, 128), dtype=np.uint8)).save(d / f"img{i + 1}.png")
    return d


def _roi(tmp_path: Path, height: int) -> Path:
    roi = np.zeros((96, 128), np.uint8)
    roi[30:30 + height, 10:120] = 255
    path = tmp_path / f"roi_{height}.png"
    Image.fromarray(roi).save(path)
    return path


def test_edge_trim_out_of_range_fails_before_tracking(folder: Path) -> None:
    assert main(["run", str(folder), "--strain", "--edge-trim", "1.5", "--quiet"]) == 2


def test_trim_wider_than_the_roi_fails_before_tracking(folder: Path, tmp_path: Path) -> None:
    narrow = _roi(tmp_path, 20)                     # thinner than 2 x 0.7 x 20 px
    assert main(["run", str(folder), "--roi", str(narrow), "--strain", "--quiet"]) == 2
    with pytest.raises(_TrackingStarted):          # without the trim the settings are fine
        main(["run", str(folder), "--roi", str(narrow), "--strain", "--edge-trim", "0", "--quiet"])
    with pytest.raises(_TrackingStarted):
        main(["run", str(folder), "--roi", str(_roi(tmp_path, 60)), "--strain", "--quiet"])
