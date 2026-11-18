"""The documented public API (README, notebook): raftcorr.track / compute_strain / CLI run."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import raftcorr
from raftcorr import synthetic as syn

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]


@pytest.fixture()
def folder(tmp_path):
    frames = syn.render_sequence(192, 224, [syn.translation(1.5 * k, -0.5 * k) for k in (1, 2)], seed=9)
    d = tmp_path / "imgs"
    d.mkdir()
    for i, f in enumerate(frames):
        Image.fromarray(np.round(f).astype(np.uint8)).save(d / f"img{i + 1}.png")
    roi = np.zeros((192, 224), np.uint8)
    roi[40:150, 50:180] = 255
    Image.fromarray(roi).save(tmp_path / "roi.png")
    return d


def test_track_from_a_folder_with_a_roi_file(folder, weights, require_cuda):
    weights("s8")
    series = raftcorr.track(folder, roi=folder.parent / "roi.png", model="s8")
    assert series.n_frames == 3 and series.done.all()
    assert series.frame_names == ("img1.png", "img2.png", "img3.png")
    assert abs(np.nanmedian(series.u[2][series.mask]) - 3.0) < 0.1
    assert abs(np.nanmedian(series.v[2][series.mask]) + 1.0) < 0.1
    strain = raftcorr.compute_strain(series)
    assert abs(np.nanmedian(strain.components["exx"][2])) < 2e-3   # rigid translation: no strain


def test_cli_run_writes_results(folder, tmp_path, weights, require_cuda):
    from raftcorr.cli.main import main
    from raftcorr.export.scientific import read_results

    weights("s8")
    out = tmp_path / "out" / "r.npz"
    rc = main(["run", str(folder), "--roi", str(folder.parent / "roi.png"), "--model", "s8", "--strain",
               "--out", str(out), "--quiet"])
    assert rc == 0 and out.is_file()
    d = read_results(out)
    assert d["U"].shape[0] == 3 and "exx" in d and "Residual" in d
    assert main(["run", str(folder), "--model", "s8", "--out", str(out), "--quiet"]) == 2  # exists, no --overwrite
