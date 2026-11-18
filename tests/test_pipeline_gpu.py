"""End-to-end tests of the sequence pipeline on synthetic speckle with known fields (GPU + weights)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from raftcorr import synthetic as syn
from raftcorr.callbacks import FunctionCallbacks
from raftcorr.config import RunConfig, TilingConfig, TrackingConfig
from raftcorr.engine.pipeline import Pipeline
from raftcorr.models import load_model

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]

H, W = 320, 384


def fields_growing(n: int):
    """Frame k: translation (3k, -2k) px plus 0.6 % k stretch in x about the centre."""
    return [syn.affine(0.006 * k, 0.0, 0.0, -0.002 * k, tx=3.0 * k, ty=-2.0 * k, cx=W / 2, cy=H / 2)
            for k in range(1, n)]


@pytest.fixture(scope="module")
def sequence():
    fields = fields_growing(4)
    frames = syn.render_sequence(H, W, fields, seed=3)
    truth = [np.zeros((2, H, W), np.float32)] + [syn.evaluate(f, H, W) for f in fields]
    return frames, truth


@pytest.fixture(scope="module")
def roi():
    m = np.zeros((H, W), bool)
    m[60:260, 70:320] = True
    m[140:180, 170:220] = False  # a hole
    return m


@pytest.fixture(scope="module")
def s4(weights, require_cuda):
    return load_model(weights("s4"), device="cuda")


def _errors(series, truth, k, border=12):
    from scipy import ndimage

    inner = ndimage.binary_erosion(series.mask, iterations=border)
    gt = truth[k][:, series.grid.slices[0], series.grid.slices[1]]
    e = np.abs(series.field(k) - gt)[:, inner]
    return float(np.nanmedian(e)), float(np.nanpercentile(e, 99)), float(np.isnan(e).mean())


def test_accumulative_matches_truth(sequence, roi, s4):
    frames, truth = sequence
    series = Pipeline(RunConfig(model="s4"), loaded=s4).run(frames, roi)
    assert series.done.all()
    assert np.all(series.u[0][series.mask] == 0)
    assert np.isnan(series.u[1][~series.mask]).all()
    for k in range(1, len(frames)):
        med, p99, nan_frac = _errors(series, truth, k)
        assert med < 0.05, (k, med)
        assert p99 < 0.3, (k, p99)
        assert nan_frac == 0.0


def test_incremental_composition(sequence, roi, s4):
    frames, truth = sequence
    cfg = RunConfig(model="s4", tracking=TrackingConfig(mode="incremental"))
    series = Pipeline(cfg, loaded=s4).run(frames, roi)
    assert series.schedule.ref_indices == (0, 1, 2)
    for k in range(1, len(frames)):
        med, p99, nan_frac = _errors(series, truth, k)
        assert med < 0.08, (k, med)
        assert nan_frac < 0.01


def test_tiles_agree_with_one_window(sequence, roi, s4):
    frames, _ = sequence
    one = Pipeline(RunConfig(model="s4"), loaded=s4).run(frames[:2], roi)
    cfg = RunConfig(model="s4", tiling=TilingConfig(max_tile_pixels=230 * 230, margin_px=40))
    pipe = Pipeline(cfg, loaded=s4)
    tiled = pipe.run(frames[:2], roi)
    assert len(pipe.last_plan.tiles) > 1
    d = np.abs(tiled.field(1) - one.field(1))[:, tiled.mask]
    assert np.nanmedian(d) < 0.02
    assert np.nanpercentile(d, 99.5) < 0.2


def test_large_translation_s8(weights, require_cuda):
    s8 = load_model(weights("s8"), device="cuda")
    frames = syn.render_sequence(384, 448, [syn.translation(60.0, -35.0)], seed=5)
    m = np.zeros((384, 448), bool)
    m[140:250, 160:300] = True
    series = Pipeline(RunConfig(model="s8"), loaded=s8).run(frames, m)
    u = series.field(1)[:, series.mask]
    assert abs(np.nanmedian(u[0]) - 60.0) < 0.3
    assert abs(np.nanmedian(u[1]) + 35.0) < 0.3


def test_frame_mask_excludes_points(sequence, roi, s4):
    frames, truth = sequence
    crack = np.ones((H, W), bool)
    crack[:, 200:206] = False  # excluded band in frame 2's coordinates
    series = Pipeline(RunConfig(model="s4"), loaded=s4).run(frames[:3], roi, frame_masks={2: crack})
    X, Y = series.grid.coords()
    xd = X + truth[2][0][series.grid.slices]
    inside_band = series.mask & (xd >= 200.5) & (xd <= 204.5)
    assert inside_band.any()
    assert np.isnan(series.u[2][inside_band]).all()
    assert np.isfinite(series.u[1][inside_band]).all()  # frame 1 has no mask


def test_stop_keeps_completed_frames(sequence, roi, s4):
    frames, _ = sequence
    calls = {"n": 0}

    def progress(f, msg):
        if msg.startswith("Frame "):         # "Frame k/N (reference r): ..."
            calls["n"] += 1

    cb = FunctionCallbacks(progress=progress, should_stop=lambda: calls["n"] >= 1)
    series = Pipeline(RunConfig(model="s4"), callbacks=cb, loaded=s4).run(frames, roi)
    assert series.done[0] and series.done[1]
    assert not series.done[2:].any()
    assert any("stopped" in n for n in series.info.notes)


def test_subset_of_frames_includes_references(sequence, roi, s4):
    frames, _ = sequence
    cfg = RunConfig(model="s4", tracking=TrackingConfig(mode="incremental"))
    series = Pipeline(cfg, loaded=s4).run(frames, roi, frames=[3])
    assert series.done.tolist() == [True, True, True, True]
    cfg2 = replace(cfg, tracking=TrackingConfig(mode="accumulative"))
    s2 = Pipeline(cfg2, loaded=s4).run(frames, roi, frames=[3])
    assert s2.done.tolist() == [True, False, False, True]
