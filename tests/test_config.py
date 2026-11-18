"""Unit tests for raftcorr.config (user-facing settings and their JSON round trip)."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from raftcorr.config import PAPER_STRAIN, RunConfig, StrainConfig, TilingConfig, TrackingConfig


# -- defaults -------------------------------------------------------------------------------------
def test_run_defaults_follow_the_paper():
    c = RunConfig()
    assert c.iters == 20
    assert c.smoothing_sigma == 0.0
    assert c.tracking.mode == "accumulative"
    assert c.tracking.every_n == 10 and c.tracking.key_frames == ()
    assert c.model == "s4" and c.device == "auto" and c.corr_mode is None
    assert c.tiling == TilingConfig()
    assert c.tiling.margin_px is None and c.tiling.max_tile_pixels is None and c.tiling.blend_px == 32


def test_strain_defaults_follow_the_paper():
    s = StrainConfig()
    assert s.weighting == "uniform"
    assert s.poly_order == 1          # plane fit
    assert s.measure == "green_lagrange"
    assert s.gauge_px == 41
    assert s.step == 1 and s.smoothing_sigma == 0.0 and s.connectivity_aware
    assert PAPER_STRAIN == s


def test_configs_are_frozen():
    with pytest.raises(FrozenInstanceError):
        RunConfig().iters = 5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        StrainConfig().gauge_px = 5  # type: ignore[misc]


def test_tracking_key_frames_coerced_to_int_tuple():
    t = TrackingConfig(mode="custom", key_frames=[3, 1.0])  # type: ignore[arg-type]
    assert t.key_frames == (3, 1)
    assert all(type(k) is int for k in t.key_frames)


# -- validation -----------------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [{"mode": "sometimes"}, {"every_n": 0}, {"every_n": -2}])
def test_tracking_validation(kwargs):
    with pytest.raises(ValueError):
        TrackingConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"margin_px": -1}, {"max_tile_pixels": 64 * 64 - 1}, {"memory_fraction": 0.04}, {"memory_fraction": 1.01},
     {"blend_px": -1}],
)
def test_tiling_validation(kwargs):
    with pytest.raises(ValueError):
        TilingConfig(**kwargs)


def test_tiling_boundary_values_are_accepted():
    TilingConfig(margin_px=0, max_tile_pixels=64 * 64, memory_fraction=0.05, blend_px=0)
    TilingConfig(memory_fraction=1.0)


@pytest.mark.parametrize("kwargs", [{"iters": 0}, {"smoothing_sigma": -0.5}, {"corr_mode": "dense"}])
def test_run_validation(kwargs):
    with pytest.raises(ValueError):
        RunConfig(**kwargs)


@pytest.mark.parametrize("mode", [None, "volume", "tile", "rows"])
def test_run_corr_modes(mode):
    assert RunConfig(corr_mode=mode).corr_mode == mode


@pytest.mark.parametrize(
    "kwargs",
    [{"measure": "hencky"}, {"weighting": "triangular"}, {"gauge_px": 2}, {"gauge_px": 40}, {"gauge_px": 1},
     {"poly_order": 0}, {"poly_order": 3}, {"step": 0}, {"min_coverage": 0.0}, {"min_coverage": 1.5}],
)
def test_strain_validation(kwargs):
    with pytest.raises(ValueError):
        StrainConfig(**kwargs)


def test_strain_boundary_values_are_accepted():
    StrainConfig(gauge_px=3, min_coverage=1.0, poly_order=2, weighting="gaussian", measure="euler_almansi")


# -- serialisation --------------------------------------------------------------------------------
def _custom_run() -> RunConfig:
    return RunConfig(
        model="s8", device="cuda:1", iters=12, corr_mode="tile", smoothing_sigma=1.5,
        tracking=TrackingConfig(mode="custom", every_n=4, key_frames=(2, 5)),
        tiling=TilingConfig(margin_px=40, max_tile_pixels=512 * 512, memory_fraction=0.5, blend_px=16),
    )


@pytest.mark.parametrize("cfg", [RunConfig(), _custom_run()])
def test_run_dict_round_trip(cfg):
    d = cfg.to_dict()
    assert isinstance(d["tracking"]["key_frames"], list)
    json.dumps(d)  # JSON-serialisable
    assert RunConfig.from_dict(d) == cfg


def test_run_save_load_round_trip(tmp_path: Path):
    cfg = _custom_run()
    p = tmp_path / "run.json"
    cfg.save(p)
    assert json.loads(p.read_text(encoding="utf-8"))["tracking"]["key_frames"] == [2, 5]
    assert RunConfig.load(p) == cfg


def test_run_from_dict_ignores_unknown_keys_and_fills_defaults():
    d = {"iters": 7, "future_option": True,
         "tracking": {"mode": "incremental", "unknown": 1},
         "tiling": {"blend_px": 8, "gpu_magic": "yes"}}
    cfg = RunConfig.from_dict(d)
    assert cfg.iters == 7 and cfg.tracking.mode == "incremental" and cfg.tiling.blend_px == 8
    assert cfg.model == "s4" and cfg.tracking.every_n == 10 and cfg.tiling.memory_fraction == 0.85
    assert RunConfig.from_dict({}) == RunConfig()
    assert RunConfig.from_dict(None) == RunConfig()  # type: ignore[arg-type]
    assert RunConfig.from_dict({"tracking": None, "tiling": None}) == RunConfig()


def test_run_from_dict_does_not_modify_input():
    d = _custom_run().to_dict()
    snapshot = json.loads(json.dumps(d))
    RunConfig.from_dict(d)
    assert d == snapshot


def test_run_from_dict_validates():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"tracking": {"mode": "bogus"}})
    with pytest.raises(ValueError):
        RunConfig.from_dict({"iters": 0})


@pytest.mark.parametrize(
    "cfg",
    [StrainConfig(), StrainConfig(measure="infinitesimal", gauge_px=21, weighting="gaussian", poly_order=2, step=3,
                                  gaussian_sigma=4.5, smoothing_sigma=1.0, connectivity_aware=False,
                                  condition_max=500.0, min_coverage=0.5)],
)
def test_strain_round_trip(cfg, tmp_path: Path):
    assert StrainConfig.from_dict(cfg.to_dict()) == cfg
    p = tmp_path / "strain.json"
    p.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    assert StrainConfig.from_dict(json.loads(p.read_text(encoding="utf-8"))) == cfg


def test_strain_from_dict_ignores_unknown_keys():
    s = StrainConfig.from_dict({"gauge_px": 31, "colour": "red"})
    assert s == StrainConfig(gauge_px=31)
    assert StrainConfig.from_dict(None) == StrainConfig()  # type: ignore[arg-type]
