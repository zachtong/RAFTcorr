"""User-facing settings as frozen dataclasses with a JSON round trip.

Defaults follow the paper (decided 2026-11-18): 20 iterations, no displacement smoothing,
accumulative tracking, Green-Lagrange strain from an unweighted first-order virtual strain gauge.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, TypeVar

from .schedule import MODES as TRACKING_MODES

T = TypeVar("T")


def _from_dict(cls: type[T], d: dict[str, Any] | None) -> T:
    """Build a dataclass from a dict, ignoring unknown keys (forward compatibility)."""
    d = d or {}
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in d.items() if k in names})  # type: ignore[call-arg]


@dataclass(frozen=True)
class TrackingConfig:
    mode: str = "accumulative"            # accumulative | incremental | every_n | custom
    every_n: int = 10
    key_frames: tuple[int, ...] = ()      # 0-based reference frames for "custom"

    def __post_init__(self) -> None:
        if self.mode not in TRACKING_MODES:
            raise ValueError(f"tracking mode must be one of {TRACKING_MODES}, got {self.mode!r}")
        if self.every_n < 1:
            raise ValueError(f"every_n must be >= 1, got {self.every_n}")
        object.__setattr__(self, "key_frames", tuple(int(k) for k in self.key_frames))


@dataclass(frozen=True)
class TilingConfig:
    margin_px: int | None = None          # displacement margin; None = the model's lookup radius (32 s)
    max_tile_pixels: int | None = None    # None = from the memory model and the free device memory
    memory_fraction: float = 0.85         # of the free device memory one forward pass may use
    blend_px: int = 32                    # feathered overlap between neighbouring tiles

    def __post_init__(self) -> None:
        if self.margin_px is not None and self.margin_px < 0:
            raise ValueError("margin_px must be >= 0")
        if self.max_tile_pixels is not None and self.max_tile_pixels < 64 * 64:
            raise ValueError("max_tile_pixels must be at least 64 x 64")
        if not 0.05 <= self.memory_fraction <= 1.0:
            raise ValueError("memory_fraction must be in [0.05, 1]")
        if self.blend_px < 0:
            raise ValueError("blend_px must be >= 0")


@dataclass(frozen=True)
class RunConfig:
    model: str = "s4"                     # released model key (s1, s2, s4, s8) or a checkpoint path
    device: str = "auto"
    iters: int = 20
    corr_mode: str | None = None          # None = stored volume for s8, on demand for s <= 4
    smoothing_sigma: float = 0.0          # Gaussian smoothing of the displacement (px); 0 = raw (paper)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)

    def __post_init__(self) -> None:
        if self.iters < 1:
            raise ValueError("iters must be >= 1")
        if self.smoothing_sigma < 0:
            raise ValueError("smoothing_sigma must be >= 0")
        if self.corr_mode not in (None, "volume", "tile", "rows"):
            raise ValueError(f"unknown correlation mode {self.corr_mode!r}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tracking"]["key_frames"] = list(self.tracking.key_frames)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        d = dict(d or {})
        tracking = _from_dict(TrackingConfig, d.pop("tracking", None))
        tiling = _from_dict(TilingConfig, d.pop("tiling", None))
        base = _from_dict(cls, {k: v for k, v in d.items() if k not in ("tracking", "tiling")})
        return replace(base, tracking=tracking, tiling=tiling)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


STRAIN_MEASURES: tuple[str, ...] = ("green_lagrange", "infinitesimal", "euler_almansi")
WEIGHTINGS: tuple[str, ...] = ("uniform", "gaussian")


@dataclass(frozen=True)
class StrainConfig:
    measure: str = "green_lagrange"
    gauge_px: int = 41                    # virtual strain gauge side Lg (odd)
    weighting: str = "uniform"            # uniform (paper) | gaussian
    poly_order: int = 1                   # 1 = plane (paper), 2 = quadratic
    step: int = 1                         # output spacing (px); 1 = every pixel
    gaussian_sigma: float | None = None   # None = Lg / 4 (gaussian weighting only)
    smoothing_sigma: float = 0.0          # Gaussian pre-smoothing of the displacement (px)
    connectivity_aware: bool = True       # never fit a gauge across a crack / hole
    condition_max: float = 1000.0         # reject ill-conditioned fits (sparse gauges)
    min_coverage: float = 0.25            # fraction of the gauge that must hold valid points
    edge_trim_alpha: float = 0.70         # trim points within alpha * (Lg - 1) / 2 of an edge (pyALDIC)

    def __post_init__(self) -> None:
        if self.measure not in STRAIN_MEASURES:
            raise ValueError(f"measure must be one of {STRAIN_MEASURES}")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}")
        if self.gauge_px < 3 or self.gauge_px % 2 == 0:
            raise ValueError("gauge_px must be an odd integer >= 3")
        if self.poly_order not in (1, 2):
            raise ValueError("poly_order must be 1 or 2")
        if self.step < 1:
            raise ValueError("step must be >= 1")
        if not 0.0 < self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in (0, 1]")
        if not 0.0 <= self.edge_trim_alpha <= 1.0:
            raise ValueError("edge_trim_alpha must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrainConfig":
        return _from_dict(cls, d)


PAPER_STRAIN = StrainConfig()
