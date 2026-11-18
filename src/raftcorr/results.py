"""Results of a run: displacement series (and, after post-processing, strain series).

Indexing: frame ``k`` of a series is image ``k`` of the sequence. Frame 0 is the reference and
has zero displacement inside the ROI. (The 1.x application shifted indices by one.)

Fields live on a :class:`~raftcorr.engine.mapping.Grid`: the bounding box of the frame-0 ROI.
Values outside the ROI, and at points the run could not track, are NaN. Arrays may be numpy
memmaps for long sequences; treat them as read-only.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .engine.mapping import Grid
from .schedule import FrameSchedule


@dataclass(frozen=True)
class RunInfo:
    """Provenance of a run (written into every export)."""

    software_version: str
    model_name: str
    model_sha256: str
    downsample_factor: int
    iters: int
    corr_mode: str
    device: str
    config: dict[str, Any]
    started: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))
    seconds: float = 0.0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "software_version": self.software_version,
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
            "downsample_factor": self.downsample_factor,
            "iters": self.iters,
            "corr_mode": self.corr_mode,
            "device": self.device,
            "config": self.config,
            "started": self.started,
            "seconds": self.seconds,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunInfo":
        return cls(
            software_version=str(d.get("software_version", "")),
            model_name=str(d.get("model_name", "")),
            model_sha256=str(d.get("model_sha256", "")),
            downsample_factor=int(d.get("downsample_factor", 0)),
            iters=int(d.get("iters", 0)),
            corr_mode=str(d.get("corr_mode", "")),
            device=str(d.get("device", "")),
            config=dict(d.get("config", {})),
            started=str(d.get("started", "")),
            seconds=float(d.get("seconds", 0.0)),
            notes=tuple(d.get("notes", ())),
        )


@dataclass(frozen=True)
class DisplacementSeries:
    """Lagrangian displacement of every frame relative to frame 0."""

    grid: Grid
    mask: np.ndarray          # (h, w) bool: frame-0 ROI on the grid
    u: np.ndarray             # (T, h, w) float32, +right
    v: np.ndarray             # (T, h, w) float32, +down
    done: np.ndarray          # (T,) bool: frame computed (frame 0 always True)
    frame_names: tuple[str, ...]
    schedule: FrameSchedule
    info: RunInfo
    residual: np.ndarray | None = None  # (T, h, w) float16/32: local RMS intensity residual (grey levels)

    def __post_init__(self) -> None:
        T = len(self.frame_names)
        if self.residual is not None and self.residual.shape != self.u.shape:
            raise ValueError(f"residual {self.residual.shape} does not match u {self.u.shape}")
        if self.u.shape != self.v.shape or self.u.shape[0] != T or self.u.shape[1:] != self.grid.shape:
            raise ValueError(f"u {self.u.shape} / v {self.v.shape} do not match {T} frames on grid {self.grid.shape}")
        if self.mask.shape != self.grid.shape:
            raise ValueError(f"mask {self.mask.shape} does not match grid {self.grid.shape}")

    @property
    def n_frames(self) -> int:
        return len(self.frame_names)

    @property
    def completed_frames(self) -> list[int]:
        return [int(k) for k in np.flatnonzero(self.done)]

    def field(self, frame: int) -> np.ndarray:
        """``(2, h, w)`` float32 of one frame."""
        return np.stack([self.u[frame], self.v[frame]])

    def magnitude(self, frame: int) -> np.ndarray:
        return np.hypot(self.u[frame], self.v[frame])

    def frame_mask(self, frame: int) -> np.ndarray:
        """ROI points with a finite value in ``frame``."""
        return self.mask & np.isfinite(self.u[frame]) & np.isfinite(self.v[frame])

    def coverage(self, frame: int) -> float:
        n = int(self.mask.sum())
        return float(self.frame_mask(frame).sum()) / n if n else 0.0


@dataclass(frozen=True)
class StrainSettings:
    """How a strain series was computed (recorded with the results)."""

    measure: str
    gauge_px: int
    weighting: str
    poly_order: int
    step: int
    gaussian_sigma: float | None
    smoothing_sigma: float
    connectivity_aware: bool
    edge_trim_alpha: float = 0.0          # 0: nothing trimmed (also results saved before the trim existed)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class StrainSeries:
    """Strain components per frame on a (possibly coarser) grid inside the displacement grid.

    ``components[name]`` is ``(T, hs, ws)`` float32; sample ``(i, j)`` sits at image pixel
    ``(grid.x0 + j * step, grid.y0 + i * step)``. Points of the edge-trimmed band (see
    :func:`raftcorr.post.strain.edge_trimmed`) are NaN in every component; ``trimmed`` marks them,
    ``(T, hs, ws)`` bool, or is None when nothing was trimmed.
    """

    grid: Grid
    step: int
    components: dict[str, np.ndarray]
    done: np.ndarray
    settings: StrainSettings
    trimmed: np.ndarray | None = None

    @property
    def names(self) -> list[str]:
        return list(self.components)

    def get(self, name: str, frame: int) -> np.ndarray:
        return self.components[name][frame]

    def trimmed_in(self, frame: int) -> np.ndarray | None:
        """Edge-trimmed points of one frame, or None when nothing is trimmed."""
        if self.trimmed is None or not self.done[frame]:
            return None
        return self.trimmed[frame]
