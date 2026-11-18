"""Peak-memory model of one forward pass, the memory budget, and a running ETA.

Peak memory of a forward pass on ``N`` pixels with downsample factor ``s``:

* every mode: a part linear in ``N`` (encoder activations at full and reduced resolution, GRU
  state, convex-upsampling masks), ``N * (A / s^2 + B)`` bytes;
* stored volume (``"volume"``) adds the all-pairs correlation, quadratic in the feature-grid size
  ``n = N / s^2``: the fp32 product and its scaled copy coexist briefly, ``2 * 4 * n^2`` bytes.

``A`` and ``B`` were measured on this implementation (``scripts/calibrate_memory.py``, CUDA,
mixed precision, 20 iterations). The estimate is for go / no-go tiling decisions; the pipeline
still recovers from an out-of-memory error by shrinking the tiles.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

#: Linear coefficients (bytes per pixel = A / s^2 + B). Measured by scripts/calibrate_memory.py
#: on an RTX 4070 Ti (fit 14445 / s^2 + 72), rounded up slightly.
LINEAR_A = 14_600.0
LINEAR_B = 100.0
#: Fixed overhead per forward pass (cuDNN workspace, allocator slack); the weights are resident.
OVERHEAD_BYTES = 128 * 1024**2
#: Fallback budget when the device cannot report its free memory (CPU).
CPU_BUDGET_BYTES = 6 * 1024**3


def linear_bytes_per_pixel(downsample_factor: int) -> float:
    return LINEAR_A / downsample_factor**2 + LINEAR_B


def estimate_peak_bytes(n_pixels: int, downsample_factor: int, corr_mode: str) -> int:
    """Peak memory (bytes) of one forward pass on ``n_pixels`` pixels."""
    s = downsample_factor
    mem = n_pixels * linear_bytes_per_pixel(s) + OVERHEAD_BYTES
    if corr_mode == "volume":
        n = n_pixels / s**2
        mem += 8.0 * n * n
    return int(mem)


def max_pixels_for_budget(budget_bytes: float, downsample_factor: int, corr_mode: str) -> int:
    """Largest pixel count whose estimated peak fits ``budget_bytes``."""
    s = downsample_factor
    avail = budget_bytes - OVERHEAD_BYTES
    if avail <= 0:
        return 0
    a = linear_bytes_per_pixel(s)
    if corr_mode != "volume":
        return int(avail / a)
    q = 8.0 / s**4  # bytes per pixel^2
    return int((-a + math.sqrt(a * a + 4 * q * avail)) / (2 * q))


def device_budget_bytes(device: str, fraction: float = 0.85, reserve_bytes: int = 384 * 1024**2) -> int:
    """Memory one forward pass may use on ``device``: a fraction of the free memory now."""
    if device.startswith("cuda"):
        import torch

        idx = torch.device(device).index
        free, _total = torch.cuda.mem_get_info(idx if idx is not None else torch.cuda.current_device())
        return max(0, int(free * fraction) - reserve_bytes)
    return CPU_BUDGET_BYTES


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    name: str
    total_bytes: int | None
    free_bytes: int | None

    def describe(self) -> str:
        if self.total_bytes is None:
            return f"{self.name}"
        return f"{self.name}: {self.free_bytes / 1024**3:.1f} of {self.total_bytes / 1024**3:.1f} GB free"


def device_info(device: str) -> DeviceInfo:
    if device.startswith("cuda"):
        import torch

        idx = torch.device(device).index
        idx = idx if idx is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(idx)
        return DeviceInfo(device, torch.cuda.get_device_name(idx), int(total), int(free))
    return DeviceInfo(device, "CPU", None, None)


class Eta:
    """Exponential-moving-average throughput for a loop of ``total`` units."""

    def __init__(self, total: int, alpha: float = 0.3) -> None:
        self.total = max(1, int(total))
        self.alpha = alpha
        self.done = 0
        self._rate: float | None = None
        self._t_last = time.perf_counter()
        self.t_start = self._t_last

    def update(self, n: int = 1) -> None:
        now = time.perf_counter()
        rate = n / max(now - self._t_last, 1e-9)
        self._rate = rate if self._rate is None else (1 - self.alpha) * self._rate + self.alpha * rate
        self._t_last = now
        self.done += n

    @property
    def fraction(self) -> float:
        return min(1.0, self.done / self.total)

    @property
    def seconds_left(self) -> float | None:
        if self._rate is None or self._rate <= 0:
            return None
        return (self.total - self.done) / self._rate

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t_start

    def message(self, unit: str = "pairs") -> str:
        left = self.seconds_left
        eta = f", about {_fmt_seconds(left)} left" if left is not None else ""
        return f"{self.done}/{self.total} {unit}{eta}"


def _fmt_seconds(s: float) -> str:
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"
