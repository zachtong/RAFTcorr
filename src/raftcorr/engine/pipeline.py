"""The sequence pipeline: images + ROI -> Lagrangian displacement of every frame.

For each pair ``(r, k)`` of the schedule:

1. the ROI in frame ``r``: the frame-0 ROI for ``r = 0``; the user's mask for frame ``r`` if one
   was given; otherwise the frame-0 ROI carried to frame ``r`` by ``U_r``;
2. the pair is solved over that ROI (one window or tiles, :mod:`raftcorr.engine.tiling`);
3. the step is composed into frame-0 coordinates, ``U_k(X) = U_r(X) + d_rk(X + U_r(X))``;
4. points outside the frame-0 ROI, and points whose deformed position falls outside a user mask
   given for frame ``k`` (cracks, holes), become NaN;
5. optional Gaussian smoothing (off by default: the paper reports raw displacements).

Stopping keeps every completed frame (``DisplacementSeries.done``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .._version import __version__
from ..callbacks import Callbacks, StopRequested, SubProgress, check_stop, coerce
from ..config import RunConfig
from ..results import DisplacementSeries, RunInfo
from ..roi.masks import bbox_of
from ..schedule import FrameSchedule
from .backend import Predictor, empty_cache
from .mapping import Grid, compose, forward_mask_check, inside_frame, intensity_residual, warp_mask
from .memory import Eta, device_budget_bytes, max_pixels_for_budget
from .tiling import TilePlan, blend_weights, plan_tiles

OOM_RETRIES = 3


def _is_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # torch without CUDA
        pass
    return isinstance(exc, (RuntimeError, MemoryError)) and "out of memory" in str(exc).lower()


class Pipeline:
    """Holds one loaded model; solves pairs and whole sequences."""

    def __init__(self, config: RunConfig | None = None, callbacks: Callbacks | None = None,
                 loaded: Any | None = None) -> None:
        self.config = config or RunConfig()
        self.callbacks = coerce(callbacks)
        self._loaded = loaded
        self._predictor: Predictor | None = None
        self.last_plan: TilePlan | None = None

    # -- model ------------------------------------------------------------------------------------
    @property
    def loaded(self) -> Any:
        if self._loaded is None:
            from ..models import load_model, resolve_weights

            path = resolve_weights(self.config.model, allow_download=True, callbacks=self.callbacks)
            self._loaded = load_model(path, self.config.device, self.config.corr_mode, self.callbacks)
        return self._loaded

    @property
    def predictor(self) -> Predictor:
        if self._predictor is None or self._predictor.loaded is not self.loaded:
            self._predictor = Predictor(self.loaded, iters=self.config.iters)
        elif self._predictor.iters != self.config.iters:
            self._predictor = Predictor(self.loaded, iters=self.config.iters)
        return self._predictor

    @property
    def margin_px(self) -> int:
        m = self.config.tiling.margin_px
        return int(self.loaded.lookup_radius_px if m is None else m)

    def max_tile_pixels(self) -> int:
        t = self.config.tiling
        if t.max_tile_pixels is not None:
            return int(t.max_tile_pixels)
        budget = device_budget_bytes(self.loaded.device, t.memory_fraction)
        return max_pixels_for_budget(budget, self.loaded.downsample_factor, self.loaded.corr_mode)

    def plan(self, image_shape: tuple[int, int], roi: np.ndarray, max_pixels: int | None = None) -> TilePlan:
        bbox = bbox_of(roi)
        if bbox is None:
            raise ValueError("the ROI is empty")
        return plan_tiles(image_shape, bbox, self.margin_px, max_pixels or self.max_tile_pixels(),
                          self.predictor.min_side, self.config.tiling.blend_px)

    # -- one pair ---------------------------------------------------------------------------------
    def solve_pair(self, reference: np.ndarray, deformed: np.ndarray, roi: np.ndarray,
                   callbacks: Callbacks | None = None) -> tuple[np.ndarray, int, int]:
        """Displacement ``reference -> deformed`` over the ROI's bounding box (plus a few px).

        Returns ``(field, x0, y0)``: ``field`` is ``(2, h, w)`` float32 placed at image pixel
        ``(x0, y0)``, in reference-frame coordinates, not masked by the ROI.
        """
        import torch

        cb = coerce(callbacks) if callbacks is not None else self.callbacks
        if reference.shape != deformed.shape or reference.shape != roi.shape:
            raise ValueError(f"sizes differ: reference {reference.shape}, deformed {deformed.shape}, ROI {roi.shape}")
        pred = self.predictor
        dev = pred.device
        ref_t = torch.from_numpy(np.array(reference, dtype=np.float32, copy=True)).to(dev)
        dfm_t = torch.from_numpy(np.array(deformed, dtype=np.float32, copy=True)).to(dev)
        max_px: int | None = None
        try:
            for attempt in range(OOM_RETRIES + 1):
                plan = self.plan(reference.shape, roi, max_px)
                if self.last_plan is None or plan.tiles != self.last_plan.tiles:
                    cb.log(f"tiling: {plan.describe()}", "info" if self.last_plan is None else "debug")
                self.last_plan = plan
                try:
                    out = self._run_plan(plan, ref_t, dfm_t, cb)
                    return out, plan.output.x0, plan.output.y0
                except StopRequested:
                    raise
                except Exception as exc:
                    if not _is_oom(exc) or attempt == OOM_RETRIES:
                        raise
                    empty_cache(dev)
                    max_px = int(plan.max_window_pixels * 0.55)
                    cb.log(f"out of device memory; retrying with windows of at most {max_px:,} px", "warning")
        finally:
            del ref_t, dfm_t
        raise MemoryError("out of device memory")  # unreachable

    def _run_plan(self, plan: TilePlan, ref_t, dfm_t, cb: Callbacks) -> np.ndarray:
        out = plan.output
        pred = self.predictor
        if plan.single:
            tile = plan.tiles[0]
            ys, xs = tile.window.slices
            flow = pred.predict_tensor(ref_t[ys, xs], dfm_t[ys, xs])
            cy, cx = out.y0 - tile.window.y0, out.x0 - tile.window.x0
            return flow[:, cy : cy + out.height, cx : cx + out.width].cpu().numpy()
        accum = np.zeros((2, out.height, out.width), dtype=np.float64)
        wsum = np.zeros((out.height, out.width), dtype=np.float64)
        n = len(plan.tiles)
        for i, tile in enumerate(plan.tiles):
            check_stop(cb)
            cb.progress(i / n, f"tile {i + 1}/{n}")
            ys, xs = tile.window.slices
            flow = pred.predict_tensor(ref_t[ys, xs], dfm_t[ys, xs])
            c = tile.core
            cy, cx = c.y0 - tile.window.y0, c.x0 - tile.window.x0
            core = flow[:, cy : cy + c.height, cx : cx + c.width].cpu().numpy()
            wts = blend_weights(tile, out, plan.blend_px)
            oy, ox = c.y0 - out.y0, c.x0 - out.x0
            accum[:, oy : oy + c.height, ox : ox + c.width] += core * wts[None]
            wsum[oy : oy + c.height, ox : ox + c.width] += wts
        cb.progress(1.0, f"tile {n}/{n}")
        with np.errstate(invalid="ignore", divide="ignore"):
            res = accum / wsum[None]
        res[:, wsum <= 0] = np.nan
        return res.astype(np.float32)

    # -- sequence ---------------------------------------------------------------------------------
    def run(
        self,
        images: Sequence[np.ndarray] | Any,
        roi: np.ndarray | None = None,
        frame_masks: dict[int, np.ndarray] | None = None,
        frame_names: Sequence[str] | None = None,
        frames: Sequence[int] | None = None,
        storage_dir: str | Path | None = None,
    ) -> DisplacementSeries:
        """Track every frame of ``images`` (an :class:`~raftcorr.io.ImageSequence` or a list of
        2-D arrays) relative to frame 0.

        ``roi``: frame-0 ROI (whole image when None). ``frame_masks``: optional masks in the
        coordinates of other frames (used as the ROI when that frame is a reference, and to drop
        points that move into excluded areas). ``frames``: subset of frames to compute (their
        references are added automatically). ``storage_dir``: write the result arrays as
        memory-mapped ``.npy`` files there instead of keeping them in RAM.
        """
        cb = self.callbacks
        t_start = time.perf_counter()
        n = len(images)
        cfg = self.config
        sched = FrameSchedule.from_mode(cfg.tracking.mode, n, cfg.tracking.every_n, cfg.tracking.key_frames)
        first = np.asarray(images[0])
        H, W = first.shape
        roi0 = np.ones((H, W), dtype=bool) if roi is None else np.asarray(roi, dtype=bool)
        if roi0.shape != (H, W):
            raise ValueError(f"ROI {roi0.shape} does not match the images {(H, W)}")
        bbox = bbox_of(roi0)
        if bbox is None:
            raise ValueError("the ROI is empty")
        grid = Grid(bbox[0], bbox[1], bbox[2], bbox[3], H, W)
        mask0 = roi0[grid.slices].copy()
        masks = {int(k): np.asarray(m, dtype=bool) for k, m in (frame_masks or {}).items()}
        for k, m in masks.items():
            if m.shape != (H, W):
                raise ValueError(f"mask of frame {k} has shape {m.shape}, expected {(H, W)}")
        names = tuple(frame_names) if frame_names is not None else tuple(getattr(images, "names", [f"frame_{i:04d}" for i in range(n)]))
        wanted = _with_references(sched, frames, n)

        u, v, res = _allocate((n,) + grid.shape, storage_dir)
        done = np.zeros(n, dtype=bool)
        u[0] = np.where(mask0, 0.0, np.nan)
        v[0] = np.where(mask0, 0.0, np.nan)
        res[0] = np.where(mask0, 0.0, np.nan)
        done[0] = True

        loaded = self.loaded
        cb.log(
            f"{loaded.name}: {n} frames, {cfg.tracking.mode} tracking, {cfg.iters} iterations, "
            f"ROI {grid.width} x {grid.height} px at ({grid.x0}, {grid.y0})",
            "info",
        )
        pairs = [(r, k) for r, k in sched.pairs() if k in wanted]
        eta = Eta(len(pairs))
        notes: list[str] = []
        try:
            for i, (r, k) in enumerate(pairs):
                check_stop(cb)
                sub = SubProgress(cb, i / max(1, len(pairs)), (i + 1) / max(1, len(pairs)))
                if not done[r]:
                    notes.append(f"frame {k}: reference frame {r} has no result; skipped")
                    cb.log(notes[-1], "warning")
                    continue
                u_r = np.stack([u[r], v[r]])
                roi_r = roi0 if r == 0 else masks.get(r)
                if roi_r is None:
                    roi_r = warp_mask(mask0, u_r, grid)
                if not roi_r.any():
                    notes.append(f"frame {k}: the ROI carried to reference frame {r} is empty; skipped")
                    cb.log(notes[-1], "warning")
                    continue
                step, sx0, sy0 = self.solve_pair(np.asarray(images[r]), np.asarray(images[k]), roi_r, sub)
                if r == 0:
                    oy, ox = grid.y0 - sy0, grid.x0 - sx0
                    u_k = step[:, oy : oy + grid.height, ox : ox + grid.width].astype(np.float32)
                else:
                    u_k = compose(u_r, grid, step, sx0, sy0)
                u_k[:, ~mask0] = np.nan
                u_k[:, ~inside_frame(u_k, grid)] = np.nan  # points that left the field of view
                if k in masks:
                    keep = forward_mask_check(u_k, grid, masks[k])
                    u_k[:, ~keep] = np.nan
                if cfg.smoothing_sigma > 0:
                    from ..post.smoothing import nan_gaussian

                    u_k = nan_gaussian(u_k, cfg.smoothing_sigma, mask0)
                u[k], v[k] = u_k[0], u_k[1]
                res[k] = intensity_residual(np.asarray(images[0]), np.asarray(images[k]), u_k, grid)
                done[k] = True
                eta.update()
                cov = float(np.isfinite(u_k[0][mask0]).mean()) if mask0.any() else 0.0
                cb.progress((i + 1) / max(1, len(pairs)), f"frame {k} (reference {r}): {eta.message('pairs')}")
                if cov < 0.9:
                    cb.log(f"frame {k}: only {cov:.0%} of the ROI could be tracked", "warning")
        except StopRequested:
            notes.append(f"stopped by the user after {int(done.sum()) - 1} of {len(pairs)} pairs")
            cb.log(notes[-1], "warning")
        finally:
            empty_cache(loaded.device)

        from ..post.quality import describe, frame_residuals, suspect_frames

        medians = frame_residuals(res, done)
        hint = describe(suspect_frames(medians), medians, sched.is_accumulative)
        if hint:
            notes.append(hint)
            cb.log(hint, "warning")

        info = RunInfo(
            software_version=__version__,
            model_name=loaded.name,
            model_sha256=loaded.sha256,
            downsample_factor=loaded.downsample_factor,
            iters=cfg.iters,
            corr_mode=loaded.corr_mode,
            device=loaded.device,
            config=cfg.to_dict(),
            seconds=time.perf_counter() - t_start,
            notes=tuple(notes),
        )
        cb.log(f"done: {int(done.sum()) - 1} frames in {info.seconds:.1f} s", "info")
        return DisplacementSeries(grid, mask0, u, v, done, names, sched, info, residual=res)


def _with_references(sched: FrameSchedule, frames: Sequence[int] | None, n: int) -> set[int]:
    if frames is None:
        return set(range(1, n))
    out: set[int] = set()
    for k in frames:
        k = int(k)
        while k > 0 and k not in out:
            out.add(k)
            k = sched.reference_of(k)
    return out


def _allocate(shape: tuple[int, ...], storage_dir: str | Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """u, v (float32) and the residual (float16), NaN-filled, in RAM or as memmaps in ``storage_dir``."""
    specs = (("u", np.float32), ("v", np.float32), ("residual", np.float16))  # the residual is a diagnostic
    if storage_dir is None:
        return tuple(np.full(shape, np.nan, dtype=dt) for _, dt in specs)  # type: ignore[return-value]
    d = Path(storage_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for name, dt in specs:
        mm = np.lib.format.open_memmap(d / f"{name}.npy", mode="w+", dtype=dt, shape=shape)
        mm[:] = np.nan
        out.append(mm)
    return out[0], out[1], out[2]
