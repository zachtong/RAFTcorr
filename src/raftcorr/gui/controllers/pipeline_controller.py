"""Pipeline execution controller -- QThread worker with pause / stop (RAFTcorr engine).

The interface is pyALDIC's (``start``, ``stop``, ``pause``, ``resume``, ``_worker``,
``_start_time``); the work is :class:`raftcorr.engine.pipeline.Pipeline`. Pausing blocks the
worker at the pipeline's next check point (between tiles and frames).
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QCoreApplication, QThread, Signal

from raftcorr.callbacks import StopRequested
from raftcorr.i18n import tr_args
from raftcorr.models.zoo import ModelUnavailableError
from raftcorr.gui.app_state import AppState, RunState
from raftcorr.gui.controllers.pipeline_messages import PipelineMessages, memory_advice

logger = logging.getLogger(__name__)

#: Results larger than this are kept in memory-mapped files in a temporary folder.
RAM_RESULTS_LIMIT = 2 * 1024**3


class _Callbacks:
    """raftcorr callbacks forwarding to the worker's signals; ``should_stop`` blocks while paused."""

    def __init__(self, worker: "PipelineWorker") -> None:
        self._w = worker

    def progress(self, fraction: float, message: str = "") -> None:
        self._w.progress.emit(float(fraction), str(message))

    def should_stop(self) -> bool:
        return self._w._should_stop()

    def log(self, message: str, level: str = "info") -> None:
        self._w.log.emit(str(message), "info" if level == "debug" else str(level))


class PipelineWorker(QThread):
    """Runs the RAFTcorr pipeline in a background thread."""

    progress = Signal(float, str)       # (fraction, message)
    log = Signal(str, str)              # (message, level)
    finished_result = Signal(object)    # DisplacementSeries or None on error
    fatal_error = Signal(str, str)      # (title, message) -- modal-worthy
    download_status = Signal(float, str)    # (fraction, text) of a model download; "" ends it
    model_unavailable = Signal(str, str)    # (model key, reason): not installed, not downloaded

    def __init__(self, run_config, sequence, roi, frame_masks, loaded_cache: dict,
                 storage_dir: str | None = None) -> None:
        super().__init__()
        self._cfg = run_config
        self._seq = sequence
        self.sequence = sequence  # read by the controller: results belong to these images
        self._roi = roi
        self._masks = dict(frame_masks)
        self._cache = loaded_cache
        self._storage = storage_dir
        self._stop_requested = False
        self._pause_event = threading.Event()
        self._pause_event.set()  # not paused initially

    def run(self) -> None:
        cb = _Callbacks(self)
        try:
            from raftcorr.engine.pipeline import Pipeline

            loaded = self._load_model(cb)
            series = Pipeline(self._cfg, cb, loaded).run(self._seq, self._roi, frame_masks=self._masks,
                                                         storage_dir=self._storage)
        except StopRequested:
            self.log.emit(QCoreApplication.translate("PipelineController", "Stopped."), "warn")
            self.finished_result.emit(None)
            return
        except ModelUnavailableError as exc:      # the Models window offers a download or an import
            self.log.emit(str(exc), "error")
            self.model_unavailable.emit(exc.spec.key, exc.short)
            self.finished_result.emit(None)
            return
        except Exception as exc:  # report, never crash the GUI thread
            self.log.emit(traceback.format_exc(), "error")
            text = str(exc).strip() or type(exc).__name__
            advice = memory_advice(exc)
            if advice is not None:
                text = advice + "\n\n" + text
            self.fatal_error.emit(QCoreApplication.translate("PipelineController", "Analysis failed"), text)
            self.finished_result.emit(None)
            return
        self.finished_result.emit(series)

    def _load_model(self, cb):
        from raftcorr.gui.support.model_weights import ensure_weights
        from raftcorr.models import load_model

        cfg = self._cfg
        path = ensure_weights(cfg.model, cb, self.download_status.emit)    # first use downloads it
        key = (str(path), cfg.device, cfg.corr_mode)
        hit = self._cache.get(key)
        if hit is None:
            self._cache.clear()  # one model at a time on the GPU
            hit = load_model(path, cfg.device, cfg.corr_mode, cb)
            self._cache[key] = hit
        return hit

    def _should_stop(self) -> bool:
        self._pause_event.wait()
        return self._stop_requested

    def request_stop(self) -> None:
        self._stop_requested = True
        self._pause_event.set()

    def request_pause(self) -> None:
        self._pause_event.clear()

    def request_resume(self) -> None:
        self._pause_event.set()


class PipelineController:
    """Coordinates pipeline execution between UI and worker thread."""

    def __init__(self, state: AppState, image_ctrl) -> None:
        self._state = state
        self._image_ctrl = image_ctrl
        self._worker: PipelineWorker | None = None
        self._start_time: float = 0.0
        self._loaded_cache: dict = {}
        self._temp_dirs: list[str] = []
        # Other images while a run is going: cancel it (its results would not fit them).
        state.images_changed.connect(self._on_images_changed)

    def _fatal(self, title: str, message: str, detail: str = "") -> None:
        self._state.log_message.emit(message, "error")
        if detail:
            self._state.log_message.emit(detail, "error")
        self._state.fatal_error.emit(title, message)

    def start(self) -> None:
        """Build the RAFTcorr configuration from AppState and launch the worker."""
        state = self._state
        if state.run_state in (RunState.RUNNING, RunState.PAUSED):
            return
        if self._worker is not None:
            if self._worker.isRunning():
                self._worker.request_stop()
                self._worker.wait(5000)
            self._worker = None

        m = PipelineMessages()
        if len(state.image_files) < 2 or state.sequence is None:
            self._fatal(m.cannot_start_title(), m.too_few_images())
            return
        if state.roi_mask is None:
            self._fatal(m.cannot_start_title(), m.no_roi())
            return
        if not state.roi_mask.any():   # Batch Import can store an all-black mask
            self._fatal(m.cannot_start_title(), m.empty_roi())
            return
        if state.tracking_mode == "incremental" and state.inc_ref_mode == "custom" and state.inc_custom_refs_error:
            self._fatal(m.cannot_start_title(), state.inc_custom_refs_error)   # never the older frames
            return
        try:
            cfg = state.run_config()
        except ValueError as exc:
            self._fatal(m.cannot_start_title(), str(exc))
            return
        missing = sorted(f for f in state.reference_frames() if f != 0 and f not in state.per_frame_rois)
        if missing:
            state.log_message.emit(
                tr_args(QCoreApplication.translate(
                    "PipelineController",
                    "Reference frames without a Region of Interest of their own carry the one of frame 1 "
                    "forward: %1"),
                    ", ".join(str(f + 1) for f in missing[:20])), "info")   # 1-based, as on screen

        state.roi_editing = False
        state.display_changed.emit()
        state.log_message.emit(
            f"RAFTcorr-{cfg.model}: {len(state.image_files)} frames, {cfg.tracking.mode}, "
            f"{cfg.iters} iterations, device={cfg.device}", "info")
        masks = {k: v for k, v in state.per_frame_rois.items() if k != 0}
        self._worker = PipelineWorker(cfg, state.sequence, state.roi_mask, masks, self._loaded_cache,
                                      self._storage_for(state))
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(lambda msg, lvl: state.log_message.emit(msg, lvl))
        self._worker.fatal_error.connect(lambda t, msg: state.fatal_error.emit(t, msg))
        self._worker.download_status.connect(self._on_download_status)
        self._worker.model_unavailable.connect(lambda key, why: state.model_unavailable.emit(key, why))
        worker = self._worker
        # Bound to this worker: a cancelled worker's finish signal can arrive after a new one started.
        worker.finished_result.connect(lambda result, w=worker: self._on_finished(w, result))
        self._start_time = time.perf_counter()
        state.set_progress(0.0, QCoreApplication.translate("PipelineController", "Starting..."))
        state.progress = 0.0
        state.set_run_state(RunState.RUNNING)
        self._worker.start()

    def _storage_for(self, state: AppState) -> str | None:
        from raftcorr.roi.masks import bbox_of

        b = bbox_of(state.roi_mask)
        nbytes = len(state.image_files) * b[2] * b[3] * (4 + 4 + 2)
        if nbytes <= RAM_RESULTS_LIMIT:
            return None
        d = tempfile.mkdtemp(prefix="raftcorr_results_")
        self._temp_dirs.append(d)
        state.log_message.emit(f"results of {nbytes / 1024**3:.1f} GB are kept on disk in {d}", "info")
        return d

    def stop(self) -> None:
        if self._worker:
            self._worker.request_stop()

    def _on_images_changed(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning() and worker.sequence is not self._state.sequence:
            worker.request_stop()
            self._state.log_message.emit(
                QCoreApplication.translate("PipelineController", "The images changed: the running analysis is cancelled."), "warn")

    def pause(self) -> None:
        if self._worker and self._state.run_state == RunState.RUNNING:
            self._worker.request_pause()
            self._state.set_run_state(RunState.PAUSED)

    def resume(self) -> None:
        if self._worker and self._state.run_state == RunState.PAUSED:
            self._worker.request_resume()
            self._state.set_run_state(RunState.RUNNING)

    def _on_download_status(self, fraction: float, text: str) -> None:
        """The bar follows a model download, then starts again from 0 for the tracking (whose
        progress never goes back, see _on_progress)."""
        self._state.set_progress_status(text)
        self._state.set_progress(fraction if text else 0.0, "")

    def _on_progress(self, fraction: float, message: str) -> None:
        fraction = max(fraction, self._state.progress)
        self._state.set_progress(fraction, message)
        self._state.elapsed_seconds = time.perf_counter() - self._start_time

    def _on_finished(self, worker, result) -> None:
        worker.wait(2000)
        if worker is not self._worker:
            return  # a cancelled run's late signal; the current run (if any) is not touched
        self._worker = None
        state = self._state
        if result is not None and getattr(worker, "sequence", state.sequence) is not state.sequence:
            state.log_message.emit(QCoreApplication.translate("PipelineController", "The images changed during the run: its results are discarded."), "warn")
            state.set_run_state(RunState.DONE if state.results is not None else RunState.IDLE)
            return
        try:
            if result is not None and int(np.sum(result.done)) > 1:
                n = int(np.sum(result.done)) - 1
                state.log_message.emit(PipelineMessages().results_received(n), "success")
                state.set_results(result)
                state.set_run_state(RunState.DONE)
                state.set_current_frame(result.completed_frames[-1])
                self._report_tracking_quality(result)
            else:
                if result is not None:
                    state.log_message.emit(QCoreApplication.translate("PipelineController", "No frame was computed; the previous results are kept."), "warn")
                state.set_run_state(RunState.DONE if state.results is not None else RunState.IDLE)
        except Exception as e:
            tb = traceback.format_exc()
            m = PipelineMessages()
            self._fatal(m.failed_to_process_title(), m.see_traceback(f"{type(e).__name__}: {e}"), detail=tb)
            state.set_run_state(RunState.IDLE)

    def _report_tracking_quality(self, result) -> None:
        """The residual-jump warning (translated; the engine's English note goes to the log)."""
        from raftcorr.gui.controllers.pipeline_messages import tracking_quality_warning

        warning = tracking_quality_warning(result.residual, result.done, result.schedule.is_accumulative)
        if warning is not None:
            self._state.run_warning.emit(*warning)

    def cleanup(self) -> None:
        """Release memory maps and remove temporary result folders (on exit)."""
        import gc
        import shutil

        gc.collect()
        for d in self._temp_dirs:
            shutil.rmtree(d, ignore_errors=True)
