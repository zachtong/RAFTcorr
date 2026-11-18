"""Post-run strain computation controller (pyALDIC's design, RAFTcorr's strain).

The correlation run produces displacement only. This controller offers the
user-driven, decoupled path of pyALDIC's strain window: read the existing
displacement series (``state.results``, a
:class:`raftcorr.results.DisplacementSeries`), run
:func:`raftcorr.post.strain.compute_strain` with a
:class:`raftcorr.config.StrainConfig`, and publish the
:class:`raftcorr.results.StrainSeries` through
:meth:`AppState.set_strain_results`.

Design principles
-----------------
* The displacement series is never modified: strain is a separate result that
  the :class:`~raftcorr.viz.fields.FieldProvider` combines with it.
* :meth:`StrainController.compute_all_frames` is pure (no state writes), so it
  runs in a :class:`StrainWorker` thread; publishing
  (:meth:`StrainController.store`) happens on the GUI thread and refuses a
  result whose displacement series has been replaced meanwhile.
* Cancellation is checked between frames and surfaces as
  :class:`StrainComputationCancelled`; the previous strain result is kept.
"""

from __future__ import annotations

import logging
from typing import Callable

from PySide6.QtCore import QCoreApplication, QThread, Signal

from raftcorr.callbacks import FunctionCallbacks, StopRequested
from raftcorr.config import StrainConfig
from raftcorr.gui.app_state import AppState
from raftcorr.post.strain import compute_strain
from raftcorr.results import DisplacementSeries, StrainSeries, StrainSettings

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


class StrainComputationCancelled(Exception):
    """Raised by :meth:`StrainController.compute_all_frames` when the caller's
    ``should_stop`` predicate returns True, so a long multi-frame strain run can
    be interrupted at a frame boundary. The GUI catches it and leaves the
    previous strain result untouched."""


class StrainController:
    """Drive ``compute_strain`` over the displacement series of the state."""

    def __init__(self, state: AppState) -> None:
        self._state = state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all_frames(
        self,
        config: StrainConfig,
        progress_cb: ProgressCallback | None = None,
        should_stop: Callable[[], bool] | None = None,
        series: DisplacementSeries | None = None,
    ) -> StrainSeries:
        """Strain of every computed frame of the displacement series.

        Args:
            config: Strain settings (measure, gauge, weighting, fit order ...).
            progress_cb: Optional ``(fraction, message)`` callback invoked once
                per processed frame; the message names the frame 1-based, as
                the interface does.
            should_stop: Optional predicate checked at each frame boundary;
                when it returns True the computation is aborted by raising
                :class:`StrainComputationCancelled`.
            series: The displacement series to differentiate; defaults to
                ``state.results`` (pass it explicitly from a worker thread so
                the series cannot change under the computation).

        Raises:
            RuntimeError: If there is no displacement series.
            TypeError: If *config* is not a :class:`StrainConfig`.
            StrainComputationCancelled: If ``should_stop`` returns True.
        """
        if not isinstance(config, StrainConfig):
            raise TypeError(f"config must be a StrainConfig, got {type(config).__name__}")
        series = series if series is not None else self._require_results()
        n_todo = max(1, len(series.completed_frames))

        def on_progress(fraction: float, _message: str) -> None:
            # compute_strain reports once per frame: "i/n" counts the frames
            # processed, as pyALDIC's message did.
            if progress_cb is None:
                return
            done = max(1, min(n_todo, round(fraction * n_todo)))
            label = QCoreApplication.translate("StrainController", "Strain frame %1/%2").replace("%1", str(done)).replace("%2", str(n_todo))
            progress_cb(float(fraction), label)

        callbacks = FunctionCallbacks(
            progress=on_progress,
            should_stop=should_stop if should_stop is not None else (lambda: False),
        )
        try:
            return compute_strain(series, config, callbacks=callbacks)
        except StopRequested as exc:
            raise StrainComputationCancelled() from exc

    def compute_and_store(
        self,
        config: StrainConfig,
        progress_cb: ProgressCallback | None = None,
    ) -> StrainSeries:
        """Compute strain for all frames and publish it in the state.

        ``state.strain_config`` records the settings; listeners receive one
        ``strain_changed`` notification at the end.
        """
        series = self._require_results()
        strain = self.compute_all_frames(config, progress_cb=progress_cb, series=series)
        self.store(series, config, strain)
        return strain

    def store(self, series: DisplacementSeries, config: StrainConfig, strain: StrainSeries) -> bool:
        """Publish *strain* (computed from *series* with *config*) in the state.

        Returns False, publishing nothing, when the state's displacement series
        is no longer *series* (a new run finished meanwhile): strain of other
        displacements must never be attached to it.
        """
        if self._state.results is not series:
            return False
        self._state.strain_config = config
        self._state.set_strain_results(strain)
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_results(self) -> DisplacementSeries:
        result = self._state.results
        if result is None:
            raise RuntimeError(
                "StrainController: no displacement results available. "
                "Run the correlation first before computing strain."
            )
        return result


class StrainWorker(QThread):
    """Background thread for strain computation.

    Emits ``progress(fraction, message)`` once per frame,
    ``completed(strain_series)`` on success, ``cancelled()`` when the user
    aborts, and ``error(message)`` on failure. Computation happens in the
    thread; state updates happen in the caller's slot (main thread) to avoid
    cross-thread Qt issues. (``completed``, not pyALDIC's ``finished``: that
    name would hide QThread's own signal.) ``config`` and ``series`` say what
    was computed, so the caller can publish it with :meth:`StrainController.store`.
    """

    progress = Signal(float, str)
    completed = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(self, strain_ctrl: StrainController, config: StrainConfig,
                 series: DisplacementSeries) -> None:
        super().__init__()
        self._ctrl = strain_ctrl
        self.config = config
        self.series = series
        self._cancel = False

    def cancel(self) -> None:
        """Request cancellation; takes effect at the next frame boundary."""
        self._cancel = True

    def run(self) -> None:
        try:
            strain = self._ctrl.compute_all_frames(
                self.config,
                progress_cb=lambda f, m: self.progress.emit(f, m),
                should_stop=lambda: self._cancel,
                series=self.series,
            )
            self.completed.emit(strain)
        except StrainComputationCancelled:
            self.cancelled.emit()
        except Exception as exc:  # reported to the GUI thread
            logger.exception("strain computation failed")
            self.error.emit(f"{type(exc).__name__}: {exc}")


_MEASURE_NAMES: dict[str, str] = {
    "green_lagrange": "Green-Lagrangian",
    "infinitesimal": "infinitesimal",
    "euler_almansi": "Eulerian-Almansi",
}


def settings_summary(s: StrainSettings) -> dict[str, str]:
    """How a strain series was computed, as ``{name: value}`` text (export headers)."""
    if s.weighting != "gaussian":
        weighting = s.weighting
    elif s.gaussian_sigma:
        weighting = f"gaussian (sigma {s.gaussian_sigma:g} px)"
    else:
        weighting = "gaussian (sigma = window / 4)"
    return {
        "strain measure": _MEASURE_NAMES.get(s.measure, s.measure),
        "strain window": f"{s.gauge_px} px (virtual strain gauge, square)",
        "strain fit": {1: "plane", 2: "quadratic"}.get(s.poly_order, f"order {s.poly_order}"),
        "strain weighting": weighting,
        "strain spacing": f"{s.step} px",
        "strain displacement smoothing": f"{s.smoothing_sigma:g} px" if s.smoothing_sigma > 0 else "off",
        "strain across cracks": "no" if s.connectivity_aware else "yes",
    }


__all__ = ["StrainComputationCancelled", "StrainController", "StrainWorker", "settings_summary"]
