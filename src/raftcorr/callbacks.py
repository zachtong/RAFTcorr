"""Progress / stop / log callbacks shared by the GUI, the CLI and the Python API.

The library never prints. It reports through a :class:`Callbacks` object:

* ``progress(fraction, message)``: fraction in ``[0, 1]``;
* ``should_stop()``: polled between units of work. Returning True ends the work early, in one
  of two ways. :meth:`~raftcorr.engine.pipeline.Pipeline.run` stops tracking and returns the
  frames computed so far (``DisplacementSeries.done`` says which; ``info.notes`` records the
  stop); it does not raise. Everything else raises :class:`StopRequested`: a single pair
  (``Pipeline.solve_pair``), :func:`~raftcorr.post.strain.compute_strain`, the exporters,
  saving a project and downloading model weights;
* ``log(message, level)``: level is one of :data:`LEVELS`.

:func:`coerce` turns ``None``, a plain callable or a Callbacks into a Callbacks.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Protocol, TextIO, runtime_checkable

LEVELS: tuple[str, ...] = ("debug", "info", "warning", "error")


class StopRequested(RuntimeError):
    """Raised by :func:`check_stop` when ``should_stop()`` returns True (``Pipeline.run`` catches it)."""


@runtime_checkable
class Callbacks(Protocol):
    def progress(self, fraction: float, message: str = "") -> None: ...

    def should_stop(self) -> bool: ...

    def log(self, message: str, level: str = "info") -> None: ...


class NullCallbacks:
    """Silent callbacks; the default of the API."""

    def progress(self, fraction: float, message: str = "") -> None:
        return None

    def should_stop(self) -> bool:
        return False

    def log(self, message: str, level: str = "info") -> None:
        return None


class ConsoleCallbacks:
    """Writes progress and log lines to a stream (stdout by default), rate-limited."""

    def __init__(self, stream: TextIO | None = None, min_interval: float = 0.5, min_level: str = "info") -> None:
        if min_level not in LEVELS:
            raise ValueError(f"min_level must be one of {LEVELS}, got {min_level!r}")
        self.stream = stream or sys.stdout
        self.min_interval = min_interval
        self._min_rank = LEVELS.index(min_level)
        self._last = -1.0
        self._stop = False

    def progress(self, fraction: float, message: str = "") -> None:
        now = time.perf_counter()
        if fraction < 1.0 and self._last >= 0 and now - self._last < self.min_interval:
            return
        self._last = now
        pct = max(0.0, min(1.0, fraction)) * 100.0
        self.stream.write(f"[{pct:5.1f}%] {message}\n")
        self.stream.flush()

    def should_stop(self) -> bool:
        return self._stop

    def request_stop(self) -> None:
        self._stop = True

    def log(self, message: str, level: str = "info") -> None:
        if level not in LEVELS:
            level = "info"
        if LEVELS.index(level) < self._min_rank:
            return
        self.stream.write(f"[{level}] {message}\n")
        self.stream.flush()


class FunctionCallbacks:
    """Adapts plain functions to the Callbacks protocol."""

    def __init__(
        self,
        progress: Callable[[float, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        log: Callable[[str, str], None] | None = None,
    ) -> None:
        self._progress = progress
        self._should_stop = should_stop
        self._log = log

    def progress(self, fraction: float, message: str = "") -> None:
        if self._progress is not None:
            self._progress(fraction, message)

    def should_stop(self) -> bool:
        return bool(self._should_stop()) if self._should_stop is not None else False

    def log(self, message: str, level: str = "info") -> None:
        if self._log is not None:
            self._log(message, level)


class SubProgress:
    """Maps a sub-task's ``[0, 1]`` progress onto ``[lo, hi]`` of a parent."""

    def __init__(self, parent: Callbacks, lo: float, hi: float) -> None:
        self.parent, self.lo, self.hi = parent, lo, hi

    def progress(self, fraction: float, message: str = "") -> None:
        f = max(0.0, min(1.0, fraction))
        self.parent.progress(self.lo + (self.hi - self.lo) * f, message)

    def should_stop(self) -> bool:
        return self.parent.should_stop()

    def log(self, message: str, level: str = "info") -> None:
        self.parent.log(message, level)


def coerce(callbacks: object) -> Callbacks:
    """Return a Callbacks for ``None`` (silent), a callable (progress only) or a Callbacks."""
    if callbacks is None:
        return NullCallbacks()
    if isinstance(callbacks, Callbacks):
        return callbacks
    if callable(callbacks):
        return FunctionCallbacks(progress=callbacks)  # type: ignore[arg-type]
    raise TypeError(f"callbacks must be None, a callable or a Callbacks, got {type(callbacks).__name__}")


def check_stop(callbacks: Callbacks) -> None:
    """Raise :class:`StopRequested` if the caller asked to stop."""
    if callbacks.should_stop():
        raise StopRequested("stopped by the user")
