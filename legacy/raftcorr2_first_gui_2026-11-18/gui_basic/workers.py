"""Background execution: any ``fn(callbacks) -> result`` on a QThread.

Progress, log lines and the result travel back through Qt signals (queued connections, so the
slots run on the GUI thread). ``request_stop`` makes ``callbacks.should_stop()`` return True; the
library then stops at its next check point.
"""

from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal

from ..callbacks import StopRequested


class _Bridge:
    def __init__(self, worker: "TaskWorker") -> None:
        self._w = worker

    def progress(self, fraction: float, message: str = "") -> None:
        self._w.progress.emit(float(fraction), str(message))

    def should_stop(self) -> bool:
        return self._w.stop_requested

    def log(self, message: str, level: str = "info") -> None:
        self._w.log.emit(str(message), str(level))


class TaskWorker(QThread):
    progress = Signal(float, str)
    log = Signal(str, str)
    done = Signal(object)
    failed = Signal(str, str)   # short message, traceback
    stopped = Signal()

    def __init__(self, fn: Callable[[Any], Any], name: str = "task") -> None:
        super().__init__()
        self.fn = fn
        self.name = name
        self.stop_requested = False
        self.callbacks = _Bridge(self)

    def request_stop(self) -> None:
        self.stop_requested = True

    def run(self) -> None:  # QThread entry point
        try:
            result = self.fn(self.callbacks)
        except StopRequested:
            self.stopped.emit()
            return
        except Exception as exc:  # report, never crash the GUI
            tb = traceback.format_exc()
            self.failed.emit(_friendly(exc), tb)
            return
        self.done.emit(result)


def _friendly(exc: BaseException) -> str:
    name = type(exc).__name__
    text = str(exc).strip() or name
    if "out of memory" in text.lower():
        return ("The GPU ran out of memory. Close other GPU programs, choose a model with a larger s, "
                "or lower the memory fraction in the tiling settings.\n\n" + text)
    if isinstance(exc, (FileNotFoundError, ValueError, MemoryError, RuntimeError, KeyError)):
        return text
    return f"{name}: {text}"
