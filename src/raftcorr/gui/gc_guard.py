"""Python's cyclic garbage collector, run in the GUI thread only.

Qt objects must be destroyed in the thread that owns them. Python runs its cyclic collector in
whichever thread allocates when a threshold is crossed, so a worker (tracking, strain, export)
can collect a reference cycle that holds a widget and destroy it off the GUI thread: an access
violation (seen on CI in an export worker while matplotlib allocated). The usual remedy, as in
pyqtgraph: switch automatic collection off and collect from a timer in the GUI thread, with the
interpreter's own thresholds.
"""

from __future__ import annotations

import gc

from PySide6.QtCore import QObject, QTimer


class MainThreadGarbageCollector(QObject):
    """Disable automatic collection; collect from a GUI-thread timer when the counts call for it."""

    def __init__(self, interval_ms: int = 1000, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._threshold = gc.get_threshold()
        gc.disable()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.check)
        self._timer.start(interval_ms)

    def check(self) -> None:
        """Collect the generations whose counts passed their thresholds (GUI thread)."""
        c0, c1, c2 = gc.get_count()
        t0, t1, t2 = self._threshold
        if c0 > t0:
            gc.collect(0)
            if c1 > t1:
                gc.collect(1)
                if c2 > t2:
                    gc.collect(2)

    def stop(self) -> None:
        """Back to automatic collection."""
        self._timer.stop()
        gc.enable()


__all__ = ["MainThreadGarbageCollector"]
