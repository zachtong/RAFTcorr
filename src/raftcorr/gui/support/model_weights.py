"""Model weights in the GUI's background threads: a first-use download reported as a status line.

:func:`ensure_weights` is :func:`raftcorr.models.zoo.resolve_weights` with the download allowed,
plus a translated status ("Downloading model RAFTcorr-s4: 5.2 of 20.7 MB") for the progress
area, cleared when the download ends. A model that is still unavailable raises
:class:`~raftcorr.models.zoo.ModelUnavailableError`, which the windows answer by opening the
Models window rather than an error box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QCoreApplication

from raftcorr.i18n import tr_args
from raftcorr.models import zoo

#: ``status(fraction, text)``; an empty text ends the download status.
StatusFn = Callable[[float, str], None]


def download_status_text(name: str, done_bytes: float, total_bytes: float | None) -> str:
    """The status line of a model download (the total may be unknown)."""
    if total_bytes:
        return tr_args(QCoreApplication.translate("ModelWeights", "Downloading model %1: %2 of %3 MB"),
                       name, f"{done_bytes / 1e6:.1f}", f"{total_bytes / 1e6:.1f}")
    return tr_args(QCoreApplication.translate("ModelWeights", "Downloading model %1…"), name)


class _StatusCallbacks:
    """Turns the download's progress into status lines; stop and log go to the caller's callbacks."""

    def __init__(self, inner, spec: zoo.ModelSpec, status: StatusFn) -> None:
        self._inner, self._spec, self._status = inner, spec, status

    def progress(self, fraction: float, message: str = "") -> None:
        total = self._spec.size_bytes
        self._status(fraction, download_status_text(self._spec.name, fraction * (total or 0), total))

    def should_stop(self) -> bool:
        return self._inner.should_stop()

    def log(self, message: str, level: str = "info") -> None:
        self._inner.log(message, level)


def ensure_weights(key_or_path: str | Path, callbacks, status: StatusFn) -> Path:
    """Path of a model's weights, downloading a released model on first use.

    *status* hears about a download only: ``(0, text)`` before it, ``(fraction, text)`` while it
    runs and ``(0, "")`` when it ends, however it ends.
    """
    key = str(key_or_path).removeprefix("RAFTcorr-")
    if key not in zoo.MODELS:                         # an explicit checkpoint file
        return zoo.resolve_weights(key_or_path, callbacks=callbacks)
    spec = zoo.get_spec(key)
    found = zoo.find_weights(spec, callbacks=callbacks)
    if found is not None:
        return found
    status(0.0, download_status_text(spec.name, 0, spec.size_bytes))
    try:
        return zoo.resolve_weights(key, allow_download=True, callbacks=_StatusCallbacks(callbacks, spec, status))
    finally:
        status(0.0, "")
