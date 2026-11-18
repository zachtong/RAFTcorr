"""Shared helpers and fixtures for the offscreen GUI tests (``tests/test_gui_*.py``).

* synthetic image folders (speckle rendered by :mod:`raftcorr.synthetic`, saved as 8-bit PNG);
* synthetic displacement series with a known field, so that display, probes, exports and project
  files can be tested without a GPU;
* stubs for every modal dialog (message boxes, file dialogs) that record their calls;
* a window factory that keeps ``QSettings`` out of the user's registry and never leaves a
  background task running after a test.

Import the fixtures into a test module with ``from .gui_helpers import ...``.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QPointF, QSettings  # noqa: E402
from PySide6.QtWidgets import QFileDialog, QMessageBox  # noqa: E402

H, W = 256, 320
#: Imposed translation (u, v) in px of frames 1..3 relative to frame 0.
SHIFTS: tuple[tuple[float, float], ...] = ((2.0, 1.0), (4.0, 2.0), (6.0, 3.0))
#: Default frame-0 ROI of the synthetic tests: x in [40, 280), y in [40, 216).
ROI_BOX = (40, 40, 280, 216)
#: Slope of the synthetic linear field: u_k = k * (SLOPE * (X - X_C)) + shift_u.
SLOPE = 0.01
X_C = 160.0

WORKER_TIMEOUT_MS = 60_000


# =================================================================================== image folders
def write_sequence(folder: Path, shifts=SHIFTS, h: int = H, w: int = W, seed: int = 0,
                   pattern: str = "img_{:02d}.png") -> list[Path]:
    """Render ``[reference, deformed_1, ...]`` (translations ``shifts``) as 8-bit PNG files."""
    from PIL import Image

    from raftcorr.synthetic import render_sequence, translation

    frames = render_sequence(h, w, [translation(du, dv) for du, dv in shifts], seed=seed)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, f in enumerate(frames):
        p = folder / pattern.format(i)
        Image.fromarray(np.clip(np.round(f), 0, 255).astype(np.uint8)).save(p)
        paths.append(p)
    return paths


def write_plain_images(folder: Path, names: list[str], h: int = 24, w: int = 32) -> list[Path]:
    """Tiny random grey images with the given file names (for sorting tests)."""
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    out = []
    for n in names:
        p = folder / n
        Image.fromarray(rng.integers(0, 255, (h, w), dtype=np.uint8)).save(p)
        out.append(p)
    return out


def rect_mask(box=ROI_BOX, shape=(H, W)) -> np.ndarray:
    x0, y0, x1, y1 = box
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


# ============================================================================== synthetic results
def linear_field(k: int, X: np.ndarray, Y: np.ndarray, shifts=SHIFTS) -> tuple[np.ndarray, np.ndarray]:
    """Frame ``k``: the translation of :data:`SHIFTS` plus a uniaxial stretch ``exx = k * SLOPE``."""
    if k == 0:
        return np.zeros_like(X), np.zeros_like(Y)
    du, dv = shifts[k - 1]
    return du + k * SLOPE * (X - X_C), np.full_like(Y, dv)


def translation_field(k: int, X: np.ndarray, Y: np.ndarray, shifts=SHIFTS) -> tuple[np.ndarray, np.ndarray]:
    if k == 0:
        return np.zeros_like(X), np.zeros_like(Y)
    du, dv = shifts[k - 1]
    return np.full_like(X, du), np.full_like(Y, dv)


def make_series(roi: np.ndarray, n_frames: int = 4, field_fn: Callable = linear_field,
                done: list[bool] | None = None, names: list[str] | None = None, notes: tuple[str, ...] = ()):
    """A :class:`~raftcorr.results.DisplacementSeries` holding ``field_fn`` exactly."""
    from raftcorr.config import RunConfig
    from raftcorr.engine.mapping import Grid
    from raftcorr.results import DisplacementSeries, RunInfo
    from raftcorr.roi.masks import bbox_of
    from raftcorr.schedule import FrameSchedule

    Hh, Ww = roi.shape
    bx, by, bw, bh = bbox_of(roi)
    grid = Grid(bx, by, bw, bh, Hh, Ww)
    mask = roi[grid.slices].copy()
    X, Y = grid.coords()
    done_arr = np.ones(n_frames, bool) if done is None else np.asarray(done, bool)
    done_arr[0] = True
    u = np.full((n_frames,) + grid.shape, np.nan, np.float32)
    v = np.full_like(u, np.nan)
    for k in range(n_frames):
        if done_arr[k]:
            uk, vk = field_fn(k, X, Y)
            u[k] = np.where(mask, uk, np.nan)
            v[k] = np.where(mask, vk, np.nan)
    info = RunInfo(software_version="test", model_name="synthetic", model_sha256="", downsample_factor=8,
                   iters=20, corr_mode="volume", device="cpu", config=RunConfig().to_dict(), notes=tuple(notes))
    names = names or [f"img_{k:02d}.png" for k in range(n_frames)]
    return DisplacementSeries(grid, mask, u, v, done_arr, tuple(names), FrameSchedule.accumulative(n_frames), info)


# ======================================================================================== dialogs
@dataclass
class DialogStub:
    """Replaces every modal dialog; records ``(name, args)`` of each call."""

    calls: list[tuple[str, tuple]] = field(default_factory=list)
    question_answer: Any = QMessageBox.StandardButton.No
    open_file: str = ""
    save_file: str = ""
    directory: str = ""

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def texts(self, name: str) -> list[str]:
        return [" | ".join(str(a) for a in args[1:]) for n, args in self.calls if n == name]

    def install(self, monkeypatch) -> None:
        def rec(name, ret):
            def fn(*args, **kwargs):
                self.calls.append((name, args))
                return ret() if callable(ret) else ret
            return fn

        monkeypatch.setattr(QMessageBox, "question", rec("question", lambda: self.question_answer))
        for n in ("critical", "warning", "information"):
            monkeypatch.setattr(QMessageBox, n, rec(n, QMessageBox.StandardButton.Ok))
        monkeypatch.setattr(QMessageBox, "about", rec("about", None))
        monkeypatch.setattr(QFileDialog, "getOpenFileName", rec("getOpenFileName", lambda: (self.open_file, "")))
        monkeypatch.setattr(QFileDialog, "getSaveFileName", rec("getSaveFileName", lambda: (self.save_file, "")))
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", rec("getExistingDirectory", lambda: self.directory))


@pytest.fixture
def dialogs(monkeypatch) -> DialogStub:
    stub = DialogStub()
    stub.install(monkeypatch)
    return stub


@pytest.fixture(autouse=True)
def _restore_theme(qapp):
    from raftcorr.gui_basic import theme

    yield
    theme.set_current(theme.DEFAULT_THEME)
    qapp.setStyleSheet("")


# ========================================================================================= window
@pytest.fixture
def make_window(qtbot, monkeypatch, tmp_path, dialogs):
    """Factory of shown MainWindows whose QSettings live in ``tmp_path`` (never the registry)."""
    from raftcorr.gui_basic import main_window as mw

    ini = str(tmp_path / "raftcorr_settings.ini")
    monkeypatch.setattr(mw, "QSettings", lambda *_a: QSettings(ini, QSettings.Format.IniFormat))

    def _stop_worker(w) -> None:
        # pytest-qt closes and deletes the window before fixture teardown: finish any task first,
        # while the window can still receive the worker's "finished" signal
        if w._worker is not None:
            w._worker.request_stop()
            w._worker.wait(WORKER_TIMEOUT_MS)
            qtbot.waitUntil(lambda: w._worker is None, timeout=5000)

    def _make():
        w = mw.MainWindow()
        qtbot.addWidget(w, before_close_func=_stop_worker)
        w.show()
        qtbot.waitExposed(w)
        return w

    return _make


@pytest.fixture
def win(make_window):
    return make_window()


@pytest.fixture(scope="session")
def seq_folder(tmp_path_factory) -> Path:
    """4 frames, 256 x 320 px, translations :data:`SHIFTS` (read-only: never modify it)."""
    folder = tmp_path_factory.mktemp("seq") / "speckle"
    write_sequence(folder)
    return folder


@pytest.fixture
def loaded(win, seq_folder):
    assert win.load_folder(seq_folder)
    return win


@pytest.fixture
def with_results(loaded):
    """Window with the frame-0 ROI :data:`ROI_BOX` and a synthetic (linear) displacement series."""
    st = loaded.state
    st.set_roi(0, rect_mask())
    st.set_results(make_series(rect_mask(), names=st.sequence.names))
    render(loaded)
    return loaded


# ======================================================================================== actions
def render(win) -> None:
    """Run the pending (timer-driven) render synchronously."""
    win._render_timer.stop()
    win._render()


def wait_idle(qtbot, win, timeout: int = WORKER_TIMEOUT_MS) -> None:
    qtbot.waitUntil(lambda: win._worker is None, timeout=timeout)


def log_lines(win) -> str:
    return win.logview.toPlainText()


def view_pos(canvas, x: float, y: float):
    """Viewport position (QPoint) of image coordinate ``(x, y)``."""
    return canvas.mapFromScene(QPointF(x, y))


def scene_xy(canvas, pos) -> tuple[float, float]:
    p = canvas.mapToScene(pos)
    return float(p.x()), float(p.y())


# ================================================================================ fake pipeline
class FakePipeline:
    """Stands in for :class:`raftcorr.engine.pipeline.Pipeline` in ``run_correlation``'s job.

    ``run`` emits a progress message per frame (``frame k ...``), honours ``should_stop`` like the
    real pipeline (stopping keeps the computed frames) and returns the synthetic linear series.
    ``gate`` (a threading.Event) holds the run until the test sets it (or a stop is requested);
    ``error`` makes the run raise.
    """

    gate: threading.Event | None = None
    delay: float = 0.0
    error: Exception | None = None
    notes: tuple[str, ...] = ()
    calls: list[dict] = []

    def __init__(self, config=None, callbacks=None, loaded=None) -> None:
        self.config = config
        self.cb = callbacks

    def run(self, images, roi=None, frame_masks=None, **_kw):
        cls = type(self)
        cls.calls.append({"roi": None if roi is None else np.array(roi), "frame_masks": dict(frame_masks or {}),
                          "config": self.config})
        if cls.gate is not None:
            deadline = time.perf_counter() + 20.0
            while not cls.gate.wait(0.01) and not self.cb.should_stop() and time.perf_counter() < deadline:
                pass
        if cls.error is not None:
            raise cls.error
        n = len(images)
        done = [True] + [False] * (n - 1)
        for k in range(1, n):
            if self.cb.should_stop():
                break
            time.sleep(cls.delay)
            done[k] = True
            self.cb.progress(k / (n - 1), f"frame {k} (reference 0)")
        r = np.ones(images.shape, bool) if roi is None else np.asarray(roi, bool)
        return make_series(r, n, done=done, names=list(images.names), notes=cls.notes)


@pytest.fixture
def fake_pipeline(monkeypatch, win):
    """Patch the pipeline and model loading of ``win``; yields a fresh FakePipeline subclass."""
    import raftcorr.engine.pipeline as pipeline_mod

    class Fake(FakePipeline):
        gate = None
        delay = 0.0
        error = None
        notes = ()
        calls = []

    monkeypatch.setattr(pipeline_mod, "Pipeline", Fake)
    monkeypatch.setattr(win, "loaded_model", lambda cfg, cb: None)
    yield Fake
    if Fake.gate is not None:
        Fake.gate.set()


def accept_with(monkeypatch, dialog_cls, setup: Callable[[Any], None]) -> list:
    """Make ``dialog_cls.exec`` run ``setup(dialog)`` then accept, as if the user pressed OK.

    Returns the list of dialogs shown (for later inspection).
    """
    shown: list = []

    def fake_exec(self):
        shown.append(self)
        setup(self)
        self.accept()
        return self.result()  # what QDialog.exec returns: an int result code

    monkeypatch.setattr(dialog_cls, "exec", fake_exec)
    return shown
