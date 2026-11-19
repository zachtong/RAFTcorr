"""A new user without model files: the Models window, the download status and the failure path.

The network is never touched: ``urllib.request.urlopen`` is replaced by a fake server that serves
a stand-in for RAFTcorr-s4 (its hash recorded in the model table for the test).
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
import time
import urllib.error
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")     # the real fonts: real widths
pytest.importorskip("pytestqt")

from raftcorr import synthetic as syn  # noqa: E402
from raftcorr.models import zoo  # noqa: E402

PAYLOAD = b"weights!" * 50_000          # 400 kB: two chunks of the download loop


class _Response(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


@pytest.fixture(autouse=True)
def no_modal_boxes(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    shown: list = []
    for name in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(
            lambda *a, _n=name, **k: shown.append((_n, a[1:3])) or QMessageBox.StandardButton.Ok))
    return shown


@pytest.fixture(autouse=True)
def fresh_state(qapp):
    from PySide6.QtCore import QCoreApplication

    QCoreApplication.setOrganizationName("RAFTcorr-tests")
    QCoreApplication.setApplicationName("RAFTcorr-tests")
    from raftcorr.gui.app_state import AppState

    AppState._instance = None
    yield
    AppState._instance = None


@pytest.fixture()
def weights(tmp_path, monkeypatch):
    """No local weights anywhere; the fake release serves RAFTcorr-s4 (or fails, or waits)."""
    user = tmp_path / "models"
    monkeypatch.setattr(zoo, "user_weights_dir", lambda: user)
    monkeypatch.setattr(zoo, "checkout_models_dir", lambda: None)
    monkeypatch.delenv(zoo.ENV_WEIGHTS, raising=False)
    spec = replace(zoo.MODELS["s4"], sha256=hashlib.sha256(PAYLOAD).hexdigest(), size_bytes=len(PAYLOAD))
    monkeypatch.setitem(zoo.MODELS, "s4", spec)
    server = SimpleNamespace(error=None, gate=None, asked=[])

    def fake_urlopen(req, timeout=None):
        server.asked.append(req.full_url)
        if server.error is not None:
            raise server.error
        if server.gate is not None:
            server.gate.wait(10)
        return _Response(PAYLOAD)

    monkeypatch.setattr(zoo.urllib.request, "urlopen", fake_urlopen)
    return SimpleNamespace(user=user, spec=spec, server=server, file=user / spec.filename)


@pytest.fixture()
def opened(monkeypatch):
    """ModelsDialog.exec records how the window was opened instead of blocking."""
    from raftcorr.gui.dialogs import models_dialog

    calls: list = []
    monkeypatch.setattr(models_dialog.ModelsDialog, "exec",
                        lambda self: calls.append((self.missing_key, self.detail)) or 0)
    return calls


def _good_and_bad(tmp_path: Path) -> tuple[Path, Path]:
    d = tmp_path / "Downloads"
    d.mkdir()
    good, bad = d / "RAFTcorr-s4 (1).pth", d / "notes.pth"
    good.write_bytes(PAYLOAD)
    bad.write_bytes(b"not a model")
    return good, bad


# ------------------------------------------------------------------------------ download status
def test_the_download_status_names_the_model_and_the_megabytes():
    from raftcorr.gui.support.model_weights import download_status_text

    assert download_status_text("RAFTcorr-s4", 5.24e6, 20.66e6) == "Downloading model RAFTcorr-s4: 5.2 of 20.7 MB"


def test_ensure_weights_reports_a_download_then_clears_the_status(weights):
    from raftcorr.callbacks import NullCallbacks
    from raftcorr.gui.support.model_weights import ensure_weights

    seen: list = []
    path = ensure_weights("s4", NullCallbacks(), lambda f, t: seen.append((f, t)))
    assert path == weights.file and path.read_bytes() == PAYLOAD
    assert seen[0] == (0.0, "Downloading model RAFTcorr-s4: 0.0 of 0.4 MB")
    assert any(f == 1.0 for f, _ in seen) and seen[-1] == (0.0, "")
    seen.clear()
    assert ensure_weights("s4", NullCallbacks(), lambda f, t: seen.append((f, t))) == path
    assert seen == [] and len(weights.server.asked) == 1      # installed: no status, no request


# ---------------------------------------------------------------------------------- the window
def test_the_window_lists_every_model_with_its_state(qtbot, weights):
    from raftcorr.gui.dialogs.models_dialog import ModelsDialog

    weights.user.mkdir()
    weights.file.write_bytes(PAYLOAD)
    dlg = ModelsDialog()
    qtbot.addWidget(dlg)
    assert dlg.row_status("s4") == "Installed"
    assert not dlg.download_button("s4").isVisibleTo(dlg) and dlg.download_button("s8").isVisibleTo(dlg)
    assert dlg.row_status("s8") == "Not installed (21.1 MB)"
    assert str(weights.user) in dlg.folder_label.text()
    assert zoo.RELEASE_PAGE_URL in dlg.manual_label.text()


def test_download_from_the_window(qtbot, weights):
    from raftcorr.gui.dialogs.models_dialog import ModelsDialog

    dlg = ModelsDialog()
    qtbot.addWidget(dlg)
    assert dlg.row_status("s4") == "Not installed (0.4 MB)"
    dlg.download_button("s4").click()
    qtbot.waitUntil(lambda: not dlg.busy, timeout=10000)
    assert weights.file.read_bytes() == PAYLOAD
    assert dlg.row_status("s4") == "Installed" and "RAFTcorr-s4" in dlg.message_label.text()


def test_a_failed_download_says_why_and_offers_the_way_out(qtbot, weights):
    from raftcorr.gui.dialogs.models_dialog import ModelsDialog

    weights.server.error = urllib.error.URLError("getaddrinfo failed")
    dlg = ModelsDialog()
    qtbot.addWidget(dlg)
    dlg.download_button("s4").click()
    qtbot.waitUntil(lambda: not dlg.busy, timeout=10000)
    text = dlg.message_label.text()
    assert "RAFTcorr-s4" in text and "internet connection" in text
    assert dlg.download_button("s4").isEnabled() and dlg.import_button.isEnabled()
    assert not weights.user.exists() or not any(weights.user.iterdir())


def test_cancel_stops_the_download(qtbot, weights):
    from raftcorr.gui.dialogs.models_dialog import ModelsDialog

    weights.server.gate = threading.Event()
    dlg = ModelsDialog()
    qtbot.addWidget(dlg)
    dlg.download_button("s4").click()
    assert dlg.busy and dlg.cancel_button.isVisibleTo(dlg)
    dlg.cancel_button.click()
    weights.server.gate.set()
    qtbot.waitUntil(lambda: not dlg.busy, timeout=10000)
    assert not weights.file.exists() and "cancelled" in dlg.message_label.text().lower()


def test_closing_the_window_during_a_download_neither_waits_nor_leaves_a_file(qtbot, weights):
    """The stop is only seen between chunks; a stalled connection must not hold the window (it
    once waited 15 s, then let go of a running QThread, which aborts the process)."""
    from raftcorr.gui.dialogs.models_dialog import ModelsDialog

    weights.server.gate = threading.Event()           # the server does not answer yet
    dlg = ModelsDialog()
    qtbot.addWidget(dlg)
    dlg.download_button("s4").click()
    t0 = time.perf_counter()
    dlg.reject()
    assert time.perf_counter() - t0 < 1.0 and not dlg.busy
    weights.server.gate.set()
    qtbot.waitUntil(lambda: not any(t.name == "raftcorr-model-download" for t in threading.enumerate()),
                    timeout=10000)
    assert not weights.file.exists() and not list(weights.user.glob("*.part"))


def test_a_model_in_the_weights_folder_of_the_environment_counts_as_installed(qtbot, weights, tmp_path,
                                                                              monkeypatch):
    from raftcorr.callbacks import NullCallbacks
    from raftcorr.gui.dialogs.models_dialog import ModelsDialog
    from raftcorr.gui.support.model_weights import ensure_weights

    shared = tmp_path / "shared weights"
    shared.mkdir()
    (shared / weights.spec.filename).write_bytes(PAYLOAD)
    monkeypatch.setenv(zoo.ENV_WEIGHTS, str(shared))
    dlg = ModelsDialog()
    qtbot.addWidget(dlg)
    assert dlg.row_status("s4") == "Installed" and dlg.row_status_tooltip("s4") == str(shared / weights.spec.filename)
    seen: list = []
    assert ensure_weights("s4", NullCallbacks(), lambda f, t: seen.append(t)) == shared / weights.spec.filename
    assert seen == [] and weights.server.asked == []


def test_import_installs_a_released_model_and_refuses_anything_else(qtbot, weights, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from raftcorr.gui.dialogs.models_dialog import ModelsDialog

    good, bad = _good_and_bad(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([str(good), str(bad)], "")))
    dlg = ModelsDialog()
    qtbot.addWidget(dlg)
    dlg.import_button.click()
    assert weights.file.read_bytes() == PAYLOAD and dlg.row_status("s4") == "Installed"
    text = dlg.message_label.text()
    assert "RAFTcorr-s4" in text and "notes.pth" in text


def test_open_the_model_folder(qtbot, weights, monkeypatch):
    from PySide6.QtGui import QDesktopServices

    from raftcorr.gui.dialogs.models_dialog import ModelsDialog

    urls: list = []
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: urls.append(url) or True))
    dlg = ModelsDialog()
    qtbot.addWidget(dlg)
    dlg.open_folder_button.click()
    assert weights.user.is_dir() and Path(urls[0].toLocalFile()) == weights.user


def test_opened_after_a_failed_download_it_explains_then_confirms(qtbot, weights, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from raftcorr.gui.dialogs.models_dialog import ModelsDialog

    dlg = ModelsDialog(missing_key="s4", detail="cannot download RAFTcorr-s4.pth (no route to host)")
    qtbot.addWidget(dlg)
    banner = dlg.problem_label.text()
    assert "RAFTcorr-s4" in banner and "no route to host" in banner
    good, _ = _good_and_bad(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *a, **k: ([str(good)], "")))
    dlg.import_button.click()
    assert "installed now" in dlg.problem_label.text() and "again" in dlg.problem_label.text()


# --------------------------------------------------------------------------- the main window
def _folder(tmp_path: Path) -> Path:
    d = tmp_path / "imgs"
    d.mkdir()
    for i, f in enumerate(syn.render_sequence(96, 128, [syn.translation(1.0, 0.5)], seed=2)):
        Image.fromarray(np.round(f).astype(np.uint8)).save(d / f"img{i + 1}.png")
    return d


@pytest.fixture()
def win(qtbot):
    from raftcorr.gui.app import MainWindow

    w = MainWindow()
    qtbot.addWidget(w)
    w.resize(1400, 900)
    w.show()
    yield w
    w._state.set_results(None)


class _StopAtOnce:
    """Stands in for the pipeline once the model is there: the run ends as a stop would."""

    def __init__(self, *a, **k) -> None:
        pass

    def run(self, *a, **k):
        from raftcorr.callbacks import StopRequested

        raise StopRequested()


def _start(win, tmp_path) -> None:
    win._image_ctrl.load_folder(str(_folder(tmp_path)))
    win._state.set_frame_roi(0, np.ones((96, 128), bool))
    win._pipeline_ctrl.start()


def _idle(qtbot, state) -> None:
    from raftcorr.gui.app_state import RunState

    qtbot.waitUntil(lambda: state.run_state not in (RunState.RUNNING, RunState.PAUSED), timeout=15000)


def test_run_offline_without_the_model_opens_the_models_window(win, qtbot, tmp_path, weights, opened):
    weights.server.error = urllib.error.URLError("getaddrinfo failed")
    fatal: list = []
    win._state.fatal_error.connect(lambda *a: fatal.append(a))
    _start(win, tmp_path)
    _idle(qtbot, win._state)
    qtbot.waitUntil(lambda: bool(opened), timeout=5000)
    key, detail = opened[0]
    assert key == "s4" and "internet connection" in detail
    assert fatal == []                                # no crash box on top of it


def test_run_downloads_the_model_with_a_status_line(win, qtbot, tmp_path, weights, opened, monkeypatch):
    import raftcorr.engine.pipeline as pl
    import raftcorr.models as models

    monkeypatch.setattr(pl, "Pipeline", _StopAtOnce)
    monkeypatch.setattr(models, "load_model", lambda *a, **k: object())
    statuses: list = []
    win._state.progress_status.connect(statuses.append)
    _start(win, tmp_path)
    _idle(qtbot, win._state)
    assert weights.file.read_bytes() == PAYLOAD and opened == []
    assert statuses[0].startswith("Downloading model RAFTcorr-s4") and statuses[-1] == ""


class _TrackAQuarter:
    """Stands in for the pipeline: reports a quarter of the tracking, then the run is stopped."""

    def __init__(self, cfg, callbacks, loaded) -> None:
        self._cb = callbacks

    def run(self, *a, **k):
        from raftcorr.callbacks import StopRequested

        self._cb.progress(0.25, "Frame 2/3 (reference 1): 0.1 s")
        raise StopRequested()


def test_after_a_download_the_run_progress_starts_from_zero(win, qtbot, tmp_path, weights, monkeypatch):
    """Through the worker's signals: the download reaches 100 %, then the tracking reports 25 %,
    which the bar must show (it never goes back, so without the reset it stayed at 100 %)."""
    import raftcorr.engine.pipeline as pl
    import raftcorr.models as models

    monkeypatch.setattr(pl, "Pipeline", _TrackAQuarter)
    monkeypatch.setattr(models, "load_model", lambda *a, **k: object())
    shown: list = []
    win._state.progress_updated.connect(lambda f, m: shown.append(f))
    _start(win, tmp_path)
    _idle(qtbot, win._state)
    assert 1.0 in shown and 0.25 in shown[shown.index(1.0):]


def test_cancelling_a_background_task_is_not_an_error(win, qtbot, no_modal_boxes):
    """Cancel of the model survey's progress dialog stops it; no error box, no error line."""
    from raftcorr.callbacks import StopRequested

    errors: list = []
    win._state.log_message.connect(lambda m, lvl: errors.append(m) if lvl == "error" else None)

    def survey(cb):
        raise StopRequested("download of RAFTcorr-s8.pth cancelled")

    win._run_task("Measuring the motion", survey, lambda r: None)
    qtbot.waitUntil(lambda: win._task is None, timeout=5000)
    assert no_modal_boxes == [] and errors == []


def test_the_progress_label_shows_the_download_and_the_bar_restarts_after_it(win):
    rs, ctrl, st = win._right_sidebar, win._pipeline_ctrl, win._state
    ctrl._on_download_status(1.0, "Downloading model RAFTcorr-s4: 20.7 of 20.7 MB")
    assert "Downloading model RAFTcorr-s4" in rs._progress_label.text()
    assert rs._progress_bar.value() == 1000
    ctrl._on_download_status(0.0, "")
    assert st.progress == 0.0 and "Downloading" not in rs._progress_label.text()
    ctrl._on_progress(0.25, "Frame 2/3 (reference 1): 1.0 s")
    assert st.progress == 0.25                        # not held at the download's 100 %
    assert "Frame 2/3" in rs._progress_label.text()


def test_suggest_a_model_offline_opens_the_models_window(win, qtbot, tmp_path, weights, opened):
    weights.server.error = urllib.error.URLError("getaddrinfo failed")
    win._image_ctrl.load_folder(str(_folder(tmp_path)))
    win._on_suggest_model()
    qtbot.waitUntil(lambda: bool(opened), timeout=10000)
    assert opened[0][0] == "s8"


def test_help_menu_opens_the_models_window(win, opened):
    actions = [a for m in win.menuBar().actions() for a in m.menu().actions() if a.text() == "Models…"]
    assert len(actions) == 1
    actions[0].trigger()
    assert opened == [(None, "")]


@pytest.mark.parametrize("lang", ["zh_CN", "zh_TW", "ja", "ko", "de", "fr", "es"])
def test_no_button_of_the_window_is_squeezed(qtbot, qapp, weights, lang):
    """German button texts are long: the window grows instead of clipping them."""
    from PySide6.QtWidgets import QPushButton

    from raftcorr.gui.dialogs.models_dialog import ModelsDialog
    from raftcorr.gui.theme_manager import theme_manager
    from raftcorr.i18n import LanguageManager

    theme_manager(qapp).set_theme("light", persist=False)   # its padding and weight make the widths
    languages = LanguageManager(qapp)
    assert languages.load(lang)
    try:
        dlg = ModelsDialog(missing_key="s4", detail="no connection to the download server")
        qtbot.addWidget(dlg)
        dlg.show()
        qtbot.waitExposed(dlg)
        assert dlg.windowTitle() != "Models"          # the catalog is in use
        # never narrower than the layout needs (a fixed minimum width once kept it at 600 px, under
        # the 620 px of the German buttons, and with the Fusion style of the application they were cut)
        assert dlg.minimumWidth() >= dlg.layout().minimumSize().width(), lang
        for button in dlg.findChildren(QPushButton):
            if button.isVisible():
                assert button.width() >= button.minimumSizeHint().width(), (lang, button.text())
    finally:
        languages.load("en")
