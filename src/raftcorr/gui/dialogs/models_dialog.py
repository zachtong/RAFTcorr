"""The Models window (Help > Models…, and opened by a run whose model could not be downloaded).

One row per released model (installed or not, its size, a Download button), Download All,
Import Model Files… for files downloaded by hand on another computer (the SHA-256 says which model
a file is, so a renamed download is recognised and anything else is refused), Open Model Folder,
and the link to the release page. Downloads run in a thread and can be cancelled.

Opened for a failed download (``missing_key``), a card on top says which model is missing and
why, and turns into "installed now" once it is there.
"""

from __future__ import annotations

import html
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QStandardPaths, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from raftcorr.gui.support.model_weights import download_status_text
from raftcorr.gui.theme import current_colors
from raftcorr.gui.theme_manager import themed
from raftcorr.i18n import tr_args
from raftcorr.models import zoo

_CARD = ("QLabel {{ color: {TEXT_SECONDARY}; background: {BG_INPUT}; border: 1px solid {BORDER}; "
         "border-left: 3px solid {%s}; border-radius: 3px; padding: 8px 10px; }}")
_PROBLEM_STYLE = _CARD % "WARNING"
_RESOLVED_STYLE = _CARD % "SUCCESS"
_INSTALLED_STYLE = "QLabel {{ color: {SUCCESS}; }}"
_MISSING_STYLE = "QLabel {{ color: {TEXT_MUTED}; }}"


class _DownloadJob(QObject):
    """Downloads the models *keys* one after the other; stops at the first failure or a cancel.

    The work runs in a daemon thread and reports through signals, which Qt queues to the window.
    Not a QThread: a window closed during a stalled download (a read blocks up to the timeout)
    lets go of the job at once, and a download still blocked when the application quits neither
    holds the exit up nor aborts it, as a QThread destroyed while running does.
    """

    progress = Signal(float, str)
    downloaded = Signal(str)        # model key
    failed = Signal(str, str)       # model key, reason
    cancelled = Signal()
    finished = Signal()

    def __init__(self, keys: list[str]) -> None:
        super().__init__()
        self._keys = keys
        self.stop_requested = False
        self._thread = threading.Thread(target=self._run, name="raftcorr-model-download", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            self._download_all()
        finally:
            self.finished.emit()

    def _download_all(self) -> None:
        from raftcorr.callbacks import FunctionCallbacks, StopRequested

        for key in self._keys:
            spec = zoo.get_spec(key)
            if zoo.find_weights(spec) is not None:
                continue

            def report(fraction: float, message: str = "", spec=spec) -> None:
                done = fraction * (spec.size_bytes or 0)
                self.progress.emit(fraction, download_status_text(spec.name, done, spec.size_bytes))

            report(0.0)
            try:
                zoo.download(spec, callbacks=FunctionCallbacks(progress=report,
                                                               should_stop=lambda: self.stop_requested))
            except StopRequested:
                self.cancelled.emit()
                return
            except zoo.WeightsUnavailableError as exc:
                self.failed.emit(key, exc.short)
                return
            except OSError as exc:                   # anything download() does not foresee
                self.failed.emit(key, str(exc))
                return
            self.downloaded.emit(key)


class ModelsDialog(QDialog):
    """Which models are installed; download, import and the model folder."""

    def __init__(self, parent: QWidget | None = None, missing_key: str | None = None, detail: str = "") -> None:
        super().__init__(parent)
        self.missing_key = missing_key
        self.detail = detail
        self._job: _DownloadJob | None = None
        self._done: list[str] = []          # models downloaded by the current Download
        self._rows: dict[str, tuple[QLabel, QPushButton]] = {}
        self.setWindowTitle(self.tr("Models"))
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        self.problem_label = QLabel()
        self.problem_label.setWordWrap(True)
        self.problem_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.problem_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.problem_label.setVisible(missing_key is not None)
        lay.addWidget(self.problem_label)

        intro = QLabel(self.tr("RAFTcorr downloads a model the first time an analysis uses it and keeps it "
                               "for later runs. Each model is about 20 MB."))
        intro.setWordWrap(True)
        intro.setProperty("class", "secondary")
        lay.addWidget(intro)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        for row, spec in enumerate(zoo.MODELS.values()):
            name = QLabel(f"<b>{spec.name}</b>")
            info = QLabel(tr_args(self.tr("s = %1, reach %2 px"), spec.downsample_factor, spec.lookup_radius_px))
            info.setProperty("class", "muted")
            status = QLabel()
            button = QPushButton(self.tr("Download"))
            button.clicked.connect(lambda _checked=False, key=spec.key: self._download([key]))
            grid.addWidget(name, row, 0)
            grid.addWidget(info, row, 1)
            grid.addWidget(status, row, 2)
            grid.addWidget(button, row, 3)
            self._rows[spec.key] = (status, button)
        grid.setColumnStretch(2, 1)
        lay.addLayout(grid)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(8)
        self._progress_text = QLabel()
        self._progress_text.setProperty("class", "secondary")
        lay.addWidget(self._bar)
        lay.addWidget(self._progress_text)

        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.message_label.setVisible(False)
        lay.addWidget(self.message_label)

        self.folder_label = QLabel(tr_args(self.tr("Model folder: %1"), str(zoo.user_weights_dir())))
        self.folder_label.setProperty("class", "muted")
        self.folder_label.setWordWrap(True)
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.folder_label)

        manual = tr_args(self.tr(
            "No internet on this computer? Download the .pth files from the <a href=\"%1\">RAFTcorr model "
            "release</a> on another computer, then import them here."), zoo.RELEASE_PAGE_URL)
        # in the theme's accent: the default link blue is unreadable on the dark theme
        self.manual_label = QLabel(manual.replace("<a href=", f'<a style="color: {current_colors().ACCENT};" href='))
        self.manual_label.setWordWrap(True)
        self.manual_label.setOpenExternalLinks(True)
        lay.addWidget(self.manual_label)

        buttons = QHBoxLayout()
        self.download_all_button = QPushButton(self.tr("Download All"))
        self.download_all_button.clicked.connect(lambda: self._download(list(zoo.MODELS)))
        self.import_button = QPushButton(self.tr("Import Model Files…"))
        self.import_button.clicked.connect(self._on_import)
        self.open_folder_button = QPushButton(self.tr("Open Model Folder"))
        self.open_folder_button.clicked.connect(self._on_open_folder)
        self.cancel_button = QPushButton(self.tr("Cancel Download"))
        self.cancel_button.setProperty("class", "btn-danger")
        self.cancel_button.clicked.connect(self._on_cancel)
        close = QPushButton(self.tr("Close"))
        close.clicked.connect(self.reject)
        for b in (self.download_all_button, self.import_button, self.open_folder_button):
            buttons.addWidget(b)
        buttons.addStretch()
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(close)
        lay.addLayout(buttons)

        self._set_busy(False)
        # 600 px to start with, wider when the buttons need it (German): an explicit minimum width
        # would stop the layout from enforcing its own and the buttons would be squeezed.
        self.resize(max(600, self.sizeHint().width()), self.sizeHint().height())

    # -- state ------------------------------------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._job is not None

    def row_status(self, key: str) -> str:
        return self._rows[key][0].text()

    def row_status_tooltip(self, key: str) -> str:
        return self._rows[key][0].toolTip()

    def download_button(self, key: str) -> QPushButton:
        return self._rows[key][1]

    def _refresh(self) -> None:
        installed = {key: zoo.find_weights(spec) for key, spec in zoo.MODELS.items()}
        for key, (status, button) in self._rows.items():
            path, spec = installed[key], zoo.MODELS[key]
            if path is not None:
                status.setText(self.tr("Installed"))
                status.setToolTip(str(path))
                themed(status, _INSTALLED_STYLE)
            else:
                status.setText(tr_args(self.tr("Not installed (%1 MB)"), f"{spec.size_bytes / 1e6:.1f}")
                               if spec.size_bytes else self.tr("Not installed"))
                status.setToolTip("")
                themed(status, _MISSING_STYLE)
            button.setVisible(path is None)
            button.setEnabled(not self.busy)
        self.download_all_button.setEnabled(not self.busy and any(p is None for p in installed.values()))
        self.import_button.setEnabled(not self.busy)
        if self.missing_key is not None:
            name = zoo.get_spec(self.missing_key).name
            if installed.get(self.missing_key) is not None:
                self.problem_label.setText(tr_args(self.tr(
                    "<b>%1 is installed now.</b> Close this window and start again."), name))
                themed(self.problem_label, _RESOLVED_STYLE)
            else:
                self.problem_label.setText(tr_args(self.tr(
                    "<b>%1 is needed, but it is not installed and could not be downloaded.</b> Download it "
                    "again below, or import the file."), name) + "<br>" + html.escape(self.detail))
                themed(self.problem_label, _PROBLEM_STYLE)

    def _say(self, text: str) -> None:
        self.message_label.setText(text)
        self.message_label.setVisible(bool(text))

    def _set_busy(self, busy: bool) -> None:
        self._bar.setVisible(busy)
        self._progress_text.setVisible(busy)
        self.cancel_button.setVisible(busy)
        if busy:
            self._bar.setValue(0)
        self._refresh()

    # -- download ---------------------------------------------------------------------------------
    def _download(self, keys: list[str]) -> None:
        if self._job is not None:
            return
        self._done = []
        self._say("")
        job = _DownloadJob(keys)
        self._job = job
        job.progress.connect(self._on_progress)
        job.downloaded.connect(self._on_downloaded)
        job.failed.connect(self._on_failed)
        job.cancelled.connect(self._on_cancelled)
        job.finished.connect(self._on_finished)
        self._set_busy(True)
        job.start()

    def _on_progress(self, fraction: float, text: str) -> None:
        self._bar.setValue(int(fraction * 1000))
        self._progress_text.setText(text)

    def _on_downloaded(self, key: str) -> None:
        self._done.append(zoo.get_spec(key).name)
        self._say(tr_args(self.tr("Downloaded: %1."), ", ".join(self._done)))

    def _on_failed(self, key: str, reason: str) -> None:
        self._say(tr_args(self.tr("%1 could not be downloaded: %2"), zoo.get_spec(key).name, reason))

    def _on_cancelled(self) -> None:
        self._say(self.tr("Download cancelled."))

    def _on_finished(self) -> None:
        self._job = None
        self._set_busy(False)

    def _on_cancel(self) -> None:
        if self._job is not None:
            self._job.stop_requested = True

    def done(self, result: int) -> None:  # noqa: D401 - Qt override
        """Closing the window cancels a download in progress without waiting for it: the job
        stops at its next chunk (or when a stalled read times out) and leaves no file."""
        job, self._job = self._job, None
        if job is not None:
            job.stop_requested = True
            for signal in (job.progress, job.downloaded, job.failed, job.cancelled, job.finished):
                signal.disconnect()
        super().done(result)

    # -- import and folder ------------------------------------------------------------------------
    def _on_import(self) -> None:
        downloads = QStandardPaths.StandardLocation.DownloadLocation
        start = QStandardPaths.writableLocation(downloads) or str(Path.home())
        names, _ = QFileDialog.getOpenFileNames(self, self.tr("Import Model Files"), start,
                                                self.tr("RAFTcorr models (*.pth)"))
        if not names:
            return
        imported, refused = [], []
        for name in names:
            try:
                spec, _path = zoo.import_weights(name)
                imported.append(spec.name)
            except ValueError:
                refused.append(Path(name).name)
            except OSError as exc:
                refused.append(f"{Path(name).name} ({exc.strerror or exc})")
        lines = []
        if imported:
            lines.append(tr_args(self.tr("Imported: %1."), ", ".join(imported)))
        if refused:
            lines.append(tr_args(self.tr("Not imported: %1. Only the model files of the release can be "
                                         "imported."), ", ".join(refused)))
        self._say("\n".join(lines))
        self._refresh()

    def _on_open_folder(self) -> None:
        folder = zoo.user_weights_dir()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
