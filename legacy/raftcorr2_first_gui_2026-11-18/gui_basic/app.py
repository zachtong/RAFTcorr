"""Entry point of the desktop application: ``raftcorr`` / ``raftcorr gui`` / ``python -m raftcorr``."""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

from . import theme

log = logging.getLogger("raftcorr")


def _excepthook(exc_type, exc, tb) -> None:
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    log.error(text)
    sys.stderr.write(text)
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        if QApplication.instance() is not None:
            box = QMessageBox(QMessageBox.Icon.Critical, "Unexpected error",
                              f"{exc_type.__name__}: {exc}\n\nThe application may be unstable; save your project.")
            box.setDetailedText(text)
            box.exec()
    except Exception:
        pass


def build_app(argv: list[str] | None = None):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QGuiApplication
    from PySide6.QtWidgets import QApplication

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("RAFTcorr")
    app.setOrganizationName("RAFTcorr")
    app.setFont(QFont("Segoe UI", 9))
    return app


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = build_app(argv)
    from PySide6.QtCore import QSettings

    from .main_window import MainWindow

    name = QSettings("RAFTcorr", "RAFTcorr").value("theme", theme.DEFAULT_THEME)
    theme.set_current(name if name in theme.THEMES else theme.DEFAULT_THEME)
    app.setStyleSheet(theme.stylesheet())
    sys.excepthook = _excepthook
    win = MainWindow()
    geo = win.settings.value("geometry")
    if geo is not None:
        win.restoreGeometry(geo)
    win.show()
    for arg in argv[1:]:
        win.open_path(Path(arg))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
