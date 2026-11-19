"""The real RAFTcorr window, built off-screen at a device-pixel ratio of 2.

Shared by the documentation pictures (``screenshots.py``) and the quick-start video
(``quickstart_video.py``). Importing this module configures Qt, so import it before PySide6:

* the ``offscreen`` platform: Qt lays out and paints every window normally, nothing appears on the
  desktop, and it works the same on any computer;
* an exact device-pixel ratio of 2 (sharp text in pictures and in camera zooms);
* the symbol fonts a desktop finds by itself (the scissors of "Cut", arrows), which the offscreen
  platform does not look for;
* settings in a throw-away folder, so the user's own RAFTcorr settings are never read or changed;
* blocking message boxes raise instead of waiting for a click nobody can give.

The demo data is the aluminium plate with two holes in tension, 44 images of 552 x 3008 px. It is
not part of the repository; pass its folder on the command line.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DPR = 2


def _offscreen_platform() -> str:
    """The offscreen platform with a 1920 x 1200 (logical) screen; its default screen is so small that
    dialogs fitted to the screen would shrink and popups would be pushed around."""
    cfg = Path(tempfile.mkdtemp(prefix="raftcorr_media_qpa_")) / "offscreen.json"
    cfg.write_text('{"synchronousWindowSystemEvents": false, "windowFrameMargins": false, "screens": '
                   '[{"name": "media", "x": 0, "y": 0, "width": 3840, "height": 2400, '
                   '"logicalDpi": 96, "logicalBaseDpi": 96, "dpr": 1}]}', encoding="utf-8")
    path = cfg.resolve().as_posix()
    # The platform string splits its options at ':', so a drive letter cannot appear in it.
    if len(path) > 1 and path[1] == ":":
        if Path.cwd().drive.lower() != cfg.resolve().drive.lower():
            raise RuntimeError("run this script from the drive of the temporary folder")
        path = path[2:]
    return f"offscreen:configfile={path}"


os.environ["QT_QPA_PLATFORM"] = _offscreen_platform()
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
os.environ["QT_SCALE_FACTOR"] = str(DPR)
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", "C:/Windows/Fonts")

import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, QRectF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QFont, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget  # noqa: E402

#: Logical size of the main window (the pictures are twice as large).
DESK_W, DESK_H = 1600, 1000


@dataclass(frozen=True)
class Specimen:
    """Geometry of the demo specimen in image pixels (x right, y down), measured on frame 1."""

    gauge: tuple[float, float, float, float] = (150.0, 1300.0, 375.0, 2750.0)    # x0, y0, x1, y1
    holes: tuple[tuple[float, float], ...] = ((261.0, 1975.0), (262.0, 2192.0))
    hole_cut_radius: float = 55.0
    extensometer: tuple[float, float, float, float] = (215.0, 1420.0, 215.0, 1820.0)
    point: tuple[float, float] = (262.0, 2083.0)                                 # between the holes
    area: tuple[float, float, float, float] = (180.0, 2330.0, 345.0, 2620.0)
    gauge_view: tuple[float, float, float, float] = (0.0, 1180.0, 552.0, 1700.0)  # x, y, w, h
    holes_view: tuple[float, float, float, float] = (90.0, 1720.0, 372.0, 720.0)  # x, y, w, h


SPECIMEN = Specimen()


class ScriptAbort(RuntimeError):
    """A dialog the script did not expect tried to block the run."""


def _fail_modal(*args, **_kwargs):
    raise ScriptAbort(f"unexpected modal dialog: {args[1:3]!r}")


def make_app(lang: str = "en", theme: str = "light") -> tuple[QApplication, QWidget]:
    """Return (app, main window) laid out at DESK_W x DESK_H logical pixels."""
    settings_dir = tempfile.mkdtemp(prefix="raftcorr_media_settings_")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, settings_dir)

    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setOrganizationName("RAFTcorr")
    app.setApplicationName("RAFTcorr")
    app.setStyle("Fusion")
    for family in ("Segoe UI", "Sans Serif", "Helvetica Neue", "SF Pro Text", "Microsoft YaHei UI"):
        QFont.insertSubstitutions(family, ["Segoe UI Symbol", "Segoe UI Emoji", "Apple Symbols"])

    from raftcorr.gui.support.matplotlib_fonts import configure_matplotlib_fonts
    from raftcorr.gui.theme_manager import theme_manager

    theme_manager(app).set_theme(theme, persist=False)
    configure_matplotlib_fonts()
    if lang != "en":
        from raftcorr.i18n import LanguageManager

        manager = LanguageManager(app)
        if not manager.load(lang):
            raise RuntimeError(f"no interface translation for {lang!r}")
        app._raftcorr_media_languages = manager       # the translator lives as long as its manager

    for name in ("question", "critical", "warning", "information"):
        setattr(QMessageBox, name, staticmethod(_fail_modal))

    from raftcorr.gui.app import MainWindow

    win = MainWindow()
    win.move(0, 0)
    win.resize(DESK_W, DESK_H)
    win.show()
    settle(app, 0.3)
    if abs(win.devicePixelRatioF() - DPR) > 1e-6:
        raise RuntimeError(f"expected a device-pixel ratio of {DPR}, got {win.devicePixelRatioF()}")
    return app, win


def settle(app: QApplication, seconds: float) -> None:
    """Pump the event loop for ``seconds`` of wall time."""
    end = time.perf_counter() + seconds
    while True:
        app.processEvents()
        if time.perf_counter() >= end:
            return
        time.sleep(0.01)


def wait_until(app: QApplication, done, timeout: float = 900.0) -> None:
    t0 = time.perf_counter()
    while not done():
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError("the application did not reach the expected state in time")
        settle(app, 0.05)


def offscreen(w: QWidget) -> None:
    """Kept for symmetry with a desktop platform; the offscreen platform shows nothing anyway."""
    w.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)


def qimage_to_rgb(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    h, w, bpl = img.height(), img.width(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), np.uint8, count=h * bpl).reshape(h, bpl)
    return buf[:, : w * 3].reshape(h, w, 3).copy()


def grab(w: QWidget) -> np.ndarray:
    """RGB capture of a widget at DPR 2."""
    return qimage_to_rgb(w.grab().toImage())


def grab_with_popups(root: QWidget) -> QImage:
    """``root`` with its visible menus, combo lists and dialogs painted where they are."""
    image = root.grab().toImage()
    painter = QPainter(image)
    origin = root.mapToGlobal(QPoint(0, 0))
    for w in QApplication.topLevelWidgets():
        if w is root or not w.isVisible():
            continue
        floating = w.windowType() in (Qt.WindowType.Popup, Qt.WindowType.ToolTip)
        if floating or _descends_from(w, root):
            painter.drawPixmap(w.mapToGlobal(QPoint(0, 0)) - origin, w.grab())
    painter.end()
    return image


def _descends_from(widget: QWidget, root: QWidget) -> bool:
    p = widget.parentWidget()
    while p is not None:
        if p is root:
            return True
        p = p.parentWidget()
    return False


# ---------------------------------------------------------------------------- shortcuts
def specimen_roi(win, spec: Specimen = SPECIMEN) -> np.ndarray:
    """The gauge section minus both holes, as the ROI tools draw it."""
    from raftcorr.roi.shapes import Circle, Rect, apply_shape

    h, w = win._state.sequence.shape
    roi = apply_shape(np.zeros((h, w), bool), Rect(*spec.gauge), "add")
    for cx, cy in spec.holes:
        roi = apply_shape(roi, Circle(cx, cy, spec.hole_cut_radius), "cut")
    return roi


def zoom_canvas(canvas, rect: tuple[float, float, float, float]) -> None:
    """Show the image rectangle ``(x, y, w, h)`` as large as the canvas allows."""
    canvas.resetTransform()
    canvas.fitInView(QRectF(*rect), Qt.AspectRatioMode.KeepAspectRatio)
    canvas._zoom_level = canvas.transform().m11()
    canvas.view_changed.emit()


def find_ffmpeg() -> str:
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    import imageio_ffmpeg   # a RAFTcorr dependency (MP4 export)

    return imageio_ffmpeg.get_ffmpeg_exe()
