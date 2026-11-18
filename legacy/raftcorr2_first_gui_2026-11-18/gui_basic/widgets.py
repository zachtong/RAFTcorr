"""Small reusable widgets: collapsible section, frame navigator, colour bar, log view, help hint."""

from __future__ import annotations

import datetime as _dt

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QLinearGradient, QPainter, QPen, QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..viz.colormaps import lut
from ..viz.render import format_tick, nice_ticks
from . import theme


class Section(QFrame):
    """A titled card whose body can be collapsed."""

    def __init__(self, title: str, parent: QWidget | None = None, expanded: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("section")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 2, 8, 8)
        lay.setSpacing(4)
        self.header = QToolButton()
        self.header.setObjectName("sectionHeader")
        self.header.setText(title.upper())
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header.toggled.connect(self._toggle)
        lay.addWidget(self.header)
        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(6)
        lay.addWidget(self.body)
        self.body.setVisible(expanded)

    def _toggle(self, on: bool) -> None:
        self.body.setVisible(on)
        self.header.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

    def add(self, w: QWidget) -> QWidget:
        self.body_layout.addWidget(w)
        return w

    def add_layout(self, layout) -> None:
        self.body_layout.addLayout(layout)


def muted(text: str, wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    lab.setProperty("role", "muted")
    lab.setWordWrap(wrap)
    return lab


def row(*widgets: QWidget | str, stretch_last: bool = False) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    for i, w in enumerate(widgets):
        if isinstance(w, str):
            w = QLabel(w)
        lay.addWidget(w, 1 if (stretch_last and i == len(widgets) - 1) else 0)
    if not stretch_last:
        lay.addStretch(1)
    return lay


def hint(text: str) -> QLabel:
    """A small (?) label that shows ``text`` as its tooltip."""
    lab = QLabel("?")
    lab.setToolTip(text)
    lab.setFixedSize(16, 16)
    lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
    p = theme.current()
    lab.setStyleSheet(f"QLabel {{ border: 1px solid {p.BORDER}; border-radius: 8px; color: {p.TEXT_MUTED};"
                      f" font-size: 10px; }}")
    return lab


class FrameNavigator(QWidget):
    """Slider + first / previous / play / next / last + speed + loop + frame label."""

    frame_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(4)
        self._n = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._frames: list[int] | None = None

        style = self.style()
        SP = QStyle.StandardPixmap
        self._icon_play = style.standardIcon(SP.SP_MediaPlay)
        self._icon_pause = style.standardIcon(SP.SP_MediaPause)

        def btn(icon, tip: str, slot) -> QToolButton:
            b = QToolButton()
            b.setIcon(style.standardIcon(icon) if not hasattr(icon, "isNull") else icon)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            lay.addWidget(b)
            return b

        btn(SP.SP_MediaSkipBackward, "First frame (Home)", lambda: self.request(0))
        btn(SP.SP_MediaSeekBackward, "Previous frame (Left)", lambda: self.step(-1))
        self.play_btn = btn(self._icon_play, "Play / pause (Space)", self.toggle_play)
        btn(SP.SP_MediaSeekForward, "Next frame (Right)", lambda: self.step(1))
        btn(SP.SP_MediaSkipForward, "Last frame (End)", lambda: self.request(self._n - 1))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._slid)
        lay.addWidget(self.slider, 1)
        self.label = QLabel("–")
        self.label.setMinimumWidth(120)
        self.label.setProperty("role", "secondary")
        lay.addWidget(self.label)
        lay.addWidget(QLabel("fps"))
        self.speed = QComboBox()
        self.speed.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        for v in ("1", "2", "5", "10", "20", "30"):
            self.speed.addItem(v)
        self.speed.setCurrentText("5")
        self.speed.setToolTip("Playback speed (frames per second)")
        self.speed.currentTextChanged.connect(self._speed_changed)
        lay.addWidget(self.speed)
        self.loop = QToolButton()
        self.loop.setText("Loop")
        self.loop.setCheckable(True)
        self.loop.setChecked(True)
        self.loop.setToolTip("Loop playback")
        lay.addWidget(self.loop)
        self._names: list[str] = []
        self.set_count(0)

    def set_count(self, n: int, names: list[str] | None = None) -> None:
        self._n = n
        self._names = list(names or [])
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, n - 1))
        self.slider.blockSignals(False)
        self.setEnabled(n > 0)
        self.set_frame(min(self.slider.value(), max(0, n - 1)))

    def set_playable(self, frames: list[int] | None) -> None:
        """Restrict playback to these frames (e.g. frames with results); None = all."""
        self._frames = sorted(frames) if frames else None

    def set_frame(self, k: int) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(k)
        self.slider.blockSignals(False)
        name = self._names[k] if 0 <= k < len(self._names) else ""
        self.label.setText(f"frame {k} / {max(0, self._n - 1)}" + (f"  {name}" if name else ""))
        self.label.setToolTip(name)

    def request(self, k: int) -> None:
        if self._n:
            self.frame_requested.emit(int(np.clip(k, 0, self._n - 1)))

    def step(self, d: int) -> None:
        cur = self.slider.value()
        if self._frames:
            seq = self._frames
            if d > 0:
                nxt = next((f for f in seq if f > cur), None)
            else:
                nxt = next((f for f in reversed(seq) if f < cur), None)
            if nxt is None:
                nxt = seq[0] if (d > 0 and self.loop.isChecked()) else (seq[-1] if self.loop.isChecked() else cur)
            self.request(nxt)
            return
        k = cur + d
        if k >= self._n:
            k = 0 if self.loop.isChecked() else self._n - 1
        elif k < 0:
            k = self._n - 1 if self.loop.isChecked() else 0
        self.request(k)

    def toggle_play(self) -> None:
        if self._timer.isActive():
            self.stop()
        elif self._n > 1:
            self._timer.start(int(1000 / float(self.speed.currentText())))
            self.play_btn.setIcon(self._icon_pause)

    def stop(self) -> None:
        self._timer.stop()
        self.play_btn.setIcon(self._icon_play)

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def _tick(self) -> None:
        cur = self.slider.value()
        last = self._frames[-1] if self._frames else self._n - 1
        if cur >= last and not self.loop.isChecked():
            self.stop()
            return
        self.step(1)

    def _speed_changed(self, _t: str) -> None:
        if self._timer.isActive():
            self._timer.start(int(1000 / float(self.speed.currentText())))

    def _slid(self, v: int) -> None:
        self.request(v)


class ColorBar(QWidget):
    """Vertical colour bar with ticks and a label, drawn with QPainter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(92)
        self.setMaximumWidth(120)
        self._cmap = "turbo"
        self._vmin, self._vmax = 0.0, 1.0
        self._label = ""
        self._levels = 0

    def set_scale(self, colormap: str, vmin: float, vmax: float, label: str, levels: int = 0) -> None:
        self._cmap, self._vmin, self._vmax, self._label, self._levels = colormap, vmin, vmax, label, levels
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pal = theme.current()
        text = QColor(pal.TEXT)
        f = QFont(self.font())
        f.setPointSizeF(9.5)
        p.setFont(f)
        fm = QFontMetrics(f)
        top, bottom = 22, self.height() - 14
        x0, bw = 8, 16
        if bottom - top < 20:
            p.end()
            return
        table = lut(self._cmap)
        grad = QLinearGradient(0, bottom, 0, top)
        n = 32 if self._levels <= 0 else self._levels
        for i in range(n + 1):
            t = i / n
            c = table[min(255, int(t * 255))]
            if self._levels > 0:
                c = table[min(255, int(min(i, n - 1) / max(1, n - 1) * 255))] if n > 1 else c
            grad.setColorAt(t, QColor(int(c[0]), int(c[1]), int(c[2])))
        if self._levels > 0:
            for i in range(self._levels):
                c = table[min(255, int((i + 0.5) / self._levels * 255))]
                y1 = bottom - (bottom - top) * i / self._levels
                y2 = bottom - (bottom - top) * (i + 1) / self._levels
                p.fillRect(QRectF(x0, y2, bw, y1 - y2), QColor(int(c[0]), int(c[1]), int(c[2])))
        else:
            p.fillRect(QRectF(x0, top, bw, bottom - top), grad)
        p.setPen(QPen(QColor(pal.BORDER), 1))
        p.drawRect(QRectF(x0, top, bw, bottom - top))
        p.setPen(text)
        span = self._vmax - self._vmin
        for t in nice_ticks(self._vmin, self._vmax, 6):
            y = bottom - (t - self._vmin) / span * (bottom - top) if span else bottom
            p.drawLine(int(x0 + bw), int(y), int(x0 + bw + 4), int(y))
            p.drawText(int(x0 + bw + 6), int(y + fm.ascent() / 2 - 1), format_tick(t, span))
        p.drawText(QRectF(0, 2, self.width(), 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   self._label)
        p.end()


class LogView(QPlainTextEdit):
    """Timestamped log lines; warnings and errors coloured."""

    MAX_LINES = 3000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_LINES)
        f = QFont("Consolas")
        f.setStyleHint(QFont.StyleHint.Monospace)
        f.setPointSizeF(9.0)
        self.setFont(f)
        self.show_debug = False

    def append_line(self, message: str, level: str = "info") -> None:
        if level == "debug" and not self.show_debug:
            return
        pal = theme.current()
        color = {"warning": pal.WARNING, "error": pal.DANGER, "debug": pal.TEXT_MUTED}.get(level, pal.TEXT)
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        safe = message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        self.appendHtml(f'<span style="color:{pal.TEXT_MUTED}">{ts}</span> <span style="color:{color}">{safe}</span>')
        self.moveCursor(QTextCursor.MoveOperation.End)


class IntField(QSpinBox):
    """Spin box that ignores the mouse wheel unless focused (no accidental edits while scrolling)."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoWheelCombo(QComboBox):
    """Combo box that does not widen its panel to its longest item and ignores the wheel unless focused."""

    def __init__(self, parent: QWidget | None = None, min_chars: int = 8) -> None:
        super().__init__(parent)
        self.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(min_chars)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()
