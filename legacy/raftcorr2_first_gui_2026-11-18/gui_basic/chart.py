"""Time series of the probes (matplotlib embedded in Qt)."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..post.probes import extensometer, probe_series
from ..viz.fields import ALL_FIELDS
from . import theme
from .state import AppState


class ProbeChart(QWidget):
    frame_clicked = Signal(int)

    def __init__(self, state: AppState, metric_source=None) -> None:
        super().__init__()
        self.state = state
        self._metric_source = metric_source  # callable -> metric name
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        bar = QHBoxLayout()
        self.title = QLabel("")
        self.title.setProperty("role", "secondary")
        self.ext = QCheckBox("Lines as extensometers (strain (L - L0)/L0)")
        self.ext.toggled.connect(self.refresh)
        bar.addWidget(self.title)
        bar.addStretch(1)
        bar.addWidget(self.ext)
        lay.addLayout(bar)
        self.fig = Figure(figsize=(6, 2.2), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        lay.addWidget(self.canvas, 1)
        self.canvas.mpl_connect("button_press_event", self._clicked)
        self._vline = None
        self._cache: dict = {}
        for sig in (state.probes_changed, state.results_changed, state.strain_changed, state.units_changed):
            sig.connect(self.refresh)
        state.view_changed.connect(self._view_changed)
        state.frame_changed.connect(self._move_cursor)
        self._last_field = None
        self.refresh()

    def _view_changed(self) -> None:
        if self.state.view.field != self._last_field:
            self.refresh()

    def _style(self) -> None:
        p = theme.current()
        self.fig.set_facecolor(p.BG_WINDOW)
        self.ax.set_facecolor(p.BG_WINDOW)
        for s in self.ax.spines.values():
            s.set_color(p.BORDER)
        self.ax.tick_params(colors=p.TEXT_SECONDARY, labelsize=9)
        self.ax.xaxis.label.set_color(p.TEXT_SECONDARY)
        self.ax.yaxis.label.set_color(p.TEXT_SECONDARY)

    def refresh(self, *_a) -> None:
        st = self.state
        key = st.view.field
        self._last_field = key
        self.ax.clear()
        self._style()
        self._vline = None
        spec = ALL_FIELDS.get(key)
        if st.provider is None or not st.probes or spec is None:
            self.title.setText("Place probes on the canvas to plot their time series." if st.provider is not None
                               else "Run the correlation to plot probe time series.")
            self.canvas.draw_idle()
            return
        metric = self._metric_source() if self._metric_source else "mean"
        units = st.units
        frames = np.arange(st.series.n_frames)
        t = frames / units.frame_rate if units.frame_rate else frames
        any_line = False
        for p in st.probes:
            if self.ext.isChecked() and p.kind == "line":
                _L, y = extensometer(p, st.series)
                label = f"{p.label} strain"
            else:
                if spec.is_strain and not (st.strain is not None and key in st.strain.components):
                    continue
                y = probe_series(p, st.provider, key, metric) * units.factor(spec)
                label = p.label
            ok = np.isfinite(y)
            if ok.any():
                self.ax.plot(t[ok], y[ok], "-o", ms=3, lw=1.4, color=p.color, label=label)
                any_line = True
        self.ax.set_xlabel("time (s)" if units.frame_rate else "frame")
        ylab = "strain (-)" if self.ext.isChecked() and not spec.is_strain else f"{spec.label} ({units.unit(spec)})"
        self.ax.set_ylabel(ylab if len(ylab) < 40 else f"{key} ({units.unit(spec)})")
        if any_line:
            self.ax.legend(fontsize=8, frameon=False, ncol=min(4, len(st.probes)), loc="best")
        self.ax.grid(True, alpha=0.25)
        self.title.setText(f"{spec.label}; areas and lines: {metric}")
        self.fig.tight_layout(pad=0.6)
        self._move_cursor(st.current_frame)

    def _move_cursor(self, k: int) -> None:
        if self.state.series is None:
            return
        units = self.state.units
        x = k / units.frame_rate if units.frame_rate else k
        if self._vline is None:
            self._vline = self.ax.axvline(x, color=theme.current().ACCENT, lw=1, ls="--")
        else:
            self._vline.set_xdata([x, x])
        self.canvas.draw_idle()

    def _clicked(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or self.state.series is None:
            return
        units = self.state.units
        k = int(round(event.xdata * units.frame_rate)) if units.frame_rate else int(round(event.xdata))
        self.frame_clicked.emit(int(np.clip(k, 0, self.state.series.n_frames - 1)))

    def save_png(self, path: str) -> None:
        self.fig.savefig(path, dpi=200)
