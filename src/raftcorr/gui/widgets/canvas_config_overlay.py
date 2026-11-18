"""Top-left canvas overlay that summarises the run configuration (ported from pyALDIC).

A small always-visible legend of the three choices that shape a run: tracking mode, model and
refinement iterations. It follows the sidebar live, so a glance at the canvas tells what Run
will do. Invisible until at least two images are loaded.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from raftcorr.gui.app_state import AppState
from raftcorr.gui.theme_manager import themed
from raftcorr.i18n import tr_args
from raftcorr.models.zoo import MODELS


class CanvasConfigOverlay(QFrame):
    """Semi-transparent card pinned to the canvas top-left: ``Mode``, ``Model``, ``Iterations``."""

    MARGIN = 12  # px from the canvas edge

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.instance()
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # Scoped to the card itself: a rule without a selector would box every row inside it.
        self.setObjectName("canvasConfigOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        themed(self,
            "QWidget#canvasConfigOverlay {{ background-color: {OVERLAY_BG}; "
            "border: 1px solid {BORDER}; "
            "border-radius: 6px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)

        self._mode_lbl = self._make_row(layout, self.tr("Mode"))
        self._model_lbl = self._make_row(layout, self.tr("Model"))
        self._iters_lbl = self._make_row(layout, self.tr("Iterations"))
        self.adjustSize()

        self._state.params_changed.connect(self._refresh)
        self._state.images_changed.connect(self._refresh)
        self._refresh()

    def _make_row(self, layout: QVBoxLayout, key: str) -> QLabel:
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)
        key_lbl = QLabel(key.upper())
        themed(key_lbl,
            "color: {TEXT_SECONDARY}; font-size: 9px; "
            "font-weight: bold; letter-spacing: 1px; "
            "background: transparent;"
        )
        row_layout.addWidget(key_lbl)
        val_lbl = QLabel("—")
        themed(val_lbl,
            "color: {TEXT_PRIMARY}; font-size: 12px; "
            "font-weight: bold; background: transparent;"
        )
        row_layout.addWidget(val_lbl)
        layout.addWidget(row)
        return val_lbl

    # -- state -> text -------------------------------------------------------------------------
    def _refresh(self) -> None:
        if len(self._state.image_files) < 2:
            self.setVisible(False)
            return
        self._mode_lbl.setText(self._format_mode())
        self._model_lbl.setText(self._format_model())
        self._iters_lbl.setText(str(int(self._state.iters)))
        self.adjustSize()
        self.setVisible(True)
        self.reposition()

    def _format_mode(self) -> str:
        st = self._state
        if st.tracking_mode != "incremental":
            return self.tr("Accumulative")
        if st.inc_ref_mode == "every_n":
            return tr_args(self.tr("Incremental, every %1 frames"), int(st.inc_ref_interval))
        if st.inc_ref_mode == "custom":
            return self.tr("Custom reference frames")
        return self.tr("Incremental")

    def _format_model(self) -> str:
        spec = MODELS.get(str(self._state.model))
        return str(self._state.model) if spec is None else spec.name

    # -- geometry ------------------------------------------------------------------------------
    def reposition(self) -> None:
        """Re-anchor to the parent's top-left corner (called when the viewport resizes)."""
        if self.parentWidget() is not None:
            self.move(self.MARGIN, self.MARGIN)


__all__ = ["CanvasConfigOverlay"]
