"""Display field toggle (pyALDIC style): Disp U / Disp V / |U| / Residual."""

from PySide6.QtWidgets import QGridLayout, QPushButton, QWidget

from raftcorr.gui.app_state import AppState
from raftcorr.gui.theme_manager import themed


class FieldSelector(QWidget):
    """Exclusive toggle buttons, one per displayed field."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._buttons: dict[str, QPushButton] = {}
        tips = {
            "disp_u": self.tr("Horizontal displacement (+ right)"),
            "disp_v": self.tr("Vertical displacement (+ down)"),
            "disp_magnitude": self.tr("Displacement magnitude"),
            "residual": self.tr("Intensity residual: local RMS difference between the reference image and\n"
                                "the deformed image mapped back by the displacement. Large where the\n"
                                "tracking failed."),
        }
        for i, (key, text) in enumerate((("disp_u", self.tr("Disp U")), ("disp_v", self.tr("Disp V")),
                                         ("disp_magnitude", self.tr("Magnitude")), ("residual", self.tr("Residual")))):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setToolTip(tips[key])
            btn.clicked.connect(lambda _c=False, k=key: self._select(k))
            layout.addWidget(btn, i // 2, i % 2)
            self._buttons[key] = btn
        state = AppState.instance()
        self._buttons.get(state.display_field, self._buttons["disp_u"]).setChecked(True)
        self._update_styles()
        state.display_changed.connect(self._sync)

    def _select(self, field: str) -> None:
        for k, b in self._buttons.items():
            b.setChecked(k == field)
        self._update_styles()
        AppState.instance().set_display_field(field)

    def _sync(self) -> None:
        field = AppState.instance().display_field
        if field in self._buttons and not self._buttons[field].isChecked():
            for k, b in self._buttons.items():
                b.setChecked(k == field)
            self._update_styles()

    def _update_styles(self) -> None:
        for btn in self._buttons.values():
            if btn.isChecked():
                themed(btn, "background: {ACCENT}; color: white; "
                            "border: none; border-radius: 4px; font-weight: bold;")
            else:
                themed(btn, "background: {BG_INPUT}; color: {TEXT_SECONDARY}; "
                            "border: 1px solid {BORDER}; border-radius: 4px;")
