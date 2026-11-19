"""Workflow-type parameter panel.

Holds the parameters that define WHAT kind of analysis the user is
running — which images are compared against which, which model
tracks them, and how often the reference frame refreshes. These choices constrain
the rest of the workflow: accumulative mode compares every frame to
frame 1, so only frame 1 needs a Region of Interest; incremental mode
with a Reference Update schedule needs a Region of Interest at every
ref frame.

Surfaced above the Region of Interest section in the left sidebar so
the user sees which regions are required BEFORE they start drawing.

Exposes:
* ``tracking_mode`` -> AppState.tracking_mode  ("incremental" or
  "accumulative")
* ``model`` -> AppState.model (with Suggest a Model and its advice card)
* ``inc_ref_mode``   -> AppState.inc_ref_mode
* ``inc_ref_interval``, ``inc_custom_refs`` -> AppState
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from raftcorr.gui.app_state import AppState
from raftcorr.gui.icons import icon_suggest
from raftcorr.gui.theme_manager import set_icon, themed
from raftcorr.gui.widgets.roi_hint import HINT_STYLE
from raftcorr.i18n import tr_args

#: Border of a field whose text was refused (the run keeps the last valid value).
_INVALID_STYLE = "QLineEdit {{ border: 1px solid {DANGER}; }}"


class _ModelCombo(QComboBox):
    """A combo box that says when its list is about to open (to refresh the item tooltips)."""

    popup_about_to_show = Signal()

    def showPopup(self) -> None:  # noqa: N802 - Qt override
        self.popup_about_to_show.emit()
        super().showPopup()


class WorkflowTypePanel(QWidget):
    """Panel for high-level workflow parameters (tracking, model, refs)."""

    suggest_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        state = AppState.instance()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 0, 4, 0)

        # --- Tracking Mode ---
        # Display text is translated; stable English code stored as userData
        # so state-sync doesn't break when language changes.
        self._tracking_mode = QComboBox()
        self._tracking_mode.addItem(self.tr("Accumulative"), "accumulative")
        self._tracking_mode.addItem(self.tr("Incremental"), "incremental")
        _idx = self._tracking_mode.findData(state.tracking_mode)
        if _idx >= 0:
            self._tracking_mode.setCurrentIndex(_idx)
        self._tracking_mode.setToolTip(self.tr(
            "Accumulative: every frame is compared directly with frame 1.\n"
            "Preferred whenever the motion stays within the model's reach:\n"
            "no error accumulates from frame to frame.\n\n"
            "Incremental: each frame is compared with an earlier reference frame\n"
            "and the steps are composed. Needed when the motion from frame 1\n"
            "exceeds the model's reach (large stretch or rotation); every step\n"
            "adds its error."
        ))
        row = QHBoxLayout()
        lbl = QLabel(self.tr("Tracking Mode"))
        lbl.setFixedWidth(120)
        row.addWidget(lbl)
        row.addWidget(self._tracking_mode)
        layout.addLayout(row)

        # --- Model: the four RAFTcorr networks differ only in the encoder downsampling s ---
        from raftcorr.models.zoo import MODELS

        self._model = _ModelCombo()
        for key, spec in MODELS.items():
            self._model.addItem(f"RAFTcorr-{key} ({spec.lookup_radius_px} px)", key)
        self._refresh_model_tooltips()
        # Weights arrive while the application runs (Models window, first-use download, import):
        # whether they are installed is checked again each time the list opens.
        self._model.popup_about_to_show.connect(self._refresh_model_tooltips)
        _midx = self._model.findData(state.model)
        self._model.setCurrentIndex(_midx if _midx >= 0 else 2)
        self._model.setToolTip(self.tr(
            "RAFTcorr model. A smaller s resolves finer detail; a larger s\n"
            "reaches further (the coarsest correlation level looks 32 s px away).\n\n"
            "s1, s2: small displacements, strain concentrations.\n"
            "s4: balanced.\n"
            "s8: large motion or rotation, or no prior knowledge."
        ))
        model_row = QHBoxLayout()
        model_lbl = QLabel(self.tr("Model"))
        model_lbl.setFixedWidth(120)
        model_row.addWidget(model_lbl)
        model_row.addWidget(self._model)
        layout.addLayout(model_row)

        # A sidebar button as pyALDIC's (Title Case, icon). No fixed height: the theme's
        # min-height (36 px) would exceed it and the button would overflow its section.
        self._suggest_btn = QPushButton(self.tr("Suggest a Model"))
        set_icon(self._suggest_btn, icon_suggest)
        self._suggest_btn.setToolTip(self.tr(
            "Measure the motion of the sequence quickly with s8 and pick the\n"
            "finest model whose reach covers it."
        ))
        self._suggest_btn.clicked.connect(self.suggest_requested)
        layout.addWidget(self._suggest_btn)
        self._advice = QLabel("")
        self._advice.setWordWrap(True)
        themed(self._advice, HINT_STYLE)      # the card of the Region of Interest hint
        self._advice.setVisible(False)
        layout.addWidget(self._advice)

        # ADMM iterations moved to ADVANCED > AdvancedTuningWidget —
        # it's a numerical tuning knob, not a workflow choice.

        # --- Incremental sub-panel (Reference Update) ---
        self._inc_panel = QWidget()
        inc_layout = QVBoxLayout(self._inc_panel)
        inc_layout.setContentsMargins(12, 0, 0, 0)
        inc_layout.setSpacing(4)

        self._ref_mode = QComboBox()
        # Display translated; stable code stored as userData.
        self._ref_mode.addItem(self.tr("Every Frame"), "every_frame")
        self._ref_mode.addItem(self.tr("Every N Frames"), "every_n")
        self._ref_mode.addItem(self.tr("Custom Frames"), "custom")
        self._ref_mode.setCurrentIndex(0)
        self._ref_mode.setToolTip(self.tr(
            "When the reference frame refreshes during incremental "
            "tracking.\n"
            "Every Frame: reset reference every frame (smallest per-step "
            "displacement,\nmost robust for large deformation).\n"
            "Every N Frames: reset every N frames (balance speed vs "
            "robustness).\n"
            "Custom Frames: user-defined list of reference frame indices."
        ))
        ref_row = QHBoxLayout()
        ref_lbl = QLabel(self.tr("Reference Update"))
        ref_lbl.setFixedWidth(108)
        ref_row.addWidget(ref_lbl)
        ref_row.addWidget(self._ref_mode)
        inc_layout.addLayout(ref_row)

        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(2, 100)
        self._interval_spin.setValue(state.inc_ref_interval)
        self._interval_spin.setToolTip(self.tr(
            "Update reference every N frames"))
        self._interval_row = QHBoxLayout()
        self._interval_lbl = QLabel(self.tr("Interval"))
        self._interval_lbl.setFixedWidth(108)
        self._interval_row.addWidget(self._interval_lbl)
        self._interval_row.addWidget(self._interval_spin)
        inc_layout.addLayout(self._interval_row)

        self._custom_edit = QLineEdit()
        self._custom_edit.setPlaceholderText(self.tr("e.g. 5, 10, 20"))
        self._custom_tip = self.tr(
            "Frame numbers (as in the image list) to use as reference frames, separated by "
            "commas. Frame 1 is always the first reference."
        )
        self._custom_edit.setToolTip(self._custom_tip)
        self._custom_row = QHBoxLayout()
        self._custom_lbl = QLabel(self.tr("Reference Frames"))
        self._custom_lbl.setFixedWidth(108)
        self._custom_row.addWidget(self._custom_lbl)
        self._custom_row.addWidget(self._custom_edit)
        inc_layout.addLayout(self._custom_row)

        layout.addWidget(self._inc_panel)

        # --- Wiring ---
        self._tracking_mode.currentIndexChanged.connect(
            self._on_tracking_mode_changed
        )
        self._model.currentIndexChanged.connect(self._on_model_changed)
        state.params_changed.connect(self._sync_from_state)
        state.images_changed.connect(self._sync_from_state)     # the valid reference frames change
        self._ref_mode.currentIndexChanged.connect(self._on_ref_mode_changed)
        self._interval_spin.valueChanged.connect(
            lambda v: state.set_param("inc_ref_interval", v)
        )
        self._custom_edit.editingFinished.connect(self._on_custom_refs_changed)

        # Initial visibility
        self._on_tracking_mode_changed(self._tracking_mode.currentIndex())
        self._on_ref_mode_changed(0)

        # Block scroll-wheel changes on unfocused widgets
        for widget in [
            self._tracking_mode, self._model,
            self._ref_mode, self._interval_spin,
        ]:
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            widget.installEventFilter(self)

    # ------------------------------------------------------------------

    def _on_model_changed(self, _index: int) -> None:
        state = AppState.instance()
        key = self._model.currentData()
        if key and key != state.model:
            state.set_param("model", key)

    def _refresh_model_tooltips(self) -> None:
        """Item tooltips: what each model is for, and whether its weights are installed."""
        from raftcorr.models import zoo

        avail = zoo.available_models()
        summaries = {  # raftcorr.models.zoo summaries, translatable
            "s1": self.tr("finest spatial resolution; small displacements"),
            "s2": self.tr("fine resolution, several times faster than s1; small to moderate displacements"),
            "s4": self.tr("balanced resolution and capture range"),
            "s8": self.tr("largest capture range, coarsest resolution; large motion and rotation"),
        }
        for i in range(self._model.count()):
            key = self._model.itemData(i)
            spec = zoo.MODELS[key]
            tip = tr_args(self.tr("%1: %2.\nReach (nominal lookup radius 32 s): %3 px."),
                          spec.name, summaries.get(key, spec.summary), spec.lookup_radius_px)
            if avail.get(key) is None:
                tip += "\n" + self.tr("Weights not installed.")
            self._model.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)

    def _sync_from_state(self) -> None:
        """Follow changes made elsewhere (model suggestion, project loading, new images)."""
        state = AppState.instance()
        self._drop_out_of_range_refs()
        for combo, value in ((self._model, state.model), (self._tracking_mode, state.tracking_mode),
                             (self._ref_mode, state.inc_ref_mode)):
            idx = combo.findData(value)
            if idx >= 0 and idx != combo.currentIndex():
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)
        self._inc_panel.setVisible(state.tracking_mode == "incremental")
        mode = state.inc_ref_mode
        self._interval_spin.setVisible(mode == "every_n")
        self._interval_lbl.setVisible(mode == "every_n")
        self._custom_edit.setVisible(mode == "custom")
        self._custom_lbl.setVisible(mode == "custom")
        if self._interval_spin.value() != state.inc_ref_interval:
            self._interval_spin.blockSignals(True)
            self._interval_spin.setValue(state.inc_ref_interval)
            self._interval_spin.blockSignals(False)
        shown = ", ".join(str(k + 1) for k in state.inc_custom_refs)   # 1-based, as on screen
        if not self._custom_edit.hasFocus() and self._custom_edit.text() != shown:
            self._custom_edit.setText(shown)
            self._mark_custom_refs("")                 # the field shows what the run uses again
        self._advice.setText(state.model_advice)
        self._advice.setVisible(bool(state.model_advice))

    def _on_tracking_mode_changed(self, _index: int) -> None:
        """Set tracking mode and swap init-guess default + inc sub-panel."""
        state = AppState.instance()
        mode = self._tracking_mode.currentData() or "accumulative"
        state.set_param("tracking_mode", mode)
        self._inc_panel.setVisible(mode == "incremental")

    def _on_ref_mode_changed(self, _index: int) -> None:
        """Toggle Interval / Custom row visibility based on ref mode."""
        state = AppState.instance()
        mode = self._ref_mode.currentData() or "every_frame"
        state.set_param("inc_ref_mode", mode)
        self._interval_spin.setVisible(mode == "every_n")
        self._interval_lbl.setVisible(mode == "every_n")
        self._custom_edit.setVisible(mode == "custom")
        self._custom_lbl.setVisible(mode == "custom")

    def custom_refs_invalid(self) -> bool:
        """True while the Reference Frames field holds text that was refused."""
        return bool(AppState.instance().inc_custom_refs_error)

    def _mark_custom_refs(self, problem: str) -> None:
        """Mark the Reference Frames field as refused (``problem`` says why) or as valid ("")."""
        AppState.instance().inc_custom_refs_error = problem
        themed(self._custom_edit, _INVALID_STYLE if problem else "")
        self._custom_edit.setToolTip(problem or self._custom_tip)

    @staticmethod
    def _last_valid_ref() -> int | None:
        """The last frame (1-based) that can be a reference, N - 1; None before images are loaded."""
        n = len(AppState.instance().image_files)
        return n - 1 if n >= 2 else None

    def _drop_out_of_range_refs(self) -> None:
        """Keep reference frames the run cannot use (fewer images, an older project) out of it."""
        state = AppState.instance()
        last = self._last_valid_ref()
        if last is None:
            return
        dropped = [k for k in state.inc_custom_refs if k + 1 > last]
        if not dropped:
            return
        state.inc_custom_refs = [k for k in state.inc_custom_refs if k + 1 <= last]
        state.log_message.emit(tr_args(self.tr(
            "Reference frames %1 dropped: a reference must be one of the frames 1 to %2 (the last frame "
            "cannot be a reference)."), ", ".join(str(k + 1) for k in dropped), last), "warn")
        state.params_changed.emit()

    def _on_custom_refs_changed(self) -> None:
        """Parse the frame numbers (1-based, as on screen) into 0-based reference indices.

        Refused text is marked and explained in the log; the run keeps the last valid frames.
        """
        state = AppState.instance()
        text = self._custom_edit.text().strip()
        if not text:
            self._mark_custom_refs("")
            state.inc_custom_refs = []
            state.params_changed.emit()
            return
        try:
            numbers = [int(x.strip()) for x in text.split(",") if x.strip()]
        except ValueError:
            numbers = []
        last = self._last_valid_ref()
        if not numbers or min(numbers) < 1:
            problem = self.tr("Invalid reference frames: use frame numbers from 1 on, separated by commas.")
        elif last is not None and max(numbers) > last:
            beyond = ", ".join(str(n) for n in sorted(set(numbers)) if n > last)
            problem = tr_args(self.tr(
                "Invalid reference frames %1: use frames 1 to %2, separated by commas (the last frame "
                "cannot be a reference)."), beyond, last)
        else:
            problem = ""
        self._mark_custom_refs(problem)
        if problem:
            state.log_message.emit(problem, "warn")
            return
        state.inc_custom_refs = sorted({n - 1 for n in numbers if n > 1})
        state.params_changed.emit()

    def eventFilter(self, obj, event):
        """Block wheel events on unfocused spinboxes and combos."""
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)
