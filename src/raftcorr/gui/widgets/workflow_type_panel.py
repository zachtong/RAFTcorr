"""Workflow-type parameter panel.

Holds the parameters that define WHAT kind of analysis the user is
running — which images are compared against which, which solver runs,
and how often the reference frame refreshes. These choices constrain
the rest of the workflow: accumulative mode compares every frame to
frame 1, so only frame 1 needs a Region of Interest; incremental mode
with a Reference Update schedule needs a Region of Interest at every
ref frame.

Surfaced above the Region of Interest section in the left sidebar so
the user sees which regions are required BEFORE they start drawing.

Exposes:
* ``tracking_mode`` -> AppState.tracking_mode  ("incremental" or
  "accumulative")
* ``use_admm`` / ``admm_max_iter`` -> ``AppState`` (Solver choice)
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
from raftcorr.i18n import tr_args


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
            "Incremental: each frame is compared with a later reference frame\n"
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
        from raftcorr.models.zoo import MODELS, available_models

        self._model = QComboBox()
        avail = available_models()
        summaries = {  # raftcorr.models.zoo summaries, translatable
            "s1": self.tr("finest spatial resolution; small displacements"),
            "s2": self.tr("fine resolution, several times faster than s1; small to moderate displacements"),
            "s4": self.tr("balanced resolution and capture range"),
            "s8": self.tr("largest capture range, coarsest resolution; large motion and rotation"),
        }
        for key, spec in MODELS.items():
            self._model.addItem(f"RAFTcorr-{key}  ({spec.lookup_radius_px} px)", key)
            tip = tr_args(self.tr("%1: %2.\nReach (nominal lookup radius 32 s): %3 px."),
                          spec.name, summaries.get(key, spec.summary), spec.lookup_radius_px)
            if avail.get(key) is None:
                tip += "\n" + self.tr("Weights not installed.")
            self._model.setItemData(self._model.count() - 1, tip, Qt.ItemDataRole.ToolTipRole)
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

        self._suggest_btn = QPushButton(self.tr("Suggest a model"))
        self._suggest_btn.setToolTip(self.tr(
            "Measure the motion of the sequence quickly with s8 and pick the\n"
            "finest model whose reach covers it."
        ))
        self._suggest_btn.clicked.connect(self.suggest_requested)
        layout.addWidget(self._suggest_btn)
        self._advice = QLabel("")
        self._advice.setWordWrap(True)
        self._advice.setStyleSheet("font-size: 11px;")
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
        self._custom_edit.setPlaceholderText("e.g. 0, 5, 10, 20")
        self._custom_edit.setToolTip(self.tr(
            "Comma-separated frame indices to use as reference frames "
            "(0-based)"
        ))
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

    def _sync_from_state(self) -> None:
        """Follow changes made elsewhere (model suggestion, project loading)."""
        state = AppState.instance()
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

    def _on_custom_refs_changed(self) -> None:
        """Parse comma-separated frame indices and store in AppState."""
        state = AppState.instance()
        text = self._custom_edit.text().strip()
        if not text:
            state.inc_custom_refs = []
            return
        try:
            refs = [int(x.strip()) for x in text.split(",") if x.strip()]
            state.inc_custom_refs = sorted(set(refs))
        except ValueError:
            state.log_message.emit(
                "Invalid frame indices — use comma-separated numbers", "warn"
            )
        state.params_changed.emit()

    def eventFilter(self, obj, event):
        """Block wheel events on unfocused spinboxes and combos."""
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return super().eventFilter(obj, event)
