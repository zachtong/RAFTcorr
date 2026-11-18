"""Dynamic hint above the Region of Interest toolbar.

Shows which frames need a Region of Interest based on the current
workflow type (tracking mode + reference-update policy). Updates live
whenever the workflow parameters change so users never have to guess
whether they should draw a region on frame 1 only or on several
reference frames.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget

from raftcorr.gui.app_state import AppState
from raftcorr.gui.theme_manager import themed
from raftcorr.i18n import tr_args


# A style template (theme_manager.themed fills the {FIELD}s)
_HINT_STYLE = (
    "QLabel {{ "
    "color: {TEXT_SECONDARY}; "
    "background: {BG_INPUT}; "
    "border: 1px solid {BORDER}; "
    "border-left: 3px solid {ACCENT}; "
    "border-radius: 3px; "
    "padding: 6px 8px; "
    "font-size: 11px; "
    "}}"
)


class ROIHint(QLabel):
    """Info line showing which frames need a Region of Interest.

    Listens to ``AppState.params_changed`` and rewrites itself whenever
    ``tracking_mode``, ``inc_ref_mode``, ``inc_ref_interval``, or
    ``inc_custom_refs`` change. Also listens to ``images_changed`` so
    the frame count in the hint stays current.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = AppState.instance()
        themed(self, _HINT_STYLE)
        self.setWordWrap(True)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )
        self._state.params_changed.connect(self._refresh)
        self._state.images_changed.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        """Recompute the hint text based on current state."""
        s = self._state
        n = len(s.image_files)

        # No images yet: skip mode-specific advice, just tell the user
        # they need to load images first.
        if n == 0:
            self.setText(self.tr(
                "Load images first, then draw a Region of Interest on "
                "frame 1."
            ))
            return

        if s.tracking_mode == "accumulative":
            self.setText(self.tr(
                "<b>Accumulative mode</b> — only frame 1 needs a "
                "Region of Interest. All later frames are compared "
                "against it directly."
            ))
            return

        # Incremental (RAFTcorr): only frame 1 needs a region. The pipeline carries it to each
        # reference frame with the displacement measured so far; a mask drawn on another frame
        # replaces the carried region there and removes points that move into excluded areas.
        refs = sorted(s.reference_frames())
        refs_display = [str(r + 1) for r in refs]
        preview = ", ".join(refs_display[:8]) + (", …" if len(refs_display) > 8 else "")
        self.setText(tr_args(self.tr(
            "<b>Incremental</b> — only frame 1 needs a Region of Interest. "
            "RAFTcorr carries it to the reference frames (<b>%1</b>) with the "
            "measured displacement. Draw on another frame only to replace that "
            "region or to leave out cracks and holes."
        ), preview))
        return

        # Fallback — unrecognized mode
        self.setText(self.tr(
            "Draw a Region of Interest on frame 1."
        ))
