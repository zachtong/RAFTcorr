"""Field selector for the strain post-processing window.

Exposes displacement, derived displacement, and strain fields in two
clearly separated sub-grids:

    DISPLACEMENT
        [U]         [V]
        [Magnitude] [Velocity]

    STRAIN
        [εxx]       [εyy]       [εxy]
        [ε₁]        [ε₂]        [max shear]
        [von Mises] [ω rot]     [ε₁ angle]
        [ΔA/A₀]

Displacement fields are read from the displacement series and are visible
*before* running Compute Strain. Strain fields require a completed strain
computation.

Field names are pyALDIC's (``disp_u``, ``strain_exx`` ...); :func:`field_key`
maps them to the keys of :class:`raftcorr.viz.fields.FieldProvider`
(``u``, ``exx`` ...). ``velocity`` is the speed ``|du/dt|``.
"""

from __future__ import annotations

from PySide6.QtCore import QT_TRANSLATE_NOOP, QCoreApplication, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from raftcorr.gui.support.fields import provider_key
from raftcorr.gui.theme_manager import themed
from raftcorr.viz.fields import ALL_FIELDS as PROVIDER_FIELDS

# ---- Displacement section -------------------------------------------------

DISP_FIELD_NAMES: tuple[str, ...] = (
    "disp_u",
    "disp_v",
    "disp_magnitude",
    "velocity",
)

# ---- Strain section -------------------------------------------------------

STRAIN_FIELD_NAMES: tuple[str, ...] = (
    "strain_exx",
    "strain_eyy",
    "strain_exy",
    "strain_principal_max",
    "strain_principal_min",
    "strain_maxshear",
    "strain_von_mises",
    "strain_rotation",
    "strain_principal_angle",
    "strain_area_change",
)

# ---- Combined ---------------------------------------------------------------

FIELD_NAMES: tuple[str, ...] = DISP_FIELD_NAMES + STRAIN_FIELD_NAMES

#: GUI field name -> FieldProvider key.
FIELD_KEYS: dict[str, str] = {name: provider_key(name) for name in FIELD_NAMES}
FIELD_KEYS["velocity"] = "speed"

from raftcorr.gui.support.fields import STRAIN_FIELDS, strain_symbol  # noqa: E402

_FIELD_LABELS: dict[str, str] = {
    "disp_u":                 "U",               # U
    "disp_v":                 "V",               # V
    "disp_magnitude":         QT_TRANSLATE_NOOP("StrainFieldSelector", "Magnitude"),
    "velocity":               QT_TRANSLATE_NOOP("StrainFieldSelector", "Velocity"),
    # One label per component everywhere (raftcorr.post.strain.COMPONENT_LABELS).
    **{name: strain_symbol(name) for name in STRAIN_FIELDS},
}

# (row, col, colspan) inside each sub-grid, 2 columns for displacement and
# 3 columns for strain.
_DISP_POSITIONS: dict[str, tuple[int, int, int]] = {
    "disp_u":         (0, 0, 1),
    "disp_v":         (0, 1, 1),
    "disp_magnitude": (1, 0, 1),
    "velocity":       (1, 1, 1),
}

_STRAIN_POSITIONS: dict[str, tuple[int, int, int]] = {
    "strain_exx":             (0, 0, 1),
    "strain_eyy":             (0, 1, 1),
    "strain_exy":             (0, 2, 1),
    "strain_principal_max":   (1, 0, 1),
    "strain_principal_min":   (1, 1, 1),
    "strain_maxshear":        (1, 2, 1),
    "strain_von_mises":       (2, 0, 1),
    "strain_rotation":        (2, 1, 1),
    "strain_principal_angle": (2, 2, 1),
    "strain_area_change":     (3, 0, 1),
}

_ANGLE_FIELDS: frozenset[str] = frozenset({"strain_rotation", "strain_principal_angle"})


def field_key(name: str) -> str:
    """FieldProvider key of a field name (a pyALDIC name or a provider key).

    Raises:
        ValueError: If *name* is neither.
    """
    if name in FIELD_KEYS:
        return FIELD_KEYS[name]
    key = provider_key(name)
    if key in PROVIDER_FIELDS:
        return key
    raise ValueError(f"Unknown field '{name}'. Allowed: {FIELD_NAMES}")


def is_disp_field(name: str) -> bool:
    """True for the displacement-family fields (available before Compute Strain)."""
    try:
        return not PROVIDER_FIELDS[field_key(name)].is_strain
    except (KeyError, ValueError):
        return False


def field_label(name: str) -> str:
    """Short display name of a field (button text, colorbar title)."""
    if name in _FIELD_LABELS:
        # Symbols (εxx, U) come back unchanged; words (Magnitude, Velocity) are translated.
        return QCoreApplication.translate("StrainFieldSelector", _FIELD_LABELS[name])
    spec = PROVIDER_FIELDS.get(provider_key(name))
    return spec.label if spec is not None else name


def field_colorbar_label(
    field_name: str,
    use_physical: bool = False,
    pixel_unit: str = "mm",
    frame_rate: float = 1.0,
) -> str:
    """Return a colorbar label string with appropriate units.

    Displacement fields show ``px`` or the chosen *pixel_unit* depending on
    *use_physical*. Velocity shows ``px/frame``, ``<unit>/s`` (with a frame
    rate) or ``<unit>/frame`` -- the units of
    :meth:`raftcorr.gui.app_state.AppState.units`, so the label always names
    the unit of the drawn numbers. Strains are dimensionless (no suffix);
    rotation and the principal angle are in degrees.
    """
    base = field_label(field_name)
    try:
        kind = PROVIDER_FIELDS[field_key(field_name)].kind
    except (KeyError, ValueError):
        kind = "strain"
    if kind == "displacement":
        return f"{base} ({pixel_unit})" if use_physical else f"{base} (px)"
    if kind == "velocity":
        if not use_physical:
            return f"{base} (px/frame)"
        return f"{base} ({pixel_unit}/s)" if frame_rate > 0 else f"{base} ({pixel_unit}/frame)"
    if kind == "angle" or field_name in _ANGLE_FIELDS:
        return f"{base} (deg)"
    if kind == "quality":
        return f"{base} (grey levels)"
    return base  # strain fields: dimensionless


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    themed(lbl,
        "color: {TEXT_SECONDARY}; font-size: 10px; "
        "font-weight: bold; letter-spacing: 1px; margin-top: 4px;"
    )
    return lbl


class StrainFieldSelector(QWidget):
    """Two-section exclusive selector for displacement and strain fields."""

    field_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # Single shared button group -- ensures cross-section mutual exclusion.
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

        # --- DISPLACEMENT section ---
        outer.addWidget(_section_label(self.tr("DISPLACEMENT")))
        disp_grid = QGridLayout()
        disp_grid.setContentsMargins(0, 0, 0, 0)
        disp_grid.setHorizontalSpacing(4)
        disp_grid.setVerticalSpacing(4)
        for name in DISP_FIELD_NAMES:
            row, col, span = _DISP_POSITIONS[name]
            self._add_button(name, disp_grid, row, col, span)
        outer.addLayout(disp_grid)

        # --- STRAIN section ---
        outer.addWidget(_section_label(self.tr("STRAIN")))
        strain_grid = QGridLayout()
        strain_grid.setContentsMargins(0, 0, 0, 0)
        strain_grid.setHorizontalSpacing(4)
        strain_grid.setVerticalSpacing(4)
        for name in STRAIN_FIELD_NAMES:
            row, col, span = _STRAIN_POSITIONS[name]
            self._add_button(name, strain_grid, row, col, span)
        # Equal columns, also for the last, partly empty row.
        for col in range(3):
            strain_grid.setColumnStretch(col, 1)
        outer.addLayout(strain_grid)

        self._set_tooltips()
        self._current: str = "disp_u"
        self._buttons[self._current].setChecked(True)
        self._update_styles()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current_field(self) -> str:
        """Return the currently selected field name."""
        return self._current

    def set_current_field(self, name: str) -> None:
        """Programmatically activate *name*. Emits ``field_changed`` only on
        an actual change.

        Raises:
            ValueError: If *name* is not a recognised field.
        """
        if name not in self._buttons:
            raise ValueError(
                f"Unknown field '{name}'. Allowed: {FIELD_NAMES}"
            )
        if name == self._current:
            return
        self._current = name
        self._buttons[name].setChecked(True)
        self._update_styles()
        self.field_changed.emit(name)

    def button(self, name: str) -> QPushButton:
        """The button of field *name* (tests, keyboard navigation)."""
        return self._buttons[name]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_tooltips(self) -> None:
        tips = {
            "disp_u": self.tr("Horizontal displacement u (+ right)"),
            "disp_v": self.tr("Vertical displacement v (+ down)"),
            "disp_magnitude": self.tr("Displacement magnitude |u|"),
            "velocity": self.tr(
                "Speed |du/dt|: central difference of the displacement over "
                "the neighbouring computed frames, times the frame rate"),
            "strain_exx": self.tr("Normal strain εxx"),
            "strain_eyy": self.tr("Normal strain εyy"),
            "strain_exy": self.tr(
                "Shear strain εxy (tensor component: half the engineering "
                "shear γxy)"),
            "strain_principal_max": self.tr("Major principal strain ε₁"),
            "strain_principal_min": self.tr("Minor principal strain ε₂"),
            "strain_maxshear": self.tr(
                "Maximum in-plane shear strain (ε₁ − ε₂) / 2 (tensor "
                "component: half the engineering shear)"),
            "strain_von_mises": self.tr(
                "Von Mises equivalent strain (incompressible material)"),
            "strain_rotation": self.tr(
                "Rigid rotation in degrees (polar decomposition), clockwise "
                "on screen positive"),
            "strain_principal_angle": self.tr(
                "Direction of ε₁ in degrees from +x, clockwise on screen "
                "positive"),
            "strain_area_change": self.tr(
                "Relative area change ΔA/A₀ = det F − 1"),
        }
        for name, tip in tips.items():
            self._buttons[name].setToolTip(tip)

    def _add_button(
        self,
        name: str,
        grid: QGridLayout,
        row: int,
        col: int,
        col_span: int,
    ) -> None:
        btn = QPushButton(field_label(name))
        btn.setCheckable(True)
        btn.setFixedHeight(28)
        grid.addWidget(btn, row, col, 1, col_span)
        self._group.addButton(btn)
        self._buttons[name] = btn
        btn.clicked.connect(
            lambda _checked=False, n=name: self._on_clicked(n)
        )

    def _on_clicked(self, name: str) -> None:
        if name == self._current:
            self._buttons[name].setChecked(True)
            return
        self._current = name
        self._update_styles()
        self.field_changed.emit(name)

    def _update_styles(self) -> None:
        for name, btn in self._buttons.items():
            active = name == self._current
            if active:
                themed(btn,
                    "background: {ACCENT}; color: white; "
                    "border: none; border-radius: 4px; font-weight: bold;"
                )
            else:
                themed(btn,
                    "background: {BG_INPUT}; "
                    "color: {TEXT_SECONDARY}; "
                    "border: 1px solid {BORDER}; border-radius: 4px;"
                )
