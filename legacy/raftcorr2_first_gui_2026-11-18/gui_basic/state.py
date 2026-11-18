"""Central application state with Qt signals.

Panels never talk to each other: they read the state, change it through its methods, and react to
its signals. Workers report back through the controllers, which update the state on the GUI
thread. Settings objects are frozen dataclasses replaced as a whole (``dataclasses.replace``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Signal

from ..config import RunConfig, StrainConfig
from ..io.images import ImageSequence
from ..post.probes import Probe
from ..results import DisplacementSeries, StrainSeries
from ..roi.masks import MaskHistory
from ..viz.colormaps import DEFAULT_COLORMAP
from ..viz.fields import FieldProvider, Units


@dataclass(frozen=True)
class FieldDisplay:
    """Colour settings of one field."""

    colormap: str = DEFAULT_COLORMAP
    auto: bool = True
    vmin: float = 0.0
    vmax: float = 1.0
    levels: int = 0  # 0 = continuous; otherwise discrete colour bands


@dataclass(frozen=True)
class ViewSettings:
    field: str = "u"
    configuration: str = "reference"   # reference | deformed (where the field is drawn)
    background: str = "auto"           # auto (matches the configuration) | frame | reference | none
    opacity: float = 0.75
    show_roi: bool = True
    show_overlay: bool = True
    range_scope: str = "frame"         # frame | sequence: auto range of one frame or of all frames
    fields: dict[str, FieldDisplay] = dc_field(default_factory=dict)

    def display(self, key: str) -> FieldDisplay:
        return self.fields.get(key, FieldDisplay())

    def with_display(self, key: str, **changes: Any) -> "ViewSettings":
        fields = dict(self.fields)
        fields[key] = replace(self.display(key), **changes)
        return replace(self, fields=fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field, "configuration": self.configuration, "background": self.background,
            "opacity": self.opacity, "show_roi": self.show_roi, "show_overlay": self.show_overlay,
            "range_scope": self.range_scope,
            "fields": {k: dict(v.__dict__) for k, v in self.fields.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ViewSettings":
        base = cls()
        fields = {k: FieldDisplay(**{kk: vv for kk, vv in v.items() if kk in FieldDisplay.__dataclass_fields__})
                  for k, v in (d.get("fields") or {}).items()}
        known = {k: d[k] for k in ("field", "configuration", "background", "opacity", "show_roi", "show_overlay",
                                   "range_scope") if k in d}
        return replace(base, fields=fields, **known)


class AppState(QObject):
    images_changed = Signal()
    frame_changed = Signal(int)
    roi_changed = Signal(int)          # frame index, -1 = several
    run_config_changed = Signal()
    strain_config_changed = Signal()
    run_state_changed = Signal(str)    # idle | running | stopping
    progress = Signal(float, str)
    results_changed = Signal()
    strain_changed = Signal()
    view_changed = Signal()
    units_changed = Signal()
    probes_changed = Signal()
    log = Signal(str, str)             # message, level
    stale_changed = Signal(bool)
    project_path_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.sequence: ImageSequence | None = None
        self.image_folder: Path | None = None
        self.natural_sort: bool = True
        self.current_frame: int = 0
        self.roi_masks: dict[int, np.ndarray] = {}
        self.roi_history = MaskHistory(depth=40)
        self.run_config = RunConfig()
        self.strain_config = StrainConfig()
        self.series: DisplacementSeries | None = None
        self.strain: StrainSeries | None = None
        self.provider: FieldProvider | None = None
        self.view = ViewSettings()
        self.units = Units()
        self.probes: list[Probe] = []
        self.run_state: str = "idle"
        self.project_path: Path | None = None
        self._last_run_signature: str | None = None

    # -- images -----------------------------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return 0 if self.sequence is None else len(self.sequence)

    @property
    def image_shape(self) -> tuple[int, int] | None:
        return None if self.sequence is None else self.sequence.shape

    def set_sequence(self, sequence: ImageSequence | None, folder: Path | None) -> None:
        """A new image sequence: everything that belonged to the previous one is dropped.

        All fields change first and the signals follow, images first, so that no slot ever sees
        the new images together with the old results (or the reverse).
        """
        self.sequence = sequence
        self.image_folder = folder
        self.current_frame = 0
        self.roi_masks = {}
        self.roi_history.clear()
        self.probes = []
        self.series = None
        self.strain = None
        self.provider = None
        self._last_run_signature = None
        self.project_path = None  # a new session: Ctrl+S must not overwrite the previous project
        if self.view.field not in ("u", "v", "magnitude"):
            self.view = replace(self.view, field="u")
        self.images_changed.emit()
        self.project_path_changed.emit("")
        self.results_changed.emit()
        self.strain_changed.emit()
        self.roi_changed.emit(-1)
        self.probes_changed.emit()
        self.frame_changed.emit(0)
        self.view_changed.emit()
        self._update_stale()

    def set_frame(self, frame: int) -> None:
        if self.sequence is None:
            return
        frame = int(np.clip(frame, 0, self.n_frames - 1))
        if frame != self.current_frame:
            self.current_frame = frame
            self.frame_changed.emit(frame)

    # -- ROI --------------------------------------------------------------------------------------
    def roi(self, frame: int = 0) -> np.ndarray | None:
        return self.roi_masks.get(frame)

    def set_roi(self, frame: int, mask: np.ndarray | None, record: bool = True) -> None:
        """Replace the mask of ``frame`` (None removes it), with an undo record."""
        if record:
            self.roi_history.record(frame, self.roi_masks.get(frame))
        if mask is None or not np.asarray(mask).any():
            self.roi_masks.pop(frame, None)
        else:
            m = np.asarray(mask, dtype=bool).copy()
            m.setflags(write=False)
            self.roi_masks[frame] = m
        self.roi_changed.emit(frame)
        self._update_stale()

    def undo_roi(self) -> None:
        step = self.roi_history.undo(self.roi_masks)
        if step is not None:
            self.set_roi(step[0], step[1], record=False)
            self.set_frame(step[0])

    def redo_roi(self) -> None:
        step = self.roi_history.redo(self.roi_masks)
        if step is not None:
            self.set_roi(step[0], step[1], record=False)
            self.set_frame(step[0])

    # -- settings ---------------------------------------------------------------------------------
    def set_run_config(self, cfg: RunConfig) -> None:
        if cfg != self.run_config:
            self.run_config = cfg
            self.run_config_changed.emit()
            self._update_stale()

    def set_strain_config(self, cfg: StrainConfig) -> None:
        if cfg != self.strain_config:
            self.strain_config = cfg
            self.strain_config_changed.emit()

    def set_view(self, view: ViewSettings) -> None:
        if view != self.view:
            self.view = view
            self.view_changed.emit()

    def update_view(self, **changes: Any) -> None:
        self.set_view(replace(self.view, **changes))

    def set_units(self, units: Units) -> None:
        if units != self.units:
            self.units = units
            self.units_changed.emit()
            self.view_changed.emit()

    # -- results ----------------------------------------------------------------------------------
    def set_results(self, series: DisplacementSeries | None, strain: StrainSeries | None = None) -> None:
        self.series = series
        self.strain = strain
        self.provider = None if series is None else FieldProvider(series, strain)
        if series is not None:
            self._last_run_signature = self.run_signature()
        else:
            self._last_run_signature = None
        if series is None and self.view.field not in ("u", "v", "magnitude"):
            self.view = replace(self.view, field="u")
        self.results_changed.emit()
        self.strain_changed.emit()
        self.view_changed.emit()
        self._update_stale()

    def set_strain(self, strain: StrainSeries | None) -> None:
        self.strain = strain
        if self.provider is not None:
            self.provider = self.provider.with_strain(strain)
        if strain is None and self.view.field not in ("u", "v", "magnitude", "vel_u", "vel_v", "speed"):
            self.view = replace(self.view, field="u")
        self.strain_changed.emit()
        self.view_changed.emit()

    def set_run_state(self, s: str) -> None:
        if s != self.run_state:
            self.run_state = s
            self.run_state_changed.emit(s)

    # -- probes -----------------------------------------------------------------------------------
    def add_probe(self, probe: Probe) -> None:
        self.probes = [*self.probes, probe]
        self.probes_changed.emit()

    def remove_probe(self, probe_id: int) -> None:
        self.probes = [p for p in self.probes if p.id != probe_id]
        self.probes_changed.emit()

    def clear_probes(self) -> None:
        self.probes = []
        self.probes_changed.emit()

    def next_probe_id(self) -> int:
        return 1 + max((p.id for p in self.probes), default=0)

    # -- stale results ----------------------------------------------------------------------------
    def run_signature(self) -> str:
        """Hash of what determines the displacement: images, ROIs, model and run settings."""
        h = hashlib.sha1()
        if self.sequence is not None:
            h.update("|".join(str(p) for p in self.sequence.files).encode())
        for k in sorted(self.roi_masks):
            h.update(str(k).encode())
            h.update(np.packbits(self.roi_masks[k]).tobytes())
        cfg = self.run_config.to_dict()
        h.update(repr(sorted(cfg.items())).encode())
        return h.hexdigest()

    @property
    def is_stale(self) -> bool:
        return self.series is not None and self._last_run_signature is not None and \
            self._last_run_signature != self.run_signature()

    def mark_stale(self) -> None:
        """The results no longer match the inputs (e.g. restored as such from a project)."""
        self._last_run_signature = "out of date"
        self._update_stale()

    def _update_stale(self) -> None:
        self.stale_changed.emit(self.is_stale)

    def emit_log(self, message: str, level: str = "info") -> None:
        self.log.emit(message, level)
