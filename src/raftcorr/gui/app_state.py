"""Central application state with Qt signals (the pyALDIC design, RAFTcorr fields).

All GUI panels subscribe to signals -- no direct panel-to-panel communication.
Workers emit signals back to the main thread; controllers update AppState.

Field and signal names follow pyALDIC so its widgets work unchanged; the analysis
parameters are RAFTcorr's (model, iterations, tracking, tiling) and the results are
:class:`raftcorr.results.DisplacementSeries` / :class:`raftcorr.results.StrainSeries`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal

from raftcorr.config import RunConfig, StrainConfig, TilingConfig, TrackingConfig


class RunState(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"


@dataclass
class FieldColorState:
    """Per-field color range and colormap settings."""
    auto: bool = True
    vmin: float = 0.0
    vmax: float = 1.0
    colormap: str = "turbo"


#: Display fields of the main window (pyALDIC names) -> raftcorr.viz field keys.
DISPLAY_FIELDS: dict[str, str] = {
    "disp_u": "u",
    "disp_v": "v",
    "disp_magnitude": "magnitude",
    "residual": "residual",
}


def _default_colormap(field_name: str) -> str:
    """Return the default colormap for a given field name."""
    return "magma" if field_name == "residual" else "turbo"


class AppState(QObject):
    """Singleton state container for the entire GUI."""

    _instance: AppState | None = None

    # --- Signals (pyALDIC names) ---
    images_changed = Signal()
    current_frame_changed = Signal(int)
    roi_changed = Signal()
    params_changed = Signal()
    run_state_changed = Signal(RunState)
    progress_updated = Signal(float, str)
    results_changed = Signal()
    display_changed = Signal()
    log_message = Signal(str, str)  # (message, level)
    # Fatal errors that should be surfaced as a modal dialog in addition to
    # being written to the console log: (title, message), user-facing text.
    fatal_error = Signal(str, str)
    # A complete run whose tracking quality is poor: (title, text, informative,
    # details) of a non-blocking warning dialog. The same findings are in the log.
    run_warning = Signal(str, str, str, str)
    # A line the progress area shows instead of the frame while it is set ("" ends it): the
    # download of a model on its first use.
    progress_status = Signal(str)
    # A model the run needs is not installed and could not be downloaded: (model key, reason).
    # The main window opens the Models window on it.
    model_unavailable = Signal(str, str)
    physical_units_changed = Signal()
    seeds_changed = Signal()      # kept for widget compatibility; RAFTcorr has no seeds
    load_data_changed = Signal()
    # --- RAFTcorr additions ---
    strain_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._init_state()

    def _init_state(self) -> None:
        # Images
        self.image_folder: Path | None = None
        self.image_files: list[str] = []
        self.current_frame: int = 0
        self.sequence = None  # raftcorr.io.images.ImageSequence of the loaded files
        # ROI — per-frame system (frame_idx -> bool mask)
        self.per_frame_rois: dict[int, NDArray[np.bool_]] = {}
        # Per-frame masks in deformed coordinates (unused by RAFTcorr; per_frame_rois
        # already exclude regions in the deformed frames)
        self.deformed_masks: dict[int, NDArray[np.bool_]] | None = None
        # Tracking (pyALDIC names)
        self.tracking_mode: str = "accumulative"
        self.inc_ref_mode: str = "every_frame"
        self.inc_ref_interval: int = 5
        self.inc_custom_refs: list[int] = []
        # Why the Reference Frames text was refused ("" = it shows what the run uses); a run with
        # custom reference frames is refused while it is set, instead of using older frames.
        self.inc_custom_refs_error: str = ""
        # RAFTcorr analysis parameters (paper defaults)
        self.model: str = "s4"
        self.device: str = "auto"
        self.iters: int = 20
        self.smoothing_sigma: float = 0.0
        self.corr_mode: str | None = None
        self.margin_px: int | None = None
        self.memory_fraction: float = 0.85
        self.blend_px: int = 32
        self.model_advice: str = ""
        # Strain (paper defaults)
        self.strain_config: StrainConfig = StrainConfig()
        # Computation
        self.run_state: RunState = RunState.IDLE
        self.progress: float = 0.0
        self.progress_message: str = ""
        self.progress_status_text: str = ""
        self.elapsed_seconds: float = 0.0
        self.results = None          # raftcorr.results.DisplacementSeries
        self.strain_results = None   # raftcorr.results.StrainSeries
        self.provider = None         # raftcorr.viz.fields.FieldProvider of the results
        # Display
        self.display_field: str = "disp_u"
        #: Where the field is drawn: deformed positions, or reference.
        self.show_deformed: bool = True
        #: Whether an image sits behind the field at all.
        self.show_background: bool = True
        #: What replaces the image when it is hidden: "white", "black" or "transparent".
        self.hidden_bg_color: str = "white"
        #: "Hide untracked regions": leave points without a result transparent.
        self.hide_untracked: bool = True
        self.roi_editing: bool = False
        self._field_states: dict[str, FieldColorState] = {}
        self.overlay_alpha: float = 0.7
        self.show_mesh: bool = False
        self.show_subset_window: bool = False
        # Physical units
        self.use_physical_units: bool = False
        self.pixel_size: float = 1.0    # value in units of pixel_unit per px
        self.pixel_unit: str = "mm"     # unit string: nm, μm, mm, cm, m, inch
        self.frame_rate: float = 1.0    # fps
        # Post-processing probes (frame-0 image pixels). Filled by the analysis tab.
        from raftcorr.analysis.probes import ProbeSet

        self.probes = ProbeSet()
        self.load_data = None
        # Project file of this session (None until saved or opened)
        self.project_path: Path | None = None

    # ------------------------------------------------------------------
    # RAFTcorr configuration
    # ------------------------------------------------------------------
    def tracking_config(self) -> TrackingConfig:
        if self.tracking_mode == "accumulative":
            return TrackingConfig(mode="accumulative")
        if self.inc_ref_mode == "every_n":
            return TrackingConfig(mode="every_n", every_n=max(1, int(self.inc_ref_interval)))
        if self.inc_ref_mode == "custom":
            return TrackingConfig(mode="custom", key_frames=tuple(self.inc_custom_refs))
        return TrackingConfig(mode="incremental")

    def run_config(self) -> RunConfig:
        return RunConfig(
            model=self.model, device=self.device, iters=int(self.iters), corr_mode=self.corr_mode,
            smoothing_sigma=float(self.smoothing_sigma), tracking=self.tracking_config(),
            tiling=TilingConfig(margin_px=self.margin_px, memory_fraction=float(self.memory_fraction),
                                blend_px=int(self.blend_px)),
        )

    def apply_run_config(self, cfg: RunConfig) -> None:
        """Set the parameter fields from a RunConfig (project loading)."""
        self.model = str(cfg.model)
        self.device = cfg.device
        self.iters = cfg.iters
        self.corr_mode = cfg.corr_mode
        self.smoothing_sigma = cfg.smoothing_sigma
        self.margin_px = cfg.tiling.margin_px
        self.memory_fraction = cfg.tiling.memory_fraction
        self.blend_px = cfg.tiling.blend_px
        t = cfg.tracking
        self.tracking_mode = "accumulative" if t.mode == "accumulative" else "incremental"
        self.inc_ref_mode = {"incremental": "every_frame", "every_n": "every_n", "custom": "custom"}.get(t.mode,
                                                                                                         "every_frame")
        self.inc_ref_interval = t.every_n
        self.inc_custom_refs = list(t.key_frames)
        self.params_changed.emit()

    def reference_frames(self) -> set[int]:
        """Frames used as references by the current tracking settings."""
        n = len(self.image_files)
        if n < 2:
            return {0}
        from raftcorr.schedule import FrameSchedule

        cfg = self.tracking_config()
        try:
            return set(FrameSchedule.from_mode(cfg.mode, n, cfg.every_n, cfg.key_frames).reference_frames)
        except ValueError:
            return {0}

    @property
    def roi_mask(self) -> NDArray[np.bool_] | None:
        """Backward-compatible access to frame-0 ROI mask."""
        return self.per_frame_rois.get(0)

    @property
    def refine_brush_mask(self) -> None:
        """pyALDIC's mesh-refinement brush; RAFTcorr has no mesh."""
        return None

    # ------------------------------------------------------------------
    # Per-field color state accessors
    # ------------------------------------------------------------------

    def _get_or_create_field_state(self, field_name: str) -> FieldColorState:
        if field_name not in self._field_states:
            self._field_states[field_name] = FieldColorState(colormap=_default_colormap(field_name))
        return self._field_states[field_name]

    def get_field_state(self, field_name: str) -> FieldColorState:
        """Return the stored color state for the given field (public API)."""
        return self._get_or_create_field_state(field_name)

    @property
    def colormap(self) -> str:
        return self._get_or_create_field_state(self.display_field).colormap

    @colormap.setter
    def colormap(self, value: str) -> None:
        self._get_or_create_field_state(self.display_field).colormap = value

    @property
    def color_auto(self) -> bool:
        return self._get_or_create_field_state(self.display_field).auto

    @color_auto.setter
    def color_auto(self, value: bool) -> None:
        self._get_or_create_field_state(self.display_field).auto = value

    @property
    def color_min(self) -> float:
        return self._get_or_create_field_state(self.display_field).vmin

    @color_min.setter
    def color_min(self, value: float) -> None:
        self._get_or_create_field_state(self.display_field).vmin = float(value)

    @property
    def color_max(self) -> float:
        return self._get_or_create_field_state(self.display_field).vmax

    @color_max.setter
    def color_max(self, value: float) -> None:
        self._get_or_create_field_state(self.display_field).vmax = float(value)

    @classmethod
    def instance(cls) -> AppState:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self) -> None:
        self._init_state()

    # --- Setters (emit signals) ---
    def set_image_files(self, files: list[str], sequence=None, folder: Path | None = None) -> None:
        """New images: results, ROIs, probes and the project of the previous set are dropped."""
        had_results = self.results is not None
        had_roi = bool(self.per_frame_rois)
        was_active = self.run_state != RunState.IDLE
        self.results = None
        self.strain_results = None
        self.provider = None
        self.deformed_masks = None
        self.per_frame_rois = {}
        self.roi_editing = False
        self.run_state = RunState.IDLE
        self.progress = 0.0
        self.progress_message = ""
        self.progress_status_text = ""
        self.elapsed_seconds = 0.0
        self.image_files = list(files)
        self.sequence = sequence
        self.image_folder = folder
        self.current_frame = 0
        self.project_path = None
        self.model_advice = ""
        if self.probes is not None and hasattr(self.probes, "clear"):
            self.probes.clear()
        had_load = self.load_data is not None
        self.load_data = None
        if had_load:
            self.load_data_changed.emit()
        if had_results:
            self.results_changed.emit()
            self.strain_changed.emit()
        if had_results or was_active:
            self.run_state_changed.emit(RunState.IDLE)
        if had_roi:
            self.roi_changed.emit()
        self.images_changed.emit()

    def set_remaining_frames(self, files: list[str], sequence, rois: dict) -> None:
        """Frames were removed from the current set (``ImageController.remove_frames``).

        The ROIs arrive re-keyed to the new frame numbers. The results go (their frame k is no
        longer image k); probes, the project and the load data stay, the latter with a warning
        because it is matched to frames by number or time.
        """
        had_results = self.results is not None
        self.results = None
        self.strain_results = None
        self.provider = None
        self.deformed_masks = None
        self.show_deformed = False
        if self.run_state != RunState.IDLE:
            self.run_state = RunState.IDLE
            self.run_state_changed.emit(RunState.IDLE)
        self.image_files = list(files)
        self.sequence = sequence
        self.per_frame_rois = dict(rois)
        self.current_frame = min(self.current_frame, len(files) - 1)
        if self.load_data is not None:
            self.log_message.emit(
                "Frames were removed: the load data is matched to frames by number or time, "
                "check its frame mapping.", "warn")
        if had_results:
            self.results_changed.emit()
            self.strain_changed.emit()
        self.roi_changed.emit()
        self.images_changed.emit()

    def set_load_data(self, data) -> None:
        self.load_data = data
        self.load_data_changed.emit()

    def set_current_frame(self, idx: int) -> None:
        if not self.image_files:
            return
        self.current_frame = max(0, min(idx, len(self.image_files) - 1))
        self.current_frame_changed.emit(self.current_frame)

    def set_roi_mask(self, mask: NDArray[np.bool_] | None) -> None:
        """Set or clear the frame-0 ROI mask."""
        self.set_frame_roi(0, mask)

    def set_frame_roi(self, frame: int, mask: NDArray[np.bool_] | None) -> None:
        """Set or clear the ROI mask of a frame."""
        if mask is None or not np.asarray(mask).any():
            self.per_frame_rois.pop(frame, None)
        else:
            self.per_frame_rois[frame] = np.asarray(mask, dtype=bool)
        self.roi_changed.emit()

    def set_refine_brush_mask(self, mask) -> None:  # pyALDIC API; no mesh refinement in RAFTcorr
        return None

    def get_effective_roi(self, frame: int, *, is_ref_frame: bool = False) -> NDArray[np.bool_] | None:
        own = self.per_frame_rois.get(frame)
        if own is not None:
            return own
        if is_ref_frame:
            return self.per_frame_rois.get(0)
        return None

    def set_run_state(self, state: RunState) -> None:
        self.run_state = state
        self.run_state_changed.emit(state)

    def set_progress(self, fraction: float, message: str = "") -> None:
        fraction = min(max(fraction, 0.0), 1.0)
        self.progress = fraction
        self.progress_message = message
        self.progress_updated.emit(fraction, message)

    def set_progress_status(self, text: str) -> None:
        self.progress_status_text = text
        self.progress_status.emit(text)

    def set_results(self, results, strain=None) -> None:
        from raftcorr.viz.fields import FieldProvider

        self.results = results
        self.strain_results = strain
        self.provider = None if results is None else FieldProvider(results, strain)
        if results is None and self.display_field not in ("disp_u", "disp_v"):
            self.display_field = "disp_u"
        self.results_changed.emit()
        self.strain_changed.emit()

    def set_strain_results(self, strain) -> None:
        self.strain_results = strain
        if self.provider is not None:
            self.provider = self.provider.with_strain(strain)
        self.strain_changed.emit()

    def set_colormap(self, cmap: str) -> None:
        self.colormap = cmap
        self.display_changed.emit()

    def set_display_field(self, field_name: str) -> None:
        self.display_field = field_name
        self.display_changed.emit()

    def set_color_range(self, auto: bool, vmin: float = 0.0, vmax: float = 1.0) -> None:
        self.color_auto = auto
        self.color_min = vmin
        self.color_max = vmax
        self.display_changed.emit()

    def set_show_mesh(self, show: bool) -> None:
        self.show_mesh = show
        self.display_changed.emit()

    def set_show_subset_window(self, show: bool) -> None:
        self.show_subset_window = show
        self.display_changed.emit()

    def set_overlay_alpha(self, alpha: float) -> None:
        self.overlay_alpha = max(0.0, min(1.0, float(alpha)))
        self.display_changed.emit()

    def set_physical_units(self, enabled: bool, pixel_size: float, frame_rate: float, pixel_unit: str = "mm") -> None:
        self.use_physical_units = bool(enabled)
        self.pixel_size = max(0.0, float(pixel_size))
        self.pixel_unit = str(pixel_unit)
        self.frame_rate = max(0.0, float(frame_rate))
        self.physical_units_changed.emit()
        self.display_changed.emit()

    def units(self):
        """The display units as :class:`raftcorr.viz.fields.Units`."""
        from raftcorr.viz.fields import Units

        if not self.use_physical_units:
            return Units()
        return Units(pixel_size=self.pixel_size, length_unit=self.pixel_unit,
                     frame_rate=self.frame_rate if self.frame_rate > 0 else None)

    def set_param(self, name: str, value: object) -> None:
        if hasattr(self, name):
            setattr(self, name, value)
            self.params_changed.emit()
