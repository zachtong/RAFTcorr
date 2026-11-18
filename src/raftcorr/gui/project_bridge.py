"""AppState <-> RAFTcorr project files (``.raftcorr``)."""

from __future__ import annotations

from pathlib import Path

from raftcorr.export.scientific import PhysicalUnits
from raftcorr.gui.app_state import AppState, RunState
from raftcorr.io.project import Project

_VIEW_KEYS = ("display_field", "show_deformed", "show_background", "hidden_bg_color", "hide_untracked",
              "overlay_alpha")


def project_from_state(state: AppState) -> Project:
    view = {k: getattr(state, k) for k in _VIEW_KEYS}
    view["field_states"] = {k: dict(v.__dict__) for k, v in state._field_states.items()}
    view["use_physical_units"] = state.use_physical_units
    view["model_advice"] = state.model_advice
    probes = []
    if state.probes is not None and hasattr(state.probes, "to_dicts"):
        probes = state.probes.to_dicts()
        # The id counter keeps a deleted probe's id from coming back on reload.
        view["probe_counters"] = state.probes.counters()
    if state.load_data is not None:
        # The testing-machine record (only the columns the frame mapping uses), as pyALDIC saves it.
        view["load_data"] = state.load_data.to_payload()
    return Project(
        image_folder=str(state.image_folder) if state.image_folder else None,
        image_files=tuple(Path(f).name for f in state.image_files),
        natural_sort=True,
        run_config=state.run_config(),
        strain_config=state.strain_config,
        roi_masks=dict(state.per_frame_rois),
        view=view,
        probes=probes,
        units=PhysicalUnits(scale=state.pixel_size, unit=state.pixel_unit,
                            frame_rate=state.frame_rate if state.frame_rate > 0 else None),
        series=state.results,
        strain=state.strain_results,
    )


def apply_project(state: AppState, image_ctrl, prj: Project, folder: Path, path: Path) -> None:
    """Load the project's images, settings, ROIs, results and view into the state."""
    files = [str(folder / f) for f in prj.image_files] if prj.image_files else None
    if files:
        missing = [f for f in files if not Path(f).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image(s) of the project are missing, e.g. {missing[0]}")
        image_ctrl._raw_files = list(files)
        image_ctrl._publish(files, folder)
    else:
        image_ctrl.load_folder(str(folder))
    if state.sequence is None:
        raise RuntimeError("the images of the project could not be read")
    for k, m in sorted(prj.roi_masks.items()):
        if m.shape == state.sequence.shape:
            state.per_frame_rois[k] = m
    state.roi_changed.emit()
    state.apply_run_config(prj.run_config)
    state.strain_config = prj.strain_config
    v = prj.view or {}
    for k in _VIEW_KEYS:
        if k in v:
            setattr(state, k, v[k])
    from raftcorr.gui.app_state import FieldColorState

    for name, d in (v.get("field_states") or {}).items():
        state._field_states[name] = FieldColorState(**{kk: vv for kk, vv in d.items()
                                                       if kk in FieldColorState.__dataclass_fields__})
    state.model_advice = v.get("model_advice", "")
    u = prj.units
    state.set_physical_units(bool(v.get("use_physical_units", u.unit != "px")), u.scale, u.frame_rate or 1.0,
                             u.unit if u.unit != "px" else state.pixel_unit)
    # Probes before the results, so the views the results reach draw them.
    if state.probes is not None and hasattr(state.probes, "load_dicts") and prj.probes:
        counters = v.get("probe_counters") or {}
        try:
            state.probes.load_dicts(prj.probes, counters.get("next_id"), counters.get("next_number"))
        except ValueError as exc:
            state.log_message.emit(f"The probes of the project could not be read: {exc}", "warn")
    # After the images: loading images clears the load data.
    if v.get("load_data"):
        from raftcorr.analysis.load_data import LoadData

        try:
            state.set_load_data(LoadData.from_payload(v["load_data"]))
        except ValueError as exc:
            state.log_message.emit(f"The load data of the project could not be read: {exc}", "warn")
    if prj.series is not None:
        state.set_results(prj.series, prj.strain)
        state.set_run_state(RunState.DONE)
    state.project_path = path
    state.params_changed.emit()
    state.display_changed.emit()
