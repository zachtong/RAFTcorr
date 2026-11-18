"""Project files (``.raftcorr``): everything needed to reopen a session.

A project file is a ZIP archive (deflate) holding:

* ``project.json``: format marker, schema version, timestamps, image folder and files, sort
  order, run and strain settings, units, GUI view settings, probes, and the index of the array
  entries below;
* ``masks/frame_<k>.npy``: the ROI of frame ``k`` (frame 0 is the main ROI) as
  ``numpy.packbits`` of the C-ordered boolean mask (uint8; the shape is in the JSON);
* ``series/u.npy``, ``series/v.npy`` (streamed frame by frame), ``series/mask.npy``,
  ``series/done.npy``: the displacement series, dtypes kept as they are;
* ``strain/component_<i>.npy`` (streamed frame by frame) and ``strain/done.npy``.

:func:`save_project` writes atomically and stamps ``modified`` with the save time (``created``
is kept). :func:`load_project` reads every array into memory and rebuilds the series exactly. It
does not fail when the images have moved: the returned :attr:`Project.warnings` says so, and when
the image folder kept its position relative to the project file (both moved together, or a
synced folder under another user name), ``image_folder`` points at the new location.

``view`` and ``probes`` must be JSON-serialisable (numpy scalars and arrays are converted);
tuples inside them come back as lists.
"""

from __future__ import annotations

import datetime as _dt
import json
import operator
import os
import zipfile
import zlib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .._version import __version__
from ..callbacks import Callbacks, check_stop, coerce
from ..config import RunConfig, StrainConfig
from ..engine.mapping import Grid
from ..export.scientific import PhysicalUnits, atomic_output, read_npy_entry, to_json, write_npy_entry
from ..results import DisplacementSeries, RunInfo, StrainSeries, StrainSettings
from ..roi.masks import PackedMask
from ..schedule import FrameSchedule

__all__ = ["PROJECT_EXTENSION", "SCHEMA_VERSION", "Project", "load_project", "save_project"]

SCHEMA_VERSION = 1
PROJECT_EXTENSION = ".raftcorr"

_MAGIC = "raftcorr-project"
_DOCUMENT = "project.json"
_COMPRESSLEVEL = 1  # displacement floats barely shrink further at higher levels, saving gets slower
_SERIES_ENTRIES = {"u": "series/u.npy", "v": "series/v.npy", "mask": "series/mask.npy", "done": "series/done.npy"}
_STRAIN_DONE = "strain/done.npy"
_CORRUPT = (KeyError, ValueError, TypeError, IndexError, EOFError, zipfile.BadZipFile, zlib.error)


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True, eq=False)
class Project:
    """A session: images, settings, ROIs, GUI state and (optionally) results.

    ``roi_masks`` maps a 0-based frame index to its boolean mask (full image size); frame 0 is
    the main ROI. ``warnings`` is filled by :func:`load_project` (never saved).
    """

    image_folder: str | None = None
    image_files: tuple[str, ...] = ()
    natural_sort: bool = True
    run_config: RunConfig = field(default_factory=RunConfig)
    strain_config: StrainConfig = field(default_factory=StrainConfig)
    roi_masks: dict[int, np.ndarray] = field(default_factory=dict)
    view: dict[str, Any] = field(default_factory=dict)
    probes: list[dict[str, Any]] = field(default_factory=list)
    units: PhysicalUnits = field(default_factory=PhysicalUnits)
    series: DisplacementSeries | None = None
    strain: StrainSeries | None = None
    created: str = field(default_factory=_now_iso)
    modified: str = field(default_factory=_now_iso)
    schema_version: int = SCHEMA_VERSION
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.strain is not None and self.series is None:
            raise ValueError("a project with a strain series needs its displacement series")
        object.__setattr__(self, "image_files", tuple(str(f) for f in self.image_files))
        object.__setattr__(self, "roi_masks", _normalise_masks(self.roi_masks))
        object.__setattr__(self, "view", dict(self.view))
        object.__setattr__(self, "probes", list(self.probes))
        object.__setattr__(self, "warnings", tuple(str(w) for w in self.warnings))


def _normalise_masks(masks: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for key, mask in masks.items():
        frame = operator.index(key)
        arr = np.asarray(mask, dtype=bool)
        if frame < 0 or arr.ndim != 2:
            raise ValueError(f"roi_masks[{key!r}]: need a frame index >= 0 and a 2-D mask, got shape {arr.shape}")
        out[int(frame)] = arr
    return dict(sorted(out.items()))


# -- save ---------------------------------------------------------------------------------------


def save_project(path: str | Path, project: Project, callbacks: object = None) -> Path:
    """Write ``project`` to ``path`` atomically (an existing file is replaced only on success).

    Raises TypeError when ``view`` or ``probes`` hold something JSON cannot represent (before
    any file is touched). Progress is reported per frame of every stored series array.
    """
    path = Path(path)
    cb = coerce(callbacks)
    try:
        text = to_json(_document(project, path), indent=2)
    except TypeError as exc:
        raise TypeError(f"project view / probes must be JSON-serialisable: {exc}") from exc
    total = max(1, _frame_steps(project))
    with atomic_output(path) as tmp:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=_COMPRESSLEVEL,
                             allowZip64=True) as zf:
            zf.writestr(_DOCUMENT, text)
            for frame, mask in project.roi_masks.items():
                _write_array(zf, _mask_entry(frame), np.frombuffer(PackedMask.pack(mask).bits, dtype=np.uint8))
            steps = 0 if project.series is None else _write_series(zf, project.series, cb, total)
            if project.strain is not None:
                _write_strain(zf, project.strain, cb, steps, total)
    cb.progress(1.0, "project saved")
    return path


def _document(project: Project, path: Path) -> dict[str, Any]:
    return {
        "format": _MAGIC,
        "schema_version": SCHEMA_VERSION,
        "software_version": __version__,
        "created": project.created,
        "modified": _now_iso(),
        "image_folder": project.image_folder,
        "image_folder_relative": _relative_folder(project.image_folder, path),
        "image_files": list(project.image_files),
        "natural_sort": bool(project.natural_sort),
        "run_config": project.run_config.to_dict(),
        "strain_config": project.strain_config.to_dict(),
        "units": project.units.to_dict(),
        "view": project.view,
        "probes": project.probes,
        "roi_masks": [{"frame": k, "shape": list(m.shape), "entry": _mask_entry(k)}
                      for k, m in project.roi_masks.items()],
        "series": None if project.series is None else _series_document(project.series),
        "strain": None if project.strain is None else _strain_document(project.strain),
    }


_RESIDUAL_ENTRY = "series/residual.npy"


def _series_document(series: DisplacementSeries) -> dict[str, Any]:
    entries = dict(_SERIES_ENTRIES)
    if series.residual is not None:
        entries["residual"] = _RESIDUAL_ENTRY
    return {
        "grid": series.grid.to_dict(),
        "frame_names": list(series.frame_names),
        "schedule": series.schedule.to_dict(),
        "info": series.info.to_dict(),
        "entries": entries,
    }


def _strain_document(strain: StrainSeries) -> dict[str, Any]:
    return {
        "grid": strain.grid.to_dict(),
        "step": int(strain.step),
        "settings": strain.settings.to_dict(),
        "components": [{"name": name, "entry": _component_entry(i)} for i, name in enumerate(strain.names)],
        "done": _STRAIN_DONE,
    }


def _mask_entry(frame: int) -> str:
    return f"masks/frame_{frame:05d}.npy"


def _component_entry(index: int) -> str:
    return f"strain/component_{index:02d}.npy"


def _relative_folder(folder: str | None, project_path: Path) -> str | None:
    if folder is None:
        return None
    try:
        return os.path.relpath(Path(folder).absolute(), project_path.absolute().parent)
    except ValueError:  # another drive (Windows)
        return None


def _frame_steps(project: Project) -> int:
    steps = 0 if project.series is None else 2 * project.series.n_frames
    if project.strain is not None:
        steps += sum(int(a.shape[0]) for a in project.strain.components.values())
    return steps


def _write_array(zf: zipfile.ZipFile, entry: str, arr: np.ndarray) -> None:
    write_npy_entry(zf, entry, arr.shape, arr.dtype, [arr])


def _frames(arr: np.ndarray, cb: Callbacks, offset: int, total: int, label: str) -> Iterator[np.ndarray]:
    """Frames of ``arr`` one by one (memmaps stay on disk), reporting progress after each."""
    n = int(arr.shape[0])
    for k in range(n):
        check_stop(cb)
        yield np.asarray(arr[k])
        cb.progress((offset + k + 1) / total, f"saving {label}: frame {k + 1}/{n}")


def _write_series(zf: zipfile.ZipFile, series: DisplacementSeries, cb: Callbacks, total: int) -> int:
    T = series.n_frames
    for i, (key, arr) in enumerate((("u", series.u), ("v", series.v))):
        write_npy_entry(zf, _SERIES_ENTRIES[key], arr.shape, arr.dtype, _frames(arr, cb, i * T, total, key))
    if series.residual is not None:
        res = series.residual
        write_npy_entry(zf, _RESIDUAL_ENTRY, res.shape, np.dtype(np.float16),
                        (np.asarray(f, dtype=np.float16) for f in res))  # a diagnostic: half precision is enough
    _write_array(zf, _SERIES_ENTRIES["mask"], np.asarray(series.mask))
    _write_array(zf, _SERIES_ENTRIES["done"], np.asarray(series.done))
    return 2 * T


def _write_strain(zf: zipfile.ZipFile, strain: StrainSeries, cb: Callbacks, offset: int, total: int) -> None:
    steps = offset
    for i, (name, arr) in enumerate(strain.components.items()):
        write_npy_entry(zf, _component_entry(i), arr.shape, arr.dtype, _frames(arr, cb, steps, total, name))
        steps += int(arr.shape[0])
    _write_array(zf, _STRAIN_DONE, np.asarray(strain.done))


# -- load ---------------------------------------------------------------------------------------


def load_project(path: str | Path) -> Project:
    """Read a project file. Raises FileNotFoundError, or ValueError for a file that is not a
    project, is damaged, or was written by a newer RAFTcorr (newer ``schema_version``)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such project file: {path}")
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path} is not a RAFTcorr project file (not a ZIP archive)") from exc
    with archive as zf:
        doc = _read_document(zf, path)
        try:
            parts = _read_parts(zf, doc)
        except _CORRUPT as exc:
            raise ValueError(f"{path}: damaged or incomplete project file ({type(exc).__name__}: {exc})") from exc
    folder, warnings = _resolve_image_folder(doc, path)
    return Project(image_folder=folder, warnings=warnings, **parts)


def _read_document(zf: zipfile.ZipFile, path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(zf.read(_DOCUMENT).decode("utf-8"))
    except KeyError as exc:
        raise ValueError(f"{path} is not a RAFTcorr project file (no {_DOCUMENT})") from exc
    except (UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile, zlib.error) as exc:
        raise ValueError(f"{path}: damaged project file ({_DOCUMENT} unreadable: {exc})") from exc
    if not isinstance(doc, dict) or doc.get("format") != _MAGIC:
        raise ValueError(f"{path} is not a RAFTcorr project file (format marker missing)")
    version = doc.get("schema_version")
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"{path}: invalid project schema_version {version!r}")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"{path} was saved by a newer version of RAFTcorr (project schema {version}; this version "
            f"{__version__} reads schema {SCHEMA_VERSION} and older). Update RAFTcorr to open it."
        )
    return doc


def _read_parts(zf: zipfile.ZipFile, doc: dict[str, Any]) -> dict[str, Any]:
    """Every Project field stored in the file except the (resolved) image folder."""
    return {
        "image_files": tuple(str(f) for f in doc.get("image_files", ())),
        "natural_sort": bool(doc.get("natural_sort", True)),
        "run_config": RunConfig.from_dict(doc.get("run_config")),
        "strain_config": StrainConfig.from_dict(doc.get("strain_config")),
        "roi_masks": _read_masks(zf, doc.get("roi_masks", [])),
        "view": dict(doc.get("view") or {}),
        "probes": list(doc.get("probes") or []),
        "units": PhysicalUnits.from_dict(doc.get("units")),
        "series": _read_series(zf, doc.get("series")),
        "strain": _read_strain(zf, doc.get("strain")),
        "created": str(doc.get("created", "")),
        "modified": str(doc.get("modified", "")),
        "schema_version": int(doc["schema_version"]),
    }


def _read_masks(zf: zipfile.ZipFile, entries: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for e in entries:
        h, w = (int(s) for s in e["shape"])
        bits = read_npy_entry(zf, e["entry"]).tobytes()
        if len(bits) != -(-h * w // 8):
            raise ValueError(f"mask {e['entry']}: {len(bits)} bytes do not hold a {h} x {w} mask")
        out[int(e["frame"])] = PackedMask((h, w), bits).unpack()
    return out


def _read_series(zf: zipfile.ZipFile, d: dict[str, Any] | None) -> DisplacementSeries | None:
    if d is None:
        return None
    entries = d["entries"]
    return DisplacementSeries(
        grid=Grid.from_dict(d["grid"]),
        mask=read_npy_entry(zf, entries["mask"]),
        u=read_npy_entry(zf, entries["u"]),
        v=read_npy_entry(zf, entries["v"]),
        done=read_npy_entry(zf, entries["done"]),
        frame_names=tuple(str(n) for n in d["frame_names"]),
        schedule=FrameSchedule.from_dict(d["schedule"]),
        info=RunInfo.from_dict(d["info"]),
        residual=(read_npy_entry(zf, entries["residual"]).astype(np.float32) if "residual" in entries else None),
    )


def _read_strain(zf: zipfile.ZipFile, d: dict[str, Any] | None) -> StrainSeries | None:
    if d is None:
        return None
    components = {str(c["name"]): read_npy_entry(zf, c["entry"]) for c in d["components"]}
    return StrainSeries(
        grid=Grid.from_dict(d["grid"]),
        step=int(d["step"]),
        components=components,
        done=read_npy_entry(zf, d["done"]),
        settings=_strain_settings(d["settings"]),
    )


def _strain_settings(d: dict[str, Any]) -> StrainSettings:
    """StrainSettings from its ``to_dict`` (unknown keys ignored, missing keys are an error)."""
    return StrainSettings(**{f.name: d[f.name] for f in fields(StrainSettings)})


def _resolve_image_folder(doc: dict[str, Any], project_path: Path) -> tuple[str | None, tuple[str, ...]]:
    """``(image folder to use, warnings)``: the saved folder, else the same relative position."""
    folder = doc.get("image_folder")
    if folder is None:
        return None, ()
    files = [str(f) for f in doc.get("image_files", ())]
    if Path(folder).is_dir():
        return folder, _missing_files(Path(folder), files)
    relative = doc.get("image_folder_relative")
    if relative is not None:
        candidate = Path(os.path.normpath(project_path.absolute().parent / relative))
        if candidate.is_dir():
            moved = f"image folder {folder} not found; using {candidate} (same place relative to the project file)"
            return str(candidate), (moved,) + _missing_files(candidate, files)
    return folder, (f"image folder not found: {folder}",)


def _missing_files(folder: Path, files: list[str]) -> tuple[str, ...]:
    missing = [f for f in files if not (folder / f).is_file()]
    if not missing:
        return ()
    return (f"{len(missing)} of {len(files)} image files are missing from {folder} (first: {missing[0]})",)
