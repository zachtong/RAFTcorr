"""Scientific export of a displacement series (and its strain): HDF5, MATLAB v7.3 and NumPy NPZ.

The format follows the file extension: ``.h5`` / ``.hdf5`` (HDF5), ``.mat`` (MATLAB v7.3, an
HDF5 file MATLAB opens with ``load``) or ``.npz`` (NumPy). Every file is written to a temporary
file in the destination folder and moved into place at the end, so a failed or cancelled export
never leaves a half-written file behind and never damages an existing one.

Variables (``variables=None`` exports every available one):

==============  ===============  =========  ===============================================
name            shape            dtype      content
==============  ===============  =========  ===============================================
U, V            (T, h, w)        float32    displacement (px), +right / +down
Magnitude       (T, h, w)        float32    ``sqrt(U^2 + V^2)`` (px)
X, Y            (h, w)           float64    image pixel coordinates of the grid points
ROI             (h, w)           uint8      frame-0 region of interest (0 / 1)
Done            (T,)             uint8      1 where the frame was computed
<component>     (T, hs, ws)      float32    one per strain component (strain given only)
X_strain, ...   (hs, ws)         float64    pixel coordinates of the strain samples
==============  ===============  =========  ===============================================

``U``, ``V`` and ``Magnitude`` are NaN outside the ROI, at untracked points and in every frame
that was not computed; strain components are NaN in strain frames that were not computed.

Units: displacements and coordinates are always written in pixels. A :class:`PhysicalUnits`
adds the metadata entries ``pixel_size`` (length unit per pixel), ``length_unit`` and
``frame_rate`` (frames per second, or null); multiply by ``pixel_size`` to convert.

Metadata (every format) is one JSON document: software version, creation time, frame names,
reference indices of the schedule, run info, grid, strain settings, units, a conventions note,
and the name / shape / unit of every exported variable.

* HDF5: one gzip (level 4) + shuffle dataset per variable, one chunk per frame; the metadata JSON
  is the root attribute ``metadata``, with a few simple root attributes for quick inspection and a
  ``frame_names`` string dataset. Written frame by frame.
* MAT v7.3: the same streaming writer with ``MATLAB_class`` attributes, the axes reversed on disk
  so MATLAB reads ``U`` as ``T x h x w``, the metadata JSON as the char array ``metadata_json``
  (``jsondecode(metadata_json)`` in MATLAB), the frame names as one newline-joined char array, and
  the 512-byte MATLAB header in the HDF5 user block.
* NPZ: one ``.npy`` entry per variable, streamed frame by frame into the deflated ZIP (so the
  export never holds a whole series in memory either); the metadata JSON is the 0-d string array
  ``metadata_json``. ``numpy.load`` reads each array fully into memory when it is accessed.

The module also holds three small file helpers shared with :mod:`raftcorr.export.tables` and
:mod:`raftcorr.io.project`: :func:`atomic_output`, :func:`to_json` and the ``.npy``-in-ZIP
streaming pair :func:`write_npy_entry` / :func:`read_npy_entry`.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import struct
import sys
import tempfile
import zipfile
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .._version import __version__
from ..callbacks import Callbacks, check_stop, coerce
from ..post.strain import COMPONENT_UNITS, COMPONENTS
from ..results import DisplacementSeries, StrainSeries

__all__ = [
    "CONVENTIONS",
    "DISPLACEMENT_VARIABLES",
    "FORMATS",
    "STRAIN_COORDINATES",
    "PhysicalUnits",
    "available_variables",
    "export_results",
    "read_results",
]

FORMATS: dict[str, str] = {".npz": "npz", ".mat": "mat", ".h5": "hdf5", ".hdf5": "hdf5"}
DISPLACEMENT_VARIABLES: tuple[str, ...] = ("U", "V", "Magnitude", "X", "Y", "ROI", "Done")
STRAIN_COORDINATES: tuple[str, ...] = ("X_strain", "Y_strain")
RESIDUAL = "Residual"  # local RMS intensity residual (grey levels), a tracking-quality map
EXPORT_SCHEMA_VERSION = 1
CONVENTIONS = (
    "u +right, v +down, px; frame k = image k; frame 0 is the reference; "
    "Lagrangian (frame-0) coordinates; NaN = outside ROI or not tracked; exy is the tensor shear"
)

_DISPLACEMENT_FRAMES: tuple[str, ...] = ("U", "V", "Magnitude")
_PIXEL_VARIABLES = frozenset({"U", "V", "Magnitude", "X", "Y", *STRAIN_COORDINATES})
_GZIP_LEVEL = 4
_NPZ_LEVEL = 4
_MAT_USERBLOCK = 512
_MATLAB_CLASS: dict[np.dtype, str] = {
    np.dtype("float32"): "single",
    np.dtype("float64"): "double",
    np.dtype("uint8"): "uint8",
    np.dtype("uint16"): "uint16",
    np.dtype("int32"): "int32",
    np.dtype("int64"): "int64",
}
_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@dataclass(frozen=True)
class PhysicalUnits:
    """Calibration recorded with an export (the data itself stays in pixels).

    ``scale``: length units per pixel; ``unit``: name of the length unit; ``frame_rate``: frames
    per second, or None when unknown.
    """

    scale: float = 1.0
    unit: str = "px"
    frame_rate: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.scale) and self.scale > 0):
            raise ValueError(f"scale must be a positive finite number, got {self.scale!r}")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError(f"unit must be a non-empty string, got {self.unit!r}")
        if self.frame_rate is not None and not (math.isfinite(self.frame_rate) and self.frame_rate > 0):
            raise ValueError(f"frame_rate must be None or a positive finite number, got {self.frame_rate!r}")
        object.__setattr__(self, "scale", float(self.scale))
        if self.frame_rate is not None:
            object.__setattr__(self, "frame_rate", float(self.frame_rate))

    @property
    def is_pixels(self) -> bool:
        return self.scale == 1.0 and self.unit == "px"

    def to_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "unit": self.unit, "frame_rate": self.frame_rate}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "PhysicalUnits":
        d = d or {}
        rate = d.get("frame_rate")
        return cls(float(d.get("scale", 1.0)), str(d.get("unit", "px")), None if rate is None else float(rate))


@dataclass(frozen=True)
class _Plan:
    """What one export writes: the source series, the variables (canonical order), the metadata."""

    series: DisplacementSeries
    strain: StrainSeries | None
    names: tuple[str, ...]
    shapes: dict[str, tuple[int, ...]]
    metadata: dict[str, Any]

    @property
    def per_frame(self) -> tuple[str, ...]:
        return tuple(n for n in self.names if len(self.shapes[n]) == 3)

    @property
    def static(self) -> tuple[str, ...]:
        return tuple(n for n in self.names if len(self.shapes[n]) != 3)


# -- public API ---------------------------------------------------------------------------------


def available_variables(strain: StrainSeries | None = None,
                        series: DisplacementSeries | None = None) -> tuple[str, ...]:
    """Every variable an export can hold, in file order (strain variables only with a strain;
    ``Residual``, the intensity residual, only when ``series`` carries one)."""
    base = DISPLACEMENT_VARIABLES
    if series is not None and series.residual is not None:
        base = base + (RESIDUAL,)
    if strain is None:
        return base
    return base + _strain_components(strain) + STRAIN_COORDINATES


def export_results(
    path: str | Path,
    series: DisplacementSeries,
    strain: StrainSeries | None = None,
    variables: Iterable[str] | None = None,
    units: PhysicalUnits | None = None,
    overwrite: bool = False,
    callbacks: object = None,
) -> Path:
    """Write ``series`` (and ``strain``) to ``path``; the format follows the extension.

    Raises ValueError for an unknown extension or variable, FileExistsError when ``path`` exists
    and ``overwrite`` is False. Progress is reported once per frame; ``should_stop()`` cancels
    (:class:`~raftcorr.callbacks.StopRequested`) without touching ``path``.
    """
    path = Path(path)
    fmt = _format_of(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (pass overwrite=True to replace it)")
    if strain is not None:
        _check_strain(series, strain)
    names = _resolve_variables(variables, strain, series)
    shapes = _variable_shapes(series, strain, names)
    plan = _Plan(series, strain, names, shapes, _metadata(series, strain, units or PhysicalUnits(), names, shapes, fmt))
    cb = coerce(callbacks)
    with atomic_output(path) as tmp:
        if fmt == "npz":
            _write_npz(tmp, plan, cb)
        else:
            _write_hdf5(tmp, plan, cb, mat=fmt == "mat")
    return path


def read_results(path: str | Path) -> dict[str, Any]:
    """Arrays of an export (numpy shapes, MAT transposition undone) plus ``"metadata"`` (dict).

    Reads every array fully into memory.
    """
    path = Path(path)
    fmt = _format_of(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    if fmt == "npz":
        return _read_npz(path)
    return _read_hdf5(path, mat=fmt == "mat")


# -- shared file helpers ------------------------------------------------------------------------


@contextmanager
def atomic_output(path: str | Path) -> Iterator[Path]:
    """Yield a temporary path next to ``path``; move it onto ``path`` when the block succeeds.

    On any error (or cancellation) the temporary file is removed and ``path`` is left untouched.
    """
    target = Path(path)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        os.replace(tmp, target)
    except BaseException:
        with suppress(OSError):
            tmp.unlink()
        raise


def to_json(value: Any, indent: int | None = None) -> str:
    """JSON text of ``value``; numpy scalars / arrays and paths are converted, anything else
    that JSON cannot represent raises TypeError."""
    return json.dumps(value, default=_json_default, ensure_ascii=False, indent=indent)


def write_npy_entry(
    zf: zipfile.ZipFile, name: str, shape: tuple[int, ...], dtype: Any, chunks: Iterable[np.ndarray]
) -> None:
    """Stream an ``.npy`` array into the ZIP entry ``name`` from consecutive C-order ``chunks``
    (for example the frames of a ``(T, h, w)`` array). Raises ValueError if the chunks do not
    add up to ``shape``."""
    dt = np.dtype(dtype)
    expected = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
    header = {"descr": np.lib.format.dtype_to_descr(dt), "fortran_order": False, "shape": tuple(int(s) for s in shape)}
    written = 0
    with zf.open(name, "w", force_zip64=True) as fp:
        np.lib.format.write_array_header_1_0(fp, header)
        for chunk in chunks:
            data = np.ascontiguousarray(chunk, dtype=dt).tobytes()
            written += len(data)
            if written > expected:
                break
            fp.write(data)
    if written != expected:
        raise ValueError(f"{name}: chunks hold {written} bytes, shape {tuple(shape)} {dt} needs {expected}")


def read_npy_entry(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    """Read the ``.npy`` ZIP entry ``name`` (no pickles)."""
    with zf.open(name) as fp:
        return np.lib.format.read_array(fp, allow_pickle=False)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


# -- planning -----------------------------------------------------------------------------------


def _format_of(path: Path) -> str:
    fmt = FORMATS.get(path.suffix.lower())
    if fmt is None:
        raise ValueError(f"unknown export format {path.suffix!r}; use one of {', '.join(FORMATS)}")
    return fmt


def _strain_components(strain: StrainSeries) -> tuple[str, ...]:
    return tuple(name for name in COMPONENTS if name in strain.components)


def _check_strain(series: DisplacementSeries, strain: StrainSeries) -> None:
    if strain.grid != series.grid:
        raise ValueError(f"strain grid {strain.grid} differs from the displacement grid {series.grid}")
    shapes = {strain.components[n].shape for n in _strain_components(strain)}
    if len(shapes) > 1:
        raise ValueError(f"strain components have different shapes: {sorted(shapes)}")
    if shapes and next(iter(shapes))[0] != series.n_frames:
        raise ValueError(f"strain has {next(iter(shapes))[0]} frames, the displacement series {series.n_frames}")
    if len(strain.done) != series.n_frames:
        raise ValueError(f"strain done has {len(strain.done)} entries, expected {series.n_frames}")


def _resolve_variables(variables: Iterable[str] | None, strain: StrainSeries | None,
                       series: DisplacementSeries | None = None) -> tuple[str, ...]:
    valid = available_variables(strain, series)
    if variables is None:
        return valid
    requested = {variables} if isinstance(variables, str) else set(variables)
    unknown = sorted(requested - set(valid))
    if unknown:
        hint = "" if strain is not None else " (strain variables need a strain series)"
        raise ValueError(f"unknown variable(s) {unknown}; valid names: {', '.join(valid)}{hint}")
    if not requested:
        raise ValueError("no variables selected")
    return tuple(name for name in valid if name in requested)


def _variable_shapes(series: DisplacementSeries, strain: StrainSeries | None,
                     names: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    T, (h, w) = series.n_frames, series.grid.shape
    base: dict[str, tuple[int, ...]] = {
        "U": (T, h, w), "V": (T, h, w), "Magnitude": (T, h, w), "X": (h, w), "Y": (h, w), "ROI": (h, w), "Done": (T,),
        RESIDUAL: (T, h, w),
    }
    if strain is not None:
        comps = _strain_components(strain)
        hs, ws = strain.components[comps[0]].shape[1:] if comps else (0, 0)
        base.update({name: (T, hs, ws) for name in comps})
        base.update({name: (hs, ws) for name in STRAIN_COORDINATES})
    return {name: tuple(int(s) for s in base[name]) for name in names}


def _variable_unit(name: str) -> str:
    if name in _PIXEL_VARIABLES:
        return "px"
    if name in ("ROI", "Done"):
        return ""
    if name == RESIDUAL:
        return "grey levels"
    return COMPONENT_UNITS.get(name, "1")


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _metadata(series: DisplacementSeries, strain: StrainSeries | None, units: PhysicalUnits,
              names: tuple[str, ...], shapes: dict[str, tuple[int, ...]], fmt: str) -> dict[str, Any]:
    return {
        "software_version": __version__,
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "created": _now_iso(),
        "format": fmt,
        "n_frames": series.n_frames,
        "frame_names": list(series.frame_names),
        "ref_indices": list(series.schedule.ref_indices),
        "run_info": series.info.to_dict(),
        "grid": series.grid.to_dict(),
        "strain_settings": None if strain is None else strain.settings.to_dict(),
        "strain_step": None if strain is None else int(strain.step),
        "strain_done_frames": None if strain is None else [int(k) for k in np.flatnonzero(strain.done)],
        "pixel_size": units.scale,
        "length_unit": units.unit,
        "frame_rate": units.frame_rate,
        "displacement_unit": "px",
        "conventions": CONVENTIONS,
        "variables": list(names),
        "shapes": {name: list(shapes[name]) for name in names},
        "variable_units": {name: _variable_unit(name) for name in names},
    }


# -- data of one frame / the static arrays ------------------------------------------------------


def _displacement_frame(series: DisplacementSeries, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``(U, V)`` of frame ``k`` as written: NaN outside the ROI and in frames not computed."""
    if not bool(series.done[k]):
        empty = np.full(series.grid.shape, np.nan, dtype=np.float32)
        return empty, empty
    nan = np.float32(np.nan)
    u = np.where(series.mask, np.asarray(series.u[k], dtype=np.float32), nan)
    v = np.where(series.mask, np.asarray(series.v[k], dtype=np.float32), nan)
    return u.astype(np.float32, copy=False), v.astype(np.float32, copy=False)


def _strain_frame(strain: StrainSeries, name: str, k: int) -> np.ndarray:
    values = strain.components[name]
    if not bool(strain.done[k]):
        return np.full(values.shape[1:], np.nan, dtype=np.float32)
    return np.asarray(values[k], dtype=np.float32)


def _frame_values(plan: _Plan, k: int, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Frame ``k`` of every per-frame variable in ``names`` (U / V read once)."""
    out: dict[str, np.ndarray] = {}
    if any(n in _DISPLACEMENT_FRAMES for n in names):
        u, v = _displacement_frame(plan.series, k)
        out.update({"U": u, "V": v, "Magnitude": np.hypot(u, v)} if "Magnitude" in names else {"U": u, "V": v})
    if RESIDUAL in names:
        s = plan.series
        res = np.asarray(s.residual[k], dtype=np.float32) if bool(s.done[k]) else None
        out[RESIDUAL] = (np.where(s.mask, res, np.float32(np.nan)).astype(np.float32) if res is not None
                         else np.full(s.grid.shape, np.nan, dtype=np.float32))
    for name in names:
        if name not in out and plan.strain is not None:
            out[name] = _strain_frame(plan.strain, name, k)
    return {name: out[name] for name in names}


def _static_values(plan: _Plan) -> dict[str, np.ndarray]:
    series, wanted = plan.series, set(plan.static)
    out: dict[str, np.ndarray] = {}
    if wanted & {"X", "Y"}:
        X, Y = series.grid.coords()
        out.update({"X": X, "Y": Y})
    out["ROI"] = np.asarray(series.mask, dtype=np.uint8)
    out["Done"] = np.asarray(series.done, dtype=np.uint8)
    if plan.strain is not None and wanted & set(STRAIN_COORDINATES):
        hs, ws = plan.shapes.get("X_strain", plan.shapes.get("Y_strain"))
        g, s = plan.strain.grid, plan.strain.step
        ys, xs = np.mgrid[0:hs, 0:ws].astype(np.float64)
        out.update({"X_strain": g.x0 + xs * s, "Y_strain": g.y0 + ys * s})
    return {name: out[name] for name in plan.static}


def _frame_message(k: int, T: int) -> str:
    return f"export: frame {k + 1}/{T}"


# -- HDF5 / MAT v7.3 ----------------------------------------------------------------------------


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("HDF5 and MAT export need h5py (pip install h5py)") from exc
    return h5py


def _write_hdf5(tmp: Path, plan: _Plan, cb: Callbacks, mat: bool) -> None:
    h5py = _h5py()
    # MAT: user block for the MATLAB header. HDF5: 1.8 object format so that a large metadata
    # attribute (thousands of frame names) is stored densely; readable by any HDF5 >= 1.8.
    options: dict[str, Any] = {"userblock_size": _MAT_USERBLOCK} if mat else {"libver": ("v108", "v108")}
    T = plan.series.n_frames
    with h5py.File(tmp, "w", **options) as f:
        dsets = {name: _create_frame_dataset(f, name, plan.shapes[name], mat) for name in plan.per_frame}
        for name, arr in _static_values(plan).items():
            _write_static(f, name, arr, mat)
        if mat:
            _write_mat_metadata(f, plan.metadata)
        else:
            _write_h5_metadata(f, plan.metadata)
        for k in range(T):
            check_stop(cb)
            for name, frame in _frame_values(plan, k, plan.per_frame).items():
                if mat:
                    dsets[name][:, :, k] = frame.T
                else:
                    dsets[name][k] = frame
            cb.progress((k + 1) / T, _frame_message(k, T))
    if mat:
        _write_matlab_header(tmp)


def _create_frame_dataset(f: Any, name: str, shape: tuple[int, ...], mat: bool) -> Any:
    disk = tuple(reversed(shape)) if mat else shape
    chunks = disk[:-1] + (1,) if mat else (1,) + shape[1:]
    dset = f.create_dataset(name, shape=disk, dtype=np.float32, chunks=chunks, compression="gzip",
                            compression_opts=_GZIP_LEVEL, shuffle=True, fillvalue=np.float32(np.nan))
    _tag(dset, name, np.dtype(np.float32), mat)
    return dset


def _write_static(f: Any, name: str, arr: np.ndarray, mat: bool) -> None:
    data = arr
    if mat:
        # MATLAB arrays are at least 2-D and column-major: a vector becomes T x 1, then every
        # array is stored with its axes reversed so that MATLAB reads the numpy shape.
        data = np.ascontiguousarray((arr.reshape(-1, 1) if arr.ndim == 1 else arr).T)
    compress = {"compression": "gzip", "compression_opts": _GZIP_LEVEL, "shuffle": True} if data.size else {}
    dset = f.create_dataset(name, data=data, **compress)
    _tag(dset, name, arr.dtype, mat)


def _matlab_class(dset: Any, cls: str) -> None:
    """``MATLAB_class`` as a fixed-length ASCII string, as MATLAB writes it. (A Python ``bytes``
    value would be stored by h5py as a variable-length string.)"""
    dset.attrs.create("MATLAB_class", np.bytes_(cls))


def _tag(dset: Any, name: str, dtype: np.dtype, mat: bool) -> None:
    if mat:
        _matlab_class(dset, _MATLAB_CLASS[np.dtype(dtype)])
        return
    unit = _variable_unit(name)
    if unit:
        dset.attrs["units"] = unit


def _write_h5_metadata(f: Any, metadata: dict[str, Any]) -> None:
    h5py = _h5py()
    f.attrs["metadata"] = to_json(metadata)
    for key in ("software_version", "created", "format", "length_unit", "displacement_unit", "conventions"):
        f.attrs[key] = str(metadata[key])
    f.attrs["export_schema_version"] = int(metadata["export_schema_version"])
    f.attrs["n_frames"] = int(metadata["n_frames"])
    f.attrs["pixel_size"] = float(metadata["pixel_size"])
    f.attrs["frame_rate"] = math.nan if metadata["frame_rate"] is None else float(metadata["frame_rate"])
    f.create_dataset("frame_names", data=np.array(metadata["frame_names"], dtype=object),
                     dtype=h5py.string_dtype(encoding="utf-8"))


def _write_mat_metadata(f: Any, metadata: dict[str, Any]) -> None:
    _write_matlab_char(f, "metadata_json", to_json(metadata))
    _write_matlab_char(f, "frame_names", "\n".join(metadata["frame_names"]))
    _write_matlab_char(f, "length_unit", str(metadata["length_unit"]))
    rate = math.nan if metadata["frame_rate"] is None else float(metadata["frame_rate"])
    for key, value in (("pixel_size", float(metadata["pixel_size"])), ("frame_rate", rate)):
        dset = f.create_dataset(key, data=np.full((1, 1), value, dtype=np.float64))
        _matlab_class(dset, "double")


def _write_matlab_char(f: Any, name: str, text: str) -> None:
    """A MATLAB ``1 x N`` char array: UTF-16 code units, stored ``N x 1`` on disk."""
    units = np.frombuffer(text.encode("utf-16-le"), dtype="<u2")
    if units.size == 0:
        dset = f.create_dataset(name, data=np.zeros(2, dtype=np.uint64))
        dset.attrs.create("MATLAB_empty", np.uint8(1))
    else:
        dset = f.create_dataset(name, data=units.reshape(-1, 1).astype(np.uint16))
    _matlab_class(dset, "char")
    dset.attrs.create("MATLAB_int_decode", np.int32(2))


def _read_matlab_char(dset: Any) -> str:
    if "MATLAB_empty" in dset.attrs and int(dset.attrs["MATLAB_empty"]):
        return ""
    return np.asarray(dset[()], dtype="<u2").ravel().tobytes().decode("utf-16-le")


def _matlab_platform() -> str:
    if sys.platform.startswith("win"):
        return "PCWIN64"
    if sys.platform == "darwin":
        return "MACI64"
    return "GLNXA64"


def _write_matlab_header(path: Path) -> None:
    """The 128-byte MAT-file header at the start of the 512-byte HDF5 user block."""
    now = _dt.datetime.now()
    stamp = f"{_DAYS[now.weekday()]} {_MONTHS[now.month - 1]} {now.day:02d} {now:%H:%M:%S} {now.year}"
    text = f"MATLAB 7.3 MAT-file, Platform: {_matlab_platform()}, Created on: {stamp} HDF5 schema 1.00 ."
    header = text.encode("ascii")[:116].ljust(116, b" ") + b"\x00" * 8 + struct.pack("<H", 0x0200) + b"IM"
    with open(path, "r+b") as fp:
        fp.seek(0)
        fp.write(header)


def _read_hdf5(path: Path, mat: bool) -> dict[str, Any]:
    h5py = _h5py()
    with h5py.File(path, "r") as f:
        if mat:
            if "metadata_json" not in f:
                raise ValueError(f"{path} is not a RAFTcorr export (no metadata_json)")
            metadata = json.loads(_read_matlab_char(f["metadata_json"]))
        else:
            if "metadata" not in f.attrs:
                raise ValueError(f"{path} is not a RAFTcorr export (no metadata attribute)")
            metadata = json.loads(f.attrs["metadata"])
        out: dict[str, Any] = {}
        for name in metadata["variables"]:
            data = np.asarray(f[name][()])
            if mat:
                data = np.ascontiguousarray(data.T).reshape(metadata["shapes"][name])
            out[name] = data
    out["metadata"] = metadata
    return out


# -- NPZ ----------------------------------------------------------------------------------------


def _write_npz(tmp: Path, plan: _Plan, cb: Callbacks) -> None:
    T = plan.series.n_frames
    total = max(1, len(plan.per_frame)) * T
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=_NPZ_LEVEL,
                         allowZip64=True) as zf:
        meta = np.array(to_json(plan.metadata))
        write_npy_entry(zf, "metadata_json.npy", meta.shape, meta.dtype, [meta])
        names = np.array(plan.metadata["frame_names"], dtype=str)
        write_npy_entry(zf, "frame_names.npy", names.shape, names.dtype, [names])
        for name, arr in _static_values(plan).items():
            write_npy_entry(zf, f"{name}.npy", arr.shape, arr.dtype, [arr])
        for i, name in enumerate(plan.per_frame):
            frames = _npz_frames(plan, name, cb, offset=i * T, total=total)
            write_npy_entry(zf, f"{name}.npy", plan.shapes[name], np.float32, frames)
        if not plan.per_frame:
            cb.progress(1.0, "export: done")


def _npz_frames(plan: _Plan, name: str, cb: Callbacks, offset: int, total: int) -> Iterator[np.ndarray]:
    T = plan.series.n_frames
    for k in range(T):
        check_stop(cb)
        yield _frame_values(plan, k, (name,))[name]
        cb.progress((offset + k + 1) / total, f"{name}: {_frame_message(k, T)}")


def _read_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        if "metadata_json" not in z.files:
            raise ValueError(f"{path} is not a RAFTcorr export (no metadata_json)")
        metadata = json.loads(str(z["metadata_json"][()]))
        out: dict[str, Any] = {name: z[name] for name in metadata["variables"]}
    out["metadata"] = metadata
    return out
