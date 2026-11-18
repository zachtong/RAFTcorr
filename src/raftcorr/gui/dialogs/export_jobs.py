"""What each tab of the export dialog writes, and the thread that writes it.

A job is a frozen description of one export (output names, the results, the settings read from the
dialog). ``targets()`` lists what it would write and ``existing()`` what of that is already on
disk, so the dialog can ask before replacing anything; ``run(report, should_stop, written)`` does
the work with RAFTcorr's exporters (:mod:`raftcorr.export`) and returns the written paths.
:class:`ExportWorker` runs a job off the GUI thread with progress and cancellation.

Output names follow pyALDIC, ``{prefix}_{kind}_{timestamp}`` in the output folder:

=====================================  =================================================
``{p}_parameters_{t}.json``            run parameters (always written with the data)
``{p}_results_{t}.npz / .mat / .h5``   data files (:func:`raftcorr.export.export_results`)
``{p}_npz_{t}/frame_0003.npz``         NPZ per frame (:func:`raftcorr.export.scientific.export_npz_frames`)
``{p}_csv_{t}/frame_0003.csv``         per-frame tables (:mod:`raftcorr.export.frame_csv`)
``{p}_images_{t}/<field>/frame_0003.png``  pictures (:func:`raftcorr.export.figures.export_frames`)
``{p}_animation_{t}/<field>.gif``      animations (:func:`raftcorr.export.figures.export_animation`)
``{p}_report_{t}.html``                report (:mod:`raftcorr.export.report`)
=====================================  =================================================

Numbers in file names and titles are the dialog's frame numbers, 1-based like the frame slider
(frame 1 = the reference image). The Python API numbers from 0 (the image index) by default.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal

from raftcorr.callbacks import StopRequested, check_stop
from raftcorr.export.figures import FrameStyle, export_animation, export_frames
from raftcorr.export.frame_csv import DISPLACEMENT_COLUMNS, export_frame_csv, frame_csv_paths
from raftcorr.export.params import export_params
from raftcorr.export.report import export_html_report
from raftcorr.export.scientific import (
    STRAIN_COORDINATES,
    PhysicalUnits,
    available_variables,
    export_npz_frames,
    export_results,
    npz_frame_paths,
)
from raftcorr.results import DisplacementSeries, StrainSeries
from raftcorr.viz.fields import FieldProvider, Units

_log = logging.getLogger(__name__)

#: ``report(fraction of the whole job, item being written, done, total of that item)``.
Report = Callable[[float, str, int, int], None]
StopCheck = Callable[[], bool]

#: Always in a data file: grid coordinates, ROI and computed-frame flags (pyALDIC always wrote
#: the node coordinates).
STATIC_VARIABLES: tuple[str, ...] = ("X", "Y", "ROI", "Done")
DATA_FORMATS: tuple[str, ...] = (".npz", ".mat", ".h5")


def make_prefix(image_folder: Path | str | None) -> str:
    """File-name prefix from the image folder name (characters Windows forbids become ``_``)."""
    if image_folder is None:
        return "raftcorr"
    stem = Path(image_folder).name or "raftcorr"
    return re.sub(r'[<>:"/\\|?*\s]', "_", stem)


def make_timestamp() -> str:
    """Local time as ``YYYYMMDDHHMMSS``: a fresh one per export keeps exports apart."""
    return datetime.now().strftime("%Y%m%d%H%M%S")


@dataclass(frozen=True)
class OutputNames:
    """Where one export goes (see the module docstring)."""

    folder: Path
    prefix: str
    timestamp: str

    def _name(self, kind: str) -> str:
        return f"{self.prefix}_{kind}_{self.timestamp}"

    def parameters(self) -> Path:
        return self.folder / f"{self._name('parameters')}.json"

    def results(self, ext: str) -> Path:
        return self.folder / f"{self._name('results')}{ext}"

    def csv_folder(self) -> Path:
        return self.folder / self._name("csv")

    def npz_folder(self) -> Path:
        return self.folder / self._name("npz")

    def images_folder(self) -> Path:
        return self.folder / self._name("images")

    def animation_folder(self) -> Path:
        return self.folder / self._name("animation")

    def report(self) -> Path:
        return self.folder / f"{self._name('report')}.html"


@dataclass(frozen=True)
class FieldOutput:
    """One field of a picture export."""

    name: str                 # GUI field name ("disp_u"): folder / file name
    label: str                # user-facing label ("U"): progress messages
    style: FrameStyle         # style.field is the FieldProvider key ("u")
    frames: tuple[int, ...]   # 0-based frames to draw, computed ones only


class _Stage:
    """Callbacks of one exporter call, mapped onto ``[lo, hi]`` of the job's progress."""

    def __init__(self, report: Report, should_stop: StopCheck, lo: float, hi: float, item: str, total: int) -> None:
        self._report, self._should_stop = report, should_stop
        self._lo, self._hi, self._item, self._total = lo, hi, item, max(1, int(total))

    def progress(self, fraction: float, message: str = "") -> None:
        f = min(max(float(fraction), 0.0), 1.0)
        self._report(self._lo + (self._hi - self._lo) * f, self._item, int(round(f * self._total)), self._total)

    def should_stop(self) -> bool:
        return bool(self._should_stop())

    def log(self, message: str, level: str = "info") -> None:
        _log.log(logging.WARNING if level in ("warning", "error") else logging.DEBUG, message)


def _without_missing_images(style: FrameStyle, images: Any) -> FrameStyle:
    return replace(style, background="none") if images is None else style


# ================================================================================== data files
@dataclass(frozen=True)
class DataJob:
    """Parameters JSON, then the data files, then (optionally) the per-frame CSV tables.

    With ``npz_per_frame`` the ``.npz`` of ``formats`` is one file per computed frame instead of
    one merged file.
    """

    names: OutputNames
    series: DisplacementSeries = field(repr=False, compare=False)
    strain: StrainSeries | None = field(repr=False, compare=False)
    formats: tuple[str, ...] = (".npz", ".mat")
    csv: bool = False
    variables: tuple[str, ...] = ("U", "V", "Magnitude")   # per-frame variables chosen in the dialog
    units: PhysicalUnits = PhysicalUnits()
    npz_per_frame: bool = False

    def _per_frame_npz(self, ext: str) -> bool:
        return ext == ".npz" and self.npz_per_frame

    def _strain_used(self) -> StrainSeries | None:
        """The strain series, when a strain variable was chosen (else the files hold none)."""
        if self.strain is None or not any(v in self.strain.components for v in self.variables):
            return None
        return self.strain

    def file_variables(self) -> tuple[str, ...]:
        strain = self._strain_used()
        wanted = set(self.variables) | set(STATIC_VARIABLES)
        if strain is not None:
            wanted |= set(STRAIN_COORDINATES)
        return tuple(v for v in available_variables(strain, self.series) if v in wanted)

    def table_variables(self) -> tuple[str, ...]:
        strain = self._strain_used()
        return tuple(v for v in self.variables
                     if v in DISPLACEMENT_COLUMNS or (strain is not None and v in strain.components))

    def targets(self) -> list[Path]:
        out = [self.names.parameters()]
        for ext in self.formats:
            if self._per_frame_npz(ext):
                out += npz_frame_paths(self.names.npz_folder(), self.series, first_number=GUI_FIRST_FRAME)
            else:
                out.append(self.names.results(ext))
        if self.csv:
            out += frame_csv_paths(self.names.csv_folder(), self.series, self._strain_used(), self.table_variables(),
                                   first_number=GUI_FIRST_FRAME)
        return out

    def existing(self) -> list[Path]:
        return [p for p in self.targets() if p.exists()]

    def run(self, report: Report, should_stop: StopCheck, written: list[Path]) -> list[Path]:
        check_stop(_Stage(report, should_stop, 0, 0, "", 1))
        self.names.folder.mkdir(parents=True, exist_ok=True)
        written.append(export_params(self.names.parameters(), self.series, self.strain, self.units, overwrite=True))
        steps = max(1, len(self.formats) + int(self.csv))
        strain, variables = self._strain_used(), self.file_variables()
        for i, ext in enumerate(self.formats):
            if self._per_frame_npz(ext):
                folder = self.names.npz_folder()
                stage = _Stage(report, should_stop, i / steps, (i + 1) / steps, folder.name, 100)
                try:
                    written.extend(export_npz_frames(folder, self.series, strain, variables, self.units,
                                                     overwrite=True, callbacks=stage, first_number=GUI_FIRST_FRAME))
                except StopRequested:
                    written.extend(sorted(folder.glob("frame_*.npz")))
                    raise
                continue
            path = self.names.results(ext)
            stage = _Stage(report, should_stop, i / steps, (i + 1) / steps, path.name, 100)
            written.append(export_results(path, self.series, strain, variables, self.units, overwrite=True,
                                          callbacks=stage))
        if self.csv:
            folder = self.names.csv_folder()
            stage = _Stage(report, should_stop, (steps - 1) / steps, 1.0, folder.name, 100)
            try:
                written.extend(export_frame_csv(folder, self.series, strain, self.table_variables(), overwrite=True,
                                                callbacks=stage, first_number=GUI_FIRST_FRAME))
            except StopRequested:
                written.extend(sorted(folder.glob("*.csv")))
                raise
        return list(written)


#: The dialog shows the first image as frame 1; file names and titles use the same numbers.
GUI_FIRST_FRAME = 1

# ==================================================================================== pictures
@dataclass(frozen=True)
class ImagesJob:
    """One picture per frame and field, ``{images folder}/<field>/frame_<k>.<ext>``."""

    names: OutputNames
    series: DisplacementSeries = field(repr=False, compare=False)
    strain: StrainSeries | None = field(repr=False, compare=False)
    images: Any = field(repr=False, compare=False)
    fields: tuple[FieldOutput, ...] = ()
    units: Units = Units()
    fmt: str = ".png"
    max_side: int = 0           # long-edge cap in pixels, 0 = the size set by the DPI
    jpeg_quality: int = 92

    def folder(self, out: FieldOutput) -> Path:
        return self.names.images_folder() / out.name

    def targets(self) -> list[Path]:
        return [self.folder(f) for f in self.fields]

    def existing(self) -> list[Path]:
        return [d for d in self.targets() if d.is_dir() and any(d.iterdir())]

    def run(self, report: Report, should_stop: StopCheck, written: list[Path]) -> list[Path]:
        provider = FieldProvider(self.series, self.strain)  # private caches: this runs off the GUI thread
        total = max(1, sum(len(f.frames) for f in self.fields))
        done = 0
        for f in self.fields:
            stage = _Stage(report, should_stop, done / total, (done + len(f.frames)) / total, f.label, len(f.frames))
            folder = self.folder(f)
            try:
                written.extend(export_frames(folder, provider, self.images, f.frames,
                                             _without_missing_images(f.style, self.images), self.units, self.fmt,
                                             prefix="frame", callbacks=stage, first_number=GUI_FIRST_FRAME,
                                             max_side=self.max_side, jpeg_quality=self.jpeg_quality))
            except StopRequested:
                written.extend(sorted(folder.glob(f"frame_*{self.fmt}")))
                raise
            done += len(f.frames)
        return list(written)


@dataclass(frozen=True)
class AnimationJob:
    """One animation per field, ``{animation folder}/<field>.<ext>``, one colour range per field."""

    names: OutputNames
    series: DisplacementSeries = field(repr=False, compare=False)
    strain: StrainSeries | None = field(repr=False, compare=False)
    images: Any = field(repr=False, compare=False)
    fields: tuple[FieldOutput, ...] = ()
    units: Units = Units()
    ext: str = ".gif"
    fps: float = 10.0
    max_side: int = 1024        # long-edge cap in pixels, 0 = the images' own size

    def path(self, out: FieldOutput) -> Path:
        return self.names.animation_folder() / f"{out.name}{self.ext}"

    def targets(self) -> list[Path]:
        return [self.path(f) for f in self.fields]

    def existing(self) -> list[Path]:
        return [p for p in self.targets() if p.exists()]

    def run(self, report: Report, should_stop: StopCheck, written: list[Path]) -> list[Path]:
        provider = FieldProvider(self.series, self.strain)
        total = max(1, sum(len(f.frames) for f in self.fields))
        done = 0
        for f in self.fields:
            stage = _Stage(report, should_stop, done / total, (done + len(f.frames)) / total, f.label, len(f.frames))
            written.append(export_animation(self.path(f), provider, self.images, f.frames,
                                            _without_missing_images(f.style, self.images), self.units,
                                            fps=self.fps, max_side=self.max_side, callbacks=stage,
                                            first_number=GUI_FIRST_FRAME))
            done += len(f.frames)
        return list(written)


# ====================================================================================== report
@dataclass(frozen=True)
class ReportJob:
    """The HTML report; the pictures use each field's style (colour map, range, opacity)."""

    names: OutputNames
    series: DisplacementSeries = field(repr=False, compare=False)
    strain: StrainSeries | None = field(repr=False, compare=False)
    images: Any = field(repr=False, compare=False)
    fields: tuple[FieldOutput, ...] = ()
    units: Units = Units()
    include_parameters: bool = True
    include_statistics: bool = True
    include_images: bool = True
    sample_every: int = 5

    def targets(self) -> list[Path]:
        return [self.names.report()]

    def existing(self) -> list[Path]:
        return [p for p in self.targets() if p.exists()]

    def run(self, report: Report, should_stop: StopCheck, written: list[Path]) -> list[Path]:
        path = self.names.report()
        stage = _Stage(report, should_stop, 0.0, 1.0, path.name, 100)
        styles = {f.style.field: _without_missing_images(f.style, self.images) for f in self.fields}
        written.append(export_html_report(
            path, FieldProvider(self.series, self.strain), self.images, [f.style.field for f in self.fields],
            units=self.units, styles=styles, include_parameters=self.include_parameters,
            include_statistics=self.include_statistics, include_images=self.include_images,
            sample_every=self.sample_every, title=f"RAFTcorr report: {self.names.prefix}", overwrite=True,
            callbacks=stage))
        return list(written)


# ====================================================================================== worker
class ExportWorker(QThread):
    """Runs one job; ``request_stop()`` cancels it at the next frame."""

    progress = Signal(float, str, int, int)   # fraction of the job, item, done, total of the item
    succeeded = Signal(list)                  # written paths (str)
    failed = Signal(str)                      # error message
    cancelled = Signal(list)                  # paths written before the stop (str)

    def __init__(self, job: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self._job = job
        self._stop = threading.Event()

    @property
    def job(self) -> Any:
        return self._job

    def request_stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        written: list[Path] = []
        try:
            paths = self._job.run(self.progress.emit, self._stop.is_set, written)
        except StopRequested:
            self.cancelled.emit([str(p) for p in written])
        except Exception as exc:  # reported to the user; the traceback goes to the log
            _log.exception("export failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.succeeded.emit([str(p) for p in paths])
