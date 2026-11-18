"""Project files (.raftcorr): exact round trip of settings, masks, GUI state and results."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

from raftcorr.callbacks import FunctionCallbacks, StopRequested
from raftcorr.config import RunConfig, StrainConfig, TilingConfig, TrackingConfig
from raftcorr.export import PhysicalUnits
from raftcorr.io.project import PROJECT_EXTENSION, SCHEMA_VERSION, Project, load_project, save_project
from raftcorr.results import DisplacementSeries, StrainSeries

from .test_export_scientific import T, make_series, make_strain

IMAGE_FILES = ("img_000.tif", "img_001.tif", "img_002.tif")


def make_images(folder: Path, files: tuple[str, ...] = IMAGE_FILES) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in files:
        (folder / name).write_bytes(b"")
    return folder


def make_project(image_folder: Path | None, with_results: bool = True) -> Project:
    rng = np.random.default_rng(3)
    return Project(
        image_folder=None if image_folder is None else str(image_folder),
        image_files=IMAGE_FILES,
        natural_sort=False,
        run_config=RunConfig(model="s8", device="cpu", iters=12, corr_mode="volume", smoothing_sigma=0.5,
                             tracking=TrackingConfig(mode="custom", key_frames=(1,)),
                             tiling=TilingConfig(margin_px=16, max_tile_pixels=512 * 512, memory_fraction=0.5,
                                                 blend_px=8)),
        strain_config=StrainConfig(measure="infinitesimal", gauge_px=21, weighting="gaussian", poly_order=2, step=2,
                                   gaussian_sigma=4.0, smoothing_sigma=1.0, connectivity_aware=False,
                                   condition_max=500.0, min_coverage=0.5),
        roi_masks={0: rng.random((50, 60)) > 0.3, 2: rng.random((37, 53)) > 0.5},   # 37 * 53 is not a multiple of 8
        view={"colormap": "turbo", "zoom": 1.5, "limits": {"U": [-1.0, 1.0]}, "overlay": True, "frame": 2,
              "label": "εxx"},
        probes=[{"kind": "point", "x": 12.5, "y": 20.0, "name": "P1"},
                {"kind": "rect", "x0": 5, "y0": 5, "x1": 15, "y1": 12, "name": "R1"}],
        units=PhysicalUnits(0.01, "mm", 50.0),
        series=make_series() if with_results else None,
        strain=make_strain() if with_results else None,
        created="2026-11-01T09:00:00",
    )


def assert_series_equal(a: DisplacementSeries, b: DisplacementSeries) -> None:
    assert a.grid == b.grid
    assert a.frame_names == b.frame_names and a.schedule == b.schedule and a.info == b.info
    for name in ("mask", "u", "v", "done"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype and x.shape == y.shape, name
        np.testing.assert_array_equal(x, y)


def assert_strain_equal(a: StrainSeries, b: StrainSeries) -> None:
    assert a.grid == b.grid and a.step == b.step and a.settings == b.settings
    assert a.names == b.names
    for name in a.names:
        assert a.components[name].dtype == b.components[name].dtype
        np.testing.assert_array_equal(a.components[name], b.components[name])
    np.testing.assert_array_equal(a.done, b.done)


def test_round_trip_everything(tmp_path: Path) -> None:
    project = make_project(make_images(tmp_path / "images"))
    path = save_project(tmp_path / f"session{PROJECT_EXTENSION}", project)
    loaded = load_project(path)

    assert loaded.warnings == ()
    assert loaded.image_folder == project.image_folder
    assert loaded.image_files == IMAGE_FILES and loaded.natural_sort is False
    assert loaded.run_config == project.run_config and loaded.strain_config == project.strain_config
    assert loaded.units == project.units
    assert loaded.view == project.view and loaded.probes == project.probes
    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.created == "2026-11-01T09:00:00"
    dt.datetime.fromisoformat(loaded.modified)

    assert sorted(loaded.roi_masks) == [0, 2]
    for k, mask in project.roi_masks.items():
        assert loaded.roi_masks[k].dtype == bool and loaded.roi_masks[k].shape == mask.shape
        np.testing.assert_array_equal(loaded.roi_masks[k], mask)

    assert_series_equal(loaded.series, project.series)
    assert np.isfinite(loaded.series.u[2]).all()     # raw values of a frame not computed are kept as stored
    assert_strain_equal(loaded.strain, project.strain)


def test_round_trip_without_results(tmp_path: Path) -> None:
    project = dataclasses.replace(make_project(None, with_results=False), roi_masks={}, view={}, probes=[])
    loaded = load_project(save_project(tmp_path / "empty.raftcorr", project))
    assert loaded.series is None and loaded.strain is None
    assert loaded.image_folder is None and loaded.warnings == ()
    assert loaded.roi_masks == {} and loaded.view == {} and loaded.probes == []
    assert loaded.run_config == project.run_config


def test_zip_layout(tmp_path: Path) -> None:
    path = save_project(tmp_path / "p.raftcorr", make_project(make_images(tmp_path / "images")))
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        assert names[0] == "project.json"
        assert {"series/u.npy", "series/v.npy", "series/mask.npy", "series/done.npy", "strain/done.npy"} <= set(names)
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in zf.infolist())
        doc = json.loads(zf.read("project.json"))
    assert doc["format"] == "raftcorr-project" and doc["schema_version"] == SCHEMA_VERSION
    assert [m["frame"] for m in doc["roi_masks"]] == [0, 2]


def test_memmap_series_is_saved(tmp_path: Path) -> None:
    project = make_project(None)
    for name in ("u", "v"):
        np.save(tmp_path / f"{name}.npy", getattr(project.series, name))
    mapped = dataclasses.replace(project.series, u=np.load(tmp_path / "u.npy", mmap_mode="r"),
                                 v=np.load(tmp_path / "v.npy", mmap_mode="r"))
    loaded = load_project(save_project(tmp_path / "m.raftcorr", dataclasses.replace(project, series=mapped)))
    assert_series_equal(loaded.series, project.series)


def test_progress(tmp_path: Path) -> None:
    project = make_project(None)
    seen: list[float] = []
    save_project(tmp_path / "p.raftcorr", project, callbacks=lambda f, _msg: seen.append(f))
    assert len(seen) == 2 * T + len(project.strain.names) * T + 1
    assert seen == sorted(seen) and seen[-1] == 1.0


def test_masks_are_normalised_to_bool() -> None:
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1, 2] = 255
    project = Project(roi_masks={np.int64(0): mask})
    assert list(project.roi_masks) == [0] and project.roi_masks[0].dtype == bool
    assert project.roi_masks[0].sum() == 1 and mask[1, 2] == 255          # input untouched


@pytest.mark.parametrize("kwargs", [
    {"strain": make_strain()},
    {"roi_masks": {0: np.zeros((2, 3, 4), dtype=bool)}},
    {"roi_masks": {-1: np.zeros((2, 3), dtype=bool)}},
])
def test_project_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Project(**kwargs)


# -- file safety and errors ---------------------------------------------------------------------


def test_cancelled_save_keeps_existing_file(tmp_path: Path) -> None:
    path = save_project(tmp_path / "p.raftcorr", make_project(None, with_results=False))
    before = path.read_bytes()
    with pytest.raises(StopRequested):
        save_project(path, make_project(None), callbacks=FunctionCallbacks(should_stop=lambda: True))
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_non_json_view_is_rejected_before_writing(tmp_path: Path) -> None:
    project = dataclasses.replace(make_project(None, with_results=False), view={"widget": object()})
    with pytest.raises(TypeError, match="JSON"):
        save_project(tmp_path / "p.raftcorr", project)
    assert not any(tmp_path.iterdir())


def rewrite_document(src: Path, dst: Path, **changes: object) -> Path:
    """Copy a project file with entries of project.json replaced."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename == "project.json":
                data = json.dumps({**json.loads(data), **changes}).encode("utf-8")
            zout.writestr(info.filename, data)
    return dst


def test_newer_schema_is_rejected(tmp_path: Path) -> None:
    path = save_project(tmp_path / "p.raftcorr", make_project(None, with_results=False))
    newer = rewrite_document(path, tmp_path / "newer.raftcorr", schema_version=SCHEMA_VERSION + 1)
    with pytest.raises(ValueError, match="newer version of RAFTcorr"):
        load_project(newer)


@pytest.mark.parametrize("content", [b"not a zip file", None])
def test_not_a_project(tmp_path: Path, content: bytes | None) -> None:
    path = tmp_path / "bad.raftcorr"
    if content is None:
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("readme.txt", "hello")
    else:
        path.write_bytes(content)
    with pytest.raises(ValueError, match="not a RAFTcorr project"):
        load_project(path)


def test_missing_array_entry_is_reported(tmp_path: Path) -> None:
    path = save_project(tmp_path / "p.raftcorr", make_project(None))
    broken = tmp_path / "broken.raftcorr"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(broken, "w") as zout:
        for info in zin.infolist():
            if info.filename != "series/v.npy":
                zout.writestr(info.filename, zin.read(info.filename))
    with pytest.raises(ValueError, match="damaged or incomplete"):
        load_project(broken)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_project(tmp_path / "nothing.raftcorr")


# -- image folder resolution --------------------------------------------------------------------


def test_missing_image_folder_warns(tmp_path: Path) -> None:
    gone = tmp_path / "gone"
    loaded = load_project(save_project(tmp_path / "p.raftcorr", make_project(gone, with_results=False)))
    assert loaded.image_folder == str(gone)
    assert len(loaded.warnings) == 1 and "image folder not found" in loaded.warnings[0]


def test_missing_image_files_warn(tmp_path: Path) -> None:
    folder = make_images(tmp_path / "images", IMAGE_FILES[:1])
    loaded = load_project(save_project(tmp_path / "p.raftcorr", make_project(folder, with_results=False)))
    assert loaded.image_folder == str(folder)
    assert len(loaded.warnings) == 1 and "2 of 3 image files are missing" in loaded.warnings[0]


def test_moved_project_finds_images_relative_to_it(tmp_path: Path) -> None:
    old = tmp_path / "old"
    save_project(old / "p.raftcorr", make_project(make_images(old / "images"), with_results=False))
    new = tmp_path / "new"
    shutil.move(str(old), str(new))
    loaded = load_project(new / "p.raftcorr")
    assert Path(loaded.image_folder) == new / "images"
    assert len(loaded.warnings) == 1 and "not found; using" in loaded.warnings[0]
