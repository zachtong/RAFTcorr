"""Scientific export (HDF5 / MAT v7.3 / NPZ) and CSV tables: round trips and on-disk layout."""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from raftcorr._version import __version__
from raftcorr.callbacks import FunctionCallbacks, StopRequested
from raftcorr.engine.mapping import Grid
from raftcorr.export import (
    CONVENTIONS,
    DISPLACEMENT_VARIABLES,
    PhysicalUnits,
    available_variables,
    export_results,
    probe_table_csv,
    read_results,
    series_statistics,
)
from raftcorr.post.strain import COMPONENTS
from raftcorr.results import DisplacementSeries, RunInfo, StrainSeries, StrainSettings
from raftcorr.schedule import FrameSchedule

T, H, W, STEP = 3, 20, 30, 2
HS, WS = -(-H // STEP), -(-W // STEP)
GRID = Grid(x0=5, y0=7, width=W, height=H, image_height=50, image_width=60)
FRAME_NAMES = ("img_000.tif", "img_001.tif", "img_002.tif")
ALL_FORMATS = [".npz", ".mat", ".h5", ".hdf5"]


# -- synthetic results (also used by test_project_io) -------------------------------------------


def make_series(seed: int = 0) -> DisplacementSeries:
    """T=3 frames on a 20 x 30 grid; a hole in the ROI, untracked points, frame 2 not computed."""
    rng = np.random.default_rng(seed)
    mask = np.ones((H, W), dtype=bool)
    mask[4:9, 10:16] = False
    u = rng.normal(size=(T, H, W)).astype(np.float32)
    v = rng.normal(size=(T, H, W)).astype(np.float32)
    u[0], v[0] = 0.0, 0.0
    u[1, 2, 3] = np.nan
    v[1, 12, 20] = np.nan
    info = RunInfo(
        software_version=__version__, model_name="RAFTcorr-s4", model_sha256="ab" * 32, downsample_factor=4,
        iters=20, corr_mode="tile", device="cpu", config={"iters": 20, "tracking": {"mode": "incremental"}},
        started="2026-11-18T10:00:00", seconds=1.5, notes=("synthetic",),
    )
    return DisplacementSeries(GRID, mask, u, v, np.array([True, True, False]), FRAME_NAMES,
                              FrameSchedule.incremental(T), info)


def make_strain(seed: int = 1) -> StrainSeries:
    rng = np.random.default_rng(seed)
    comps = {name: rng.normal(size=(T, HS, WS)).astype(np.float32) for name in COMPONENTS}
    for arr in comps.values():
        arr[:, 2:4, 5:8] = np.nan
    settings = StrainSettings(measure="green_lagrange", gauge_px=41, weighting="uniform", poly_order=1, step=STEP,
                              gaussian_sigma=None, smoothing_sigma=0.0, connectivity_aware=True)
    return StrainSeries(GRID, STEP, comps, np.array([True, True, False]), settings)


@pytest.fixture
def series() -> DisplacementSeries:
    return make_series()


@pytest.fixture
def strain() -> StrainSeries:
    return make_strain()


def expected_uv(series: DisplacementSeries) -> tuple[np.ndarray, np.ndarray]:
    """What an export must hold: NaN outside the ROI and in frames not computed."""
    keep = series.mask[None] & series.done[:, None, None]
    return np.where(keep, series.u, np.nan).astype(np.float32), np.where(keep, series.v, np.nan).astype(np.float32)


def expected_strain(strain: StrainSeries, name: str) -> np.ndarray:
    return np.where(strain.done[:, None, None], strain.components[name], np.nan).astype(np.float32)


def entries(path: Path) -> set[str]:
    if path.suffix == ".npz":
        with zipfile.ZipFile(path) as zf:
            return {n[: -len(".npy")] for n in zf.namelist()}
    import h5py

    with h5py.File(path, "r") as f:
        return set(f.keys())


# -- round trips --------------------------------------------------------------------------------


@pytest.mark.parametrize("ext", ALL_FORMATS)
def test_round_trip_every_variable(tmp_path: Path, series: DisplacementSeries, strain: StrainSeries, ext: str) -> None:
    data = read_results(export_results(tmp_path / f"out{ext}", series, strain))
    assert set(data) == set(available_variables(strain)) | {"metadata"}

    u, v = expected_uv(series)
    np.testing.assert_array_equal(data["U"], u)
    np.testing.assert_array_equal(data["V"], v)
    np.testing.assert_allclose(data["Magnitude"], np.hypot(u, v), rtol=1e-6)
    assert np.isnan(data["U"][1, 5, 12])          # hole in the ROI
    assert np.isnan(data["U"][1, 2, 3])           # untracked point
    assert np.isnan(data["U"][2]).all()           # frame not computed
    assert np.isfinite(data["U"][1, 0, 0])

    X, Y = GRID.coords()
    np.testing.assert_array_equal(data["X"], X)
    np.testing.assert_array_equal(data["Y"], Y)
    assert data["X"][0, 0] == 5 and data["Y"][0, 0] == 7 and data["X"][0, -1] == 5 + W - 1
    np.testing.assert_array_equal(data["ROI"], series.mask.astype(np.uint8))
    np.testing.assert_array_equal(data["Done"], [1, 1, 0])

    for name in COMPONENTS:
        np.testing.assert_array_equal(data[name], expected_strain(strain, name))
    np.testing.assert_array_equal(data["X_strain"][0], 5 + STEP * np.arange(WS))
    np.testing.assert_array_equal(data["Y_strain"][:, 0], 7 + STEP * np.arange(HS))

    for name in ("U", "V", "Magnitude", *COMPONENTS):
        assert data[name].dtype == np.float32
    assert data["X"].dtype == data["X_strain"].dtype == np.float64
    assert data["ROI"].dtype == data["Done"].dtype == np.uint8
    assert data["U"].shape == (T, H, W) and data["exx"].shape == (T, HS, WS)
    assert data["Done"].shape == (T,) and data["X_strain"].shape == (HS, WS)


@pytest.mark.parametrize("ext", ALL_FORMATS)
def test_metadata_content(tmp_path: Path, series: DisplacementSeries, strain: StrainSeries, ext: str) -> None:
    units = PhysicalUnits(scale=0.005, unit="µm", frame_rate=25.0)
    meta = read_results(export_results(tmp_path / f"out{ext}", series, strain, units=units))["metadata"]

    assert meta["software_version"] == __version__
    dt.datetime.fromisoformat(meta["created"])
    assert meta["frame_names"] == list(FRAME_NAMES)
    assert meta["ref_indices"] == [0, 1]
    assert meta["run_info"] == json.loads(json.dumps(series.info.to_dict()))
    assert RunInfo.from_dict(meta["run_info"]) == series.info
    assert meta["grid"] == GRID.to_dict()
    assert meta["strain_settings"] == strain.settings.to_dict()
    assert meta["strain_step"] == STEP and meta["strain_done_frames"] == [0, 1]
    assert (meta["pixel_size"], meta["length_unit"], meta["frame_rate"]) == (0.005, "µm", 25.0)
    assert meta["conventions"] == CONVENTIONS
    assert "u +right, v +down" in meta["conventions"] and "tensor shear" in meta["conventions"]
    assert meta["variables"] == list(available_variables(strain))
    assert meta["shapes"]["U"] == [T, H, W] and meta["shapes"]["Done"] == [T]
    assert meta["variable_units"]["U"] == "px" and meta["variable_units"]["rotation"] == "deg"


def test_metadata_defaults_without_strain(tmp_path: Path, series: DisplacementSeries) -> None:
    meta = read_results(export_results(tmp_path / "out.npz", series))["metadata"]
    assert meta["strain_settings"] is None
    assert (meta["pixel_size"], meta["length_unit"], meta["frame_rate"]) == (1.0, "px", None)
    assert meta["variables"] == list(DISPLACEMENT_VARIABLES)


@pytest.mark.parametrize("ext", ALL_FORMATS)
def test_variables_subset(tmp_path: Path, series: DisplacementSeries, strain: StrainSeries, ext: str) -> None:
    path = export_results(tmp_path / f"out{ext}", series, strain, variables=["exy", "U", "Done"])
    data = read_results(path)
    assert set(data) == {"U", "Done", "exy", "metadata"}
    assert data["metadata"]["variables"] == ["U", "Done", "exy"]      # file order, not request order
    assert not {"V", "Magnitude", "X", "exx", "X_strain"} & entries(path)
    np.testing.assert_array_equal(data["exy"], expected_strain(strain, "exy"))


def test_single_variable_given_as_string(tmp_path: Path, series: DisplacementSeries) -> None:
    data = read_results(export_results(tmp_path / "out.h5", series, variables="Magnitude"))
    assert set(data) == {"Magnitude", "metadata"}


def test_streams_from_memmaps(tmp_path: Path, series: DisplacementSeries) -> None:
    for name, arr in (("u", series.u), ("v", series.v)):
        mm = np.lib.format.open_memmap(tmp_path / f"{name}.npy", mode="w+", dtype=np.float32, shape=arr.shape)
        mm[:] = arr
        mm.flush()
        del mm
    mapped = dataclasses.replace(series, u=np.load(tmp_path / "u.npy", mmap_mode="r"),
                                 v=np.load(tmp_path / "v.npy", mmap_mode="r"))
    u, v = expected_uv(series)
    for ext in (".h5", ".npz", ".mat"):
        data = read_results(export_results(tmp_path / f"mapped{ext}", mapped, variables=["U", "V"]))
        np.testing.assert_array_equal(data["U"], u)
        np.testing.assert_array_equal(data["V"], v)


# -- errors and file safety ---------------------------------------------------------------------


@pytest.mark.parametrize("ext", ALL_FORMATS)
def test_overwrite_protection(tmp_path: Path, series: DisplacementSeries, ext: str) -> None:
    path = tmp_path / f"out{ext}"
    export_results(path, series, variables=["U"])
    with pytest.raises(FileExistsError):
        export_results(path, series, variables=["V"])
    assert read_results(path)["metadata"]["variables"] == ["U"]
    export_results(path, series, variables=["V"], overwrite=True)
    assert read_results(path)["metadata"]["variables"] == ["V"]
    assert [p.name for p in tmp_path.iterdir()] == [path.name]        # no temporary file left


@pytest.mark.parametrize("ext", ALL_FORMATS)
def test_cancelled_export_keeps_existing_file(tmp_path: Path, series: DisplacementSeries, ext: str) -> None:
    path = export_results(tmp_path / f"out{ext}", series, variables=["U"])
    before = path.read_bytes()
    with pytest.raises(StopRequested):
        export_results(path, series, overwrite=True, callbacks=FunctionCallbacks(should_stop=lambda: True))
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_unknown_variable_lists_valid_names(tmp_path: Path, series: DisplacementSeries, strain: StrainSeries) -> None:
    with pytest.raises(ValueError, match="valid names") as exc:
        export_results(tmp_path / "out.h5", series, strain, variables=["U", "Ux"])
    assert "'Ux'" in str(exc.value) and "Magnitude" in str(exc.value) and "von_mises" in str(exc.value)
    assert not any(tmp_path.iterdir())


def test_strain_variable_needs_strain(tmp_path: Path, series: DisplacementSeries) -> None:
    with pytest.raises(ValueError, match="strain"):
        export_results(tmp_path / "out.h5", series, variables=["exx"])


def test_empty_variable_list(tmp_path: Path, series: DisplacementSeries) -> None:
    with pytest.raises(ValueError, match="no variables"):
        export_results(tmp_path / "out.npz", series, variables=[])


@pytest.mark.parametrize("name", ["out.csv", "out.txt", "out"])
def test_unknown_extension(tmp_path: Path, series: DisplacementSeries, name: str) -> None:
    with pytest.raises(ValueError, match="unknown export format"):
        export_results(tmp_path / name, series)


def test_strain_must_match_series(tmp_path: Path, series: DisplacementSeries, strain: StrainSeries) -> None:
    moved = dataclasses.replace(strain, grid=Grid(0, 0, W, H, 50, 60))
    with pytest.raises(ValueError, match="grid"):
        export_results(tmp_path / "out.h5", series, moved)


@pytest.mark.parametrize("ext", ALL_FORMATS)
def test_progress_reported_per_frame(tmp_path: Path, series: DisplacementSeries, strain: StrainSeries,
                                     ext: str) -> None:
    seen: list[float] = []
    export_results(tmp_path / f"out{ext}", series, strain, callbacks=lambda f, _msg: seen.append(f))
    assert len(seen) >= T and len(seen) % T == 0
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)


# -- on-disk layout -----------------------------------------------------------------------------


def test_mat_layout(tmp_path: Path, series: DisplacementSeries, strain: StrainSeries) -> None:
    h5py = pytest.importorskip("h5py")
    path = export_results(tmp_path / "out.mat", series, strain, units=PhysicalUnits(0.01, "µm"))
    header = path.read_bytes()[:128]
    assert header.startswith(b"MATLAB 7.3 MAT-file")
    assert header[124:126] == struct.pack("<H", 0x0200) and header[126:128] == b"IM"

    u, _ = expected_uv(series)
    with h5py.File(path, "r") as f:
        assert f.userblock_size == 512
        assert f["U"].shape == (W, H, T) and f["U"].chunks == (W, H, 1)
        assert f["exx"].shape == (WS, HS, T)
        assert f["X"].shape == (W, H) and f["X_strain"].shape == (WS, HS)
        assert f["Done"].shape == (1, T)
        np.testing.assert_array_equal(f["U"][:, :, 1].T, u[1])
        names = ("U", "exx", "X", "ROI", "Done", "metadata_json", "frame_names", "length_unit", "pixel_size")
        classes = {name: f[name].attrs["MATLAB_class"] for name in names}
        assert classes == {"U": b"single", "exx": b"single", "X": b"double", "ROI": b"uint8", "Done": b"uint8",
                           "metadata_json": b"char", "frame_names": b"char", "length_unit": b"char",
                           "pixel_size": b"double"}
        for name in names:      # MATLAB writes (and expects) fixed-length ASCII, not variable-length strings
            assert not f[name].attrs.get_id("MATLAB_class").get_type().is_variable_str()
        chars = f["frame_names"]
        assert chars.dtype == np.uint16 and chars.shape[1] == 1 and int(chars.attrs["MATLAB_int_decode"]) == 2
        assert chars[()].ravel().tobytes().decode("utf-16-le") == "\n".join(FRAME_NAMES)
        meta = json.loads(f["metadata_json"][()].ravel().tobytes().decode("utf-16-le"))
        assert meta["length_unit"] == "µm"
        assert f["length_unit"][()].ravel().tobytes().decode("utf-16-le") == "µm"
        assert f["pixel_size"].shape == (1, 1) and f["pixel_size"][0, 0] == 0.01
        assert np.isnan(f["frame_rate"][0, 0])


def test_hdf5_layout(tmp_path: Path, series: DisplacementSeries, strain: StrainSeries) -> None:
    h5py = pytest.importorskip("h5py")
    path = export_results(tmp_path / "out.h5", series, strain, units=PhysicalUnits(0.01, "mm", 100.0))
    with h5py.File(path, "r") as f:
        assert f.userblock_size == 0
        U = f["U"]
        assert U.shape == (T, H, W) and U.chunks == (1, H, W)
        assert U.compression == "gzip" and U.compression_opts == 4 and U.shuffle
        assert f["exx"].chunks == (1, HS, WS)
        assert U.attrs["units"] == "px" and f["principal_angle"].attrs["units"] == "deg"
        assert json.loads(f.attrs["metadata"])["grid"] == GRID.to_dict()
        assert f.attrs["software_version"] == __version__
        assert (f.attrs["pixel_size"], f.attrs["length_unit"], f.attrs["frame_rate"]) == (0.01, "mm", 100.0)
        assert list(f["frame_names"].asstr()[()]) == list(FRAME_NAMES)


def test_npz_layout(tmp_path: Path, series: DisplacementSeries) -> None:
    path = export_results(tmp_path / "out.npz", series)
    with zipfile.ZipFile(path) as zf:
        assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in zf.infolist())
    with np.load(path, allow_pickle=False) as z:
        assert z["metadata_json"].shape == ()
        assert json.loads(str(z["metadata_json"]))["frame_names"] == list(FRAME_NAMES)
        assert list(z["frame_names"]) == list(FRAME_NAMES)


# -- units --------------------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [{"scale": 0}, {"scale": -1.0}, {"scale": float("nan")}, {"unit": ""},
                                    {"unit": "  "}, {"frame_rate": 0}, {"frame_rate": float("inf")}])
def test_physical_units_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PhysicalUnits(**kwargs)


def test_physical_units_round_trip() -> None:
    units = PhysicalUnits(0.01, "mm", 30)
    assert PhysicalUnits.from_dict(units.to_dict()) == units
    assert isinstance(units.frame_rate, float)
    assert PhysicalUnits().is_pixels and not units.is_pixels


# -- tables -------------------------------------------------------------------------------------


def read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_probe_table_csv(tmp_path: Path) -> None:
    columns = {"U_P1": np.array([0.1, np.nan, -2.5], dtype=np.float32), "V_P1 (px)": np.array([1.0, 2.0, 3.0])}
    path = probe_table_csv(tmp_path / "probes.csv", [0, 2, 5], columns, frame_rate=10.0)
    assert read_csv(path) == [
        ["frame", "time_s", "U_P1", "V_P1 (px)"],
        ["0", "0.0", "0.1", "1.0"],
        ["2", "0.2", "", "2.0"],
        ["5", "0.5", "-2.5", "3.0"],
    ]


def test_probe_table_without_frame_rate(tmp_path: Path) -> None:
    path = probe_table_csv(tmp_path / "p.csv", [1, 2], {"εxx": [np.nan, 0.25]})
    assert read_csv(path) == [["frame", "εxx"], ["1", ""], ["2", "0.25"]]


@pytest.mark.parametrize("frames, columns, rate, error", [
    ([0, 1], {"a": [1.0]}, None, ValueError),
    ([0, 1], {"frame": [1.0, 2.0]}, None, ValueError),
    ([0, 1], {"time_s": [1.0, 2.0]}, None, ValueError),
    ([0, -1], {"a": [1.0, 2.0]}, None, ValueError),
    ([0, 1], {"a": [1.0, 2.0]}, 0.0, ValueError),
    ([0, 1.5], {"a": [1.0, 2.0]}, None, TypeError),
    ([0, 1], {"a": np.zeros((2, 2))}, None, ValueError),
])
def test_probe_table_errors(tmp_path: Path, frames: list, columns: dict, rate: float | None, error: type) -> None:
    with pytest.raises(error):
        probe_table_csv(tmp_path / "p.csv", frames, columns, frame_rate=rate)
    assert not any(tmp_path.iterdir())


def test_series_statistics(series: DisplacementSeries) -> None:
    stats = series_statistics(series, 1)
    valid = series.frame_mask(1)
    u = series.u[1][valid].astype(np.float64)
    v = series.v[1][valid].astype(np.float64)
    for name, x in (("U", u), ("V", v), ("Magnitude", np.hypot(u, v))):
        s = stats[name]
        assert s["count"] == int(valid.sum())
        assert (s["min"], s["max"]) == (pytest.approx(x.min()), pytest.approx(x.max()))
        assert (s["mean"], s["std"], s["median"]) == (pytest.approx(x.mean()), pytest.approx(x.std()),
                                                      pytest.approx(np.median(x)))


def test_series_statistics_empty_and_out_of_range(series: DisplacementSeries) -> None:
    stats = series_statistics(series, 2)                          # frame not computed
    assert stats["U"]["count"] == 0 and np.isnan(stats["U"]["mean"])
    with pytest.raises(IndexError):
        series_statistics(series, T)
