"""The analysis engine on dense fields (port of pyALDIC's ``test_analysis/test_engine.py``,
``test_extract.py`` and ``test_series.py``).

Every expected number comes from the physics of a hand-built field, not from the
implementation. pyALDIC's runs were values at mesh nodes; here a run is a
:class:`~raftcorr.results.DisplacementSeries` on a pixel grid, a crack is material whose
displacement is NaN from some frame on (what the pipeline writes for a region cut out of a
deformed frame's ROI), and a hole is a gap in the frame-0 ROI.

``build_series`` / ``build_strain`` are also used by ``test_gui_analysis.py``.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import pytest

from raftcorr.analysis.engine import AnalysisEngine, SampleStatus
from raftcorr.analysis.extract import extract_series, field_values, frame_count
from raftcorr.analysis.probes import AreaGeom, LineGeom, PointGeom, Probe
from raftcorr.analysis.series import FrameStatus, TimeSeries

SHAPE = (48, 48)

Field = Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]]


# ---------------------------------------------------------------------------
# Synthetic runs
# ---------------------------------------------------------------------------

def build_series(disp: Field, n_frames: int, *, shape: tuple[int, int] = SHAPE,
                 roi: np.ndarray | None = None, lost: dict[int, np.ndarray] | None = None,
                 done: list[bool] | None = None):
    """``n_frames`` deformed frames after the reference: frame ``f`` displaces the
    material point at pixel ``(x, y)`` by ``disp(X, Y, f)``; frame 0 is zero.

    ``roi`` is the frame-0 ROI (image shape; default the whole image). ``lost[f]``
    (image shape) marks material untracked on frame ``f`` and after: NaN displacement.
    """
    from raftcorr.engine.mapping import Grid
    from raftcorr.results import DisplacementSeries, RunInfo
    from raftcorr.roi.masks import bbox_of
    from raftcorr.schedule import FrameSchedule

    roi = np.ones(shape, dtype=bool) if roi is None else np.asarray(roi, dtype=bool)
    bx, by, bw, bh = bbox_of(roi)
    grid = Grid(bx, by, bw, bh, *shape)
    mask = roi[grid.slices].copy()
    X, Y = grid.coords()
    T = n_frames + 1
    done_arr = np.ones(T, dtype=bool) if done is None else np.asarray(done, dtype=bool)
    done_arr[0] = True
    u = np.full((T,) + grid.shape, np.nan, dtype=np.float32)
    v = np.full_like(u, np.nan)
    for f in range(T):
        if not done_arr[f]:
            continue
        uf, vf = (np.zeros_like(X), np.zeros_like(Y)) if f == 0 else disp(X, Y, f)
        u[f] = np.where(mask, uf, np.nan)
        v[f] = np.where(mask, vf, np.nan)
    for first, where in (lost or {}).items():
        gone = np.asarray(where, dtype=bool)[grid.slices]
        for f in range(first, T):
            u[f][gone] = np.nan
            v[f][gone] = np.nan
    info = RunInfo(software_version="test", model_name="synthetic", model_sha256="ab" * 32,
                   downsample_factor=8, iters=20, corr_mode="volume", device="cpu",
                   config={"tracking": {"mode": "accumulative"}, "smoothing_sigma": 0.0})
    names = tuple(f"img_{k:02d}.png" for k in range(T))
    return DisplacementSeries(grid, mask, u, v, done_arr, names, FrameSchedule.accumulative(T), info)


def build_strain(series, value: Callable[[np.ndarray, np.ndarray, int], np.ndarray], *,
                 step: int = 1, untrusted: np.ndarray | None = None,
                 done: list[bool] | None = None):
    """Every strain component ``value(X, Y, f)`` on ``series.grid`` subsampled by ``step``.

    NaN outside the ROI and where the displacement is NaN (as the strain stage leaves it),
    and where ``untrusted`` (image shape) is True: a strain gauge without enough points.
    """
    from raftcorr.post.strain import COMPONENTS
    from raftcorr.results import StrainSeries, StrainSettings

    g = series.grid
    X, Y = g.coords()
    Xs, Ys = X[::step, ::step], Y[::step, ::step]
    keep_static = series.mask[::step, ::step].copy()
    if untrusted is not None:
        keep_static &= ~np.asarray(untrusted, dtype=bool)[g.slices][::step, ::step]
    T = series.n_frames
    done_arr = np.array(series.done if done is None else done, dtype=bool)
    arr = np.full((T,) + Xs.shape, np.nan, dtype=np.float32)
    for f in range(T):
        if done_arr[f]:
            tracked = np.isfinite(series.u[f][::step, ::step])
            arr[f] = np.where(keep_static & tracked, value(Xs, Ys, f) + 0 * Xs, np.nan)
    comps = {name: arr.copy() for name in COMPONENTS}
    settings = StrainSettings(measure="green_lagrange", gauge_px=41, weighting="uniform", poly_order=1,
                              step=step, gaussian_sigma=None, smoothing_sigma=0.0, connectivity_aware=True)
    return StrainSeries(g, step, comps, done_arr, settings)


def stretch(rate: float = 0.01) -> Field:
    """Uniaxial: u = rate * f * x, v = 0 -- engineering strain rate * f."""
    return lambda x, y, f: (rate * f * x, np.zeros_like(y))


def probe(kind: str, geom) -> Probe:
    return Probe(id=1, kind=kind, geometry=geom, label="P1", color="#ef4444")


def column(x: int, shape: tuple[int, int] = SHAPE) -> np.ndarray:
    """Pixels of image column ``x``: a crack along it."""
    m = np.zeros(shape, dtype=bool)
    m[:, x] = True
    return m


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_a_point_is_bilinear_on_a_linear_field():
    s = build_series(lambda x, y, f: (0.1 * x + 0.02 * y, 0.0 * x), 1)
    eng = AnalysisEngine(s)
    smp = eng.sample(eng.plan(PointGeom(13.3, 7.6)), "disp_u", 1)
    assert smp.status[0] == SampleStatus.VALID
    assert smp.values[0] == pytest.approx(0.1 * 13.3 + 0.02 * 7.6, abs=1e-6)


def test_a_point_on_a_cropped_grid_reads_its_image_position():
    """The result grid is the ROI's bounding box; probes stay in image pixels."""
    roi = np.zeros(SHAPE, dtype=bool)
    roi[10:40, 5:45] = True
    s = build_series(lambda x, y, f: (0.1 * x, 0.2 * y), 1, roi=roi)
    assert (s.grid.x0, s.grid.y0) == (5, 10)
    ts = AnalysisEngine(s).series(probe("point", PointGeom(20.5, 30.25)), "disp_v", "value")
    assert ts.values[1] == pytest.approx(0.2 * 30.25, abs=1e-5)


def test_a_line_carries_its_reference_arc_length():
    eng = AnalysisEngine(build_series(stretch(), 1))
    plan = eng.plan(LineGeom(4.0, 8.0, 34.0, 8.0))
    assert plan.distance[0] == 0.0
    assert plan.distance[-1] == pytest.approx(30.0)
    assert np.all(np.diff(plan.distance) > 0)


def test_line_sampling_follows_the_grid_step():
    s = build_series(stretch(), 1)
    eng = AnalysisEngine(s, build_strain(s, lambda x, y, f: 0.01 * f, step=4))
    line = LineGeom(4.0, 8.0, 36.0, 8.0)
    assert eng.plan(line, "disp_u").size == 33        # one per pixel, plus the far end
    assert eng.plan(line, "strain_exx").size == 9     # one per strain cell


# ---------------------------------------------------------------------------
# Gauges: the extensometer must not be scaled by the pixel size
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pixel_size", [1.0, 0.005, 300.0])
def test_extensometer_strain_does_not_depend_on_the_unit(pixel_size):
    eng = AnalysisEngine(build_series(stretch(0.01), 3))
    ts = eng.series(probe("line", LineGeom(4.0, 20.0, 36.0, 20.0)),
                    None, "strain", pixel_size=pixel_size)
    np.testing.assert_allclose(ts.values, [0.0, 0.01, 0.02, 0.03], atol=1e-7)
    assert ts.unit == ""


def test_extensometer_reads_zero_under_a_rigid_rotation():
    theta = math.radians(4.0)
    c, s = math.cos(theta), math.sin(theta)
    rot = lambda x, y, f: ((c - 1) * x - s * y, s * x + (c - 1) * y)  # noqa: E731
    eng = AnalysisEngine(build_series(rot, 2))
    ts = eng.series(probe("line", LineGeom(8.0, 12.0, 30.0, 26.0)),
                    None, "strain", pixel_size=0.005)
    np.testing.assert_allclose(ts.values, 0.0, atol=1e-6)


def test_true_strain_and_elongation():
    eng = AnalysisEngine(build_series(stretch(0.1), 2))
    line = probe("line", LineGeom(4.0, 20.0, 24.0, 20.0))        # L0 = 20 px
    true = eng.series(line, None, "true_strain")
    elong = eng.series(line, None, "elongation", pixel_size=0.5, length_unit="mm")
    np.testing.assert_allclose(true.values, np.log([1.0, 1.1, 1.2]), atol=1e-6)
    np.testing.assert_allclose(elong.values, [0.0, 1.0, 2.0], atol=1e-5)  # 20px*0.1*0.5mm
    assert elong.unit == "mm"


def test_crack_opening_components():
    """Opening along the gauge, sliding across it, in the length unit."""
    shift = lambda x, y, f: (np.where(x > 22, 2.0 * f, 0.0),   # noqa: E731
                             np.where(x > 22, 1.0 * f, 0.0))
    eng = AnalysisEngine(build_series(shift, 1))
    gauge = probe("line", LineGeom(12.0, 20.0, 32.0, 20.0))        # along +x
    kw = dict(pixel_size=0.5, length_unit="mm")
    assert eng.series(gauge, None, "cod", **kw).values[1] == pytest.approx(1.0)
    assert eng.series(gauge, None, "cod_sliding", **kw).values[1] == pytest.approx(0.5)
    assert eng.series(gauge, None, "cod_magnitude", **kw).values[1] == pytest.approx(0.5 * math.hypot(2, 1))


def test_crack_opening_does_not_depend_on_endpoint_order():
    shift = lambda x, y, f: (np.where(x > 22, 2.0 * f, 0.0), 0 * y)  # noqa: E731
    eng = AnalysisEngine(build_series(shift, 1))
    forward = eng.series(probe("line", LineGeom(12.0, 20.0, 32.0, 20.0)), None, "cod")
    backward = eng.series(probe("line", LineGeom(32.0, 20.0, 12.0, 20.0)), None, "cod")
    assert forward.values[1] == pytest.approx(2.0)
    assert backward.values[1] == pytest.approx(2.0), "opening is positive either way"


def test_a_gauge_endpoint_off_the_grid_is_lost_on_every_frame():
    eng = AnalysisEngine(build_series(stretch(), 2))
    ts = eng.series(probe("line", LineGeom(20.0, 20.0, 60.0, 20.0)), None, "strain")
    assert set(ts.status) == {FrameStatus.ENDPOINT_LOST}


# ---------------------------------------------------------------------------
# Validity: displacement is never cut by the strain's own gaps
# ---------------------------------------------------------------------------

def _left_band() -> np.ndarray:
    band = np.zeros(SHAPE, dtype=bool)
    band[:, :10] = True
    return band


def test_a_displacement_probe_ignores_missing_strain():
    s = build_series(stretch(), 2)
    strain = build_strain(s, lambda x, y, f: 0.01 * f, untrusted=_left_band())
    ts = AnalysisEngine(s, strain).series(probe("point", PointGeom(2.0, 20.0)), "disp_u", "value")
    assert np.all(np.isfinite(ts.values)), "displacement is not trimmed"


def test_a_strain_probe_respects_missing_strain():
    s = build_series(stretch(), 2)
    strain = build_strain(s, lambda x, y, f: 0.01 * f, untrusted=_left_band())
    ts = AnalysisEngine(s, strain).series(probe("point", PointGeom(2.0, 20.0)), "strain_eyy", "value")
    assert np.all(np.isnan(ts.values))
    # Beside a hole this is the usual case, and "no data" would tell users nothing.
    assert set(ts.status) == {FrameStatus.UNRELIABLE}


def test_strain_that_was_never_computed_says_so():
    eng = AnalysisEngine(build_series(stretch(), 2))
    ts = eng.series(probe("point", PointGeom(20.0, 20.0)), "strain_eyy", "value")
    assert np.all(np.isnan(ts.values))
    assert set(ts.status) == {FrameStatus.NOT_COMPUTED}


def test_a_strain_point_reads_its_own_coarser_grid():
    s = build_series(stretch(), 2)
    strain = build_strain(s, lambda x, y, f: 0.001 * f * x, step=4)
    ts = AnalysisEngine(s, strain).series(probe("point", PointGeom(21.0, 13.0)), "strain_exx", "value")
    np.testing.assert_allclose(ts.values, [0.0, 0.021, 0.042], atol=1e-6)


def test_a_region_smaller_than_a_strain_cell_reads_its_centre():
    s = build_series(stretch(), 1)
    strain = build_strain(s, lambda x, y, f: 0.001 * f * x, step=4)
    ts = AnalysisEngine(s, strain).series(probe("area", AreaGeom.rect(9.0, 9.0, 11.0, 11.0)),
                                          "strain_exx", "mean")
    assert ts.values[1] == pytest.approx(0.010, abs=1e-6)


def test_a_strain_region_reads_the_cells_of_the_strain_grid():
    """Cells every 4 px at x = 8, 12, ..., 32: their mean x is 20. A hole in the ROI
    leaves its cells out of the statistics and out of the valid fraction."""
    s = build_series(stretch(), 2)
    strain = build_strain(s, lambda x, y, f: 0.001 * f * x, step=4)
    eng = AnalysisEngine(s, strain)
    region = probe("area", AreaGeom.rect(8.0, 8.0, 32.0, 24.0))
    ts = eng.series(region, "strain_exx", "mean")
    np.testing.assert_allclose(ts.values, [0.0, 0.020, 0.040], atol=1e-6)
    assert eng.plan(region.geometry, "strain_exx").size == 7 * 5

    roi = np.ones(SHAPE, dtype=bool)
    roi[8:25, 26:33] = False                                  # removes the cells at x = 28, 32
    holed = build_series(stretch(), 2, roi=roi)
    ts = AnalysisEngine(holed, build_strain(holed, lambda x, y, f: 0.001 * f * x, step=4)).series(
        region, "strain_exx", "mean")
    np.testing.assert_allclose(ts.values, [0.0, 0.016, 0.032], atol=1e-6)   # mean x of 8..24
    assert set(ts.valid_fraction) == {1.0}


def test_a_large_region_is_sampled_in_blocks():
    """A region over most of a large image: cells tested a block of rows at a time."""
    s = build_series(lambda x, y, f: (0.001 * x, 0.002 * y), 1, shape=(900, 1000))
    eng = AnalysisEngine(s)
    disc = probe("area", AreaGeom.circle(500.0, 450.0, 400.0))
    ts = eng.series(disc, "disp_v", "mean")
    assert ts.values[1] == pytest.approx(0.002 * 450.0, rel=1e-4)
    plan = eng.plan(disc.geometry)
    assert plan.size == pytest.approx(np.pi * 400.0 ** 2, rel=1e-3)
    assert plan.cells.dtype == bool and plan.cells.shape == (801, 801)
    hexagon = probe("area", AreaGeom.polygon([(500.0 + 300 * np.cos(t), 450.0 + 300 * np.sin(t))
                                              for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)]))
    assert eng.series(hexagon, "disp_u", "mean").values[1] == pytest.approx(0.5, rel=1e-3)


def test_frames_that_were_not_computed_say_so():
    s = build_series(stretch(), 3, done=[True, True, False, True])
    strain = build_strain(s, lambda x, y, f: 0.01 * f, done=[True, False, False, True])
    eng = AnalysisEngine(s, strain)
    pt = probe("point", PointGeom(20.0, 20.0))
    disp = eng.series(pt, "disp_u", "value")
    assert disp.status[2] is FrameStatus.NOT_COMPUTED and math.isnan(disp.values[2])
    assert disp.status[3] is FrameStatus.OK
    eps = eng.series(pt, "strain_exx", "value")
    assert [st is FrameStatus.NOT_COMPUTED for st in eps.status] == [False, True, True, False]
    gauge = eng.series(probe("line", LineGeom(4.0, 20.0, 36.0, 20.0)), None, "strain")
    assert gauge.status[2] is FrameStatus.NOT_COMPUTED


# ---------------------------------------------------------------------------
# Holes present from frame 0, and material lost on later frames
# ---------------------------------------------------------------------------

def _hole_roi(radius: float = 7.0) -> np.ndarray:
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    return np.hypot(xx - 22.0, yy - 22.0) > radius


def test_a_section_from_a_hole_edge_is_measured_not_blanked():
    eng = AnalysisEngine(build_series(stretch(), 2, roi=_hole_roi()))
    ts = eng.series(probe("line", LineGeom(30.0, 22.0, 43.0, 22.0)), "disp_u", "mean")
    assert np.all(np.isfinite(ts.values))
    assert set(ts.status) == {FrameStatus.OK}


def test_samples_inside_a_hole_are_not_material():
    eng = AnalysisEngine(build_series(stretch(), 2, roi=_hole_roi()))
    plan = eng.plan(LineGeom(10.0, 22.0, 34.0, 22.0))
    smp = eng.sample(plan, "disp_u", 1)
    inside = np.abs(plan.xy[:, 0] - 22.0) < 6.5
    assert np.all(smp.status[inside] == SampleStatus.OUTSIDE)
    # ...so they do not count against the valid fraction.
    assert smp.valid_fraction == pytest.approx(1.0)


def test_a_point_in_a_hole_reads_nothing():
    eng = AnalysisEngine(build_series(stretch(), 2, roi=_hole_roi()))
    ts = eng.series(probe("point", PointGeom(22.0, 22.0)), "disp_u", "value")
    assert np.all(np.isnan(ts.values))
    assert set(ts.status) == {FrameStatus.NO_DATA}


def test_a_growing_crack_drops_only_the_samples_it_takes():
    """The line keeps a value; the frame is marked, not blanked."""
    eng = AnalysisEngine(build_series(stretch(), 4, lost={3: column(20)}))
    ts = eng.series(probe("line", LineGeom(4.0, 22.0, 40.0, 22.0)), "disp_u", "mean")
    assert list(ts.status[:3]) == [FrameStatus.OK] * 3
    assert ts.status[3] is FrameStatus.UNTRACKED
    assert np.isfinite(ts.values[3]), "enough of the line is still material"
    assert ts.first_frame(FrameStatus.UNTRACKED) == 3


def test_an_extensometer_keeps_reading_across_a_crack():
    """Elongation at break is a standard output; the frame is annotated."""
    eng = AnalysisEngine(build_series(stretch(), 3, lost={2: column(20)}))
    ts = eng.series(probe("line", LineGeom(4.0, 22.0, 40.0, 22.0)), None, "strain")
    assert np.all(np.isfinite(ts.values))
    assert ts.status[2] is FrameStatus.UNTRACKED


def test_a_gauge_endpoint_on_lost_material_is_reported():
    eng = AnalysisEngine(build_series(stretch(), 2, lost={1: column(20)}))
    ts = eng.series(probe("line", LineGeom(20.2, 22.0, 40.0, 22.0)), None, "cod")
    assert math.isnan(ts.values[1])
    assert ts.status[1] is FrameStatus.ENDPOINT_LOST


def test_a_line_beside_a_crack_is_bit_exact_against_no_crack():
    plain = AnalysisEngine(build_series(stretch(), 3))
    cracked = AnalysisEngine(build_series(stretch(), 3, lost={2: column(20)}))
    far = probe("line", LineGeom(28.0, 4.0, 40.0, 36.0))
    a = plain.series(far, "disp_u", "mean")
    b = cracked.series(far, "disp_u", "mean")
    np.testing.assert_array_equal(a.values, b.values)
    assert a.status == b.status


def test_the_kymograph_shows_the_untracked_band():
    eng = AnalysisEngine(build_series(stretch(), 3, lost={2: column(20)}))
    line = probe("line", LineGeom(4.0, 22.0, 40.0, 22.0))
    kymo = eng.kymograph(line, "disp_u")
    x = kymo.xy[:, 0]
    at = np.abs(x - 20.0) < 0.5
    assert np.all(np.isfinite(kymo.values[1, at])), "before the crack"
    assert np.all(np.isnan(kymo.values[2, at])), "after the crack"
    assert np.all(kymo.status[2, at] == SampleStatus.UNTRACKED)
    beyond = np.abs(x - 20.0) > 1.0
    assert np.all(np.isfinite(kymo.values[2, beyond])), "the rest is untouched"


def test_a_notch_present_from_frame_zero_is_part_of_the_specimen():
    """A notch is geometry, not an event: no mark on any frame."""
    roi = np.ones(SHAPE, dtype=bool)
    roi[:, 17:20] = False
    ts = extract_series(build_series(stretch(), 3, roi=roi),
                        probe("line", LineGeom(4.0, 20.0, 36.0, 20.0)), None, "strain")
    assert np.isfinite(ts.values).all()
    assert set(ts.status) == {FrameStatus.OK}


def test_an_area_a_crack_opens_through_keeps_its_value_and_is_marked():
    slit = np.zeros(SHAPE, dtype=bool)
    slit[:, 17:20] = True
    s = build_series(stretch(), 3, lost={2: slit})
    ts = extract_series(s, probe("area", AreaGeom.rect(8.0, 8.0, 28.0, 28.0)), "disp_u", "mean")
    assert ts.status[1] is FrameStatus.OK
    assert ts.status[2] is FrameStatus.UNTRACKED
    assert np.isfinite(ts.values[2])


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------

def test_a_region_mean_is_the_centre_value_of_a_linear_field():
    eng = AnalysisEngine(build_series(lambda x, y, f: (0.01 * x, 0 * y), 1))
    ts = eng.series(probe("area", AreaGeom.rect(0.0, 0.0, 44.0, 44.0)), "disp_u", "mean")
    assert ts.values[1] == pytest.approx(0.01 * 22.0, rel=1e-5)


def test_a_region_maximum_reaches_the_peak_inside_it():
    peak = lambda x, y, f: (np.exp(-((x - 20) ** 2 + (y - 20) ** 2) / 8), 0 * y)  # noqa: E731
    eng = AnalysisEngine(build_series(peak, 1))
    ts = eng.series(probe("area", AreaGeom.circle(20.0, 20.0, 6.0)), "disp_u", "max")
    assert ts.values[1] == pytest.approx(1.0)


def test_a_region_smaller_than_a_pixel_still_measures():
    eng = AnalysisEngine(build_series(lambda x, y, f: (0.1 * x, 0 * y), 1))
    ts = eng.series(probe("area", AreaGeom.rect(9.2, 9.2, 9.8, 9.8)), "disp_u", "mean")
    assert ts.values[1] == pytest.approx(0.95, abs=1e-5)


def test_a_polygon_region_reads_the_field_inside_it():
    eng = AnalysisEngine(build_series(lambda x, y, f: (0.01 * x, 0 * y), 1))
    triangle = AreaGeom.polygon([(8.0, 8.0), (32.0, 8.0), (8.0, 32.0)])
    ts = eng.series(probe("area", triangle), "disp_u", "mean")
    # Centroid x of the triangle is 16: u = 0.16 on a linear field.
    assert ts.values[1] == pytest.approx(0.16, abs=0.01)


def test_valid_fraction_is_dimensionless_and_zero_when_empty():
    s = build_series(stretch(), 2, lost={2: np.ones(SHAPE, dtype=bool)})
    ts = AnalysisEngine(s).series(probe("area", AreaGeom.rect(4.0, 4.0, 40.0, 40.0)),
                                  "disp_u", "valid_fraction", length_unit="mm")
    assert ts.unit == ""
    assert ts.values[0] == pytest.approx(1.0)
    assert ts.values[2] == 0.0


def test_a_point_joins_mean_but_not_std():
    eng = AnalysisEngine(build_series(stretch(), 1))
    pt = probe("point", PointGeom(20.0, 20.0))
    assert eng.supports(pt, "mean")
    assert not eng.supports(pt, "std")
    assert eng.series(pt, "disp_u", "mean").values[1] == pytest.approx(0.2)


def test_std_is_the_population_standard_deviation():
    eng = AnalysisEngine(build_series(lambda x, y, f: (0.01 * x, 0 * y), 1))
    line = probe("line", LineGeom(4.0, 20.0, 36.0, 20.0))
    xs = eng.plan(line.geometry).xy[:, 0]
    ts = eng.series(line, "disp_u", "std")
    assert ts.values[1] == pytest.approx(np.std(0.01 * xs), rel=1e-5)


def test_median_shrugs_off_a_spike_the_mean_does_not():
    spike = lambda x, y, f: (np.where((x == 20) & (y == 20), 100.0, 0.0), 0 * y)  # noqa: E731
    eng = AnalysisEngine(build_series(spike, 1))
    region = probe("area", AreaGeom.rect(4.0, 4.0, 40.0, 40.0))
    assert eng.series(region, "disp_u", "median").values[1] == pytest.approx(0.0)
    assert eng.series(region, "disp_u", "mean").values[1] > 0.0


def test_a_frame_below_the_threshold_is_left_blank():
    left = np.zeros(SHAPE, dtype=bool)
    left[:, :30] = True                                    # most of the region
    eng = AnalysisEngine(build_series(stretch(), 2, lost={2: left}))
    region = probe("area", AreaGeom.rect(4.0, 4.0, 40.0, 40.0))
    blank = eng.series(region, "disp_u", "mean", min_valid_fraction=0.5)
    kept = eng.series(region, "disp_u", "mean", min_valid_fraction=0.0)
    assert blank.status[2] is FrameStatus.BELOW_THRESHOLD
    assert math.isnan(blank.values[2])
    assert math.isfinite(kept.values[2]), "a threshold of 0 keeps what is left"


def test_an_unknown_statistic_is_an_error_not_a_default():
    eng = AnalysisEngine(build_series(stretch(), 1))
    with pytest.raises(ValueError):
        eng.series(probe("area", AreaGeom.rect(4.0, 4.0, 40.0, 40.0)), "disp_u", "average")


def test_a_point_outside_the_grid_is_nan_not_clamped():
    eng = AnalysisEngine(build_series(stretch(), 1))
    ts = eng.series(probe("point", PointGeom(60.0, 20.0)), "disp_u", "value")
    assert np.all(np.isnan(ts.values))
    assert set(ts.status) == {FrameStatus.NO_DATA}


def test_the_magnitude_is_the_hypot_of_the_components():
    eng = AnalysisEngine(build_series(lambda x, y, f: (3.0 * f + 0 * x, 4.0 * f + 0 * y), 2))
    ts = eng.series(probe("area", AreaGeom.circle(20.0, 20.0, 5.0)), "disp_magnitude", "mean")
    np.testing.assert_allclose(ts.values, [0.0, 5.0, 10.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Line views
# ---------------------------------------------------------------------------

def test_profile_is_the_kymograph_row_for_that_frame():
    eng = AnalysisEngine(build_series(stretch(0.01), 3))
    line = probe("line", LineGeom(4.0, 20.0, 40.0, 20.0))
    kymo = eng.kymograph(line, "disp_u")
    prof = eng.profile(line, "disp_u", 2)
    assert kymo.values.shape == (4, len(prof.distance))
    np.testing.assert_allclose(prof.values, kymo.values[2])
    np.testing.assert_allclose(prof.values, 0.02 * (4.0 + prof.distance), atol=1e-6)
    np.testing.assert_allclose(prof.xy, kymo.xy)


def test_a_kymograph_needs_a_line():
    eng = AnalysisEngine(build_series(stretch(), 1))
    with pytest.raises(ValueError, match="line"):
        eng.kymograph(probe("point", PointGeom(1.0, 1.0)), "disp_u")


# ---------------------------------------------------------------------------
# Performance and storage
# ---------------------------------------------------------------------------

def test_a_long_run_is_interactive():
    """Plans are built once; each frame reads only the probe's window."""
    s = build_series(stretch(1e-4), 100, shape=(300, 300))
    t0 = time.perf_counter()
    eng = AnalysisEngine(s)
    line = eng.series(probe("line", LineGeom(20.0, 150.0, 280.0, 150.0)), "disp_u", "mean")
    area = eng.series(probe("area", AreaGeom.rect(50.0, 50.0, 250.0, 250.0)), "disp_magnitude", "mean")
    elapsed = time.perf_counter() - t0
    assert np.all(np.isfinite(line.values)) and np.all(np.isfinite(area.values))
    assert elapsed < 3.0, f"{elapsed:.2f}s for 101 frames"


def test_a_memory_mapped_run_reads_the_same(tmp_path):
    import dataclasses

    s = build_series(stretch(0.01), 2)
    np.save(tmp_path / "u.npy", s.u)
    np.save(tmp_path / "v.npy", s.v)
    mapped = dataclasses.replace(s, u=np.load(tmp_path / "u.npy", mmap_mode="r"),
                                 v=np.load(tmp_path / "v.npy", mmap_mode="r"))
    region = probe("area", AreaGeom.rect(4.0, 4.0, 30.0, 30.0))
    a = AnalysisEngine(s).series(region, "disp_u", "mean")
    b = AnalysisEngine(mapped).series(region, "disp_u", "mean")
    np.testing.assert_array_equal(a.values, b.values)


# ---------------------------------------------------------------------------
# The scripting front (pyALDIC test_extract.py)
# ---------------------------------------------------------------------------

def test_reference_frame_has_zero_displacement():
    vals = field_values(build_series(stretch(), 3), "disp_u", 0)
    assert np.all(vals == 0.0)


def test_displacement_follows_the_frame_index():
    s = build_series(stretch(0.01), 3)
    X, _ = s.grid.coords()
    for f in (1, 2, 3):
        np.testing.assert_allclose(field_values(s, "disp_u", f), 0.01 * f * X, atol=1e-6)


def test_pixel_size_scales_displacement_but_never_strain():
    s = build_series(stretch(), 1)
    strain = build_strain(s, lambda x, y, f: 0.01 * f)
    px = field_values(s, "disp_u", 1, pixel_size=1.0)
    np.testing.assert_allclose(field_values(s, "disp_u", 1, pixel_size=0.5), px * 0.5)
    np.testing.assert_allclose(field_values(s, "strain_exx", 1, strain=strain, pixel_size=0.5),
                               field_values(s, "strain_exx", 1, strain=strain))


def test_unknown_field_raises():
    with pytest.raises(ValueError, match="Unknown field"):
        field_values(build_series(stretch(), 1), "strain_eyx", 1)


def test_frame_count_includes_the_reference():
    assert frame_count(build_series(stretch(), 5)) == 6


def test_point_probe_series():
    p = Probe(id=1, kind="point", geometry=PointGeom(20.0, 20.0), label="p", color="#FF0000")
    ts = extract_series(build_series(stretch(0.01), 3), p, "disp_u", "value")
    np.testing.assert_allclose(ts.values, [0.0, 0.2, 0.4, 0.6], atol=1e-6)
    assert ts.unit == "px"


def test_point_probe_rejects_a_statistic_one_sample_cannot_have():
    s = build_series(stretch(0.01), 3)
    p = Probe(id=1, kind="point", geometry=PointGeom(20.0, 20.0), label="p", color="#FF0000")
    with pytest.raises(ValueError, match="does not apply"):
        extract_series(s, p, "disp_u", "std")
    np.testing.assert_allclose(extract_series(s, p, "disp_u", "median").values,
                               [0.0, 0.2, 0.4, 0.6], atol=1e-6)


def test_area_probe_mean_is_area_weighted():
    p = Probe(id=1, kind="area", geometry=AreaGeom.rect(0.0, 0.0, 8.0, 8.0), label="a", color="#00FF00")
    ts = extract_series(build_series(stretch(0.01), 3), p, "disp_u", "mean")
    np.testing.assert_allclose(ts.values, [0.0, 0.04, 0.08, 0.12], atol=1e-6)


def test_gauge_strain_recovers_the_applied_stretch():
    p = Probe(id=1, kind="line", geometry=LineGeom(4.0, 20.0, 36.0, 20.0), label="g", color="#0000FF")
    ts = extract_series(build_series(stretch(0.01), 3), p, "disp_u", "strain")
    np.testing.assert_allclose(ts.values, [0.0, 0.01, 0.02, 0.03], atol=1e-7)
    assert ts.unit == "", "engineering strain is dimensionless"


def test_gauge_cod_is_a_length():
    p = Probe(id=1, kind="line", geometry=LineGeom(4.0, 20.0, 36.0, 20.0), label="g", color="#0000FF")
    ts = extract_series(build_series(stretch(0.01), 3), p, "disp_u", "cod", length_unit="px")
    np.testing.assert_allclose(ts.values, [0.0, 0.32, 0.64, 0.96], atol=1e-6)
    assert ts.unit == "px"


def test_line_probe_mean_along_the_chord():
    p = Probe(id=1, kind="line", geometry=LineGeom(0.0, 20.0, 40.0, 20.0), label="l", color="#FFFF00")
    ts = extract_series(build_series(stretch(0.01), 3), p, "disp_u", "mean")
    np.testing.assert_allclose(ts.values, [0.0, 0.2, 0.4, 0.6], atol=1e-6)


def test_valid_fraction_is_extractable_as_its_own_series():
    p = Probe(id=1, kind="area", geometry=AreaGeom.rect(0.0, 0.0, 8.0, 8.0), label="a", color="#00FF00")
    ts = extract_series(build_series(stretch(), 3), p, "disp_u", "valid_fraction")
    assert np.all(ts.values == 1.0)


def test_probe_away_from_the_crack_is_unaffected():
    slit = np.zeros(SHAPE, dtype=bool)
    slit[:, 17:20] = True
    p = Probe(id=1, kind="area", geometry=AreaGeom.rect(24.0, 8.0, 36.0, 20.0), label="a", color="#00FF00")
    without = extract_series(build_series(stretch(0.01), 3), p, "disp_u", "mean")
    with_crack = extract_series(build_series(stretch(0.01), 3, lost={1: slit}), p, "disp_u", "mean")
    np.testing.assert_array_equal(without.values, with_crack.values)
    assert without.status == with_crack.status


# ---------------------------------------------------------------------------
# TimeSeries (pyALDIC test_series.py)
# ---------------------------------------------------------------------------

def _ts(values, status) -> TimeSeries:
    n = len(values)
    return TimeSeries(frames=np.arange(n, dtype=np.int64), values=np.asarray(values, dtype=np.float64),
                      valid_fraction=np.ones(n), status=list(status), unit="")


def test_series_never_bridges_a_gap():
    ts = _ts([0.0, 1.0, np.nan, np.nan, 4.0, 5.0],
             [FrameStatus.OK] * 2 + [FrameStatus.NO_DATA] * 2 + [FrameStatus.OK] * 2)
    assert list(ts.contiguous_runs()) == [(0, 2), (4, 6)]


def test_a_series_that_is_all_gap_has_no_runs():
    assert list(_ts([np.nan, np.nan], [FrameStatus.NO_DATA] * 2).contiguous_runs()) == []


def test_status_is_independent_of_whether_a_value_survived():
    ts = _ts([0.0, 1.0, 2.0], [FrameStatus.OK, FrameStatus.OK, FrameStatus.UNTRACKED])
    assert list(ts.contiguous_runs()) == [(0, 3)]
    assert ts.first_frame(FrameStatus.UNTRACKED) == 2


def test_first_frame_is_none_when_it_never_happens():
    assert _ts([0.0, 1.0], [FrameStatus.OK] * 2).first_frame(FrameStatus.UNTRACKED) is None
