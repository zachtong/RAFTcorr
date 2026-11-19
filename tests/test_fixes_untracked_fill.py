"""One rule for "Hide untracked regions" off, in the main window, the strain window and exports.

Untracked points inside the region of interest take the nearest value of their own connected part
of the region, so the fill never crosses a crack; the edge-trimmed strain band stays empty unless
"Fill trimmed edges" is on. Before, exported pictures and the main window filled from the nearest
value anywhere, and exported strain pictures also filled the trimmed band.
"""

from __future__ import annotations

import numpy as np

from raftcorr.config import RunConfig
from raftcorr.engine.mapping import Grid
from raftcorr.results import DisplacementSeries, RunInfo, StrainSeries, StrainSettings
from raftcorr.schedule import FrameSchedule

H, W = 20, 40
CRACK_COL = 20          # a column cut out of the region of interest: two separate parts


def _series() -> DisplacementSeries:
    """u = 1 left of the crack, 2 right of it; an untracked patch left of the crack, next to it."""
    n = 2
    g = Grid(0, 0, W, H, H, W)
    mask = np.ones(g.shape, bool)
    mask[:, CRACK_COL] = False
    u = np.where(np.arange(W) < CRACK_COL, 1.0, 2.0)[None, None].repeat(H, 1).repeat(n, 0)
    u = u.astype(np.float32)
    u[:, ~mask] = np.nan
    u[:, 2:, 15:CRACK_COL] = np.nan
    v = np.zeros_like(u)
    v[:, ~mask] = np.nan
    info = RunInfo("test", "RAFTcorr-s4", "0" * 64, 4, 20, "tile", "cpu", RunConfig().to_dict())
    return DisplacementSeries(g, mask, u, v, np.ones(n, bool), ("f0", "f1"),
                              FrameSchedule.accumulative(n), info)


def _strain(series: DisplacementSeries) -> StrainSeries:
    exx = series.u.copy()
    trimmed = np.zeros_like(exx, dtype=bool)
    trimmed[:, :2, :] = series.mask[:2, :]
    exx[trimmed] = np.nan
    settings = StrainSettings("infinitesimal", 5, "uniform", 1, 1, None, 0.0, True, 0.5)
    return StrainSeries(series.grid, 1, {"exx": exx}, series.done.copy(), settings, trimmed)


def test_shared_fill_stays_on_its_own_side_of_a_crack():
    from raftcorr.viz.fields import fill_untracked

    s = _series()
    out = fill_untracked(s.u[1], s.mask)
    assert (out[2:, 15:CRACK_COL] == 1.0).all()
    assert np.isnan(out[:, CRACK_COL]).all()


def test_main_window_uses_the_shared_fill():
    from raftcorr.gui.controllers import viz_controller
    from raftcorr.viz.fields import fill_untracked

    assert viz_controller.fill_untracked is fill_untracked
    assert not hasattr(viz_controller, "_fill_untracked")


def test_exported_displacement_fill_does_not_cross_a_crack():
    from raftcorr.export.figures import FrameStyle, _reference_values
    from raftcorr.viz.fields import FieldProvider

    s = _series()
    vals, _ = _reference_values(FieldProvider(s), FrameStyle(field="u", hide_untracked=False), 1)
    assert (vals[2:, 15:CRACK_COL] == 1.0).all()


def test_exported_strain_fill_leaves_the_trimmed_band_empty():
    from raftcorr.export.figures import FrameStyle, _reference_values
    from raftcorr.viz.fields import FieldProvider

    s = _series()
    prov = FieldProvider(s, _strain(s))
    vals, _ = _reference_values(prov, FrameStyle(field="exx", hide_untracked=False), 1)
    assert np.isnan(vals[:2, :]).all()
    assert (vals[2:, 15:CRACK_COL] == 1.0).all()


def test_exported_strain_with_fill_trimmed_edges_fills_the_band_as_that_option_does():
    from raftcorr.export.figures import FrameStyle, _reference_values
    from raftcorr.post.strain import fill_trimmed
    from raftcorr.viz.fields import FieldProvider

    s = _series()
    strain = _strain(s)
    style = FrameStyle(field="exx", hide_untracked=False, fill_trimmed_edges=True)
    vals, _ = _reference_values(FieldProvider(s, strain), style, 1)
    expected = fill_trimmed(strain.components["exx"][1], strain.trimmed[1])
    assert np.array_equal(vals[:2, :], expected[:2, :], equal_nan=True)
