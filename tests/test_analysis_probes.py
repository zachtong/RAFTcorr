"""Probe geometry, validation, the probe set, editing and the on-disk form (port of pyALDIC's
``test_analysis/test_probes.py``, ``test_geometry_edit.py`` and ``test_probe_persistence.py``).

Coordinates are frame-0 image pixels throughout, so nothing here transforms between
coordinate systems. The on-disk form is the one ``raftcorr.post.probes`` (the basic interface)
reads, so a project file carries its probes between the two interfaces.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from raftcorr.analysis.geometry_edit import control_points, hit_test, move_control_point, translate
from raftcorr.analysis.probes import (
    AreaGeom,
    LineGeom,
    PointGeom,
    Probe,
    ProbeSet,
    allowed_reductions,
    probe_from_dict,
    probe_to_dict,
    replace,
)

# --- geometry ------------------------------------------------------------------


def test_line_requires_two_distinct_endpoints():
    LineGeom(x0=0.0, y0=0.0, x1=10.0, y1=0.0)
    with pytest.raises(ValueError, match="zero length"):
        LineGeom(x0=5.0, y0=5.0, x1=5.0, y1=5.0)


def test_line_length_and_direction():
    g = LineGeom(x0=0.0, y0=0.0, x1=3.0, y1=4.0)
    assert g.length() == pytest.approx(5.0)
    assert g.direction() == pytest.approx((0.6, 0.8))


def test_area_rect_is_normalised():
    """A rectangle dragged right-to-left is the same rectangle."""
    assert AreaGeom.rect(10.0, 20.0, 0.0, 5.0).data == AreaGeom.rect(0.0, 5.0, 10.0, 20.0).data


def test_area_circle_rejects_non_positive_radius():
    AreaGeom.circle(10.0, 10.0, 4.0)
    with pytest.raises(ValueError, match="radius"):
        AreaGeom.circle(10.0, 10.0, 0.0)


def test_area_polygon_needs_three_vertices_enclosing_an_area():
    AreaGeom.polygon([(0, 0), (10, 0), (5, 8)])
    with pytest.raises(ValueError, match="three"):
        AreaGeom.polygon([(0, 0), (10, 0)])
    with pytest.raises(ValueError, match="area"):
        AreaGeom.polygon([(0.0, 0.0), (10.0, 0.0), (5.0, 0.0)])
    assert AreaGeom.polygon([(0, 0), (1, 0), (0, 1)]).shape == "polygon"


# --- probe ---------------------------------------------------------------------


def test_probe_is_immutable():
    p = Probe(id=1, kind="point", geometry=PointGeom(1.0, 2.0), label="P1", color="#FF0000")
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.label = "renamed"
    assert dataclasses.replace(p, label="renamed").label == "renamed"


def test_probe_rejects_geometry_of_the_wrong_kind():
    with pytest.raises(ValueError, match="kind"):
        Probe(id=1, kind="line", geometry=PointGeom(1.0, 2.0), label="P1", color="#FF0000")


def test_probe_rejects_malformed_colour():
    with pytest.raises(ValueError, match="color"):
        Probe(id=1, kind="point", geometry=PointGeom(0.0, 0.0), label="P1", color="red")


# --- reductions ----------------------------------------------------------------


def test_a_point_is_a_one_sample_region():
    assert allowed_reductions("point") == frozenset({"value", "mean", "median", "max", "min"})


def test_line_reductions_include_gauge_measurements():
    red = allowed_reductions("line")
    assert {"mean", "median", "max", "min", "std", "valid_fraction"} <= red
    assert {"strain", "true_strain", "elongation", "cod", "cod_sliding", "cod_magnitude"} <= red


def test_area_reductions_exclude_gauge_measurements():
    red = allowed_reductions("area")
    assert {"mean", "median", "max", "min", "std", "valid_fraction"} <= red
    assert not ({"strain", "cod"} & red)


# --- serialisation -------------------------------------------------------------

PROBES = [
    Probe(id=1, kind="point", geometry=PointGeom(12.5, 30.25), label="tip", color="#FF0000"),
    Probe(id=2, kind="line", geometry=LineGeom(0.0, 0.0, 40.0, 10.0), label="gauge", color="#00FF00",
          visible=False),
    Probe(id=3, kind="area", geometry=AreaGeom.rect(0.0, 0.0, 20.0, 20.0), label="box", color="#0000FF"),
    Probe(id=4, kind="area", geometry=AreaGeom.circle(30.0, 30.0, 8.0), label="disc", color="#FFFF00"),
    Probe(id=5, kind="area", geometry=AreaGeom.polygon([(0, 0), (9, 1), (4, 7)]), label="tri",
          color="#FF00FF"),
]


@pytest.mark.parametrize("probe", PROBES)
def test_serialisation_round_trip_through_json(probe):
    assert probe_from_dict(json.loads(json.dumps(probe_to_dict(probe)))) == probe


@pytest.mark.parametrize("probe", PROBES)
def test_the_basic_interface_reads_the_same_entries(probe):
    """``raftcorr.post.probes.Probe`` reads what a project of this interface stores."""
    from raftcorr.post.probes import Probe as BasicProbe

    d = probe_to_dict(probe)
    basic = BasicProbe.from_dict(d)
    assert basic.id == probe.id and basic.label == probe.label and basic.color == probe.color
    assert basic.kind == d["kind"]


def test_entries_written_by_the_basic_interface_load():
    from raftcorr.post.probes import Probe as BasicProbe

    entries = [BasicProbe(1, "point", ((10.0, 20.0),)).to_dict(),
               BasicProbe(2, "circle", ((30.0, 30.0), (30.0, 38.0))).to_dict(),
               {"id": 3, "kind": "poly", "points": [[0, 0], [9, 1], [4, 7]], "color": "#123456", "label": "A3"}]
    ps = ProbeSet.from_list(entries)
    assert [p.kind for p in ps] == ["point", "area", "area"]
    assert ps.get(2).geometry == AreaGeom.circle(30.0, 30.0, 8.0)
    assert ps.get(3).geometry.shape == "polygon"
    assert all(p.visible for p in ps)


@pytest.mark.parametrize("bad", [
    "not-a-dict",
    {"id": 1, "kind": "point"},                                               # no points
    {"id": 1, "kind": "wedge", "points": [[0, 0]], "label": "x", "color": "#000000"},
    {"id": 1, "kind": "line", "points": [[0, 0]], "label": "x", "color": "#000000"},
    {"id": 1, "kind": "point", "points": [["a", 0]], "label": "x", "color": "#000000"},
])
def test_a_malformed_entry_is_a_value_error(bad):
    with pytest.raises(ValueError):
        probe_from_dict(bad)


# --- probe set -----------------------------------------------------------------


def test_ids_are_unique_across_kinds_and_never_reused():
    s = ProbeSet()
    a = s.add("point", PointGeom(1.0, 1.0))
    b = s.add("line", LineGeom(0.0, 0.0, 5.0, 5.0))
    c = s.add("area", AreaGeom.rect(0.0, 0.0, 4.0, 4.0))
    assert len({a.id, b.id, c.id}) == 3
    s.remove(c.id)
    assert s.add("point", PointGeom(2.0, 2.0)).id != c.id


def test_remove_unknown_id_raises():
    with pytest.raises(KeyError):
        ProbeSet().remove(99)


def test_an_invalid_probe_moves_no_counter():
    s = ProbeSet()
    with pytest.raises(ValueError):
        s.add("line", PointGeom(1.0, 1.0))
    assert (s.add("point", PointGeom(1.0, 1.0)).id, len(s)) == (1, 1)


def test_default_labels_and_colours_are_assigned():
    s = ProbeSet()
    colours = [s.add("point", PointGeom(float(i), 0.0)).color for i in range(4)]
    assert len(set(colours)) == 4
    assert all(c.startswith("#") and len(c) == 7 for c in colours)


def test_a_new_probe_takes_a_colour_nobody_is_using():
    s = ProbeSet()
    probes = [s.add("point", PointGeom(float(i), 0.0)) for i in range(8)]
    for p in probes[:3]:
        s.remove(p.id)
    in_use = {p.color for p in s}
    fresh = [s.add("point", PointGeom(9.0 + i, 0.0)).color for i in range(3)]
    assert not (set(fresh) & in_use)
    assert len(set(fresh)) == 3


def test_default_names_restart_at_p1_once_every_probe_is_deleted():
    s = ProbeSet()
    old = [s.add("point", PointGeom(float(i), 0.0)) for i in range(3)]
    for p in old:
        s.remove(p.id)
    new = s.add("line", LineGeom(0.0, 0.0, 5.0, 5.0))
    assert new.label == "P1"
    assert new.id not in {p.id for p in old}


def test_names_are_not_reused_while_a_probe_remains():
    s = ProbeSet()
    probes = [s.add("point", PointGeom(float(i), 0.0)) for i in range(3)]
    s.remove(probes[1].id)
    assert s.add("point", PointGeom(9.0, 9.0)).label == "P4"


def test_a_given_label_still_counts_towards_the_default_names():
    s = ProbeSet()
    s.add("point", PointGeom(1.0, 1.0), label="tip")
    assert s.add("point", PointGeom(2.0, 2.0)).label == "P2"


def test_clear_all_restarts_default_names():
    s = ProbeSet()
    s.add("point", PointGeom(1.0, 1.0))
    s.clear()
    assert [s.add("point", PointGeom(float(i), 3.0)).label for i in range(2)] == ["P1", "P2"]


def test_replace_updates_in_place_by_id():
    s = ProbeSet()
    p = s.add("point", PointGeom(1.0, 1.0))
    s.replace(dataclasses.replace(p, label="renamed"))
    assert s.get(p.id).label == "renamed"
    assert len(s) == 1


# --- persistence (pyALDIC test_probe_persistence.py) -----------------------------


def test_a_deleted_id_is_not_reused_after_a_round_trip():
    ps = ProbeSet()
    ps.add("point", PointGeom(1.0, 1.0))
    ps.add("point", PointGeom(2.0, 2.0))
    ps.remove(ps.add("line", LineGeom(0.0, 0.0, 5.0, 0.0)).id)
    back = ProbeSet.from_payload(json.loads(json.dumps(ps.to_payload())))
    assert back.add("point", PointGeom(3.0, 3.0)).id == 4


def test_the_name_counter_survives_a_round_trip():
    ps = ProbeSet()
    for i in range(2):
        ps.add("point", PointGeom(float(i), 0.0))
    ps.clear()
    ps.add("point", PointGeom(5.0, 5.0))             # P1 again, as id 3
    back = ProbeSet.from_payload(ps.to_payload())
    p = back.add("point", PointGeom(6.0, 6.0))
    assert (p.label, p.id) == ("P2", 4)


def test_a_saved_empty_set_restarts_names_at_p1():
    ps = ProbeSet()
    ps.remove(ps.add("point", PointGeom(1.0, 1.0)).id)
    assert ProbeSet.from_payload(ps.to_payload()).add("point", PointGeom(2.0, 2.0)).label == "P1"


def test_a_list_without_counters_continues_from_the_ids():
    ps = ProbeSet()
    ps.add("point", PointGeom(1.0, 1.0))
    ps.add("point", PointGeom(2.0, 2.0))
    back = ProbeSet.from_payload(ps.to_list())
    p = back.add("point", PointGeom(3.0, 3.0))
    assert (p.id, p.label) == (3, "P3")


def test_duplicate_ids_are_refused():
    ps = ProbeSet()
    ps.add("point", PointGeom(1.0, 1.0))
    with pytest.raises(ValueError, match="duplicate"):
        ProbeSet.from_payload({"items": ps.to_list() * 2, "next_id": 5})


@pytest.mark.parametrize("bad", [{"items": "nope", "next_id": 2}, "nope"])
def test_a_malformed_payload_is_a_value_error(bad):
    with pytest.raises(ValueError):
        ProbeSet.from_payload(bad)


def test_load_dicts_replaces_in_place_and_is_all_or_nothing():
    """The application shares one set; a bad file must not leave it half loaded."""
    shared = ProbeSet()
    shared.add("point", PointGeom(1.0, 1.0), label="kept")
    good = ProbeSet()
    good.add("line", LineGeom(0.0, 0.0, 5.0, 0.0), label="E1")
    with pytest.raises(ValueError):
        shared.load_dicts(good.to_dicts() + [{"id": 9, "kind": "wedge"}])
    assert [p.label for p in shared] == ["kept"]
    shared.load_dicts(good.to_dicts(), **{k: v for k, v in good.counters().items()})
    assert [p.label for p in shared] == ["E1"]
    assert shared.add("point", PointGeom(2.0, 2.0)).id == 2


# --- hit testing (pyALDIC test_geometry_edit.py) ----------------------------------


def _set():
    ps = ProbeSet()
    ps.add("point", PointGeom(10.0, 10.0))                       # 1
    ps.add("line", LineGeom(20.0, 50.0, 80.0, 50.0))             # 2
    ps.add("area", AreaGeom.rect(100.0, 100.0, 140.0, 130.0))    # 3
    ps.add("area", AreaGeom.circle(200.0, 200.0, 15.0))          # 4
    ps.add("area", AreaGeom.polygon([(300, 300), (340, 300), (320, 340)]))  # 5
    return ps


@pytest.mark.parametrize("x, y, expected", [
    (11.0, 9.0, 1), (50.0, 52.0, 2), (120.0, 115.0, 3), (205.0, 205.0, 4), (320.0, 310.0, 5),
    (500.0, 500.0, None),
])
def test_a_click_finds_the_probe_under_it(x, y, expected):
    assert hit_test(_set(), x, y, tolerance=4.0) == expected


def test_a_line_is_hit_along_its_body_not_only_at_its_ends():
    assert hit_test(_set(), 50.0, 58.0, tolerance=4.0) is None
    assert hit_test(_set(), 50.0, 53.0, tolerance=4.0) == 2


def test_hidden_probes_cannot_be_clicked():
    ps = _set()
    ps.replace(replace(ps.get(1), visible=False))
    assert hit_test(ps, 11.0, 9.0, tolerance=4.0) is None


def test_a_region_drawn_later_does_not_bury_what_lies_inside_it():
    ps = ProbeSet()
    ps.add("point", PointGeom(50.0, 50.0))
    ps.add("line", LineGeom(20.0, 70.0, 80.0, 70.0))
    ps.add("area", AreaGeom.rect(0.0, 0.0, 100.0, 100.0))
    assert hit_test(ps, 50.0, 50.0, tolerance=4.0) == 1
    assert hit_test(ps, 50.0, 71.0, tolerance=4.0) == 2
    assert hit_test(ps, 30.0, 30.0, tolerance=4.0) == 3
    assert hit_test(ps, 100.0, 40.0, tolerance=4.0) == 3


# --- reshaping -----------------------------------------------------------------


def test_control_points_per_shape():
    ps = _set()
    assert control_points(ps.get(1).geometry) == [(10.0, 10.0)]
    assert control_points(ps.get(2).geometry) == [(20.0, 50.0), (80.0, 50.0)]
    assert len(control_points(ps.get(3).geometry)) == 4
    assert control_points(ps.get(4).geometry) == [(200.0, 200.0), (215.0, 200.0)]
    assert len(control_points(ps.get(5).geometry)) == 3


def test_dragging_handles():
    assert move_control_point(LineGeom(20.0, 50.0, 80.0, 50.0), 1, 90.0, 60.0) == LineGeom(20.0, 50.0, 90.0, 60.0)
    rect = AreaGeom.rect(100.0, 100.0, 140.0, 130.0)
    far = control_points(rect).index((140.0, 130.0))
    assert move_control_point(rect, far, 160.0, 150.0) == AreaGeom.rect(100.0, 100.0, 160.0, 150.0)
    assert move_control_point(rect, far, 60.0, 70.0) == AreaGeom.rect(60.0, 70.0, 100.0, 100.0)
    circle = AreaGeom.circle(200.0, 200.0, 15.0)
    assert move_control_point(circle, 1, 200.0, 230.0) == AreaGeom.circle(200.0, 200.0, 30.0)
    assert move_control_point(circle, 0, 210.0, 190.0) == AreaGeom.circle(210.0, 190.0, 15.0)
    tri = AreaGeom.polygon([(300, 300), (340, 300), (320, 340)])
    assert move_control_point(tri, 2, 320.0, 360.0) == AreaGeom.polygon([(300, 300), (340, 300), (320, 360)])


def test_a_degenerate_result_is_refused_not_stored():
    assert move_control_point(LineGeom(20.0, 50.0, 80.0, 50.0), 1, 20.0, 50.0) is None
    assert move_control_point(AreaGeom.circle(0.0, 0.0, 5.0), 1, 0.0, 0.0) is None
    rect = AreaGeom.rect(100.0, 100.0, 140.0, 130.0)
    assert move_control_point(rect, control_points(rect).index((140.0, 130.0)), 140.0, 100.0) is None
    assert move_control_point(AreaGeom.polygon([(0, 0), (10, 0), (5, 9)]), 2, 5.0, 0.0) is None


@pytest.mark.parametrize("geom", [
    PointGeom(10.0, 10.0),
    LineGeom(20.0, 50.0, 80.0, 50.0),
    AreaGeom.rect(100.0, 100.0, 140.0, 130.0),
    AreaGeom.circle(200.0, 200.0, 15.0),
    AreaGeom.polygon([(300, 300), (340, 300), (320, 340)]),
])
def test_translation_moves_every_point_by_the_same_amount(geom):
    moved = translate(geom, 5.0, -3.0)
    expected = [c for x, y in control_points(geom) for c in (x + 5.0, y - 3.0)]
    assert expected == pytest.approx([c for p in control_points(moved) for c in p])
    assert type(moved) is type(geom)
