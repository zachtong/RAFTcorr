"""Unit tests for raftcorr.schedule (reference-frame schedules)."""

from __future__ import annotations

import pytest

from raftcorr.schedule import MODES, FrameSchedule


# -- constructors ---------------------------------------------------------------------------------
def test_accumulative_refs_all_zero():
    s = FrameSchedule.accumulative(5)
    assert s.ref_indices == (0, 0, 0, 0)
    assert s.n_frames == 5
    assert len(s) == 4
    assert s.is_accumulative


def test_incremental_refs_are_predecessors():
    s = FrameSchedule.incremental(5)
    assert s.ref_indices == (0, 1, 2, 3)
    assert not s.is_accumulative


def test_every_n_new_reference_every_n_frames():
    s = FrameSchedule.every_n(3, 8)
    # frames 1..3 -> 0, frames 4..6 -> 3, frame 7 -> 6
    assert s.ref_indices == (0, 0, 0, 3, 3, 3, 6)
    assert s.reference_frames == [0, 3, 6]


def test_every_n_with_n_1_is_incremental():
    assert FrameSchedule.every_n(1, 6) == FrameSchedule.incremental(6)


def test_every_n_larger_than_sequence_is_accumulative():
    assert FrameSchedule.every_n(100, 6) == FrameSchedule.accumulative(6)


def test_custom_uses_latest_key_frame_before_each_frame():
    s = FrameSchedule.custom([3, 5], 8)
    assert s.ref_indices == (0, 0, 0, 3, 3, 5, 5)
    assert s.reference_frames == [0, 3, 5]


def test_custom_always_includes_frame_zero_and_ignores_duplicates():
    s = FrameSchedule.custom([2, 2], 4)
    assert s.ref_indices == (0, 0, 2)
    assert FrameSchedule.custom([], 4) == FrameSchedule.accumulative(4)


def test_custom_last_valid_key_frame_is_second_to_last():
    s = FrameSchedule.custom([6], 8)
    assert s.reference_of(7) == 6


@pytest.mark.parametrize("mode", MODES)
def test_from_mode_dispatches(mode):
    s = FrameSchedule.from_mode(mode, 8, every_n=3, key_frames=(4,))
    expected = {
        "accumulative": FrameSchedule.accumulative(8),
        "incremental": FrameSchedule.incremental(8),
        "every_n": FrameSchedule.every_n(3, 8),
        "custom": FrameSchedule.custom([4], 8),
    }[mode]
    assert s == expected


# -- queries --------------------------------------------------------------------------------------
def test_pairs_in_frame_order():
    s = FrameSchedule.custom([2], 5)
    assert s.pairs() == [(0, 1), (0, 2), (2, 3), (2, 4)]


def test_pairs_reference_is_always_earlier():
    for s in (FrameSchedule.incremental(7), FrameSchedule.every_n(2, 7), FrameSchedule.custom([1, 4], 7)):
        for ref, dfm in s.pairs():
            assert 0 <= ref < dfm


def test_reference_frames_includes_zero():
    assert FrameSchedule.accumulative(3).reference_frames == [0]
    assert FrameSchedule.incremental(4).reference_frames == [0, 1, 2]


def test_reference_of_and_range():
    s = FrameSchedule.incremental(4)
    assert [s.reference_of(k) for k in (1, 2, 3)] == [0, 1, 2]
    with pytest.raises(IndexError):
        s.reference_of(0)
    with pytest.raises(IndexError):
        s.reference_of(4)


def test_ref_indices_are_coerced_to_int_tuple():
    import numpy as np

    s = FrameSchedule([np.int64(0), np.int64(1)])
    assert s.ref_indices == (0, 1)
    assert all(type(r) is int for r in s.ref_indices)


# -- validation -----------------------------------------------------------------------------------
@pytest.mark.parametrize("refs", [(0, 2), (1,), (0, 0, 3), (-1,)])
def test_reference_must_precede_frame(refs):
    with pytest.raises(ValueError):
        FrameSchedule(refs)


def test_reference_equal_to_predecessor_is_allowed():
    assert FrameSchedule((0, 1, 2)).ref_indices == (0, 1, 2)


@pytest.mark.parametrize("n", [0, 1, -3])
def test_too_few_frames(n):
    with pytest.raises(ValueError):
        FrameSchedule.accumulative(n)
    with pytest.raises(ValueError):
        FrameSchedule.incremental(n)
    with pytest.raises(ValueError):
        FrameSchedule.every_n(2, n)
    with pytest.raises(ValueError):
        FrameSchedule.custom([], n)


def test_every_n_must_be_positive():
    with pytest.raises(ValueError):
        FrameSchedule.every_n(0, 5)


@pytest.mark.parametrize("key", [-1, 7, 8, 100])
def test_custom_key_frame_out_of_range(key):
    # n_frames = 8: valid key frames are 0..6 (the last frame cannot be a reference)
    with pytest.raises(ValueError):
        FrameSchedule.custom([key], 8)


def test_unknown_mode():
    with pytest.raises(ValueError):
        FrameSchedule.from_mode("sometimes", 5)


# -- serialisation --------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "s",
    [FrameSchedule.accumulative(4), FrameSchedule.incremental(6), FrameSchedule.every_n(3, 10),
     FrameSchedule.custom([2, 7], 12)],
)
def test_dict_round_trip(s):
    d = s.to_dict()
    assert d == {"ref_indices": list(s.ref_indices)}
    assert FrameSchedule.from_dict(d) == s


def test_from_dict_validates():
    with pytest.raises(ValueError):
        FrameSchedule.from_dict({"ref_indices": [0, 5]})
