"""Group co-movement and entry clustering.

The Phase 2 subset of brochure characteristic 14: enough for VideoCounter to
report group statistics without duplicating the logic (R7). Phase 12 completes
the characteristic.
"""
import pytest

from features.group_counting import GroupCounter
from tests.conftest import make_track


def tracks(*specs):
    """[(track_id, x, y), ...] -> a track list."""
    return [make_track(tid, x=x, y=y) for tid, x, y in specs]


# ------------------------------------------------------------ co-movement


def test_close_tracks_accumulate_co_movement_time():
    gc = GroupCounter(group_max_distance_px=90.0)

    gc.update_co_movement(tracks((1, 100, 100), (2, 150, 100)), 0.0, dt_s=0.1)
    gc.update_co_movement(tracks((1, 110, 100), (2, 160, 100)), 0.1, dt_s=0.1)

    assert gc.co_movement_seconds[(1, 2)] == pytest.approx(0.2)


def test_distant_tracks_accumulate_nothing():
    gc = GroupCounter(group_max_distance_px=50.0)

    gc.update_co_movement(tracks((1, 0, 0), (2, 400, 0)), 0.0, dt_s=0.1)

    assert gc.co_movement_seconds == {}


def test_each_unordered_pair_is_counted_once():
    gc = GroupCounter(group_max_distance_px=200.0)

    gc.update_co_movement(tracks((1, 0, 0), (2, 10, 0), (3, 20, 0)), 0.0, dt_s=1.0)

    assert set(gc.co_movement_seconds) == {(1, 2), (1, 3), (2, 3)}
    assert gc.co_movement_seconds[(1, 2)] == 1.0


def test_the_distance_threshold_is_configurable():
    near = GroupCounter(group_max_distance_px=30.0)
    near.update_co_movement(tracks((1, 0, 0), (2, 40, 0)), 0.0, dt_s=1.0)
    assert near.co_movement_seconds == {}

    far = GroupCounter(group_max_distance_px=80.0)
    far.update_co_movement(tracks((1, 0, 0), (2, 40, 0)), 0.0, dt_s=1.0)
    assert far.co_movement_seconds == {(1, 2): 1.0}


def test_frame_delta_reports_the_gap_since_the_last_call():
    gc = GroupCounter()

    assert gc.frame_delta(0.0) == 0.0, "the first frame has no predecessor"
    assert gc.frame_delta(0.5) == pytest.approx(0.5)
    assert gc.frame_delta(1.0) == pytest.approx(0.5)
    assert gc.frame_delta(0.8) == 0.0, "time must not run backwards"


# ------------------------------------------------------------ clustering


def entries(*times):
    return [{"t": t, "track_id": i} for i, t in enumerate(times)]


def test_entries_close_in_time_form_one_group():
    gc = GroupCounter(entry_window_s=3.0)

    groups = gc.cluster_groups(entries(0.0, 1.0, 2.5))

    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_entries_far_apart_form_separate_groups():
    gc = GroupCounter(entry_window_s=3.0)

    groups = gc.cluster_groups(entries(0.0, 1.0, 30.0, 31.0))

    assert len(groups) == 2
    assert [len(g) for g in groups] == [2, 2]


def test_clustering_is_transitive():
    """A chain of close-in-time entries is one group, not several."""
    gc = GroupCounter(entry_window_s=2.0)

    groups = gc.cluster_groups(entries(0.0, 1.9, 3.8, 20.0))

    assert len(groups) == 2
    assert len(groups[0]) == 3


def test_the_entry_window_is_configurable():
    events = entries(0.0, 5.0)

    assert len(GroupCounter(entry_window_s=1.0).cluster_groups(events)) == 2
    assert len(GroupCounter(entry_window_s=10.0).cluster_groups(events)) == 1
    # An explicit window overrides the configured one.
    gc = GroupCounter(entry_window_s=10.0)
    assert len(gc.cluster_groups(events, window_s=1.0)) == 2


def test_clustering_handles_empty_and_single_input():
    gc = GroupCounter()
    assert gc.cluster_groups([]) == []
    assert len(gc.cluster_groups(entries(0.0))) == 1


def test_unordered_events_are_sorted_before_clustering():
    gc = GroupCounter(entry_window_s=3.0)
    assert len(gc.cluster_groups(entries(2.0, 0.0, 1.0))) == 1


# --------------------------------------------------------------- summary


def test_summary_reports_group_shape():
    gc = GroupCounter(entry_window_s=3.0)

    summary = gc.summary(entries(0.0, 1.0, 20.0, 21.0))

    assert summary["groups"] == 2
    assert summary["multi_person_groups"] == 2
    assert summary["solo_visitors"] == 0
    assert summary["average_group_size"] == pytest.approx(2.0)


def test_summary_of_no_events_is_zeroed_not_an_error():
    summary = GroupCounter().summary([])

    assert summary["groups"] == 0
    assert summary["average_group_size"] == 0.0


def test_reset_clears_co_movement():
    gc = GroupCounter()
    gc.update_co_movement(tracks((1, 0, 0), (2, 5, 0)), 0.0, dt_s=1.0)
    assert gc.co_movement_seconds

    gc.reset()
    assert gc.co_movement_seconds == {}
