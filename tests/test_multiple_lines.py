"""Feature 13: multiple independent counting lines.

Each configured segment is its own gate with its own tally, and a track is
counted at most once per line. No video, no model (R4).
"""
from features.multiple_counting_line import MultipleCountingLineManager
from src.config import Line
from tests.conftest import make_detection, make_track

# Two parallel gates across a 640x360 frame, plus a vertical one.
DOOR = Line("door", (32, 180), (608, 180), "top_to_bottom")
MIDDLE = Line("middle", (32, 90), (608, 90), "top_to_bottom")
SIDE = Line("side", (200, 0), (200, 360), "left_to_right")


def manager(*lines, **kwargs):
    return MultipleCountingLineManager(list(lines) or [DOOR], **kwargs)


def walk(mgr, track_id, x, y, frame_idx, is_staff=False):
    return mgr.update([make_track(track_id, x=x, y=y, is_staff=is_staff)], frame_idx, 0.0)


# ------------------------------------------------------------ single gate


def test_single_line_tallies_in_and_out():
    mgr = manager(DOOR)

    walk(mgr, 1, 300, 100, 0)  # first sighting
    walk(mgr, 1, 300, 250, 1)  # downwards -> in
    walk(mgr, 1, 300, 100, 2)  # upwards   -> out

    assert mgr.counts["door"] == {"in": 1, "out": 1}
    summary = mgr.summary()
    assert (summary["total_in"], summary["total_out"], summary["net_inside"]) == (1, 1, 0)


def test_a_track_is_counted_once_per_line_however_often_it_crosses():
    mgr = manager(DOOR)

    walk(mgr, 5, 300, 100, 0)
    for frame_idx, y in enumerate([150, 250, 300, 300], start=1):
        walk(mgr, 5, 300, y, frame_idx)

    assert mgr.counts["door"]["in"] == 1
    assert mgr.counts["door"]["out"] == 0
    assert [e["track_id"] for e in mgr.events] == [5]


def test_crossing_direction_follows_the_configured_in_direction():
    downward = Line("down", (0, 100), (600, 100), "top_to_bottom")
    upward = Line("up", (0, 100), (600, 100), "bottom_to_top")

    for line, expected in ((downward, "in"), (upward, "out")):
        mgr = manager(line)
        walk(mgr, 1, 300, 50, 0)
        events = walk(mgr, 1, 300, 150, 1)
        assert [e["direction"] for e in events] == [expected]


def test_horizontal_lines_use_the_x_axis():
    left_to_right = Line("l2r", (100, 0), (100, 360), "left_to_right")
    mgr = manager(left_to_right)

    walk(mgr, 1, 50, 200, 0)
    events = walk(mgr, 1, 150, 200, 1)

    assert [e["direction"] for e in events] == ["in"]


# ------------------------------------------------------- multiple gates


def test_each_gate_is_counted_independently():
    mgr = manager(DOOR, MIDDLE)

    walk(mgr, 1, 300, 40, 0)   # above both
    events = walk(mgr, 1, 300, 120, 1)  # crosses MIDDLE only

    assert [e["line"] for e in events] == ["middle"]
    assert mgr.counts["middle"] == {"in": 1, "out": 0}
    assert mgr.counts["door"] == {"in": 0, "out": 0}

    events = walk(mgr, 1, 300, 220, 2)  # now crosses DOOR too

    assert [e["line"] for e in events] == ["door"]
    assert mgr.summary()["total_in"] == 2


def test_a_track_may_be_counted_once_on_each_of_several_lines():
    mgr = manager(DOOR, SIDE)

    # The same track object moves: x 150 -> 250 crosses SIDE (x=200) and
    # y 40 -> 220 crosses DOOR (y=180) in one step.
    track = make_track(3, x=150, y=40)
    mgr.update([track], 0, 0.0)

    track.detection = make_detection(x1=240, y1=180, x2=260, y2=220)
    events = mgr.update([track], 1, 0.0)

    assert sorted(e["line"] for e in events) == ["door", "side"]
    assert mgr.counts["door"]["in"] == 1
    assert mgr.counts["side"]["in"] == 1
    # Idempotency is recorded per line on the track itself.
    assert track.counted_lines == {"door", "side"}

    # Crossing again adds nothing: both lines already recorded this track.
    track.detection = make_detection(x1=100, y1=40, x2=120, y2=80)
    mgr.update([track], 2, 0.0)
    track.detection = make_detection(x1=100, y1=240, x2=120, y2=280)
    assert mgr.update([track], 3, 0.0) == []


def test_two_lines_in_series_produce_two_events_for_one_crossing():
    mgr = manager(DOOR, MIDDLE)

    walk(mgr, 7, 300, 10, 0)
    events = walk(mgr, 7, 300, 260, 1)  # a single leap across both

    assert sorted(e["line"] for e in events) == ["door", "middle"]
    assert mgr.summary()["total_in"] == 2
    assert mgr.summary()["net_inside"] == 2


def test_gates_do_not_interfere_across_tracks():
    mgr = manager(DOOR, MIDDLE)

    walk(mgr, 1, 100, 40, 0)
    walk(mgr, 2, 500, 40, 0)
    walk(mgr, 1, 100, 120, 1)   # track 1 crosses MIDDLE
    walk(mgr, 2, 500, 250, 1)   # track 2 crosses MIDDLE and DOOR

    assert mgr.counts["middle"] == {"in": 2, "out": 0}
    assert mgr.counts["door"] == {"in": 1, "out": 0}
    assert mgr.summary()["total_in"] == 3


def test_summary_totals_sum_every_line():
    mgr = manager(DOOR, MIDDLE, SIDE)
    walk(mgr, 1, 150, 40, 0)
    walk(mgr, 1, 250, 120, 1)
    walk(mgr, 1, 250, 220, 2)

    summary = mgr.summary()
    assert summary["total_in"] == sum(t["in"] for t in mgr.counts.values())
    assert summary["total_out"] == 0
    assert set(mgr.counts) == {"door", "middle", "side"}


# ------------------------------------------------------------- lifecycle


def test_a_returning_track_is_counted_again_once_forgotten():
    mgr = manager(DOOR, stale_after_frames=2)

    walk(mgr, 1, 300, 100, 0)
    walk(mgr, 1, 300, 250, 1)
    assert mgr.counts["door"]["in"] == 1

    # Long enough absent to be treated as a brand new visit.
    for frame_idx in range(2, 6):
        mgr.update([], frame_idx, 0.0)

    walk(mgr, 1, 300, 100, 6)
    events = walk(mgr, 1, 300, 250, 7)
    assert [e["direction"] for e in events] == ["in"]
    assert mgr.counts["door"]["in"] == 2


def test_reset_clears_counts_events_and_memory():
    mgr = manager(DOOR)
    walk(mgr, 1, 300, 100, 0)
    walk(mgr, 1, 300, 250, 1)
    assert mgr.summary()["total_in"] == 1

    mgr.reset()

    assert mgr.counts["door"] == {"in": 0, "out": 0}
    assert mgr.events == []
    assert mgr.summary()["total_in"] == 0
    # Memory cleared, so a fresh crossing is tallied again.
    walk(mgr, 1, 300, 100, 2)
    assert len(walk(mgr, 1, 300, 250, 3)) == 1


def test_manager_counts_whichever_tracks_it_is_given():
    """Staff policy belongs to VideoCounter; the manager is a primitive."""
    mgr = manager(DOOR)

    walk(mgr, 1, 300, 100, 0, is_staff=True)
    events = walk(mgr, 1, 300, 250, 1, is_staff=True)

    assert len(events) == 1, "the manager applies no filtering of its own"
    assert mgr.counts["door"]["in"] == 1
