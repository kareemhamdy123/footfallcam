"""Geometry predicates and the counting tally.

No video, no model, no OpenCV: everything is driven by the factories in
`conftest.py` (R4).
"""
from features.counter import VideoCounter
from src.config import Line, Zone
from src.geometry.zones import crossed_line, point_in_polygon, zones_containing
from tests.conftest import make_track

# A wide horizontal gate across the middle of a 640x360 frame.
GATE = Line(
    name="main_entrance",
    p1=(32, 180),
    p2=(608, 180),
    in_direction="top_to_bottom",
)


# ---------------------------------------------------------------- line crossing


def test_crossed_line_detects_in_and_out_directions():
    assert crossed_line((300, 100), (300, 250), GATE) == "in"
    assert crossed_line((300, 250), (300, 100), GATE) == "out"


def test_crossed_line_returns_none_when_movement_never_reaches_the_line():
    assert crossed_line((300, 20), (300, 100), GATE) is None  # stays above
    assert crossed_line((100, 100), (200, 150), GATE) is None  # no side change


def test_crossed_line_returns_none_for_parallel_movement():
    # Drifts along the gate but never changes sides.
    assert crossed_line((100, 150), (500, 170), GATE) is None
    # Movement lying exactly on the line is not a crossing either.
    assert crossed_line((100, 180), (500, 180), GATE) is None


def test_crossed_line_rejects_crossing_outside_segment_endpoints():
    """R8: the initial commit's infinite-line bug must not come back."""
    short_gate = Line(
        name="door", p1=(40, 180), p2=(140, 180), in_direction="top_to_bottom"
    )
    # Beyond p2, and beyond p1: both straddle the infinite extension only.
    assert crossed_line((500, 100), (500, 250), short_gate) is None
    assert crossed_line((10, 100), (10, 250), short_gate) is None
    # Between the endpoints, it still counts.
    assert crossed_line((90, 100), (90, 250), short_gate) == "in"


# --------------------------------------------------------------------- polygons


def test_point_in_polygon_and_zones_containing():
    square = [(0, 0), (100, 0), (100, 100), (0, 100)]
    assert point_in_polygon((50, 50), square) is True
    assert point_in_polygon((150, 50), square) is False
    assert point_in_polygon((50, 50), [(0, 0), (10, 0)]) is False  # degenerate

    zones = [Zone(name="a", polygon=square), Zone(name="b", polygon=square)]
    assert [z.name for z in zones_containing((50, 50), zones)] == ["a", "b"]
    assert zones_containing((500, 500), zones) == []


# ---------------------------------------------------------------------- counting


def _walk(counter, track_id, y, frame_idx, is_staff=False):
    """Feed one frame for a track whose foot point is at (300, y)."""
    return counter.update(
        [make_track(track_id, x=300, y=y, is_staff=is_staff)],
        frame_idx,
        frame_idx / 30.0,
        frame_shape=(640, 360),
    )


def test_counter_tallies_bi_directionally_and_nets_out():
    counter = VideoCounter(lines=[GATE], frame_shape=(640, 360))

    _walk(counter, 1, y=100, frame_idx=0)  # first sighting: no movement yet
    _walk(counter, 1, y=250, frame_idx=1)  # crosses downwards -> IN
    _walk(counter, 1, y=250, frame_idx=2)  # standing still: not counted twice
    _walk(counter, 1, y=100, frame_idx=3)  # crosses upwards -> OUT

    summary = counter.summary()
    assert summary["per_line"]["main_entrance"] == {"in": 1, "out": 1}
    assert (summary["total_in"], summary["total_out"], summary["net_inside"]) == (1, 1, 0)


def test_counter_is_idempotent_per_line_and_keeps_net_inside():
    counter = VideoCounter(lines=[GATE], frame_shape=(640, 360))

    _walk(counter, 7, y=100, frame_idx=0)
    for frame_idx, y in enumerate([150, 250, 300], start=1):
        _walk(counter, 7, y=y, frame_idx=frame_idx)

    assert counter.summary()["total_in"] == 1, "counted once per line"
    assert counter.summary()["net_inside"] == 1
    assert [event["track_id"] for event in counter.events] == [7]


def test_counter_excludes_staff():
    counter = VideoCounter(lines=[GATE], frame_shape=(640, 360))

    _walk(counter, 3, y=100, frame_idx=0, is_staff=True)
    _walk(counter, 3, y=250, frame_idx=1, is_staff=True)

    summary = counter.summary()
    assert (summary["total_in"], summary["total_out"], summary["net_inside"]) == (0, 0, 0)


def test_counter_ignores_crossings_outside_the_segment():
    counter = VideoCounter(lines=[GATE], frame_shape=(640, 360))
    # the gate spans x = 32 .. 608

    counter.update([make_track(1, x=100, y=100)], 0, 0.0, (640, 360))
    counter.update([make_track(1, x=100, y=250)], 1, 1 / 30.0, (640, 360))
    counter.update([make_track(2, x=625, y=100)], 2, 2 / 30.0, (640, 360))
    counter.update([make_track(2, x=625, y=250)], 3, 3 / 30.0, (640, 360))

    assert counter.summary()["total_in"] == 1


def test_scaling_a_normalised_line_resolves_it_against_the_frame():
    """R3: coordinates live in config, normalised, not in code."""
    normalised = Line("door", (0.05, 0.5), (0.95, 0.5), "top_to_bottom")
    resolved = normalised.scaled(640, 360)

    assert resolved.p1 == (32, 180)
    assert resolved.p2 == (608, 180)
    assert normalised.p1 == (0.05, 0.5), "scaling must not mutate the config"

    # A line resolved this way behaves identically to the literal one above.
    assert crossed_line((300, 100), (300, 250), resolved) == "in"
    assert crossed_line((300, 250), (300, 100), resolved) == "out"
