"""VideoCounter auto mode: boundary counting with no configured lines.

With no lines in the config, the counter has to infer entry and exit from the
frame boundary. The debounce is the part that matters: a detection dropout must
not be read as somebody leaving the shop.
"""
from features.counter import OUTSIDE_SIDEWALK, SCENE_FLOW, VideoCounter
from src.config import Line
from tests.conftest import make_track

WIDTH, HEIGHT = 640, 360
FPS = 30.0


def counter(**kwargs):
    kwargs.setdefault("frame_shape", (WIDTH, HEIGHT))
    return VideoCounter(**kwargs)


def feed(counter_obj, frame_idx, tracks):
    return counter_obj.update(
        tracks, frame_idx, frame_idx / FPS, frame_shape=(WIDTH, HEIGHT)
    )


# ------------------------------------------------------------------ mode


def test_auto_mode_engages_only_without_lines():
    assert counter().auto_mode is True
    assert counter(lines=[Line("door", (0, 180), (600, 180))]).auto_mode is False


def test_auto_mode_creates_the_scene_flow_tallies():
    c = counter()
    assert c.counts[SCENE_FLOW] == {"in": 0, "out": 0}
    assert c.counts[OUTSIDE_SIDEWALK] == {"in": 0, "out": 0}


def test_line_mode_creates_one_tally_per_configured_line():
    c = counter(lines=[Line("a", (0, 100), (600, 100)), Line("b", (0, 200), (600, 200))])
    assert set(c.counts) == {"a", "b"}


# ------------------------------------------------------------------ entry


def test_first_appearance_is_an_entry():
    c = counter()

    events = feed(c, 0, [make_track(1, x=320, y=180)])

    assert [(e["line"], e["direction"]) for e in events] == [(SCENE_FLOW, "in")]
    assert c.counts[SCENE_FLOW]["in"] == 1


def test_a_track_is_only_counted_in_once():
    c = counter()

    feed(c, 0, [make_track(1, x=320, y=180)])
    for frame_idx in range(1, 20):
        events = feed(c, frame_idx, [make_track(1, x=320, y=180 + frame_idx)])

    assert events == [], "a visible track must not keep generating entries"
    assert c.counts[SCENE_FLOW]["in"] == 1


def test_several_people_entering_are_counted_separately():
    c = counter()

    feed(c, 0, [make_track(1, x=100, y=180), make_track(2, x=300, y=180)])
    feed(c, 1, [make_track(1, x=100, y=180), make_track(2, x=300, y=180), make_track(3, x=500, y=180)])

    assert c.counts[SCENE_FLOW]["in"] == 3
    assert c.summary()["net_inside"] == 3


# ------------------------------------------------------------------ exit


def test_departure_needs_the_debounce_to_elapse():
    c = counter(missing_frame_debounce=8)

    feed(c, 0, [make_track(1, x=320, y=180)])
    for frame_idx in range(1, 8):  # 7 empty frames: one short of 8
        assert feed(c, frame_idx, []) == []
    assert c.counts[SCENE_FLOW]["out"] == 0

    events = feed(c, 8, [])

    assert [(e["line"], e["direction"]) for e in events] == [(SCENE_FLOW, "out")]
    assert c.counts[SCENE_FLOW]["out"] == 1


def test_a_brief_dropout_is_not_an_exit():
    """The regression this debounce exists for."""
    c = counter(missing_frame_debounce=8)

    feed(c, 0, [make_track(1, x=320, y=180)])
    for frame_idx in range(1, 4):  # three frames of lost detection
        feed(c, frame_idx, [])
    feed(c, 4, [make_track(1, x=320, y=180)])  # and back again

    assert c.counts[SCENE_FLOW]["out"] == 0
    assert c.summary()["net_inside"] == 1


def test_a_track_that_returns_after_a_real_exit_counts_again():
    c = counter(missing_frame_debounce=3)

    feed(c, 0, [make_track(1, x=320, y=180)])
    for frame_idx in range(1, 5):
        feed(c, frame_idx, [])
    assert c.counts[SCENE_FLOW]["out"] == 1

    feed(c, 5, [make_track(1, x=320, y=180)])
    assert c.counts[SCENE_FLOW]["in"] == 2
    assert c.summary()["net_inside"] == 1


# -------------------------------------------------------------- passerby


def test_a_brief_edge_grazer_is_a_passerby_not_a_visitor():
    c = counter(missing_frame_debounce=2, passerby_max_seconds=3.2)

    feed(c, 0, [make_track(1, x=20, y=180)])        # hugging the left margin
    for frame_idx in range(1, 5):
        feed(c, frame_idx, [])

    assert c.counts[SCENE_FLOW]["out"] == 1, "still a departure"
    assert 1 in c.passerby_ids
    assert c.counts[OUTSIDE_SIDEWALK]["in"] == 1
    assert c.summary()["outside_passersby"] == 1


def test_someone_who_stays_longer_is_not_a_passerby():
    c = counter(missing_frame_debounce=2, passerby_max_seconds=3.0)

    # Continuously visible for well over three seconds, then gone for good.
    for frame_idx in range(0, 120):
        feed(c, frame_idx, [make_track(1, x=20, y=180)])
    for frame_idx in range(120, 130):
        feed(c, frame_idx, [])

    assert c.counts[SCENE_FLOW]["in"] == 1
    assert c.counts[SCENE_FLOW]["out"] == 1
    assert c.passerby_ids == set()


def test_a_centre_stage_visitor_is_never_a_passerby():
    c = counter(missing_frame_debounce=2, passerby_max_seconds=3.2)

    feed(c, 0, [make_track(1, x=320, y=180)])
    for frame_idx in range(1, 5):
        feed(c, frame_idx, [])

    assert c.passerby_ids == set()


def test_passerby_thresholds_are_config_driven():
    """R3: a config change must change the classification.

    The exit lands about 0.07s after the track was first seen, so a maximum
    below that excludes it and a maximum above it includes it.
    """
    brief = counter(missing_frame_debounce=2, passerby_max_seconds=0.01)
    feed(brief, 0, [make_track(1, x=20, y=180)])
    for frame_idx in range(1, 5):
        feed(brief, frame_idx, [])
    assert brief.passerby_ids == set(), "too brief to be a passerby"

    generous = counter(missing_frame_debounce=2, passerby_max_seconds=1.0)
    feed(generous, 0, [make_track(1, x=20, y=180)])
    for frame_idx in range(1, 5):
        feed(generous, frame_idx, [])
    assert 1 in generous.passerby_ids


def test_passerby_margin_is_config_driven():
    """A track hugging the edge is only a passerby inside the margin."""
    hugging = counter(missing_frame_debounce=2, passerby_margin_fraction=0.10)
    feed(hugging, 0, [make_track(1, x=20, y=180)])  # 20px of 640 is inside 10%
    for frame_idx in range(1, 5):
        feed(hugging, frame_idx, [])
    assert 1 in hugging.passerby_ids

    tight = counter(missing_frame_debounce=2, passerby_margin_fraction=0.001)
    feed(tight, 0, [make_track(1, x=20, y=180)])
    for frame_idx in range(1, 5):
        feed(tight, frame_idx, [])
    assert tight.passerby_ids == set()


# ----------------------------------------------------------------- staff


def test_staff_are_excluded_from_auto_mode():
    c = counter(missing_frame_debounce=2)

    feed(c, 0, [make_track(1, x=320, y=180, is_staff=True)])
    for frame_idx in range(1, 5):
        feed(c, frame_idx, [make_track(1, x=320, y=180, is_staff=True)])

    assert c.counts[SCENE_FLOW] == {"in": 0, "out": 0}
    assert c.summary()["net_inside"] == 0


# ---------------------------------------------------------------- summary


def test_auto_mode_summary_shape():
    c = counter()
    feed(c, 0, [make_track(1, x=320, y=180)])
    for frame_idx in range(1, 12):
        feed(c, frame_idx, [])

    summary = c.summary()
    for key in (
        "per_line", "total_in", "total_out", "net_inside",
        "outside_passersby", "group_stats",
    ):
        assert key in summary

    assert summary["total_in"] == 1
    assert summary["total_out"] == 1
    assert summary["net_inside"] == 0
    assert "average_group_size" in summary["group_stats"]


def test_reset_clears_auto_mode_state():
    c = counter(missing_frame_debounce=2)
    feed(c, 0, [make_track(1, x=320, y=180)])
    for frame_idx in range(1, 5):
        feed(c, frame_idx, [])

    c.reset()

    assert c.counts[SCENE_FLOW] == {"in": 0, "out": 0}
    assert c.inside_ids == set()
    assert c.passerby_ids == set()
    assert c.events == []
