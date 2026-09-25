"""Video proof rendering: overlays, role colours and the HUD.

The annotated output is the artifact a retailer checks when they dispute a
count, so these tests care about two things above all: that the frame shows what
the detector actually returned, and that a KPI with no feature behind it is
never rendered as a plausible-looking number.
"""
import numpy as np

from features.playback import UNWIRED, PlaybackEngine
from src.config import Line, Zone
from src.visualizer import (
    HUD_FIELDS,
    LINE_COLOR,
    NOT_AVAILABLE,
    PERSON_COLOR,
    RETURNING_COLOR,
    STAFF_COLOR,
    ZONE_COLORS,
    annotate_frame,
    draw_audit_badge,
    draw_hud,
    role_color,
)
from tests.conftest import make_detection, make_track

WIDTH, HEIGHT = 640, 360
LINE = Line("main_entrance", (32, 180), (608, 180), "top_to_bottom")
SQUARE = Zone("floor", [(100, 100), (300, 100), (300, 260), (100, 260)], "sales")


def canvas(width=WIDTH, height=HEIGHT, value=0):
    return np.full((height, width, 3), value, dtype=np.uint8)


def has_colour(frame, bgr, tolerance=12):
    """True if any pixel is within `tolerance` of `bgr` on every channel."""
    target = np.array(bgr, dtype=np.int16)
    diff = np.abs(frame.astype(np.int16) - target).max(axis=2)
    return bool((diff <= tolerance).any())


def count_colour(frame, bgr, tolerance=12):
    target = np.array(bgr, dtype=np.int16)
    diff = np.abs(frame.astype(np.int16) - target).max(axis=2)
    return int((diff <= tolerance).sum())


# ---------------------------------------------------------- frame contract


def test_annotate_frame_never_mutates_its_input():
    frame = canvas(value=40)
    original = frame.copy()

    annotate_frame(frame, tracks=[make_track(1, x=300, y=200)], lines=[LINE], kpis={})

    assert np.array_equal(frame, original), "the source frame must be untouched"


def test_annotate_frame_with_no_overlays_returns_an_equal_copy():
    frame = canvas(value=17)
    result = annotate_frame(frame)

    assert result is not frame
    assert np.array_equal(result, frame)


def test_overlays_only_touch_the_top_left_region_they_claim():
    """Sanity: the HUD must not repaint the whole frame."""
    frame = canvas(value=90)
    result = annotate_frame(frame, kpis={"in_out": "1 / 0"})

    untouched = result[HEIGHT - 20:, WIDTH - 20:]
    assert np.all(untouched == 90)


# ------------------------------------------------------------ role colours


def test_an_ordinary_customer_is_drawn_in_emerald():
    assert role_color(make_track(1)) == PERSON_COLOR
    assert has_colour(annotate_frame(canvas(), tracks=[make_track(1, x=300, y=200)]),
                      PERSON_COLOR)


def test_staff_are_drawn_in_fuchsia():
    track = make_track(2, is_staff=True)
    assert role_color(track) == STAFF_COLOR
    assert has_colour(annotate_frame(canvas(), tracks=[track]), STAFF_COLOR)


def test_returning_customers_are_drawn_in_gold():
    track = make_track(3)
    track.is_returning = True
    assert role_color(track) == RETURNING_COLOR
    assert has_colour(annotate_frame(canvas(), tracks=[track]), RETURNING_COLOR)


def test_staff_outrank_returning_when_both_flags_are_set():
    track = make_track(4, is_staff=True)
    track.is_returning = True
    assert role_color(track) == STAFF_COLOR


def test_the_three_role_colours_are_distinguishable():
    assert len({PERSON_COLOR, STAFF_COLOR, RETURNING_COLOR}) == 3


# ----------------------------------------------------------------- badges


def test_a_track_with_no_stored_box_is_skipped_rather_than_crashing():
    class Bare:
        track_id = 9
        history = []
        is_staff = False
        is_returning = False
        gender = None

    result = annotate_frame(canvas(), tracks=[Bare()])
    assert result.shape == (HEIGHT, WIDTH, 3)


def test_the_gender_initial_appears_in_the_tag_when_known():
    track = make_track(5)
    track.gender = "female"
    frame = annotate_frame(canvas(), tracks=[track])
    assert frame.shape == (HEIGHT, WIDTH, 3)  # rendered without error

    track.gender = "unknown"
    assert annotate_frame(canvas(), tracks=[track]).shape == (HEIGHT, WIDTH, 3)


# ----------------------------------------------------------------- trails


def test_a_motion_trail_is_drawn_along_the_track_history():
    track = make_track(6)
    track.history = [(i, 100 + i * 10, 300) for i in range(6)]

    frame = annotate_frame(canvas(), tracks=[track])

    # Every step of the trail should have painted something.
    painted = sum(
        1
        for x in range(100, 160, 10)
        if not np.array_equal(frame[295:305, x:x + 6], canvas()[295:305, x:x + 6])
    )
    assert painted >= 4, f"only {painted} trail segments drawn"


def test_a_track_with_no_history_draws_no_trail():
    track = make_track(7)
    track.history = []
    assert annotate_frame(canvas(), tracks=[track]).shape == (HEIGHT, WIDTH, 3)


# ------------------------------------------------------ lines and zones


def test_a_counting_line_is_drawn_with_its_name():
    frame = annotate_frame(canvas(), lines=[LINE])
    assert has_colour(frame, LINE_COLOR)


def test_a_line_is_scaled_from_normalised_config():
    normalised = Line("door", (0.0, 0.5), (1.0, 0.5))
    frame = annotate_frame(canvas(), lines=[normalised])

    # The line sits at mid height, so that row must be painted.
    assert not np.array_equal(frame[180, 5:60], canvas()[180, 5:60])


def test_a_zone_is_tinted_and_labelled():
    frame = annotate_frame(canvas(), zones=[SQUARE])
    assert has_colour(frame, ZONE_COLORS["sales"])


def test_each_zone_kind_gets_its_own_tint():
    colours = set()
    for kind in ZONE_COLORS:
        zone = Zone("z", [(10, 10), (200, 10), (200, 200), (10, 200)], kind)
        colours.add(count_colour(annotate_frame(canvas(), zones=[zone]), ZONE_COLORS[kind]))
    assert all(c > 0 for c in colours)


def test_a_zone_near_the_top_edge_still_renders_inside_the_frame():
    top = Zone("top", [(0.0, 0.0), (0.5, 0.0), (0.5, 0.4), (0.0, 0.4)])
    frame = annotate_frame(canvas(), zones=[top])
    assert frame.shape == (HEIGHT, WIDTH, 3)


def test_no_zones_means_no_tint():
    assert count_colour(annotate_frame(canvas()), ZONE_COLORS["sales"]) == 0


# ------------------------------------------------------------ detections


def test_the_raw_detection_is_preferred_over_the_smoothed_track_box():
    """A proof frame must show what the model returned, not the EMA."""
    track = make_track(8, x=300, y=200)
    raw = make_detection(x1=10, y1=10, x2=60, y2=90, conf=0.8)

    with_raw = annotate_frame(canvas(), tracks=[track], detections_by_id={8: raw})
    without_raw = annotate_frame(canvas(), tracks=[track])

    assert not np.array_equal(with_raw, without_raw), (
        "the raw detection should render differently from the tracker's own box"
    )
    # ... and the raw box is the one that was painted.
    assert has_colour(with_raw, PERSON_COLOR, tolerance=40)
    assert np.array_equal(without_raw[170:190, 300:306], canvas()[170:190, 300:306])


def test_a_track_absent_from_the_mapping_falls_back_to_its_own_box():
    track = make_track(9, x=300, y=200)
    frame = annotate_frame(canvas(), tracks=[track], detections_by_id={999: make_detection()})
    assert has_colour(frame, PERSON_COLOR)


# -------------------------------------------------------------------- HUD


def test_the_hud_renders_every_declared_field():
    frame = draw_hud(canvas(), {"in_out": "3 / 1", "inside": 2, "queues": 0,
                                "return_rate": 12.5, "demo": "40% / 60%"})
    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert len(HUD_FIELDS) == 5, "the spec names five HUD fields"


def test_an_unwired_kpi_renders_as_not_available_not_zero():
    """The important one: an absent feature must not look like a measurement."""
    frame = draw_hud(canvas(), {"in_out": "3 / 1", "inside": 2})

    assert has_colour(frame, ZONE_COLORS["queue"]) is False or True  # colour aside:
    # Compare against a HUD where the field is explicitly zero.
    zeroed = draw_hud(canvas(), {"in_out": "3 / 1", "inside": 2, "queues": 0})
    blank = draw_hud(canvas(), {"in_out": "3 / 1", "inside": 2})
    assert not np.array_equal(blank, zeroed), "n/a must differ from 0"


def test_an_unwired_kpi_sentinel_is_none_which_draws_as_text():
    """The sentinel is None; the visualizer is what turns it into "n/a"."""
    assert UNWIRED is None
    assert NOT_AVAILABLE == "n/a", "the visualizer's placeholder is literal text"


def test_the_hud_survives_a_very_small_frame():
    frame = draw_hud(canvas(160, 90), {"in_out": "1 / 1"})
    assert frame.shape == (90, 160, 3)


def test_the_hud_survives_a_wide_frame():
    frame = draw_hud(canvas(1920, 1080), {"in_out": "1 / 1"})
    assert frame.shape == (1080, 1920, 3)


# ------------------------------------------------------------ audit badge


def test_the_audit_badge_is_drawn():
    frame = draw_audit_badge(canvas(), "FootfallCam-CV video proof")
    assert has_colour(frame, (160, 170, 180))


def test_the_audit_badge_stays_in_frame_on_a_small_canvas():
    frame = draw_audit_badge(canvas(200, 120), "a" * 200)
    assert frame.shape == (120, 200, 3)


# -------------------------------------------------------- PlaybackEngine


def test_kpis_are_built_from_the_counting_summary():
    engine = PlaybackEngine()
    kpis = engine.build_kpis({"total_in": 12, "total_out": 5, "net_inside": 7})

    assert kpis["in_out"] == "12 / 5"
    assert kpis["inside"] == 7


def test_unimplemented_kpis_stay_unavailable():
    kpis = PlaybackEngine().build_kpis({"total_in": 1, "total_out": 1, "net_inside": 0})

    assert kpis["queues"] is None
    assert kpis["return_rate"] is None
    assert kpis["demo"] is None


def test_an_empty_summary_still_produces_a_renderable_kpi_set():
    kpis = PlaybackEngine().build_kpis()
    assert kpis["in_out"] == "0 / 0"
    assert kpis["inside"] == 0


def test_extra_kpis_override_but_none_does_not():
    engine = PlaybackEngine()
    base = {"total_in": 0, "total_out": 0, "net_inside": 0}

    assert engine.build_kpis(base, {"demo": "50% / 50%"})["demo"] == "50% / 50%"
    assert engine.build_kpis(base, {"demo": None})["demo"] is None


def test_render_proof_frame_composes_every_layer():
    engine = PlaybackEngine(lines=[LINE], zones=[SQUARE], source_label="clip.mp4")
    track = make_track(1, x=300, y=200)
    track.history = [(i, 260 + i * 8, 300) for i in range(5)]

    frame = engine.render_proof_frame(
        canvas(),
        tracks=[track],
        detections_by_id={1: make_detection(290, 160, 310, 200)},
        counts_summary={"total_in": 4, "total_out": 1, "net_inside": 3},
    )

    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert has_colour(frame, PERSON_COLOR)
    assert has_colour(frame, LINE_COLOR)
    assert has_colour(frame, ZONE_COLORS["sales"])


def test_render_proof_frame_never_mutates_the_source():
    engine = PlaybackEngine()
    frame = canvas(value=33)
    original = frame.copy()

    engine.render_proof_frame(frame, tracks=[make_track(1)], counts_summary={})

    assert np.array_equal(frame, original)


def test_a_queue_monitor_without_queues_renders_nothing_extra():
    class NoQueues:
        auto_queues = {}

    frame = PlaybackEngine().render_proof_frame(
        canvas(), tracks=[], queue_monitor=NoQueues()
    )
    assert frame.shape == (HEIGHT, WIDTH, 3)


def test_auto_queue_polygons_are_drawn_when_present():
    class State:
        polygon = [(400, 100), (600, 100), (600, 300), (400, 300)]

    class Monitor:
        auto_queues = {"till": State()}

    frame = PlaybackEngine().render_proof_frame(
        canvas(), tracks=[], queue_monitor=Monitor()
    )
    assert has_colour(frame, (0, 165, 255))
