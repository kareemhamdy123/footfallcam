"""Feature 5: area profiling - dwell accumulation and engagement share.

The interesting assertions are about attribution. A track is placed by its foot
point, so a child and an adult standing at the same depth are attributed to the
same zone; and time spent outside every polygon is reported separately instead
of quietly inflating the percentages.
"""
import pytest

from features.area_profiling import AreaProfiler
from src.config import Zone
from src.geometry import auto_bands
from tests.conftest import make_track

WIDTH, HEIGHT = 640, 360
NEAR = Zone("near", [(0, 0), (640, 0), (640, 120), (0, 120)], "generic")
MID = Zone("mid", [(0, 120), (640, 120), (640, 240), (0, 240)], "generic")
FAR = Zone("far", [(0, 240), (640, 240), (640, 360), (0, 360)], "generic")
ZONES = [NEAR, MID, FAR]

# A polygon in the middle of the frame, with clear space either side.
ISLAND = Zone("island", [(200, 150), (400, 150), (400, 210), (200, 210)])


def profiler(zones=ZONES, **kwargs):
    kwargs.setdefault("frame_shape", (WIDTH, HEIGHT))
    return AreaProfiler(zones, **kwargs)


def dwell(p, tracks, seconds, frame_shape=(WIDTH, HEIGHT)):
    return p.update(tracks, seconds, frame_shape)


# ------------------------------------------------------------ accumulation


def test_dwell_accumulates_per_zone():
    p = profiler()
    dwell(p, [make_track(1, y=50)], 2.0)    # near
    dwell(p, [make_track(2, y=180)], 3.0)   # mid
    dwell(p, [make_track(2, y=180)], 1.0)   # mid again

    assert p.summary()["seconds_by_zone"] == {"near": 2.0, "mid": 4.0}


def test_update_returns_the_frames_own_headcount():
    p = profiler()
    counts = dwell(p, [make_track(1, y=50), make_track(2, y=180)], 0.1)
    assert counts == {"near": 1, "mid": 1}


def test_dwell_is_driven_by_dt_not_by_frame_count():
    """A slow frame contributes proportionally more than a fast one."""
    p = profiler()
    dwell(p, [make_track(1, y=180)], 0.5)
    dwell(p, [make_track(1, y=180)], 0.5)
    dwell(p, [make_track(1, y=180)], 0.25)
    assert p.summary()["seconds_by_zone"]["mid"] == pytest.approx(1.25)


def test_a_track_moving_between_zones_credits_each():
    p = profiler()
    dwell(p, [make_track(1, y=50)], 1.0)
    dwell(p, [make_track(1, y=300)], 1.0)
    assert p.summary()["seconds_by_zone"] == {"near": 1.0, "far": 1.0}


def test_several_people_in_one_zone_accumulate_independently():
    p = profiler()
    dwell(p, [make_track(1, y=180), make_track(2, y=180)], 2.0)
    # Both are present, so two seconds of floor time each - not one shared total.
    assert p.summary()["seconds_by_zone"]["mid"] == pytest.approx(4.0)


# ------------------------------------------------------------- attribution


def test_attribution_uses_the_foot_point_not_the_centroid():
    """A short box standing at the same depth must land in the same zone."""
    p = profiler()
    # A child-sized box: foot at y=300, centroid much higher at ~280.
    child = make_track(1, x=300, y=300, width=16, height=20)
    assert child.foot_point[1] == pytest.approx(300)

    dwell(p, [child], 1.0)
    assert "far" in p.summary()["seconds_by_zone"]


def test_a_track_outside_every_zone_is_recorded_as_unzoned():
    p = profiler([ISLAND])
    dwell(p, [make_track(1, x=50, y=50)], 2.0)   # far from the island

    summary = p.summary()
    assert summary["seconds_by_zone"] == {}
    assert summary["seconds_outside_all_zones"] == 2.0
    assert summary["total_engagement_seconds"] == 0.0


def test_unzoned_time_does_not_inflate_the_percentage_denominator():
    p = profiler([ISLAND])
    dwell(p, [make_track(1, x=50, y=50)], 100.0)   # a long walk outside
    dwell(p, [make_track(2, x=300, y=180)], 1.0)   # one second on the island

    summary = p.summary()
    assert summary["total_engagement_seconds"] == 1.0
    assert summary["engagement_percentage"] == {"island": 100.0}


def test_overlapping_zones_resolve_to_the_first_configured_match():
    first = Zone("priority", [(0, 0), (640, 0), (640, 360), (0, 360)])
    p = profiler([first, MID])
    dwell(p, [make_track(1, y=180)], 1.0)
    assert list(p.summary()["seconds_by_zone"]) == ["priority"]


# ------------------------------------------------------------- percentages


def test_engagement_percentages_sum_to_one_hundred():
    p = profiler()
    dwell(p, [make_track(1, y=50)], 1.0)
    dwell(p, [make_track(2, y=180)], 2.0)
    dwell(p, [make_track(3, y=300)], 1.0)

    percentages = p.summary()["engagement_percentage"]
    assert sum(percentages.values()) == pytest.approx(100.0)
    assert percentages == {"near": 25.0, "mid": 50.0, "far": 25.0}


def test_engagement_percentages_sum_to_one_hundred_in_auto_mode():
    """Shares sum to ~100; the shortfall is independent rounding.

    Three equal zones read 33.33 each. The plan's criterion is "~100%", and
    padding the largest zone to force an exact 100 would report time it did
    not have.
    """
    p = AreaProfiler(auto_band_count=3, frame_shape=(WIDTH, HEIGHT))
    for y in (30, 180, 330):
        dwell(p, [make_track(1, y=y)], 1.0)

    percentages = p.summary()["engagement_percentage"]
    assert percentages == {"band_near": 33.33, "band_mid": 33.33, "band_far": 33.33}
    assert sum(percentages.values()) == pytest.approx(100.0, abs=0.05)


def test_percentages_are_zero_rather_than_an_error_with_no_data():
    summary = profiler().summary()
    assert summary["engagement_percentage"] == {}
    assert summary["total_engagement_seconds"] == 0.0


# ---------------------------------------------------------------- auto mode


def test_auto_mode_builds_bands_from_relative_y():
    p = AreaProfiler(frame_shape=(WIDTH, HEIGHT), auto_band_count=3)
    dwell(p, [make_track(1, y=50)], 1.0)
    dwell(p, [make_track(2, y=180)], 1.0)
    dwell(p, [make_track(3, y=330)], 1.0)

    assert p.summary()["auto_mode"] is True
    assert p.summary()["seconds_by_zone"] == {
        "band_near": 1.0, "band_mid": 1.0, "band_far": 1.0
    }


def test_configured_zones_disable_auto_mode():
    p = profiler()
    dwell(p, [make_track(1, y=180)], 1.0)
    assert p.summary()["auto_mode"] is False


def test_auto_mode_needs_a_frame_shape_before_it_can_band():
    p = AreaProfiler()  # no zones, no frame shape
    p.update([make_track(1, y=180)], 1.0, (WIDTH, HEIGHT))
    assert p.summary()["zones_profiled"] == 1


def test_band_count_is_configurable():
    p = AreaProfiler(frame_shape=(WIDTH, HEIGHT), auto_band_count=2)
    dwell(p, [make_track(1, y=50)], 1.0)
    dwell(p, [make_track(2, y=330)], 1.0)
    assert set(p.summary()["seconds_by_zone"]) == {"band_near", "band_far"}


def test_a_two_band_split_does_not_call_its_deepest_band_middle():
    """Depth names must be chosen per count, not sliced off a fixed list."""
    names = [b.name for b in auto_bands(WIDTH, HEIGHT, 2)]
    assert names == ["band_near", "band_far"]


def test_band_names_stay_meaningful_at_awkward_counts():
    assert [b.name for b in auto_bands(WIDTH, HEIGHT, 1)] == ["band_only"]
    assert [b.name for b in auto_bands(WIDTH, HEIGHT, 3)] == [
        "band_near", "band_mid", "band_far"
    ]
    assert [b.name for b in auto_bands(WIDTH, HEIGHT, 5)] == [
        "band_near", "band_mid_1", "band_mid_2", "band_mid_3", "band_far",
    ]


def test_bands_tile_the_frame_so_nothing_falls_through():
    bands = auto_bands(WIDTH, HEIGHT, 3)
    for y in (0, 1, 119, 120, 239, 240, 359):
        assert sum(1 for b in bands if b.polygon[0][1] <= y < b.polygon[2][1]) == 1


def test_auto_bands_reject_a_nonsensical_count():
    with pytest.raises(ValueError):
        auto_bands(WIDTH, HEIGHT, 0)


# ----------------------------------------------------------------- summary


def test_summary_exposes_the_documented_shape():
    summary = profiler().summary()
    for key in (
        "seconds_by_zone", "engagement_percentage",
        "total_engagement_seconds", "zones_profiled",
    ):
        assert key in summary


def test_zones_profiled_counts_only_zones_actually_visited():
    p = profiler()
    dwell(p, [make_track(1, y=180)], 1.0)
    assert p.summary()["zones_profiled"] == 1


def test_a_caller_resolved_auto_bands_still_report_auto_mode():
    """The pipeline bands the frame itself, so it must be able to say so."""
    bands = auto_bands(WIDTH, HEIGHT, 3)
    p = AreaProfiler(bands, frame_shape=(WIDTH, HEIGHT), auto_mode=True)
    dwell(p, [make_track(1, y=180)], 1.0)

    assert p.summary()["auto_mode"] is True
    assert p.summary()["seconds_by_zone"] == {"band_mid": 1.0}


def test_configured_zones_report_not_auto_mode():
    p = AreaProfiler(ZONES, frame_shape=(WIDTH, HEIGHT), auto_mode=False)
    dwell(p, [make_track(1, y=180)], 1.0)
    assert p.summary()["auto_mode"] is False


def test_reset_clears_accumulated_dwell():
    p = profiler()
    dwell(p, [make_track(1, y=180)], 3.0)
    p.reset()
    assert p.summary()["total_engagement_seconds"] == 0.0
    assert p.summary()["seconds_by_zone"] == {}
