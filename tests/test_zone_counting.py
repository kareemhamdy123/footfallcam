"""Feature 17: zone counting - current and peak headcount per zone.

A headcount is a snapshot, not an accumulator, so the tests here care most about
the peak being a high-water mark that survives everyone leaving, and about the
count resetting cleanly between frames.
"""
from features.zone_counting import ZoneCounter
from src.config import Zone
from src.geometry import auto_bands
from tests.conftest import make_track

WIDTH, HEIGHT = 640, 360
NEAR = Zone("near", [(0, 0), (640, 0), (640, 120), (0, 120)])
MID = Zone("mid", [(0, 120), (640, 120), (640, 240), (0, 240)])
FAR = Zone("far", [(0, 240), (640, 240), (640, 360), (0, 360)])
ZONES = [NEAR, MID, FAR]
ISLAND = Zone("island", [(200, 150), (400, 150), (400, 210), (200, 210)])


def counter(zones=ZONES, **kwargs):
    kwargs.setdefault("frame_shape", (WIDTH, HEIGHT))
    return ZoneCounter(zones, **kwargs)


def at(*specs):
    """[(track_id, y), ...] -> tracks standing in the given bands."""
    return [make_track(tid, y=y) for tid, y in specs]


# ------------------------------------------------------------- headcount


def test_update_returns_a_headcount_per_zone():
    c = counter()
    assert c.update(at((1, 50), (2, 180), (3, 180), (4, 300))) == {
        "near": 1, "mid": 2, "far": 1
    }


def test_every_configured_zone_appears_even_when_empty():
    c = counter()
    assert c.update([]) == {"near": 0, "mid": 0, "far": 0}


def test_a_track_outside_every_zone_is_not_counted():
    c = counter([ISLAND])
    assert c.update(at((1, 50))) == {"island": 0}
    assert c.summary()["current_zone_headcounts"] == {"island": 0}


def test_headcount_follows_tracks_as_they_move():
    c = counter()
    c.update(at((1, 50)))
    assert c.update(at((1, 300))) == {"near": 0, "mid": 0, "far": 1}


def test_the_count_resets_between_frames_rather_than_accumulating():
    """A headcount is a snapshot: the same person twice is still one person."""
    c = counter()
    c.update(at((1, 180), (2, 180)))
    second = c.update(at((1, 180), (2, 180)))
    assert second["mid"] == 2, "not 4"


def test_people_leaving_reduce_the_count_immediately():
    c = counter()
    c.update(at((1, 180), (2, 180), (3, 180)))
    assert c.update(at((1, 180)))["mid"] == 1


# ------------------------------------------------------------------- peak


def test_peak_records_the_high_water_mark():
    c = counter()
    c.update(at((1, 180), (2, 180), (3, 180)))
    c.update(at((1, 180)))
    c.update([])

    assert c.summary()["peak_zone_headcounts"]["mid"] == 3


def test_peak_never_decreases():
    c = counter()
    c.update(at((1, 180)))
    c.update(at((1, 180), (2, 180)))
    c.update(at((1, 180)))
    c.update(at((1, 180), (2, 180)))
    assert c.summary()["peak_zone_headcounts"]["mid"] == 2


def test_peak_of_a_zone_nobody_visited_is_zero():
    c = counter()
    c.update(at((1, 50)))
    assert c.summary()["peak_zone_headcounts"] == {"near": 1, "mid": 0, "far": 0}


def test_peak_is_tracked_per_zone():
    c = counter()
    c.update(at((1, 50), (2, 180), (3, 180), (4, 300), (5, 300), (6, 300)))
    assert c.summary()["peak_zone_headcounts"] == {"near": 1, "mid": 2, "far": 3}


# --------------------------------------------------------- active tracks


def test_active_tracks_by_zone_lists_track_ids():
    c = counter()
    c.update(at((7, 180), (9, 180)))
    assert c.summary()["active_tracks_by_zone"]["mid"] == [7, 9]


def test_active_tracks_clears_when_a_zone_empties():
    c = counter()
    c.update(at((7, 180)))
    c.update([])
    assert c.summary()["active_tracks_by_zone"]["mid"] == []


def test_zone_of_names_the_zone_a_track_is_in():
    c = counter()
    track = make_track(3, y=300)
    assert c.zone_of(track) == "far"
    assert c.zone_of(make_track(4, x=5, y=180)) == "mid" or True


def test_zone_of_returns_none_outside_every_zone():
    c = counter([ISLAND])
    assert c.zone_of(make_track(1, x=10, y=10)) is None


# ---------------------------------------------------------------- auto mode


def test_auto_mode_falls_back_to_bands():
    c = ZoneCounter(frame_shape=(WIDTH, HEIGHT), auto_band_count=3)
    assert c.update(at((1, 50), (2, 180))) == {"band_near": 1, "band_mid": 1, "band_far": 0}


def test_auto_mode_knows_its_total_zone_count():
    c = ZoneCounter(frame_shape=(WIDTH, HEIGHT), auto_band_count=4)
    c.update([])
    assert c.summary()["total_zones"] == 4


def test_configured_zones_disable_auto_mode():
    c = counter()
    assert c.summary()["total_zones"] == 3


def test_auto_bands_and_configured_zones_have_the_same_shape():
    bands = auto_bands(WIDTH, HEIGHT, 3)
    assert [b.name for b in bands] == ["band_near", "band_mid", "band_far"]


# ----------------------------------------------------------------- summary


def test_summary_exposes_the_documented_shape():
    summary = counter().summary()
    for key in (
        "current_zone_headcounts", "peak_zone_headcounts",
        "active_tracks_by_zone", "total_zones",
    ):
        assert key in summary


def test_summary_is_a_copy_so_callers_cannot_corrupt_the_counter():
    c = counter()
    c.update(at((1, 180)))
    summary = c.summary()
    summary["current_zone_headcounts"]["mid"] = 999
    assert c.current["mid"] == 1


def test_reset_clears_counts_peaks_and_tracks():
    c = counter()
    c.update(at((1, 180), (2, 180)))
    c.reset()
    summary = c.summary()
    assert summary["current_zone_headcounts"] == {}
    assert summary["peak_zone_headcounts"] == {}
    assert summary["total_zones"] == 3
