"""Tracking hardening: association, lifecycle and smoothing.

Stable identities are the foundation every later phase builds on - the plan
lists characteristics 1, 3, 10 and 20 as all depending on a track surviving
long enough to be interesting.

No model, no video: the tracker is driven by synthetic `Detection` lists
(R4).
"""
import pytest

from src.config import TrackingSettings
from src.detection.dataclasses import Detection
from src.tracking.dataclasses import Track
from src.tracking.tracker import SimpleTracker, iou


def box(x1, y1, x2, y2, conf=0.9):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_id=0)


def confirm(tracker, frames):
    """Feed the same box for `frames` frames so tracks reach min_hits."""
    out = []
    for i in range(frames):
        out.append(tracker.update([box(*frames[i])], i))
    return out


# ------------------------------------------------------------------------ iou


def test_iou_of_identical_boxes_is_one_and_of_disjoint_boxes_is_zero():
    assert iou(box(0, 0, 10, 10), box(0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou(box(0, 0, 10, 10), box(50, 50, 60, 60)) == 0.0


def test_iou_of_half_overlapping_boxes_is_one_third():
    # Two 10x10 boxes sharing a 5x10 half: inter 50, union 150.
    assert iou(box(0, 0, 10, 10), box(5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_is_zero_for_a_degenerate_box():
    assert iou(box(0, 0, 0, 0), box(0, 0, 10, 10)) == 0.0


# ------------------------------------------------------------- track lifecycle


def test_new_track_is_opened_and_needs_two_hits_to_confirm():
    tracker = SimpleTracker(TrackingSettings())

    first = tracker.update([box(100, 100, 140, 240)], 0)
    assert tracker.total_opened == 1
    assert first == [], "an unconfirmed track must not be published"
    assert tracker.tracks[1].hits == 1
    assert tracker.tracks[1].is_confirmed is False

    second = tracker.update([box(100, 100, 140, 240)], 1)
    assert len(second) == 1, "min_hits=2 must publish from the second frame"
    assert second[0].is_confirmed is True


def test_high_confidence_detection_matches_the_existing_track_by_iou():
    """A confident box near the predicted position keeps the same identity."""
    tracker = SimpleTracker(TrackingSettings())
    tracker.update([box(100, 100, 140, 240)], 0)

    tracks = tracker.update([box(103, 100, 143, 240)], 1)

    assert tracker.total_opened == 1, "no new identity for a matched detection"
    assert tracks[0].track_id == 1
    assert tracks[0].missed == 0
    assert tracks[0].hits == 2


def test_low_confidence_box_recovers_a_track_lost_to_occlusion():
    """ByteTrack's second stage: a weak box is still the same person."""
    cfg = TrackingSettings()
    tracker = SimpleTracker(cfg)
    tracker.update([box(100, 100, 140, 240)], 0)
    tracker.update([box(100, 100, 140, 240)], 1)

    # Frame 2: a weak (0.30) but well-overlapping box.
    tracks = tracker.update([box(101, 100, 141, 240, conf=0.30)], 2)

    assert tracker.total_opened == 1, "a weak box must not open a second identity"
    assert tracks[0].track_id == 1
    assert tracks[0].hits == 3


def test_weak_alone_detection_does_not_open_a_track():
    """A lone weak box is usually a partial body, not a person."""
    tracker = SimpleTracker(TrackingSettings())

    tracker.update([box(100, 100, 140, 240, conf=0.30)], 0)

    assert tracker.total_opened == 0


def test_track_survives_brief_occlusion_and_is_bridged_on_prediction():
    cfg = TrackingSettings(max_occlusion_bridge=2)
    tracker = SimpleTracker(cfg)
    tracker.update([box(100, 100, 140, 240)], 0)
    tracker.update([box(100, 100, 140, 240)], 1)

    # Two empty frames: the track is still published, carried on velocity.
    for frame_idx in (2, 3):
        tracks = tracker.update([], frame_idx)
        assert len(tracks) == 1, f"track lost at frame {frame_idx}"
        assert tracks[0].missed == frame_idx - 1

    # A third empty frame exceeds the bridge and the track stops being shown.
    assert tracker.update([], 4) == []
    assert tracker.tracks[1].missed == 3


def test_track_is_deleted_after_max_missed_frames():
    cfg = TrackingSettings(max_missed=3)
    tracker = SimpleTracker(cfg)
    tracker.update([box(100, 100, 140, 240)], 0)
    tracker.update([box(100, 100, 140, 240)], 1)
    assert 1 in tracker.tracks

    for frame_idx in range(2, 2 + cfg.max_missed + 1):
        tracker.update([], frame_idx)

    assert tracker.tracks == {}, "track must be retired once max_missed is passed"


def test_predict_step_advances_tracks_without_a_detection():
    tracker = SimpleTracker(TrackingSettings())
    tracker.update([box(100, 100, 140, 240)], 0)
    tracker.update([box(110, 100, 150, 240)], 1)  # establishes velocity

    before = tracker.tracks[1].detection.x1
    tracks = tracker.predict_step(2)

    assert len(tracks) == 1
    assert tracks[0].detection.x1 > before, "predicted box should move with velocity"

# ------------------------------------------------- association correctness


def test_distance_stage_matches_the_closest_track_not_the_farthest():
    """Regression: greedy matching must rank by proximity.

    A distance is a "lower is better" metric; sorting it like an IoU would make
    the tracker adopt the most distant track.

    iou_threshold is left at its default so the box below, which overlaps
    neither track, is rejected by stage 1 and can only be claimed by the
    distance stage.
    """
    cfg = TrackingSettings(max_center_distance_px=500)
    tracker = SimpleTracker(cfg)
    tracker.update([box(0, 100, 20, 140), box(400, 100, 420, 140)], 0)
    tracker.update([box(2, 100, 22, 140), box(402, 100, 422, 140)], 1)
    assert tracker.total_opened == 2

    # Overlaps neither track, and sits beside track 2 only.
    tracker.update([box(380, 100, 400, 140)], 2)

    assert tracker.tracks[1].missed == 1, "the far track must be left unmatched"
    assert tracker.tracks[2].missed == 0
    assert tracker.tracks[2].detection.x1 == pytest.approx(390, abs=5)


def test_one_detection_is_never_assigned_to_two_tracks():
    """Regression: an earlier stage's detection must not be re-offered.

    Stage 1 claims the box on IoU; the distance stage must then see it as
    already used rather than handing it to the other track as well.
    """
    tracker = SimpleTracker(TrackingSettings(max_center_distance_px=500))
    tracker.update([box(0, 100, 20, 140), box(400, 100, 420, 140)], 0)
    tracker.update([box(2, 100, 22, 140), box(402, 100, 422, 140)], 1)
    hits_before = {tid: t.hits for tid, t in tracker.tracks.items()}

    tracker.update([box(402, 100, 422, 140)], 2)  # a single detection

    gained = sum(
        tracker.tracks[tid].hits - hits_before[tid] for tid in hits_before
    )
    assert gained == 1, "one detection may satisfy at most one track"
    assert tracker.total_opened == 2, "a matched detection must not spawn a track"
    assert tracker.tracks[1].missed == 1, "only the nearer track may claim it"


def test_two_detections_never_swap_identities():
    tracker = SimpleTracker(TrackingSettings())
    tracker.update([box(0, 100, 40, 300), box(300, 100, 340, 300)], 0)
    tracker.update([box(2, 100, 42, 300), box(302, 100, 342, 300)], 1)

    left_before = tracker.tracks[1].detection.x1

    tracks = tracker.update([box(4, 100, 44, 300), box(304, 100, 344, 300)], 2)
    by_id = {t.track_id: t for t in tracks}

    assert by_id[1].detection.x1 == pytest.approx(4, abs=2)
    assert by_id[1].detection.x1 > left_before
    assert by_id[2].detection.x1 == pytest.approx(304, abs=2)
    assert tracker.total_opened == 2


# ------------------------------------------------------------------ smoothing


def test_box_uses_exponential_moving_average():
    """A 0.70 alpha lands 70% of the way toward the new measurement.

    The two boxes overlap, so the match is made on IoU and the EMA is what
    moves the stored box.
    """
    cfg = TrackingSettings(box_ema_alpha=0.70)
    tracker = SimpleTracker(cfg)
    tracker.update([box(0, 0, 100, 100)], 0)

    tracker.update([box(10, 0, 110, 100)], 1)

    assert tracker.tracks[1].detection.x1 == pytest.approx(7.0, abs=0.5)


def test_ema_alpha_is_config_driven():
    """R3: alpha is a tunable, not a constant."""
    for alpha, expected in ((0.0, 0.0), (0.5, 5.0), (1.0, 10.0)):
        cfg = TrackingSettings(box_ema_alpha=alpha)
        tracker = SimpleTracker(cfg)
        tracker.update([box(0, 0, 100, 100)], 0)
        tracker.update([box(10, 0, 110, 100)], 1)
        assert tracker.tracks[1].detection.x1 == pytest.approx(expected, abs=0.5), (
            f"alpha={alpha}"
        )


def test_velocity_tracks_centroid_motion():
    cfg = TrackingSettings(velocity_ema_alpha=0.60)
    tracker = SimpleTracker(cfg)
    tracker.update([box(0, 0, 100, 100)], 0)
    tracker.update([box(10, 0, 110, 100)], 1)

    track = tracker.tracks[1]
    # Centroid moved 10px right; 0.60 alpha on a standing start gives 6.0.
    assert track.velocity[0] == pytest.approx(6.0, abs=0.01)
    assert track.velocity[1] == pytest.approx(0.0, abs=0.01)


def test_ema_smoothing_damps_a_single_frame_outlier():
    """Each measurement pulls the stored box 70% of the way toward it."""
    cfg = TrackingSettings(box_ema_alpha=0.70)
    tracker = SimpleTracker(cfg)
    tracker.update([box(0, 0, 100, 100)], 0)   # stored x1 = 0
    tracker.update([box(10, 0, 110, 100)], 1)  # 0.70*10 + 0.30*0  = 7.0
    assert tracker.tracks[1].detection.x1 == pytest.approx(7.0, abs=0.5)

    tracker.update([box(12, 0, 112, 100)], 2)  # 0.70*12 + 0.30*7.0 = 10.5
    assert tracker.tracks[1].detection.x1 == pytest.approx(10.5, abs=0.2)

    # The stored box lags a 12px measurement by design - that is the damping.
    assert tracker.tracks[1].detection.x1 < 12.0


# ------------------------------------------------------------ track attributes


def test_history_is_appended_and_capped():
    cfg = TrackingSettings(max_history=5)
    tracker = SimpleTracker(cfg)

    for frame_idx in range(12):
        tracker.update([box(100, 100, 140, 240)], frame_idx)

    history = tracker.tracks[1].history
    assert len(history) == 5
    assert history[-1][0] == 11, "the newest point must be kept"
    assert all(isinstance(point, tuple) and len(point) == 3 for point in history)


def test_counted_lines_starts_empty_and_is_owned_by_the_track():
    tracker = SimpleTracker(TrackingSettings())
    tracker.update([box(100, 100, 140, 240)], 0)

    track = tracker.tracks[1]
    assert track.counted_lines == set()
    track.counted_lines.add("main_entrance")
    assert "main_entrance" in tracker.tracks[1].counted_lines


def test_track_exposes_foot_point_from_the_box_bottom():
    track = Track(track_id=1, detection=box(100, 50, 140, 250))
    assert track.foot_point == pytest.approx((120.0, 250.0))
    assert track.box == (100, 50, 140, 250)
