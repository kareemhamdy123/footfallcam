"""The pipeline: the one place the layers are wired together.

    source -> enhance -> detect -> track -> count -> annotate -> write

Every tunable comes from the loaded `Config` (R3). Each run overwrites
`outputs/annotated.mp4` and `outputs/report.json` (R6; `heatmap.png` arrives
with the Phase 8 heatmap feature).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from .config import Config
from .detection.model import PersonDetector
from .geometry import auto_bands
from .tracking.tracker import SimpleTracker
from .video_io import (
    convert_to_h264_web,
    enhance_frame,
    get_stream_info,
    make_writer,
    open_source,
    read_frame,
)
# Features live outside src/ and are imported by their public name only.
from features.counter import VideoCounter
from features.gender import DemographicsAggregator, GenderClassifier, UNKNOWN
from features.playback import PlaybackEngine
from features.area_profiling import AreaProfiler
from features.zone_counting import ZoneCounter

ANNOTATED_NAME = "annotated.mp4"
REPORT_NAME = "report.json"


class Pipeline:
    @staticmethod
    def run(
        config_path: str,
        show: bool = False,
        max_frames: int | None = None,
        source_override: str | None = None,
    ) -> dict:
        cfg = Config.load(config_path)
        source = source_override or cfg.source
        os.makedirs(cfg.output_dir, exist_ok=True)

        capture = open_source(source)
        try:
            info = get_stream_info(capture)
            width, height = info["width"], info["height"]
            if not width or not height:
                raise RuntimeError(f"Source reported no frame size: {source}")

            detector = PersonDetector(cfg.detection)
            tracker = SimpleTracker(cfg.tracking)
            width, height = info["width"], info["height"]
            counter = VideoCounter(
                lines=cfg.scaled_lines(width, height),
                group_max_distance_px=cfg.counting.group_max_distance_px,
                frame_shape=(width, height),
                group_entry_window_s=cfg.counting.group_entry_window_s,
                missing_frame_debounce=cfg.counting.auto_mode.missing_frame_debounce,
                passerby_margin_fraction=cfg.counting.auto_mode.passerby_margin_fraction,
                passerby_max_seconds=cfg.counting.auto_mode.passerby_max_seconds,
            )

            # Characteristic 6 is opt-in; see features/gender.py for why.
            demographics = DemographicsAggregator()
            gender_classifier = None
            if cfg.demographics.enabled:
                gender_classifier = GenderClassifier(
                    model_path=cfg.demographics.model_path,
                    min_confidence=cfg.demographics.heuristic_min_confidence,
                )

            annotated_path = os.path.join(cfg.output_dir, ANNOTATED_NAME)
            writer = make_writer(annotated_path, info["fps"], (width, height))

            playback = PlaybackEngine(
                lines=cfg.scaled_lines(width, height),
                zones=cfg.scaled_zones(width, height),
                source_label=os.path.basename(source),
            )

            # Characteristics 5 and 17 reason over the same zones. Resolved
            # once here so both features and the overlay agree exactly, and so
            # the profiler can still report that it is running in auto mode.
            configured_zones = cfg.scaled_zones(width, height)
            using_auto_bands = not configured_zones
            analysis_zones = configured_zones or auto_bands(
                width, height, cfg.analysis.auto_band_count
            )
            profiler = AreaProfiler(
                analysis_zones,
                frame_shape=(width, height),
                auto_mode=using_auto_bands,
            )
            zone_counter = ZoneCounter(analysis_zones, frame_shape=(width, height))

            started = time.time()
            frame_idx = 0
            active_tracks: list = []
            try:
                while max_frames is None or frame_idx < max_frames:
                    ok, frame = read_frame(capture)
                    if not ok:
                        break

                    frame_detections = _process_frame(
                        frame, frame_idx, info["fps"], detector, tracker, counter
                    )
                    active_tracks, detections = frame_detections
                    if gender_classifier is not None:
                        _classify_demographics(
                            frame, active_tracks, gender_classifier,
                            demographics, cfg.demographics,
                        )
                    dt_s = 1.0 / info["fps"] if info["fps"] else 0.0
                    profiler.update(active_tracks, dt_s)
                    zone_counts = zone_counter.update(active_tracks)
                    p_summary = profiler.summary()
                    t_zones = {t.track_id: zone_counter.zone_of(t) for t in active_tracks}
                    queue_headcount = sum(zone_counts.get(z.name, 0) for z in configured_zones if getattr(z, "kind", "") == "queue")
                    annotated = playback.render_proof_frame(
                        frame,
                        active_tracks,
                        detections_by_id=_detections_by_id(detections, active_tracks),
                        counts_summary=counter.summary(),
                        kpis={
                            "demo": _demo_kpi(demographics),
                            "queues": queue_headcount,
                        },
                        zone_counts=zone_counts,
                        zone_peaks=zone_counter.peak,
                        dwell_percentages=p_summary.get("engagement_percentage", {}),
                        track_zones=t_zones,
                        fps=info["fps"] or 13.093,
                    )
                    writer.write(annotated)

                    if show and cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    frame_idx += 1
            finally:
                writer.release()
        finally:
            capture.release()
            if show:
                cv2.destroyAllWindows()

        elapsed_s = time.time() - started
        transcode_ok = convert_to_h264_web(annotated_path)

        counting_summary = counter.summary()
        counting_summary["events_log"] = counter.events

        visits = []
        for tid, t_pos in counter._first_seen.items():
            last_t = counter._last_seen.get(tid, t_pos)[0]
            visits.append({
                "track_id": tid,
                "first_seen_s": round(t_pos[0], 2),
                "last_seen_s": round(last_t, 2),
                "dwell_seconds": round(last_t - t_pos[0], 2),
            })

        report = {
            "meta": {
                "source": source,
                "config": str(config_path),
                "detector_backend": detector.backend,
                "frames_processed": frame_idx,
                "fps": info["fps"],
                "frame_size": [width, height],
                "elapsed_seconds": round(elapsed_s, 2),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "h264_web_transcode": transcode_ok,
            },
            "tracking": {
                "live_tracks": len(active_tracks),
                "total_tracks_opened": tracker.total_opened,
            },
            "counting": counting_summary,
            "counting_visits": visits,
            "demographics": demographics.summary(),
            "area_profiling": profiler.summary(),
            "zone_counting": zone_counter.summary(),
        }

        report_path = os.path.join(cfg.output_dir, REPORT_NAME)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        counting = report["counting"]
        print(
            f"[pipeline] {frame_idx} frames | IN {counting['total_in']}"
            f" | OUT {counting['total_out']} | inside {counting['net_inside']}"
            f" | {elapsed_s:.1f}s"
        )
        print(f"[pipeline] wrote {annotated_path} and {report_path}")
        return report


def _demo_kpi(aggregator: DemographicsAggregator) -> str | None:
    """The DEMO (M/F) HUD field, or None while nothing has been classified."""
    summary = aggregator.summary()
    if not summary["total_classified"]:
        return None
    return (
        f"{summary['male_percentage']}% / {summary['female_percentage']}%"
    )


def _detections_by_id(detections: list, tracks: list) -> dict:
    """Match this frame's raw detections to the tracks they produced.

    A proof frame should show the box the model actually returned, so the
    drawing prefers the detection over the tracker's smoothed box. A track that
    was bridged across an occlusion has no fresh detection and is simply
    absent from the mapping.
    """
    if not detections or not tracks:
        return {}
    mapping = {}
    for track in tracks:
        # Nearest detection to the track's own box, by centroid distance.
        tx, ty = track.detection.centroid
        best, best_distance = None, float("inf")
        for detection in detections:
            dx, dy = detection.centroid
            distance = abs(dx - tx) + abs(dy - ty)
            if distance < best_distance:
                best, best_distance = detection, distance
        if best is not None:
            mapping[track.track_id] = best
    return mapping


def _classify_demographics(
    frame,
    tracks: list,
    classifier: GenderClassifier,
    aggregator: DemographicsAggregator,
    settings,
) -> None:
    """Classify settled tracks progressively with highest confidence (characteristic 6)."""
    for track in tracks:
        if track.hits < settings.min_hits:
            continue
        cur_gender = getattr(track, "gender", None)
        cur_conf = getattr(track, "gender_confidence", 0.0)
        cur_crop_h = getattr(track, "_gender_crop_h", 0)

        crop = _person_crop(frame, track)
        if crop is None or crop.shape[0] < 45 or crop.shape[1] < 18:
            continue

        if cur_gender in ("male", "female") and cur_crop_h >= 100 and cur_conf >= 0.65:
            continue

        label, confidence = classifier.classify(crop)
        is_substantially_larger = crop.shape[0] > (cur_crop_h + 20)
        if label != UNKNOWN and (confidence > cur_conf or (is_substantially_larger and confidence >= 0.52)):
            track.gender = label
            track.gender_confidence = confidence
            track._gender_crop_h = crop.shape[0]
            aggregator.record_track(track)
        elif cur_gender is None:
            track.gender = UNKNOWN
            track.gender_confidence = 0.0
            track._gender_crop_h = crop.shape[0]


def _person_crop(frame, track) -> np.ndarray | None:
    """The pixels of the track's box, clipped to the frame."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in track.box)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _process_frame(
    frame,
    frame_idx: int,
    fps: float,
    detector: PersonDetector,
    tracker: SimpleTracker,
    counter: VideoCounter,
):
    """enhance -> detect -> track -> count for one frame.

    Returns `(tracks, detections)`; the detections are kept for the proof frame.
    """
    enhanced = enhance_frame(frame)
    timestamp_s = frame_idx / fps if fps else 0.0
    detections = detector.detect(enhanced)
    tracks = tracker.update(detections, frame_idx)
    counter.update(tracks, frame_idx, timestamp_s)
    return tracks, detections
