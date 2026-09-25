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
from .tracking.tracker import SimpleTracker
from .video_io import (
    convert_to_h264_web,
    enhance_frame,
    get_stream_info,
    make_writer,
    open_source,
    read_frame,
)
from .visualizer import annotate_frame

# Features live outside src/ and are imported by their public name only.
from features.counter import VideoCounter
from features.gender import DemographicsAggregator, GenderClassifier

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

            started = time.time()
            frame_idx = 0
            active_tracks: list = []
            try:
                while max_frames is None or frame_idx < max_frames:
                    ok, frame = read_frame(capture)
                    if not ok:
                        break

                    active_tracks = _process_frame(
                        frame, frame_idx, info["fps"], detector, tracker, counter
                    )
                    if gender_classifier is not None:
                        _classify_demographics(
                            frame, active_tracks, gender_classifier,
                            demographics, cfg.demographics,
                        )
                    annotated = annotate_frame(frame, active_tracks)
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
            "counting": counter.summary(),
            "demographics": demographics.summary(),
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


def _classify_demographics(
    frame,
    tracks: list,
    classifier: GenderClassifier,
    aggregator: DemographicsAggregator,
    settings,
) -> None:
    """Classify each settled, not-yet-classified track once (characteristic 6).

    Gated on `hits >= min_hits` so a person's label is not decided from the
    first two frames, when the box is still settling onto them.
    """
    for track in tracks:
        if track.gender is not None or track.hits < settings.min_hits:
            continue
        crop = _person_crop(frame, track)
        label, _confidence = classifier.classify(crop)
        track.gender = label
        aggregator.record_track(track)


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
    """enhance -> detect -> track -> count for one frame."""
    enhanced = enhance_frame(frame)
    timestamp_s = frame_idx / fps if fps else 0.0
    tracks = tracker.update(detector.detect(enhanced), frame_idx)
    counter.update(tracks, frame_idx, timestamp_s)
    return tracks
