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

from .config import Config
from .counting import LineCounter
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
            counter = LineCounter(cfg.scaled_lines(width, height))

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


def _process_frame(
    frame,
    frame_idx: int,
    fps: float,
    detector: PersonDetector,
    tracker: SimpleTracker,
    counter: LineCounter,
):
    """enhance -> detect -> track -> count for one frame."""
    enhanced = enhance_frame(frame)
    timestamp_s = frame_idx / fps if fps else 0.0
    tracks = tracker.update(detector.detect(enhanced), frame_idx)
    counter.update(tracks, frame_idx, timestamp_s)
    return tracks
