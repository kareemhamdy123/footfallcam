"""
FootfallCam-style people-counting pipeline, built on YOLO detection.

Usage:
    python -m src.main --config configs/zones_example.yaml

Produces (in the configured output_dir):
    annotated.mp4   - playback video with boxes, IDs, lines, zones, trails
    heatmap.png     - accumulated presence heatmap overlaid on the last frame
    report.json     - full analytics report (see analytics/metrics.py)

Press "q" during a windowed run to stop early (only when --show is passed).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2

from .analytics.counting import LineCounter, TurnInRate
from .analytics.dwell import DwellTracker, SalesConversationMonitor
from .analytics.heatmap import AreaProfiler, HeatmapAccumulator
from .analytics.metrics import build_report
from .analytics.occupancy import OccupancyMonitor
from .analytics.queue import QueueMonitor
from .analytics.staff import tag_staff
from .config import Config
from .detector import PersonDetector
from .tracker import SimpleTracker
from .video_io import enhance_frame, make_writer, open_source

COLORS = {
    "box": (0, 200, 0),
    "staff_box": (255, 0, 255),
    "line": (0, 0, 255),
    "zone_generic": (255, 200, 0),
    "zone_queue": (0, 165, 255),
    "zone_sales": (200, 0, 200),
    "zone_safe": (0, 0, 255),
    "zone_outside": (150, 150, 150),
}


def annotate(frame, tracks, cfg, detections_by_id, line_summary):
    for zone in cfg.zones:
        color = COLORS.get(f"zone_{zone.kind}", COLORS["zone_generic"])
        pts = [(int(x), int(y)) for x, y in zone.polygon]
        cv2.polylines(frame, [cv2_points(pts)], True, color, 2)
        cv2.putText(frame, f"{zone.name} ({zone.kind})", pts[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    for line in cfg.lines:
        cv2.line(frame, tuple(map(int, line.p1)), tuple(map(int, line.p2)),
                  COLORS["line"], 2)
        cv2.putText(frame, line.name, tuple(map(int, line.p1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["line"], 1)

    for track in tracks:
        det = detections_by_id.get(track.track_id)
        if det is None:
            continue
        color = COLORS["staff_box"] if track.is_staff else COLORS["box"]
        cv2.rectangle(frame, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), color, 2)
        label = f"ID {track.track_id}" + (" STAFF" if track.is_staff else "")
        cv2.putText(frame, label, (int(det.x1), int(det.y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # trail
        pts = [(int(x), int(y)) for _, x, y in track.history[-30:]]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], color, 1)

    y = 20
    for line_name, c in line_summary["per_line"].items():
        cv2.putText(frame, f"{line_name}: in={c['in']} out={c['out']}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 22
    return frame


def cv2_points(pts):
    import numpy as np
    return np.array(pts, dtype=int)


def run(config_path: str, show: bool = False, max_frames: int | None = None) -> dict:
    cfg = Config.load(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)

    detector = PersonDetector(cfg.model_path, cfg.confidence, cfg.device, cfg.person_class_id)
    tracker = SimpleTracker()

    cap = open_source(cfg.source)
    fps = cfg.fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = make_writer(os.path.join(cfg.output_dir, "annotated.mp4"), fps, (width, height))
    heatmap = HeatmapAccumulator((width, height), decay=cfg.heatmap_decay)
    area_profiler = AreaProfiler(cfg.zones)

    line_counter = LineCounter(cfg.lines, cfg.group_max_distance_px)
    queue_monitor = QueueMonitor(cfg.zones)
    occupancy_monitor = OccupancyMonitor(cfg.zones, cfg.safe_occupancy_limit)

    entrance_lines = [l.name for l in cfg.lines if "entr" in l.name.lower()] or \
        ([cfg.lines[0].name] if cfg.lines else [])
    outside_lines = [l.name for l in cfg.lines if "outside" in l.name.lower()]
    dwell_tracker = DwellTracker(entrance_lines[0] if entrance_lines else "",
                                  cfg.dwell_alert_seconds)
    sales_monitor = SalesConversationMonitor(cfg.zones)
    turn_in_rate_calc = TurnInRate(
        outside_lines[0] if outside_lines else "",
        entrance_lines[0] if entrance_lines else "",
    )

    frame_idx = 0
    last_t = None
    start_time = time.time()
    last_frame_for_heatmap = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        timestamp_s = frame_idx / fps
        dt_s = (timestamp_s - last_t) if last_t is not None else (1.0 / fps)
        last_t = timestamp_s

        proc_frame = enhance_frame(frame, cfg.night_mode)
        detections = detector.detect(proc_frame)
        tracks = tracker.update(detections, frame_idx)

        # naive nearest-detection-to-track match for annotation/staff tagging
        detections_by_id = {}
        for t in tracks:
            detections_by_id[t.track_id] = t.detection
        tag_staff(frame, tracks, detections_by_id, cfg.staff_tag_hsv_lower, cfg.staff_tag_hsv_upper)

        line_counter.update(tracks, frame_idx, timestamp_s)
        for event in line_counter.events[-len(tracks):]:
            dwell_tracker.on_line_event(event["track_id"], event["line"], event["direction"], event["t"])

        queue_monitor.update(tracks, timestamp_s)
        occupancy_monitor.update(tracks, timestamp_s)
        area_profiler.update(tracks, dt_s)
        sales_monitor.update(tracks, timestamp_s)
        heatmap.update(tracks)

        line_summary = line_counter.summary()
        frame = annotate(frame, tracks, cfg, detections_by_id, line_summary)
        writer.write(frame)
        last_frame_for_heatmap = frame

        if show:
            cv2.imshow("FootfallCam-CV", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    if last_frame_for_heatmap is not None:
        heatmap.save(os.path.join(cfg.output_dir, "heatmap.png"), last_frame_for_heatmap)

    line_summary = line_counter.summary()
    report = build_report(
        line_summary=line_summary,
        queue_summary=queue_monitor.summary(timestamp_s if frame_idx else 0.0),
        occupancy_summary=occupancy_monitor.update(list(tracker.tracks.values()), timestamp_s if frame_idx else 0.0)
        or occupancy_monitor.check_net_occupancy(line_summary["net_inside"], timestamp_s if frame_idx else 0.0),
        dwell_summary=dwell_tracker.summary(),
        sales_summary=sales_monitor.summary(),
        area_summary=area_profiler.summary(),
        turn_in_rate=turn_in_rate_calc.compute(line_summary["per_line"]),
        elapsed_seconds=time.time() - start_time,
    )

    with open(os.path.join(cfg.output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="FootfallCam-style people counting with YOLO")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--show", action="store_true", help="Show a live preview window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (debug)")
    args = parser.parse_args()

    report = run(args.config, show=args.show, max_frames=args.max_frames)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
