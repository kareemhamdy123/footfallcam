"""
A dependency-free SORT-style tracker: greedy IOU matching + centroid fallback,
with a small track-history buffer per ID (used for line-crossing, dwell time,
turn-in direction and playback trails).

This keeps the reference implementation runnable with nothing more than
`ultralytics` + `opencv-python` installed. For production accuracy on crowded
scenes, swap this for ByteTrack/DeepSORT (see README "Extending" section) -
the rest of the pipeline only depends on the `Track` interface below, so the
swap is local to this file.

Maps to brochure characteristics:
  1. Video Counting     -> stable IDs let a person be counted once, not per-frame
  3. Passenger Queue     -> track history lets us see who has been stationary
  10/20. dwell & sales conversation -> need a persistent ID over time
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .detector import Detection


def _iou(a: Detection, b: Detection) -> float:
    xx1, yy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    xx2, yy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter_w, inter_h = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
    inter = inter_w * inter_h
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    detection: Detection
    age: int = 0                 # frames since creation
    missed: int = 0               # consecutive frames without a match
    hits: int = 0                 # total matched frames
    history: list = field(default_factory=list)   # list[(frame_idx, x, y)]
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    counted_lines: set = field(default_factory=set)   # avoid double counting
    is_staff: bool = False

    def update(self, detection: Detection, frame_idx: int) -> None:
        self.detection = detection
        self.missed = 0
        self.hits += 1
        self.last_seen_frame = frame_idx
        self.history.append((frame_idx, *detection.foot_point))
        if len(self.history) > 300:
            self.history.pop(0)


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15,
                 max_center_dist: float = 80.0):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.max_center_dist = max_center_dist
        self._next_id = itertools.count(1)
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        matches: list[tuple[int, int]] = []

        # Greedy matching by IOU first
        pairs = []
        for tid in unmatched_tracks:
            for di in unmatched_dets:
                score = _iou(self.tracks[tid].detection, detections[di])
                if score >= self.iou_threshold:
                    pairs.append((score, tid, di))
        pairs.sort(reverse=True, key=lambda p: p[0])
        used_tracks, used_dets = set(), set()
        for score, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            matches.append((tid, di))
            used_tracks.add(tid)
            used_dets.add(di)

        # Fallback: centroid distance for anything IOU missed (fast motion)
        remaining_tracks = [t for t in unmatched_tracks if t not in used_tracks]
        remaining_dets = [d for d in unmatched_dets if d not in used_dets]
        dist_pairs = []
        for tid in remaining_tracks:
            cx, cy = self.tracks[tid].detection.centroid
            for di in remaining_dets:
                dx, dy = detections[di].centroid
                dist = float(np.hypot(cx - dx, cy - dy))
                if dist <= self.max_center_dist:
                    dist_pairs.append((dist, tid, di))
        dist_pairs.sort(key=lambda p: p[0])
        for dist, tid, di in dist_pairs:
            if tid in used_tracks or di in used_dets:
                continue
            matches.append((tid, di))
            used_tracks.add(tid)
            used_dets.add(di)

        # Apply matches
        for tid, di in matches:
            self.tracks[tid].update(detections[di], frame_idx)
            self.tracks[tid].age += 1

        # New tracks for unmatched detections
        for di in range(len(detections)):
            if di in used_dets:
                continue
            tid = next(self._next_id)
            t = Track(track_id=tid, detection=detections[di],
                       first_seen_frame=frame_idx, last_seen_frame=frame_idx)
            t.history.append((frame_idx, *detections[di].foot_point))
            t.hits = 1
            self.tracks[tid] = t

        # Age out unmatched tracks
        dead = []
        for tid in self.tracks:
            if tid not in used_tracks:
                self.tracks[tid].missed += 1
                self.tracks[tid].age += 1
                if self.tracks[tid].missed > self.max_missed:
                    dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

        return list(self.tracks.values())
