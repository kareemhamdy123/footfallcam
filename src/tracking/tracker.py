"""SimpleTracker - ByteTrack-style two-stage association.

Why two stages: under occlusion the detector still emits a weak box, and
discarding it would fracture the identity. Stage 1 matches confident boxes by
IoU against a velocity-predicted position; stage 2 re-matches the leftovers
with the weak boxes at a looser IoU. A centroid-distance pass then rescues
fast-moving tracks whose IoU has already decayed to zero.

`update()` is the per-frame entry point. `predict_step()` advances tracks on
frames where detection was skipped.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..config import TrackingSettings
from ..detection.dataclasses import Detection
from .dataclasses import Track

WEAK_BOX_IOU_FACTOR = 0.7  # stage-2 IoU is this much looser than stage 1


def iou(a: Detection, b: Detection) -> float:
    """Intersection over union of two boxes."""
    inter_w = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    inter_h = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = inter_w * inter_h
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(
    candidates: list[tuple[float, int, int]], threshold: float
) -> list[tuple[int, int]]:
    """Highest-utility-first one-to-one matching over (utility, track_idx, det_idx).

    `utility` must be oriented so that *higher is better*, and `threshold` is
    the minimum acceptable utility. Callers matching on a distance negate it
    first so this stays a single code path.
    """
    matches: list[tuple[int, int]] = []
    used_tracks: set[int] = set()
    used_dets: set[int] = set()
    for utility, track_idx, det_idx in sorted(candidates, key=lambda c: -c[0]):
        if utility < threshold:
            break
        if track_idx in used_tracks or det_idx in used_dets:
            continue
        used_tracks.add(track_idx)
        used_dets.add(det_idx)
        matches.append((track_idx, det_idx))
    return matches


class SimpleTracker:
    def __init__(self, cfg: TrackingSettings | None = None):
        self.cfg = cfg or TrackingSettings()
        self._ids = itertools.count(1)
        self.tracks: dict[int, Track] = {}

    @property
    def total_opened(self) -> int:
        """How many track IDs this tracker has ever issued."""
        return next(self._ids) - 1

    # ------------------------------------------------------------------- main

    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        cfg = self.cfg
        high = [d for d in detections if d.confidence >= cfg.high_confidence_threshold]
        low = [d for d in detections if d.confidence < cfg.high_confidence_threshold]

        track_ids = list(self.tracks.keys())
        matched: set[int] = set()
        used_high: set[int] = set()

        # Stage 1: confident detections against velocity-predicted positions.
        self._associate(
            track_ids, high, frame_idx, used_high, matched, by="iou",
            threshold=cfg.iou_threshold,
        )

        # Stage 2: weak detections recover tracks lost to occlusion.
        self._associate(
            track_ids, low, frame_idx, set(), matched, by="iou",
            threshold=cfg.iou_threshold * WEAK_BOX_IOU_FACTOR,
        )

        # Stage 3: centroid-distance fallback for fast motion, where IoU has
        # already decayed to zero but the identity is still obvious.
        self._associate(
            track_ids, high, frame_idx, used_high, matched, by="distance",
            threshold=cfg.max_center_distance_px,
        )

        # Age unmatched tracks and retire the ones gone too long.
        for i in range(len(track_ids)):
            if i in matched:
                continue
            track = self.tracks[track_ids[i]]
            track.missed += 1
            track.age += 1
            if track.missed > cfg.max_missed:
                del self.tracks[track_ids[i]]

        # Open new tracks for unmatched *confident* detections only: a lone weak
        # box is usually a partial body, not a person. hits starts at 1 because
        # the creating detection is itself an observation - otherwise min_hits=2
        # would need three frames to confirm.
        for j, det in enumerate(high):
            if j in used_high:
                continue
            new_id = next(self._ids)
            self.tracks[new_id] = Track(
                track_id=new_id,
                detection=det,
                age=1,
                hits=1,
                first_seen_frame=frame_idx,
                last_seen_frame=frame_idx,
                history=[(frame_idx, det.foot_point[0], det.foot_point[1])],
            )

        return self._active_tracks(frame_idx)

    def predict_step(self, frame_idx: int) -> list[Track]:
        """Advance tracks on a frame where detection was skipped."""
        for track in self.tracks.values():
            if track.missed == 0:
                track.detection = track.predict_next_box()
                track.push_history(
                    frame_idx, track.detection.foot_point, self.cfg.max_history
                )
        return [t for t in self.tracks.values() if self.is_visible(t)]

    # --------------------------------------------------------------- internals

    def _associate(
        self,
        track_ids: list[int],
        candidates: list[Detection],
        frame_idx: int,
        used: set[int],
        matched: set[int],
        *,
        by: str,
        threshold: float,
    ) -> None:
        """Match `candidates` to still-unmatched tracks and fold them in.

        `used` holds the indices of candidates already consumed by an earlier
        stage, so no detection is ever assigned to two tracks. Indices are
        added to it as they are consumed.
        """
        open_tracks = [i for i in range(len(track_ids)) if i not in matched]
        free = [j for j in range(len(candidates)) if j not in used]
        if not open_tracks or not free:
            return

        # Orient every metric so higher is better: IoU rises with similarity,
        # so a distance has to be negated.
        negate = by == "distance"
        pairs = [
            (
                -self._centroid_distance(track_ids[i], candidates[j])
                if negate
                else iou(self.tracks[track_ids[i]].predict_next_box(), candidates[j]),
                i,
                j,
            )
            for i in open_tracks
            for j in free
        ]
        for i, j in _greedy_match(pairs, -threshold if negate else threshold):
            self.tracks[track_ids[i]].update(candidates[j], frame_idx, self.cfg)
            matched.add(i)
            used.add(j)

    def _centroid_distance(self, track_id: int, det: Detection) -> float:
        pcx, pcy = self.tracks[track_id].predict_next_box().centroid
        dcx, dcy = det.centroid
        return float(np.hypot(pcx - dcx, pcy - dcy))

    def _active_tracks(self, frame_idx: int) -> list[Track]:
        """Confirmed tracks, with brief occlusions carried on prediction."""
        active: list[Track] = []
        for track in self.tracks.values():
            if track.hits < self.cfg.min_hits:
                continue
            if track.missed == 0:
                active.append(track)
            elif track.missed <= self.cfg.max_occlusion_bridge:
                track.detection = track.predict_next_box()
                track.push_history(
                    frame_idx, track.detection.foot_point, self.cfg.max_history
                )
                active.append(track)
        return active

    def is_visible(self, track: Track) -> bool:
        return track.missed == 0 and track.hits >= self.cfg.min_hits
