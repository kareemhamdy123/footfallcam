"""The Track value object - one person's identity through time.

Attribute groups:
    identity/lifecycle  track_id, age, hits, missed, is_confirmed, first/last frame
    geometry            detection, EMA-smoothed box, velocity, foot-point history
    counting state      counted_lines (per-line idempotency, owned by features)
    classification      gender, category, embedding, is_staff, customer_id,
                        is_returning

The classification fields are declared here and unused by Phase 0, so the
tracking layer stays the single owner of track state and features only ever
read or write their own documented attributes.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..config import TrackingSettings
from ..detection.dataclasses import Detection

Box = tuple[float, float, float, float]


@dataclasses.dataclass
class Track:
    track_id: int
    detection: Detection
    age: int = 0                        # frames since creation
    missed: int = 0                     # consecutive frames without a match
    hits: int = 0                       # total matched frames
    is_confirmed: bool = False          # hits >= min_hits
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    history: list[tuple] = dataclasses.field(default_factory=list)
    velocity: tuple[float, float] = (0.0, 0.0)  # (vx, vy) in px/frame
    counted_lines: set = dataclasses.field(default_factory=set)

    # --- classification / identity attributes, owned by later features ---
    is_staff: bool = False
    gender: str | None = None
    category: str = "adult"             # adult | child | cart_or_stroller
    embedding: np.ndarray | None = None
    customer_id: int | None = None
    is_returning: bool = False

    # ------------------------------------------------------------------ state

    @property
    def box(self) -> Box:
        d = self.detection
        return (d.x1, d.y1, d.x2, d.y2)

    @property
    def foot_point(self) -> tuple[float, float]:
        return self.detection.foot_point

    @property
    def is_active(self) -> bool:
        return self.missed == 0

    def predict_next_box(self) -> Detection:
        """Where this track is expected next frame, from its velocity."""
        vx, vy = self.velocity
        d = self.detection
        return Detection(
            x1=d.x1 + vx,
            y1=d.y1 + vy,
            x2=d.x2 + vx,
            y2=d.y2 + vy,
            confidence=d.confidence * 0.9,
            class_id=d.class_id,
        )

    def update(
        self, detection: Detection, frame_idx: int, cfg: TrackingSettings
    ) -> None:
        """Absorb a matched detection: smooth the box, update velocity, age."""
        old_cx, old_cy = self.detection.centroid
        new_cx, new_cy = detection.centroid
        v_alpha = cfg.velocity_ema_alpha
        self.velocity = (
            v_alpha * (new_cx - old_cx) + (1.0 - v_alpha) * self.velocity[0],
            v_alpha * (new_cy - old_cy) + (1.0 - v_alpha) * self.velocity[1],
        )

        # EMA on the box removes single-frame jitter without adding lag.
        a = cfg.box_ema_alpha
        self.detection = Detection(
            x1=a * detection.x1 + (1.0 - a) * self.detection.x1,
            y1=a * detection.y1 + (1.0 - a) * self.detection.y1,
            x2=a * detection.x2 + (1.0 - a) * self.detection.x2,
            y2=a * detection.y2 + (1.0 - a) * self.detection.y2,
            confidence=detection.confidence,
            class_id=detection.class_id,
        )

        self.missed = 0
        self.hits += 1
        self.age += 1
        self.is_confirmed = self.hits >= cfg.min_hits
        self.last_seen_frame = frame_idx
        self.push_history(frame_idx, self.detection.foot_point, cfg.max_history)

    def push_history(
        self, frame_idx: int, point: tuple[float, float], max_history: int
    ) -> None:
        """Append a foot point to the trail, trimming to `max_history`."""
        self.history.append((frame_idx, point[0], point[1]))
        if len(self.history) > max_history:
            del self.history[: len(self.history) - max_history]
