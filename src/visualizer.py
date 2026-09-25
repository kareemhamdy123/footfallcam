"""Frame annotation.

Phase 0 scope: person boxes and ID badges only - the minimum that makes
`outputs/annotated.mp4` auditable. Zone tints, counting lines, motion trails
and the HUD telemetry bar are Phase 4 and Phase 5 work, extended in place.

Presentation constants, not deployment tunables (R3 governs the latter).
"""
from __future__ import annotations

import cv2
import numpy as np

from .tracking.dataclasses import Track

BOX_COLOR = (16, 200, 120)      # BGR emerald - customers
STAFF_COLOR = (60, 210, 255)    # BGR gold - staff
BOX_THICKNESS = 2
FOOT_MARKER_RADIUS = 3
BADGE_PAD_X = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICKNESS = 1


def annotate_frame(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    """Draw a box, an ID badge and a foot marker for every track."""
    annotated = frame.copy()

    for track in tracks:
        color = STAFF_COLOR if track.is_staff else BOX_COLOR
        x1, y1, x2, y2 = (int(v) for v in track.box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # The foot marker is the point every spatial feature measures from.
        fx, fy = (int(v) for v in track.foot_point)
        cv2.circle(annotated, (fx, fy), FOOT_MARKER_RADIUS, color, -1)

        _draw_id_badge(annotated, track.track_id, x1, y1, color)

    return annotated


def _draw_id_badge(
    frame: np.ndarray, track_id: int, box_x: int, box_y: int, color: tuple
) -> None:
    """Draw a filled pill with the track id above the box's top-left corner."""
    label = f"#{track_id}"
    (text_w, text_h), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    height, width = frame.shape[:2]

    # Keep the badge inside the frame even for a box at the very top edge.
    left = max(0, min(box_x, width - text_w - 2 * BADGE_PAD_X))
    bottom = max(text_h + 2 * BADGE_PAD_X, box_y)

    cv2.rectangle(
        frame,
        (left, bottom - text_h - 2 * BADGE_PAD_X),
        (left + text_w + 2 * BADGE_PAD_X, bottom),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (left + BADGE_PAD_X, bottom - BADGE_PAD_X),
        FONT,
        FONT_SCALE,
        (0, 0, 0),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )
