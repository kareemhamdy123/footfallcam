"""Frame annotation: the drawing toolkit.

This module draws. It holds no state, reads no config file and decides nothing
about a run - it is handed what to draw and returns a new frame. The decisions
about *what* is worth showing belong to `features/playback.py` (brochure
characteristic 4), which composes these primitives.

Presentation constants live here and are deliberately not config-driven: they
are not deployment tunables (R3 governs those). Geometry, thresholds and model
paths are configured; colours and stroke widths are not.
"""
from __future__ import annotations

import cv2
import numpy as np

# BGR. Role colours must stay distinguishable at a glance in a proof video.
PERSON_COLOR = (46, 204, 113)      # emerald - ordinary tracked customer
RETURNING_COLOR = (241, 196, 15)   # gold - repeat visitor
STAFF_COLOR = (233, 30, 99)        # fuchsia - staff, excluded from counts
LINE_COLOR = (52, 152, 219)        # blue - counting lines
ZONE_COLORS = {
    "generic": (243, 156, 18),
    "queue": (230, 126, 34),
    "sales": (155, 89, 182),
    "safe_occupancy": (41, 128, 185),
    "outside": (127, 140, 141),
}
QUEUE_AUTO_COLOR = (0, 165, 255)

PANEL_COLOR = (20, 24, 28)
TEXT_COLOR = (255, 255, 255)
MUTED_TEXT = (160, 170, 180)

FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS = 2
TRAIL_THICKNESS = 1
TRAIL_LENGTH = 25
ZONE_TINT = 0.15
QUEUE_TINT = 0.18
HUD_BLEND = 0.78
# cv2.putText cannot render non-ASCII with Hershey fonts, so an unwired KPI
# shows this rather than a dash or an em dash.
NOT_AVAILABLE = "n/a"

HUD_FIELDS = (
    ("IN / OUT", "in_out", PERSON_COLOR),
    ("INSIDE", "inside", LINE_COLOR),
    ("QUEUES", "queues", ZONE_COLORS["queue"]),
    ("RETURN RATE", "return_rate", RETURNING_COLOR),
    ("DEMO (M/F)", "demo", ZONE_COLORS["sales"]),
)


def annotate_frame(
    frame: np.ndarray,
    tracks: list = (),
    detections_by_id: dict | None = None,
    zones: list = (),
    lines: list = (),
    queue_polygons: dict | None = None,
    kpis: dict | None = None,
    audit_badge: str | None = None,
) -> np.ndarray:
    """Draw every enabled overlay onto a copy of `frame`.

    All arguments are optional, so the same entry point serves a plain
    boxes-and-ids frame and a full proof frame.
    """
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    if zones:
        _draw_zones(annotated, zones, width, height)
    if queue_polygons:
        _draw_auto_queues(annotated, queue_polygons)
    if lines:
        _draw_lines(annotated, lines)

    for track in tracks:
        _draw_track(annotated, track, detections_by_id)

    if kpis is not None:
        draw_hud(annotated, kpis)
    if audit_badge:
        draw_audit_badge(annotated, audit_badge)

    return annotated


# ------------------------------------------------------------------- zones


def _draw_zones(frame: np.ndarray, zones: list, width: int, height: int) -> None:
    overlay = frame.copy()
    resolved = []
    for zone in zones:
        colour = ZONE_COLORS.get(getattr(zone, "kind", "generic"), ZONE_COLORS["generic"])
        points = _polygon(zone, width, height)
        if len(points) >= 3:
            resolved.append((zone, colour, points))
            cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], colour)

    if resolved:
        cv2.addWeighted(overlay, ZONE_TINT, frame, 1.0 - ZONE_TINT, 0, frame)

    for zone, colour, points in resolved:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], True, colour, 2)
        label = f"{zone.name} ({getattr(zone, 'kind', 'generic')})"
        _draw_label(frame, points[0], label, colour)


def _polygon(zone, width: int, height: int) -> list[tuple[int, int]]:
    scaled = zone.scaled(width, height) if hasattr(zone, "scaled") else zone
    return [(int(x), int(y)) for x, y in scaled.polygon]


# ----------------------------------------------------------- dynamic queues


def _draw_auto_queues(frame: np.ndarray, queue_polygons: dict) -> None:
    """Dynamic queue clusters. The queue feature itself is Phase 6."""
    overlay = frame.copy()
    drawn = []
    for name, polygon in queue_polygons.items():
        points = [(int(x), int(y)) for x, y in polygon]
        if len(points) >= 3:
            drawn.append((name, points))
            cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], QUEUE_AUTO_COLOR)

    if drawn:
        cv2.addWeighted(overlay, QUEUE_TINT, frame, 1.0 - QUEUE_TINT, 0, frame)

    for name, points in drawn:
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], True, QUEUE_AUTO_COLOR, 2)
        _draw_label(frame, points[0], f"ACTIVE QUEUE ({name})", QUEUE_AUTO_COLOR)


# ------------------------------------------------------------ counting lines


def _draw_lines(frame: np.ndarray, lines: list) -> None:
    height, width = frame.shape[:2]
    for line in lines:
        resolved = line.scaled(width, height) if hasattr(line, "scaled") else line
        p1 = (int(resolved.p1[0]), int(resolved.p1[1]))
        p2 = (int(resolved.p2[0]), int(resolved.p2[1]))
        cv2.line(frame, p1, p2, LINE_COLOR, 2)
        cv2.putText(frame, resolved.name, p1, FONT, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)


# ------------------------------------------------------------------- people


def role_color(track) -> tuple:
    """Box colour by role: staff, repeat visitor, or ordinary customer."""
    if getattr(track, "is_staff", False):
        return STAFF_COLOR
    if getattr(track, "is_returning", False):
        return RETURNING_COLOR
    return PERSON_COLOR


def _draw_track(frame: np.ndarray, track, detections_by_id: dict | None) -> None:
    box = _box_for(track, detections_by_id)
    if box is None:
        return
    x1, y1, x2, y2 = box
    colour = role_color(track)

    _draw_trail(frame, track, colour)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, BOX_THICKNESS)

    cx, cy = (x1 + x2) // 2, y2
    cv2.circle(frame, (cx, cy), 3, colour, -1)  # the foot point features measure

    _draw_label(frame, (x1, y1), _track_tag(track), colour, filled=False)


def _box_for(track, detections_by_id: dict | None):
    """Prefer the raw detection: a proof video should show what the model saw."""
    if detections_by_id:
        detection = detections_by_id.get(getattr(track, "track_id", None))
        if detection is not None:
            return (
                int(detection.x1), int(detection.y1),
                int(detection.x2), int(detection.y2),
            )
    box = getattr(track, "box", None)
    if box is None:
        return None
    return tuple(int(v) for v in box)


def _track_tag(track) -> str:
    parts = [f"#{getattr(track, 'track_id', '?')}"]
    if getattr(track, "is_staff", False):
        parts.append("STAFF")
    elif getattr(track, "is_returning", False):
        customer_id = getattr(track, "customer_id", None)
        parts.append(f"RETURNING #{customer_id}" if customer_id else "RETURNING")
    gender = getattr(track, "gender", None)
    if gender and gender != "unknown":
        parts.append(gender[:1].upper())
    return " | ".join(parts)


def _draw_trail(frame: np.ndarray, track, colour: tuple) -> None:
    history = getattr(track, "history", None) or []
    points = [(int(x), int(y)) for _, x, y in history[-TRAIL_LENGTH:]]
    for index in range(1, len(points)):
        cv2.line(frame, points[index - 1], points[index], colour, TRAIL_THICKNESS)


# --------------------------------------------------------------------- HUD


def draw_hud(frame: np.ndarray, kpis: dict) -> np.ndarray:
    """Draw the telemetry bar across the top of the frame.

    Any KPI the caller has not supplied renders as `n/a` rather than as a
    fabricated zero - an unwired feature must not look like a measured one.

    Draws in place and returns the frame, so it can be chained or ignored.
    """
    height, width = frame.shape[:2]
    bar_height = max(42, int(height * 0.07))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_height), PANEL_COLOR, -1)
    cv2.addWeighted(overlay, HUD_BLEND, frame, 1.0 - HUD_BLEND, 0, frame)
    cv2.line(frame, (0, bar_height), (width, bar_height), (60, 68, 76), 1)

    slot_width = width // len(HUD_FIELDS)
    font_scale = max(0.40, min(0.60, width / 1600.0))
    title_y = int(bar_height * 0.40)
    value_y = title_y + int(bar_height * 0.45)

    for index, (title, key, colour) in enumerate(HUD_FIELDS):
        left = index * slot_width + 10
        value = kpis.get(key)
        text = NOT_AVAILABLE if value is None else str(value)
        cv2.putText(frame, title, (left, title_y), FONT,
                    font_scale * 0.8, MUTED_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, text, (left, value_y), FONT,
                    font_scale * 1.05, colour, 2, cv2.LINE_AA)

    return frame


def draw_audit_badge(frame: np.ndarray, text: str) -> np.ndarray:
    """Watermark the bottom-left corner so a proof frame is self-identifying.

    Draws in place and returns the frame.
    """
    height, width = frame.shape[:2]
    font_scale = max(0.40, min(0.55, width / 1600.0))
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_scale, 1)
    left, bottom = 10, height - 10

    cv2.rectangle(
        frame,
        (left - 4, bottom - text_h - 6),
        (left + text_w + 6, bottom + 4),
        PANEL_COLOR,
        -1,
    )
    cv2.putText(frame, text, (left, bottom), FONT, font_scale, MUTED_TEXT, 1, cv2.LINE_AA)
    return frame


# ------------------------------------------------------------------ helpers


def _draw_label(
    frame: np.ndarray,
    anchor: tuple[int, int],
    text: str,
    colour: tuple,
    filled: bool = True,
) -> None:
    """A dark pill with the text, kept inside the frame at the top edge."""
    height, width = frame.shape[:2]
    font_scale = 0.42
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_scale, 1)

    left = max(0, min(anchor[0], width - text_w - 10))
    bottom = max(text_h + 6, anchor[1])

    cv2.rectangle(
        frame,
        (left, bottom - text_h - 6),
        (left + text_w + 8, bottom + 4),
        PANEL_COLOR,
        -1,
    )
    if not filled:
        cv2.rectangle(
            frame,
            (left, bottom - text_h - 6),
            (left + text_w + 8, bottom + 4),
            colour,
            1,
        )
    cv2.putText(frame, text, (left + 4, bottom), FONT, font_scale, TEXT_COLOR, 1, cv2.LINE_AA)
