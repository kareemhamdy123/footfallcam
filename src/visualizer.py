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

PANEL_COLOR = (14, 18, 24)
BORDER_COLOR = (50, 60, 75)
TEXT_COLOR = (255, 255, 255)
MUTED_TEXT = (150, 162, 178)

FONT = cv2.FONT_HERSHEY_SIMPLEX
BOX_THICKNESS = 2
TRAIL_THICKNESS = 1
TRAIL_LENGTH = 20
ZONE_TINT = 0.08
QUEUE_TINT = 0.12
HUD_BLEND = 0.88
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
    line_counts: dict | None = None,
    zone_counts: dict | None = None,
    track_zones: dict[int, str] | None = None,
    fps: float = 13.0,
    footer_telemetry: str | None = None,
) -> np.ndarray:
    """Draw every enabled overlay onto a copy of `frame`.

    All arguments are optional, so the same entry point serves a plain
    boxes-and-ids frame and a full proof frame.
    """
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    if zones:
        _draw_zones(annotated, zones, width, height, zone_counts=zone_counts)
    if queue_polygons:
        _draw_auto_queues(annotated, queue_polygons)
    if lines:
        _draw_lines(annotated, lines, line_counts=line_counts)

    for track in tracks:
        _draw_track(annotated, track, detections_by_id, track_zones=track_zones, fps=fps)

    if kpis is not None:
        draw_hud(annotated, kpis)
    if audit_badge:
        draw_audit_badge(annotated, audit_badge, footer_telemetry=footer_telemetry)

    return annotated


# ------------------------------------------------------------------- zones


def _draw_zones(
    frame: np.ndarray,
    zones: list,
    width: int,
    height: int,
    zone_counts: dict | None = None,
) -> None:
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
        pts_arr = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [pts_arr], True, colour, 2)

        # Compact zone label anchored neatly away from traffic
        cur_c = zone_counts.get(zone.name, 0) if zone_counts else 0
        name_clean = zone.name.replace("_", " ").upper()
        label = f"{name_clean} [{cur_c}]" if zone_counts else zone.name

        min_y = min(p[1] for p in points)
        top_candidates = [p for p in points if p[1] <= min_y + 15]
        if top_candidates and top_candidates[0][0] > width // 2:
            top_pt = max(top_candidates, key=lambda p: p[0])
            bx = max(10, int(top_pt[0]) - 90)
        else:
            top_pt = min(top_candidates, key=lambda p: p[0]) if top_candidates else points[0]
            bx = int(top_pt[0]) + 4
        by = max(38, int(top_pt[1]) - 4)
        _draw_label(frame, (bx, by), label, colour)


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


def _draw_lines(
    frame: np.ndarray,
    lines: list,
    line_counts: dict | None = None,
) -> None:
    height, width = frame.shape[:2]

    for line in lines:
        resolved = line.scaled(width, height) if hasattr(line, "scaled") else line
        p1 = (int(resolved.p1[0]), int(resolved.p1[1]))
        p2 = (int(resolved.p2[0]), int(resolved.p2[1]))

        # Crisp clean counting line with terminal pips
        cv2.line(frame, p1, p2, LINE_COLOR, 2, cv2.LINE_AA)
        cv2.circle(frame, p1, 3, LINE_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, p2, 3, LINE_COLOR, -1, cv2.LINE_AA)

        # Compact gate badge positioned neatly near right post to stay out of traffic
        counts = line_counts.get(resolved.name, {}) if line_counts else {}
        in_c = counts.get("in", 0)
        out_c = counts.get("out", 0)
        tally_str = f"  IN: {in_c} | OUT: {out_c}" if (in_c or out_c) else ""
        badge_text = f"{resolved.name.upper()}{tally_str}"
        bx = max(p1[0], p2[0]) - 150
        by = min(p1[1], p2[1]) - 10
        _draw_label(frame, (bx, by), badge_text, LINE_COLOR)


# ------------------------------------------------------------------- people


def role_color(track) -> tuple:
    """Box colour by role: staff, repeat visitor, or ordinary customer."""
    if getattr(track, "is_staff", False):
        return STAFF_COLOR
    if getattr(track, "is_returning", False):
        return RETURNING_COLOR
    return PERSON_COLOR


def _draw_track(
    frame: np.ndarray,
    track,
    detections_by_id: dict | None,
    track_zones: dict[int, str] | None = None,
    fps: float = 13.0,
) -> None:
    box = _box_for(track, detections_by_id)
    if box is None:
        return
    x1, y1, x2, y2 = box
    colour = role_color(track)
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) // 2

    # 1. Motion Trail on Ground (fading with time)
    _draw_trail(frame, track, colour)

    # 2. Foot anchor pip
    cv2.circle(frame, (cx, y2), 2, colour, -1, cv2.LINE_AA)

    # 3. Clean Bounding Box with Corner Accents (BoxCornerAnnotator style)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1, cv2.LINE_AA)
    clen = max(5, min(10, bw // 4))
    cv2.line(frame, (x1, y1), (x1 + clen, y1), colour, BOX_THICKNESS, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x1 + clen, y1), colour, BOX_THICKNESS, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2 - clen, y1), colour, BOX_THICKNESS, cv2.LINE_AA)
    cv2.line(frame, (x2, y1), (x2, y1 + clen), colour, BOX_THICKNESS, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1 + clen, y2), colour, BOX_THICKNESS, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x1, y2 - clen), colour, BOX_THICKNESS, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2 - clen, y2), colour, BOX_THICKNESS, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x2, y2 - clen), colour, BOX_THICKNESS, cv2.LINE_AA)

    # 4. Compact Label Pill Pinned to Top of Box with Smart Telemetry
    tid = getattr(track, "track_id", None)
    z_name = track_zones.get(tid) if (track_zones and tid is not None) else None
    tag_text = _track_tag(track, zone_name=z_name, fps=fps)
    _draw_track_pill(frame, (x1, y1, x2, y2), tag_text, colour)


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


def _track_tag(track, zone_name: str | None = None, fps: float = 13.0) -> str:
    parts = [f"#{getattr(track, 'track_id', '?')}"]
    if getattr(track, "is_staff", False):
        parts.append("STAFF")
    elif getattr(track, "is_returning", False):
        customer_id = getattr(track, "customer_id", None)
        parts.append(f"RETURNING #{customer_id}" if customer_id else "RETURNING")
    gender = getattr(track, "gender", None)
    if gender and gender != "unknown":
        parts.append(gender[:1].upper())

    if zone_name:
        short_z = "Till" if "till" in zone_name.lower() else ("Sales" if "sales" in zone_name.lower() else zone_name)
        parts.append(short_z)

    age = getattr(track, "age", 0)
    if age > 0 and fps > 0:
        dwell_s = int(age / fps)
        parts.append(f"{dwell_s}s")

    return " | ".join(parts)


def _draw_trail(frame: np.ndarray, track, colour: tuple) -> None:
    history = getattr(track, "history", None) or []
    points = [(int(x), int(y)) for _, x, y in history[-TRAIL_LENGTH:]]
    for index in range(1, len(points)):
        cv2.line(frame, points[index - 1], points[index], colour, TRAIL_THICKNESS)


def _draw_track_pill(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    text: str,
    colour: tuple,
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    font_scale = 0.33
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_scale, 1)

    pw = text_w + 8
    ph = text_h + 6

    hud_h = max(30, int(height * 0.07))
    if y1 - ph - 2 > hud_h + 2:
        px = max(2, min(width - pw - 2, x1))
        py = y1 - ph - 2
    else:
        px = max(2, min(width - pw - 2, x1))
        py = y1 + 2

    cv2.rectangle(frame, (px, py), (px + pw, py + ph), PANEL_COLOR, -1)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), colour, 1, cv2.LINE_AA)
    cv2.putText(frame, text, (px + 4, py + text_h + 2), FONT, font_scale, TEXT_COLOR, 1, cv2.LINE_AA)


# --------------------------------------------------------------------- HUD


def draw_hud(frame: np.ndarray, kpis: dict) -> np.ndarray:
    """Draw the telemetry bar across the top of the frame.

    Any KPI the caller has not supplied renders as `n/a` rather than as a
    fabricated zero - an unwired feature must not look like a measured one.

    Draws in place and returns the frame, so it can be chained or ignored.
    """
    height, width = frame.shape[:2]
    bar_height = max(30, int(height * 0.07))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, bar_height), PANEL_COLOR, -1)
    cv2.addWeighted(overlay, HUD_BLEND, frame, 1.0 - HUD_BLEND, 0, frame)
    cv2.line(frame, (0, bar_height), (width, bar_height), (40, 50, 65), 1, cv2.LINE_AA)

    slot_width = width // len(HUD_FIELDS)
    font_scale = max(0.32, min(0.50, width / 1800.0))
    title_y = int(bar_height * 0.38)
    value_y = title_y + int(bar_height * 0.46)

    for index, (title, key, colour) in enumerate(HUD_FIELDS):
        left = index * slot_width + 12
        value = kpis.get(key)
        text = NOT_AVAILABLE if value is None else str(value)
        cv2.putText(frame, title, (left, title_y), FONT,
                    font_scale * 0.85, MUTED_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, text, (left, value_y), FONT,
                    font_scale * 1.05, colour, 1, cv2.LINE_AA)
        if index > 0:
            cv2.line(frame, (index * slot_width, 4), (index * slot_width, bar_height - 4),
                     (35, 44, 56), 1, cv2.LINE_AA)

    return frame


def draw_audit_badge(
    frame: np.ndarray,
    text: str,
    footer_telemetry: str | None = None,
) -> np.ndarray:
    """Watermark the bottom area with audit info and live store telemetry.

    Draws in place and returns the frame.
    """
    height, width = frame.shape[:2]
    font_scale = max(0.28, min(0.35, width / 1800.0))
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_scale, 1)

    if footer_telemetry:
        bar_h = max(22, int(height * 0.06))
        bar_y = height - bar_h
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (width, height), PANEL_COLOR, -1)
        cv2.addWeighted(overlay, HUD_BLEND, frame, 1.0 - HUD_BLEND, 0, frame)
        cv2.line(frame, (0, bar_y), (width, bar_y), BORDER_COLOR, 1, cv2.LINE_AA)

        text_y = bar_y + int(bar_h * 0.65)
        cv2.putText(frame, text, (10, text_y), FONT, font_scale, MUTED_TEXT, 1, cv2.LINE_AA)

        (mw, mh), _ = cv2.getTextSize(footer_telemetry, FONT, font_scale, 1)
        right_x = max(10, width - mw - 12)
        cv2.putText(frame, footer_telemetry, (right_x, text_y), FONT, font_scale, (220, 225, 230), 1, cv2.LINE_AA)
    else:
        left, bottom = 10, height - 8
        cv2.rectangle(
            frame,
            (left - 4, bottom - text_h - 5),
            (left + text_w + 5, bottom + 3),
            PANEL_COLOR,
            -1,
        )
        cv2.rectangle(
            frame,
            (left - 4, bottom - text_h - 5),
            (left + text_w + 5, bottom + 3),
            BORDER_COLOR,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(frame, text, (left + 1, bottom - 1), FONT, font_scale, MUTED_TEXT, 1, cv2.LINE_AA)

    return frame


# ------------------------------------------------------------------ helpers


def _draw_label(
    frame: np.ndarray,
    anchor: tuple[int, int],
    text: str,
    colour: tuple,
    filled: bool = True,
) -> None:
    """A compact, clean pill label with 1px border."""
    height, width = frame.shape[:2]
    font_scale = 0.32
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_scale, 1)

    left = max(0, min(anchor[0], width - text_w - 10))
    bottom = max(text_h + 4, anchor[1])

    pw = text_w + 8
    ph = text_h + 6
    py = max(0, bottom - text_h - 4)

    cv2.rectangle(frame, (left, py), (left + pw, py + ph), PANEL_COLOR, -1)
    cv2.rectangle(frame, (left, py), (left + pw, py + ph), colour, 1, cv2.LINE_AA)
    cv2.putText(frame, text, (left + 4, py + text_h + 2), FONT, font_scale, TEXT_COLOR, 1, cv2.LINE_AA)
