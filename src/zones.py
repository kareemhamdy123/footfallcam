"""
Geometry helpers for counting lines and polygon zones.

Maps to brochure characteristics:
  8.  Outside traffic         -> zone kind="outside"
  10. Sales conversation      -> zone kind="sales"
  13. Multiple counting line  -> `lines` is a list, each independently counted
  15. Safe occupancy          -> zone kind="safe_occupancy"
  17. Zone counting           -> zone kind="generic"
  3.  Passenger Queue         -> zone kind="queue"
"""
from __future__ import annotations

from .config import Line, Zone


def point_in_polygon(point: tuple, polygon: list) -> bool:
    """Standard ray-casting point-in-polygon test."""
    x, y = point
    n = len(polygon)
    inside = False
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if y > min(y1, y2):
            if y <= max(y1, y2):
                if x <= max(x1, x2):
                    if y1 != y2:
                        xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
                    else:
                        xinters = x1
                    if x1 == x2 or x <= xinters:
                        inside = not inside
        x1, y1 = x2, y2
    return inside


def _side(p, a, b) -> float:
    """Signed area to know which side of line a-b point p is on."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossed_line(prev_point: tuple, curr_point: tuple, line: Line) -> str | None:
    """
    Returns "in", "out", or None depending on whether the segment
    prev_point -> curr_point crosses the configured line, and in which
    direction relative to `line.in_direction`.

    Crossing itself is detected via a sign change of the cross-product
    "side" test (robust regardless of how p1/p2 happen to be ordered).
    The in/out *direction* is then read directly off the movement vector's
    dominant-axis sign, which is independent of the line's own point order
    and therefore doesn't need any sign-convention bookkeeping.
    """
    a, b = line.p1, line.p2
    s_prev = _side(prev_point, a, b)
    s_curr = _side(curr_point, a, b)
    if s_prev == 0 or s_curr == 0 or (s_prev > 0) == (s_curr > 0):
        return None  # no crossing

    dx = curr_point[0] - prev_point[0]
    dy = curr_point[1] - prev_point[1]

    direction_checks = {
        "left_to_right": dx > 0,
        "right_to_left": dx < 0,
        "top_to_bottom": dy > 0,
        "bottom_to_top": dy < 0,
    }
    is_in = direction_checks.get(line.in_direction, dy > 0)
    return "in" if is_in else "out"


def zones_containing(point: tuple, zones: list[Zone]) -> list[Zone]:
    return [z for z in zones if point_in_polygon(point, z.polygon)]
