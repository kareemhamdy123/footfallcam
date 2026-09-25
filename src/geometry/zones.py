"""Point-in-polygon and finite-segment line crossing.

The crossing test is deliberately two-stage:

1. A sign change in the cross product shows the movement segment straddles the
   line's *infinite* extension.
2. A second sign test on the line's own endpoints proves the intersection
   actually falls *between* those endpoints.

Stage 2 is what rejects the classic infinite-line bug (R8): without it, someone
walking past the far end of a short gate segment would still be counted.
"""
from __future__ import annotations

from typing import Protocol, Sequence

Point = tuple[float, float]


class HasEndpoints(Protocol):
    """Structural type for `config.Line`, so this layer needs no config import."""

    p1: Point
    p2: Point
    in_direction: str


class HasPolygon(Protocol):
    """Structural type for `config.Zone`."""

    polygon: Sequence[Point]


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting point-in-polygon test. Boundary handling is unspecified."""
    x, y = point
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if min(y1, y2) < y <= max(y1, y2):
            x_inters = x1 if y1 == y2 else (y - y1) * (x2 - x1) / (y2 - y1) + x1
            if x <= x_inters:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def _side(p: Point, a: Point, b: Point) -> float:
    """Signed cross product: which side of the directed line a->b point p is on."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossed_line(prev_point: Point, curr_point: Point, line: HasEndpoints) -> str | None:
    """Return "in", "out" or None for the movement prev_point -> curr_point.

    None means "no crossing", and covers three distinct cases: the movement
    never reached the line, it ran parallel to it, or it crossed the infinite
    extension somewhere beyond the segment's endpoints.
    """
    a, b = line.p1, line.p2

    s_prev = _side(prev_point, a, b)
    s_curr = _side(curr_point, a, b)
    if s_prev == 0.0 or s_curr == 0.0 or (s_prev > 0) == (s_curr > 0):
        return None  # never changed sides of the infinite line

    # The line's endpoints must straddle the movement segment, i.e. the
    # intersection has to lie within the finite span of a->b (R8).
    s_a = _side(a, prev_point, curr_point)
    s_b = _side(b, prev_point, curr_point)
    if s_a != 0.0 and s_b != 0.0 and (s_a > 0) == (s_b > 0):
        return None  # crossed the infinite extension, outside the segment

    dx = curr_point[0] - prev_point[0]
    dy = curr_point[1] - prev_point[1]
    is_in = {
        "left_to_right": dx > 0,
        "right_to_left": dx < 0,
        "top_to_bottom": dy > 0,
        "bottom_to_top": dy < 0,
    }.get(line.in_direction, dy > 0)
    return "in" if is_in else "out"


def zones_containing(point: Point, zones: Sequence[HasPolygon]) -> list:
    """Every zone whose polygon contains `point`, in configuration order."""
    return [zone for zone in zones if point_in_polygon(point, zone.polygon)]
