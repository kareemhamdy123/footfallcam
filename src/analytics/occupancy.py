"""
Maps to brochure characteristic:
  15. Safe occupancy -> real-time headcount inside a zone (or the whole
                         store, via net_inside from the LineCounter) checked
                         against a configurable capacity limit.
"""
from __future__ import annotations

from ..config import Zone
from ..tracker import Track
from ..zones import point_in_polygon


class OccupancyMonitor:
    def __init__(self, zones: list[Zone], limit: int):
        self.zones = [z for z in zones if z.kind == "safe_occupancy"]
        self.limit = limit
        self.alert_log: list[dict] = []

    def update(self, tracks: list[Track], timestamp_s: float) -> dict:
        result = {}
        for zone in self.zones:
            count = sum(
                1 for t in tracks
                if t.history and not t.is_staff
                and point_in_polygon(t.history[-1][1:], zone.polygon)
            )
            over_limit = count > self.limit
            if over_limit:
                self.alert_log.append({"t": timestamp_s, "zone": zone.name, "count": count})
            result[zone.name] = {
                "count": count,
                "limit": self.limit,
                "over_limit": over_limit,
                "utilisation_pct": round(100 * count / self.limit, 1) if self.limit else 0.0,
            }
        return result

    def check_net_occupancy(self, net_inside: int, timestamp_s: float) -> dict:
        """Fallback: whole-store occupancy from entrance/exit line counts,
        useful when no explicit safe_occupancy polygon is configured."""
        over_limit = net_inside > self.limit
        if over_limit:
            self.alert_log.append({"t": timestamp_s, "zone": "__store__", "count": net_inside})
        return {
            "count": net_inside,
            "limit": self.limit,
            "over_limit": over_limit,
            "utilisation_pct": round(100 * net_inside / self.limit, 1) if self.limit else 0.0,
        }
