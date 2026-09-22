"""
Maps to brochure characteristics:
  10. Sales conversation        -> time a customer track spends inside a
                                    'sales' zone next to a staff track, used
                                    as a proxy for staff-customer interaction
  20. Visitor in & out dwell time -> total time between a track's first
                                      'in' crossing and its 'out' crossing
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Zone
from ..tracker import Track
from ..zones import point_in_polygon


@dataclass
class VisitRecord:
    track_id: int
    entered_at: float
    exited_at: float | None = None

    @property
    def dwell_seconds(self) -> float | None:
        if self.exited_at is None:
            return None
        return round(self.exited_at - self.entered_at, 1)


class DwellTracker:
    def __init__(self, entrance_line_name: str, alert_seconds: float = 600.0):
        self.entrance_line_name = entrance_line_name
        self.alert_seconds = alert_seconds
        self.visits: dict[int, VisitRecord] = {}

    def on_line_event(self, track_id: int, line_name: str, direction: str, t: float) -> None:
        if line_name != self.entrance_line_name:
            return
        if direction == "in":
            self.visits[track_id] = VisitRecord(track_id=track_id, entered_at=t)
        elif direction == "out" and track_id in self.visits:
            self.visits[track_id].exited_at = t

    def long_dwell_alerts(self, now_s: float) -> list[int]:
        return [
            tid for tid, v in self.visits.items()
            if v.exited_at is None and (now_s - v.entered_at) > self.alert_seconds
        ]

    def summary(self) -> dict:
        completed = [v.dwell_seconds for v in self.visits.values() if v.exited_at is not None]
        return {
            "visits_recorded": len(self.visits),
            "completed_visits": len(completed),
            "average_dwell_seconds": round(sum(completed) / len(completed), 1) if completed else 0.0,
            "max_dwell_seconds": round(max(completed), 1) if completed else 0.0,
        }


class SalesConversationMonitor:
    """
    Flags a 'conversation' whenever a non-staff track and a staff track are
    both inside the same 'sales' zone at the same time for longer than
    `min_seconds`. This is a spatial/temporal proxy - it does not use audio
    or true interaction detection.
    """

    def __init__(self, zones: list[Zone], min_seconds: float = 5.0):
        self.zones = [z for z in zones if z.kind == "sales"]
        self.min_seconds = min_seconds
        self._active_since: dict[tuple, float] = {}
        self.completed: list[dict] = []

    def update(self, tracks: list[Track], timestamp_s: float) -> None:
        for zone in self.zones:
            staff_in_zone = [t for t in tracks if t.is_staff and t.history
                              and point_in_polygon(t.history[-1][1:], zone.polygon)]
            customers_in_zone = [t for t in tracks if not t.is_staff and t.history
                                  and point_in_polygon(t.history[-1][1:], zone.polygon)]
            active_pairs = {
                (zone.name, s.track_id, c.track_id)
                for s in staff_in_zone for c in customers_in_zone
            }
            for key in active_pairs:
                self._active_since.setdefault(key, timestamp_s)
            for key in list(self._active_since):
                if key[0] != zone.name:
                    continue
                if key not in active_pairs:
                    started = self._active_since.pop(key)
                    duration = timestamp_s - started
                    if duration >= self.min_seconds:
                        self.completed.append({
                            "zone": key[0], "staff_id": key[1],
                            "customer_id": key[2], "duration_s": round(duration, 1),
                        })

    def summary(self) -> dict:
        return {
            "conversations_detected": len(self.completed),
            "average_duration_s": (
                round(sum(c["duration_s"] for c in self.completed) / len(self.completed), 1)
                if self.completed else 0.0
            ),
        }
