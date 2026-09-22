"""
Maps to brochure characteristics:
  2.  Dynamic Queue Counting -> live count of people inside a queue zone
  3.  Passenger Queue        -> per-person wait-time while inside the queue zone
  16. Queue prediction       -> short-horizon linear trend on queue length
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Zone
from ..tracker import Track
from ..zones import point_in_polygon


@dataclass
class QueueState:
    zone_name: str
    history: list = field(default_factory=list)   # (timestamp_s, count)
    wait_start: dict = field(default_factory=dict)  # track_id -> t_entered


class QueueMonitor:
    def __init__(self, zones: list[Zone], max_history: int = 600):
        self.queue_zones = [z for z in zones if z.kind == "queue"]
        self.states = {z.name: QueueState(zone_name=z.name) for z in self.queue_zones}
        self.max_history = max_history

    def update(self, tracks: list[Track], timestamp_s: float) -> None:
        for zone in self.queue_zones:
            state = self.states[zone.name]
            present_ids = set()
            for track in tracks:
                if not track.history or track.is_staff:
                    continue
                point = track.history[-1][1:]
                if point_in_polygon(point, zone.polygon):
                    present_ids.add(track.track_id)
                    state.wait_start.setdefault(track.track_id, timestamp_s)
            # drop wait-timers for people no longer in the zone
            for tid in list(state.wait_start):
                if tid not in present_ids:
                    del state.wait_start[tid]

            state.history.append((timestamp_s, len(present_ids)))
            if len(state.history) > self.max_history:
                state.history.pop(0)

    def current_wait_times(self, zone_name: str, timestamp_s: float) -> dict:
        state = self.states[zone_name]
        return {tid: round(timestamp_s - t0, 1) for tid, t0 in state.wait_start.items()}

    def predict_next(self, zone_name: str, horizon_s: float = 60.0) -> float:
        """
        Very lightweight linear-regression trend over the recent history,
        extrapolated `horizon_s` seconds ahead. Good enough for a
        "queue is growing / shrinking" style alert; swap for a proper
        time-series model if you need precise forecasts.
        """
        state = self.states[zone_name]
        recent = state.history[-30:]
        if len(recent) < 2:
            return recent[-1][1] if recent else 0.0
        ts = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        n = len(ts)
        t_mean = sum(ts) / n
        y_mean = sum(ys) / n
        num = sum((t - t_mean) * (y - y_mean) for t, y in zip(ts, ys))
        den = sum((t - t_mean) ** 2 for t in ts) or 1e-9
        slope = num / den
        intercept = y_mean - slope * t_mean
        predicted = slope * (ts[-1] + horizon_s) + intercept
        return max(0.0, round(predicted, 1))

    def summary(self, timestamp_s: float) -> dict:
        out = {}
        for zone in self.queue_zones:
            state = self.states[zone.name]
            current = state.history[-1][1] if state.history else 0
            out[zone.name] = {
                "current_length": current,
                "wait_times_s": self.current_wait_times(zone.name, timestamp_s),
                "predicted_length_60s": self.predict_next(zone.name, 60.0),
            }
        return out
