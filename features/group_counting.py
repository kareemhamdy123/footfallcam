"""Group co-movement tracking and entry clustering.

Phase 2 provides only what `VideoCounter` needs in order to report group
statistics without duplicating the logic itself (R7). Brochure characteristic 14
("Group counting") is completed in Phase 12 - purchasing-unit reporting, group
dwell and the full statistics surface land there.

A "group" here is people who moved together: pairs accumulate co-movement time
while they stay within `group_max_distance_px`, and entry events that fall
inside the same time window are clustered into one purchasing unit.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only (R1)
    from src.tracking.dataclasses import Track

Point = tuple[float, float]


class GroupCounter:
    def __init__(
        self,
        group_max_distance_px: float = 90.0,
        entry_window_s: float = 3.0,
    ):
        self.group_max_distance_px = group_max_distance_px
        self.entry_window_s = entry_window_s
        # Unordered pair of track ids -> seconds spent moving together.
        self.co_movement_seconds: dict[tuple[int, int], float] = {}
        self._last_position: dict[int, Point] = {}

    def update_co_movement(
        self, tracks: Sequence["Track"], timestamp_s: float, dt_s: float = 0.0
    ) -> None:
        """Accumulate co-movement time for every close enough pair of tracks.

        `dt_s` is the frame interval; when omitted the first frame of a pair
        contributes nothing and later frames use the gap since the last update.
        """
        positions = {t.track_id: t.foot_point for t in tracks}
        for a_id, a_pos in positions.items():
            for b_id, b_pos in positions.items():
                if a_id >= b_id:
                    continue  # each unordered pair once
                if _distance(a_pos, b_pos) > self.group_max_distance_px:
                    continue
                key = (a_id, b_id)
                self.co_movement_seconds[key] = (
                    self.co_movement_seconds.get(key, 0.0) + dt_s
                )

        self._last_position = positions

    def frame_delta(self, timestamp_s: float) -> float:
        """Seconds since the last `update_co_movement`, for the next call."""
        previous = getattr(self, "_last_timestamp_s", None)
        self._last_timestamp_s = timestamp_s
        return 0.0 if previous is None else max(0.0, timestamp_s - previous)

    def cluster_groups(
        self, in_events: Iterable[dict], window_s: float | None = None
    ) -> list[list[dict]]:
        """Cluster entry events into purchasing units.

        Two entries join the same group when they are within `window_s` of each
        other in time. Clusters are transitive: a chain of close-in-time entries
        forms one group, which is what a family walking in together looks like.
        """
        window = self.entry_window_s if window_s is None else window_s
        events = sorted(in_events, key=lambda e: e.get("t", 0.0))
        groups: list[list[dict]] = []
        current: list[dict] = []

        for event in events:
            if not current:
                current = [event]
                continue
            latest = max(e.get("t", 0.0) for e in current)
            if event.get("t", 0.0) - latest <= window:
                current.append(event)
            else:
                groups.append(current)
                current = [event]
        if current:
            groups.append(current)
        return groups

    def summary(self, in_events: Iterable[dict] = ()) -> dict:
        events = list(in_events)
        groups = self.cluster_groups(events)
        multi = [g for g in groups if len(g) >= 2]
        solo = [g for g in groups if len(g) == 1]
        average = (
            sum(len(g) for g in groups) / len(groups) if groups else 0.0
        )
        return {
            "groups": len(groups),
            "multi_person_groups": len(multi),
            "solo_visitors": len(solo),
            "average_group_size": average,
            "co_movement_pairs": len(self.co_movement_seconds),
        }

    def reset(self) -> None:
        self.co_movement_seconds.clear()
        self._last_position.clear()
        self._last_timestamp_s = None


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
