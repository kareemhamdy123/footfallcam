"""Phase 0 counting skeleton.

SCOPE NOTE: deliberately minimal scaffolding, so Phase 0 can emit real
`counting.total_in` / `total_out` / `net_inside` numbers. It implements one
responsibility only - bi-directional crossing of a finite line segment, once
per line per track, excluding staff.

Phase 2 replaces this module with `features/counter.py` (VideoCounter),
`features/multiple_counting_line.py` (MultipleCountingLineManager) and
`features/group_counting.py` (GroupCounter), then deletes this file. Do not grow
it: multi-gate management, auto / scene-flow mode and group clustering belong
to the Phase 2 feature modules (R2, R7).

It sits under `src/` rather than `features/` only because `features/` must stay
empty until Phase 2. `Track` is imported for typing only, which is the pattern
`features/` must follow (R1).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from .geometry.zones import crossed_line

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps the layer decoupled
    from .config import Line
    from .tracking.dataclasses import Track

Point = tuple[float, float]


class LineCounter:
    """Bi-directional crossing tally over one or more finite line segments."""

    def __init__(self, lines: Sequence["Line"], stale_after_frames: int = 30):
        self.lines = list(lines)
        self.stale_after_frames = stale_after_frames
        self.counts: dict[str, dict[str, int]] = {
            line.name: {"in": 0, "out": 0} for line in self.lines
        }
        self.events: list[dict] = []

        # Foot point seen on the previous frame, per track. Held here rather
        # than read from Track.history, so crossing detection stays correct
        # for a track that was bridged across an occlusion.
        self._last_position: dict[int, Point] = {}
        self._missing_frames: dict[int, int] = {}

    def update(
        self,
        tracks: list["Track"],
        frame_idx: int,
        timestamp_s: float,
    ) -> list[dict]:
        """Tally this frame's crossings. Returns only the new events."""
        new_events: list[dict] = []
        present: set[int] = set()

        for track in tracks:
            present.add(track.track_id)
            self._missing_frames.pop(track.track_id, None)
            position = track.foot_point
            previous = self._last_position.get(track.track_id)
            self._last_position[track.track_id] = position

            if previous is None or track.is_staff:
                # No movement to judge yet, or staff never affect the
                # customer footfall tally.
                continue

            new_events += self._count_crossings(track, previous, position, frame_idx, timestamp_s)

        self._forget_absent_tracks(present)
        return new_events

    def _count_crossings(
        self,
        track: "Track",
        previous: Point,
        position: Point,
        frame_idx: int,
        timestamp_s: float,
    ) -> list[dict]:
        events: list[dict] = []
        for line in self.lines:
            if line.name in track.counted_lines:
                continue  # idempotent per line (R7)
            direction = crossed_line(previous, position, line)
            if direction is None:
                continue

            track.counted_lines.add(line.name)
            self.counts[line.name][direction] += 1
            events.append(
                {
                    "frame": frame_idx,
                    "t": timestamp_s,
                    "line": line.name,
                    "direction": direction,
                    "track_id": track.track_id,
                    "x": position[0],
                    "y": position[1],
                }
            )
        self.events += events
        return events

    def _forget_absent_tracks(self, present: set[int]) -> None:
        """Drop tracks gone long enough that their return is a new visit."""
        for track_id in self._last_position.keys() - present:
            misses = self._missing_frames.get(track_id, 0) + 1
            self._missing_frames[track_id] = misses
            if misses > self.stale_after_frames:
                del self._last_position[track_id]
                del self._missing_frames[track_id]

    def summary(self) -> dict:
        total_in = sum(c["in"] for c in self.counts.values())
        total_out = sum(c["out"] for c in self.counts.values())
        return {
            "per_line": self.counts,
            "total_in": total_in,
            "total_out": total_out,
            "net_inside": max(0, total_in - total_out),
            "events_count": len(self.events),
        }
