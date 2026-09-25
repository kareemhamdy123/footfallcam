"""Feature 13: Multiple counting lines.

Owns every multi-gate concern for counting: a per-line tally, the previous
foot point needed to detect a crossing, and per-line idempotency via
`track.counted_lines`. Brochure characteristic 13 ("Multiple counting line").

This is a primitive. It counts whatever tracks it is handed and applies no
policy about *who* counts - excluding staff is `VideoCounter`'s job, not this
module's. Keeping that split is what stops the two from drifting apart (R7).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from src.geometry.zones import crossed_line

if TYPE_CHECKING:  # pragma: no cover - typing only (R1)
    from src.config import Line
    from src.tracking.dataclasses import Track

Point = tuple[float, float]


class MultipleCountingLineManager:
    """Independent bi-directional tallies, one per configured line segment."""

    def __init__(self, lines: Sequence["Line"], stale_after_frames: int = 30):
        self.lines = list(lines)
        self.stale_after_frames = stale_after_frames
        self.counts: dict[str, dict[str, int]] = {
            line.name: {"in": 0, "out": 0} for line in self.lines
        }
        self.events: list[dict] = []

        # Foot point seen on the previous frame, per track. Held here rather
        # than read from Track.history, so a track bridged across an occlusion
        # still has a valid "previous" position to compare against.
        self._last_position: dict[int, Point] = {}
        self._missing_frames: dict[int, int] = {}

    def update(
        self,
        tracks: list["Track"],
        frame_idx: int,
        timestamp_s: float,
    ) -> list[dict]:
        """Tally crossings for this frame. Returns only the new events."""
        new_events: list[dict] = []
        present: set[int] = set()

        for track in tracks:
            present.add(track.track_id)
            self._missing_frames.pop(track.track_id, None)

            position = track.foot_point
            previous = self._last_position.get(track.track_id)
            self._last_position[track.track_id] = position
            if previous is None:
                continue  # first sighting: no movement to judge yet

            new_events += self._count_lines(track, previous, position, frame_idx, timestamp_s)

        self._forget_absent_tracks(present)
        return new_events

    def _count_lines(
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
                continue  # one count per line per track, ever (R7)
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
        for track_id in list(self._last_position):
            if track_id in present:
                continue
            misses = self._missing_frames.get(track_id, 0) + 1
            self._missing_frames[track_id] = misses
            if misses > self.stale_after_frames:
                del self._last_position[track_id]
                del self._missing_frames[track_id]

    def reset(self) -> None:
        """Clear all tallies, events and per-track memory."""
        for tally in self.counts.values():
            tally["in"] = 0
            tally["out"] = 0
        self.events.clear()
        self._last_position.clear()
        self._missing_frames.clear()

    def summary(self) -> dict:
        total_in = sum(t["in"] for t in self.counts.values())
        total_out = sum(t["out"] for t in self.counts.values())
        return {
            "per_line": self.counts,
            "total_in": total_in,
            "total_out": total_out,
            "net_inside": max(0, total_in - total_out),
            "events_count": len(self.events),
        }
