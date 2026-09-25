"""Feature 1: Video Counting.

Bi-directional footfall counting, the counter every other feature's numbers
depend on. Brochure characteristic 1 ("Video Counting").

Two modes:

* **Line mode** (default when lines are configured) - every crossing of every
  configured segment is tallied by `MultipleCountingLineManager`.
* **Auto mode** (no lines configured) - zero-config boundary counting: a track's
  first appearance is an IN, and a track that stays missing long enough is an
  OUT. A track that only ever grazes the frame edge for a moment is classified
  as a passerby rather than a visitor.

Both the multi-gate tally and the group clustering live in their own modules and
are delegated to, never reimplemented here (R7). This class owns only the
policy that sits above them: who counts as a customer, and what an exit means.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from features.group_counting import GroupCounter
from features.multiple_counting_line import MultipleCountingLineManager

if TYPE_CHECKING:  # pragma: no cover - typing only (R1)
    from src.config import Line
    from src.tracking.dataclasses import Track

Point = tuple[float, float]

SCENE_FLOW = "scene_flow"
OUTSIDE_SIDEWALK = "outside_sidewalk"


class VideoCounter:
    """Customer footfall counter over lines, or over the frame boundary."""

    def __init__(
        self,
        lines: Sequence["Line"] | None = None,
        group_max_distance_px: float = 90.0,
        frame_shape: tuple[int, int] | None = None,
        group_entry_window_s: float = 3.0,
        missing_frame_debounce: int = 8,
        passerby_margin_fraction: float = 0.10,
        passerby_max_seconds: float = 3.2,
    ):
        self.lines = list(lines or [])
        self.frame_shape = frame_shape  # (width, height)

        # R7: delegated, never duplicated.
        self.line_manager = MultipleCountingLineManager(self.lines)
        self.group_counter = GroupCounter(
            group_max_distance_px=group_max_distance_px,
            entry_window_s=group_entry_window_s,
        )

        self.missing_frame_debounce = missing_frame_debounce
        self.passerby_margin_fraction = passerby_margin_fraction
        self.passerby_max_seconds = passerby_max_seconds

        self.auto_mode = not self.lines
        if self.auto_mode:
            self.counts[SCENE_FLOW] = {"in": 0, "out": 0}
            self.counts[OUTSIDE_SIDEWALK] = {"in": 0, "out": 0}

        self.events: list[dict] = []
        self.passerby_ids: set[int] = set()
        self.inside_ids: set[int] = set()
        self._first_seen: dict[int, tuple[float, Point]] = {}
        self._last_seen: dict[int, tuple[float, Point]] = {}
        self._missing_frames: dict[int, int] = {}

    # ------------------------------------------------------------------ input

    @property
    def counts(self) -> dict[str, dict[str, int]]:
        """The per-line tally, owned by the line manager.

        A property rather than a copied reference, so the counter can never
        hold a tally that has drifted from the manager's (R7).
        """
        return self.line_manager.counts

    def update(
        self,
        tracks: list["Track"],
        frame_idx: int,
        timestamp_s: float,
        frame_shape: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Count this frame. Returns only the events raised by this frame."""
        if frame_shape is not None:
            self.frame_shape = frame_shape
        if not self.auto_mode:
            return self._update_line_mode(tracks, frame_idx, timestamp_s)
        return self._update_auto_mode(tracks, frame_idx, timestamp_s)

    def _update_line_mode(
        self, tracks: list["Track"], frame_idx: int, timestamp_s: float
    ) -> list[dict]:
        customers = [t for t in tracks if not t.is_staff]
        self._track_co_movement(customers, timestamp_s)
        events = self.line_manager.update(customers, frame_idx, timestamp_s)
        self.events += events
        return events

    def _update_auto_mode(
        self, tracks: list["Track"], frame_idx: int, timestamp_s: float
    ) -> list[dict]:
        customers = [t for t in tracks if not t.is_staff]
        self._track_co_movement(customers, timestamp_s)

        new_events: list[dict] = []
        present: set[int] = set()

        for track in customers:
            present.add(track.track_id)
            self._missing_frames.pop(track.track_id, None)
            position = track.foot_point
            self._last_seen[track.track_id] = (timestamp_s, position)

            if track.track_id in self.inside_ids:
                continue

            # First sighting in the scene: an entry.
            self.inside_ids.add(track.track_id)
            self._first_seen[track.track_id] = (timestamp_s, position)
            self.counts[SCENE_FLOW]["in"] += 1
            new_events.append(
                self._event(frame_idx, timestamp_s, SCENE_FLOW, "in", track.track_id, position)
            )

        new_events += self._settle_departures(present, frame_idx, timestamp_s)
        self.events += new_events
        return new_events

    def _settle_departures(
        self, present: set[int], frame_idx: int, timestamp_s: float
    ) -> list[dict]:
        """Turn vanished tracks into OUT events, once they stay vanished.

        The debounce is the whole point: a detection dropout for a few frames
        must not read as the person leaving the store.
        """
        events: list[dict] = []
        width, height = self._frame_size()

        for track_id in list(self.inside_ids):
            if track_id in present:
                continue
            misses = self._missing_frames.get(track_id, 0) + 1
            self._missing_frames[track_id] = misses
            if misses < self.missing_frame_debounce:
                continue

            self.inside_ids.discard(track_id)
            self.counts[SCENE_FLOW]["out"] += 1
            _, position = self._last_seen.get(track_id, (timestamp_s, (width / 2, height / 2)))
            events.append(
                self._event(frame_idx, timestamp_s, SCENE_FLOW, "out", track_id, position)
            )

            if self._is_passerby(track_id, timestamp_s, width, height):
                self.passerby_ids.add(track_id)
                self.counts[OUTSIDE_SIDEWALK]["in"] += 1

        return events

    def _is_passerby(
        self, track_id: int, timestamp_s: float, width: float, height: float
    ) -> bool:
        """Someone who only clipped the edge of the frame is not a visitor."""
        first_t, first_pos = self._first_seen.get(track_id, (timestamp_s, (0.0, 0.0)))
        if (timestamp_s - first_t) > self.passerby_max_seconds:
            return False

        margin_x = width * self.passerby_margin_fraction
        margin_y = height * self.passerby_margin_fraction
        x, y = first_pos
        return (
            x < margin_x or x > width - margin_x or y < margin_y or y > height - margin_y
        )

    def _track_co_movement(
        self, tracks: Sequence["Track"], timestamp_s: float
    ) -> None:
        dt_s = self.group_counter.frame_delta(timestamp_s)
        self.group_counter.update_co_movement(tracks, timestamp_s, dt_s)

    # ----------------------------------------------------------------- output

    @property
    def co_movement_seconds(self) -> dict[tuple[int, int], float]:
        """Co-movement time maintained by the delegated GroupCounter."""
        return self.group_counter.co_movement_seconds

    def grouped_entries(self, window_s: float | None = None) -> list[list[dict]]:
        """Entry events clustered into purchasing units."""
        in_events = [e for e in self.events if e.get("direction") == "in"]
        return self.group_counter.cluster_groups(in_events, window_s)

    def summary(self) -> dict:
        line_sums = self.line_manager.summary()
        per_line = self.counts
        in_events = [e for e in self.events if e.get("direction") == "in"]
        groups = self.grouped_entries()
        multi = [g for g in groups if len(g) >= 2]

        return {
            "per_line": per_line,
            "total_in": line_sums["total_in"],
            "total_out": line_sums["total_out"],
            "net_inside": line_sums["net_inside"],
            "outside_passersby": len(self.passerby_ids),
            "group_entries": len(groups),
            "multi_person_groups": len(multi),
            "events_count": len(self.events),
            "group_stats": self.group_counter.summary(in_events),
        }

    def reset(self) -> None:
        self.line_manager.reset()
        self.group_counter.reset()
        self.events.clear()
        self.passerby_ids.clear()
        self.inside_ids.clear()
        self._first_seen.clear()
        self._last_seen.clear()
        self._missing_frames.clear()
        for name in (SCENE_FLOW, OUTSIDE_SIDEWALK):
            if name in self.counts:
                self.counts[name] = {"in": 0, "out": 0}

    # --------------------------------------------------------------- internals

    def _frame_size(self) -> tuple[float, float]:
        return self.frame_shape if self.frame_shape else (640.0, 480.0)

    def _event(
        self,
        frame_idx: int,
        timestamp_s: float,
        line: str,
        direction: str,
        track_id: int,
        position: Point,
    ) -> dict:
        return {
            "frame": frame_idx,
            "t": timestamp_s,
            "line": line,
            "direction": direction,
            "track_id": track_id,
            "x": position[0],
            "y": position[1],
        }


# The plan names this feature LineCounter / VideoCounter interchangeably.
LineCounter = VideoCounter
