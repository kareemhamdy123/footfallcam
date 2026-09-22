"""
Maps to brochure characteristics:
  1.  Video Counting
  8.  Outside traffic (a line/zone placed outside the entrance)
  9.  Turn in rate            = entries / outside-traffic passers-by
  13. Multiple counting line  -> independent tally per configured Line
  14. Group counting          -> nearby simultaneous entries counted as one group
"""
from __future__ import annotations

from collections import defaultdict

from ..config import Line
from ..tracker import Track
from ..zones import crossed_line


class LineCounter:
    def __init__(self, lines: list[Line], group_max_distance_px: float = 80.0):
        self.lines = lines
        self.group_max_distance_px = group_max_distance_px
        self.counts = {l.name: {"in": 0, "out": 0} for l in lines}
        self.events: list[dict] = []  # raw crossing events, used for grouping

    def update(self, tracks: list[Track], frame_idx: int, timestamp_s: float) -> None:
        for track in tracks:
            if len(track.history) < 2:
                continue
            _, px, py = track.history[-2]
            _, cx, cy = track.history[-1]
            for line in self.lines:
                key = line.name
                if key in track.counted_lines:
                    continue
                direction = crossed_line((px, py), (cx, cy), line)
                if direction is None:
                    continue
                if track.is_staff:
                    continue  # staff excluded from footfall counts (#12)
                self.counts[key][direction] += 1
                track.counted_lines.add(key)
                self.events.append({
                    "frame": frame_idx, "t": timestamp_s, "line": key,
                    "direction": direction, "track_id": track.track_id,
                    "x": cx, "y": cy,
                })

    def grouped_entries(self, window_s: float = 2.0) -> list[list[dict]]:
        """
        Cluster near-simultaneous 'in' events that are also spatially close
        into groups (e.g. a family walking in together counts as one group,
        while still being counted individually in `self.counts`).
        """
        in_events = sorted(
            (e for e in self.events if e["direction"] == "in"),
            key=lambda e: e["t"],
        )
        groups: list[list[dict]] = []
        used = set()
        for i, e in enumerate(in_events):
            if i in used:
                continue
            group = [e]
            used.add(i)
            for j in range(i + 1, len(in_events)):
                if j in used:
                    continue
                other = in_events[j]
                if other["t"] - e["t"] > window_s:
                    break
                dist = ((other["x"] - e["x"]) ** 2 + (other["y"] - e["y"]) ** 2) ** 0.5
                if dist <= self.group_max_distance_px:
                    group.append(other)
                    used.add(j)
            groups.append(group)
        return groups

    def summary(self) -> dict:
        total_in = sum(c["in"] for c in self.counts.values())
        total_out = sum(c["out"] for c in self.counts.values())
        groups = self.grouped_entries()
        return {
            "per_line": self.counts,
            "total_in": total_in,
            "total_out": total_out,
            "net_inside": total_in - total_out,
            "group_entries": len(groups),
            "average_group_size": (
                sum(len(g) for g in groups) / len(groups) if groups else 0
            ),
        }


class TurnInRate:
    """
    Compares people who merely pass an 'outside' line (outdoor foot traffic)
    against people who actually cross an 'entrance' line into the store.
    """

    def __init__(self, outside_line_name: str, entrance_line_name: str):
        self.outside_line_name = outside_line_name
        self.entrance_line_name = entrance_line_name

    def compute(self, counts: dict) -> float:
        passers_by = counts.get(self.outside_line_name, {}).get("in", 0) + \
            counts.get(self.outside_line_name, {}).get("out", 0)
        entries = counts.get(self.entrance_line_name, {}).get("in", 0)
        if passers_by == 0:
            return 0.0
        return round(entries / passers_by, 4)
