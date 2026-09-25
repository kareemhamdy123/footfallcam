"""Feature 17: Zone counting.

How many people are in each part of the floor right now, and the peak each area
ever reached. Brochure characteristic 17 ("Zone counting").

Distinct from area profiling in kind, not in location: profiling accumulates
dwell time, this answers "how full is it" and "how full did it get". Both read
the same zones and the same foot point, but neither derives one from the other.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from src.geometry.zones import auto_bands, zones_containing

if TYPE_CHECKING:  # pragma: no cover - typing only (R1)
    from src.tracking.dataclasses import Track


class ZoneCounter:
    def __init__(
        self,
        zones: Sequence = (),
        auto_band_count: int = 3,
        frame_shape: tuple[int, int] | None = None,
    ):
        self.zones = list(zones)
        self.auto_band_count = auto_band_count
        self.frame_shape = frame_shape
        self.current: dict[str, int] = {}
        self.peak: dict[str, int] = {}
        self.active_tracks: dict[str, set[int]] = {}
        self._auto_bands_built = bool(self.zones)

    def update(self, tracks: list["Track"]) -> dict[str, int]:
        """Recompute the current headcount. Returns zone name -> people now."""
        if not self._auto_bands_built:
            self._build_bands()

        self.current = {zone.name: 0 for zone in self.zones}
        self.active_tracks = {zone.name: set() for zone in self.zones}
        # Every configured zone reports a peak, including one nobody ever
        # entered: a missing key reads as "not measured" in a report, when the
        # truth is "measured, and it stayed empty".
        for name in self.current:
            self.peak.setdefault(name, 0)

        for track in tracks:
            zone = self._zone_for(track)
            if zone is None:
                continue
            self.current[zone.name] += 1
            self.active_tracks[zone.name].add(track.track_id)
            # Peak is a high-water mark, so it only ever rises.
            self.peak[zone.name] = max(self.peak.get(zone.name, 0), self.current[zone.name])

        return dict(self.current)

    def _build_bands(self) -> None:
        self._auto_bands_built = True
        if self.zones or not self.frame_shape:
            return
        width, height = self.frame_shape
        self.zones = auto_bands(width, height, self.auto_band_count)

    def _zone_for(self, track: "Track"):
        matches = zones_containing(track.foot_point, self.zones)
        return matches[0] if matches else None

    def zone_of(self, track: "Track") -> str | None:
        """Which zone a track is in, or None. Useful for other features."""
        zone = self._zone_for(track)
        return zone.name if zone else None

    def summary(self) -> dict:
        return {
            "current_zone_headcounts": dict(self.current),
            "peak_zone_headcounts": dict(self.peak),
            "active_tracks_by_zone": {
                name: sorted(ids) for name, ids in self.active_tracks.items()
            },
            "total_zones": len(self.zones),
        }

    def reset(self) -> None:
        self.current.clear()
        self.peak.clear()
        self.active_tracks.clear()
        self._auto_bands_built = bool(self.zones)
