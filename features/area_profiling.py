"""Feature 5: Area profiling.

How long people spend in each part of the floor, and what share of attention
each area actually gets. Brochure characteristic 5 ("Area profiling").

A track is attributed to the zone containing its **foot point**, not its
centroid: two people at the same depth have the same foot y whatever their
height, so dwell time stays comparable between an adult and a child. Using the
centroid would make every child look like they were standing further forward.

Percentages are shares of the time actually spent inside *some* zone. Time
outside every configured polygon is reported separately rather than inflating
the denominator - nobody standing in an unzoned corridor is engaging with
nothing in particular.

Each share is rounded independently, so a three-way split reads 33.33 + 33.33 +
33.33 = 99.99. That is rounding, not a missing percent. The remainder is not
dumped onto the largest zone to force an exact 100, because that would report
time that zone did not have.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from src.geometry.zones import auto_bands, zones_containing

if TYPE_CHECKING:  # pragma: no cover - typing only (R1)
    from src.tracking.dataclasses import Track


class AreaProfiler:
    def __init__(
        self,
        zones: Sequence = (),
        auto_band_count: int = 3,
        frame_shape: tuple[int, int] | None = None,
        auto_mode: bool | None = None,
    ):
        self.zones = list(zones)
        self.auto_band_count = auto_band_count
        self.frame_shape = frame_shape
        self.seconds_by_zone: dict[str, float] = {}
        self.seconds_unzoned = 0.0
        # Recorded, not inferred: building auto bands *populates* self.zones,
        # so "is this auto mode" cannot be answered by asking whether any
        # zones exist any more. A caller that resolved bands itself passes
        # auto_mode explicitly.
        self.auto_mode = (not self.zones) if auto_mode is None else bool(auto_mode)
        # Whether zones exist yet. Independent of auto_mode: a caller may hand
        # over bands it resolved itself while still being "auto mode".
        self._auto_bands_built = bool(self.zones)

    def update(
        self,
        tracks: list["Track"],
        dt_s: float,
        frame_shape: tuple[int, int] | None = None,
    ) -> dict[str, int]:
        """Accumulate one frame of dwell. Returns the frame's per-zone headcount."""
        if frame_shape is not None:
            self.frame_shape = frame_shape
        if not self._auto_bands_built:
            self._build_bands()

        per_zone: dict[str, int] = {}
        for track in tracks:
            zone = self._zone_for(track)
            if zone is None:
                self.seconds_unzoned += dt_s
                continue
            self.seconds_by_zone[zone.name] = (
                self.seconds_by_zone.get(zone.name, 0.0) + dt_s
            )
            per_zone[zone.name] = per_zone.get(zone.name, 0) + 1

        return per_zone

    def _build_bands(self) -> None:
        """Zero-config mode: slice the frame by relative Y."""
        self._auto_bands_built = True
        if self.zones or not self.frame_shape:
            return
        width, height = self.frame_shape
        self.zones = auto_bands(width, height, self.auto_band_count)
        self.auto_mode = True

    def _zone_for(self, track: "Track"):
        # Overlapping zones: the first configured match wins, so a deliberate
        # ordering in the config is respected.
        matches = zones_containing(track.foot_point, self.zones)
        return matches[0] if matches else None

    def summary(self) -> dict:
        total = sum(self.seconds_by_zone.values())
        return {
            "seconds_by_zone": {k: round(v, 2) for k, v in self.seconds_by_zone.items()},
            "engagement_percentage": {
                name: _share(seconds, total) for name, seconds in self.seconds_by_zone.items()
            },
            "total_engagement_seconds": round(total, 2),
            "zones_profiled": len(self.seconds_by_zone),
            "seconds_outside_all_zones": round(self.seconds_unzoned, 2),
            "auto_mode": self.auto_mode,
        }

    def reset(self) -> None:
        self.seconds_by_zone.clear()
        self.seconds_unzoned = 0.0
        self.auto_mode = not self.zones
        self._auto_bands_built = bool(self.zones)


def _share(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0
