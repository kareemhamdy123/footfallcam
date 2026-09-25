"""The Detection value object.

Coordinates are pixels in the source frame's own coordinate space.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def foot_point(self) -> tuple[float, float]:
        """Bottom-centre of the box - where the person meets the floor.

        Every spatial feature measures from this point rather than the
        centroid, because the centroid of a partially occluded person sits too
        high and drifts as the occlusion changes.
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)
