"""
Maps to brochure characteristic:
  18. Heatmap -> accumulates a density map of where people stand/walk over
                 the session, exportable as a colourised PNG overlay.
  5.  Area Profiling -> per-zone occupied-time totals, derived from the same
                         accumulator, broken down by configured zone.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import Zone
from ..tracker import Track
from ..zones import point_in_polygon


class HeatmapAccumulator:
    def __init__(self, frame_size: tuple, decay: float = 0.0, radius: int = 18):
        w, h = frame_size
        self.grid = np.zeros((h, w), dtype=np.float32)
        self.decay = decay
        self.radius = radius

    def update(self, tracks: list[Track]) -> None:
        if self.decay > 0:
            self.grid *= (1.0 - self.decay)
        for t in tracks:
            if not t.history:
                continue
            x, y = t.history[-1][1:]
            cv2.circle(self.grid, (int(x), int(y)), self.radius, 1.0, thickness=-1)

    def export_overlay(self, base_frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        norm = self.grid.copy()
        if norm.max() > 0:
            norm = (norm / norm.max() * 255).astype(np.uint8)
        else:
            norm = norm.astype(np.uint8)
        norm = cv2.GaussianBlur(norm, (0, 0), sigmaX=9)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        return cv2.addWeighted(colored, alpha, base_frame, 1 - alpha, 0)

    def save(self, path: str, base_frame: np.ndarray) -> None:
        cv2.imwrite(path, self.export_overlay(base_frame))


class AreaProfiler:
    """Accumulates seconds-of-presence per zone, per frame tick."""

    def __init__(self, zones: list[Zone]):
        self.zones = zones
        self.seconds_by_zone = {z.name: 0.0 for z in zones}

    def update(self, tracks: list[Track], dt_s: float) -> None:
        for zone in self.zones:
            for t in tracks:
                if not t.history or t.is_staff:
                    continue
                if point_in_polygon(t.history[-1][1:], zone.polygon):
                    self.seconds_by_zone[zone.name] += dt_s

    def summary(self) -> dict:
        return {k: round(v, 1) for k, v in self.seconds_by_zone.items()}
