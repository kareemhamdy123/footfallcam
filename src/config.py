"""
Configuration loader.

Reads a YAML file describing:
  - camera / source settings
  - detection thresholds
  - counting lines (for in/out counting, multiple-line support)
  - zones (polygons) for: queue area, sales/counter area, safe-occupancy area,
    outside-store area, custom heatmap zones
  - staff-exclusion tag color range
  - behavioural thresholds (dwell time, queue length, group distance, etc.)

Keeping every tunable value in one YAML file means the pipeline can be
re-pointed at a different camera/store layout without touching code.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class Line:
    name: str
    p1: tuple
    p2: tuple
    # direction considered "in": "left_to_right", "right_to_left",
    # "top_to_bottom", "bottom_to_top"
    in_direction: str = "left_to_right"


@dataclasses.dataclass
class Zone:
    name: str
    polygon: list  # list of (x, y) tuples
    kind: str = "generic"  # generic | queue | sales | safe_occupancy | outside


@dataclasses.dataclass
class Config:
    source: str
    model_path: str
    confidence: float
    device: str
    person_class_id: int
    lines: list
    zones: list
    dwell_alert_seconds: float
    queue_min_people: int
    queue_max_gap_px: float
    group_max_distance_px: float
    safe_occupancy_limit: int
    staff_tag_hsv_lower: tuple
    staff_tag_hsv_upper: tuple
    night_mode: str  # "auto" | "on" | "off"
    heatmap_decay: float
    output_dir: str
    fps_override: float | None = None

    @staticmethod
    def load(path: str) -> "Config":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())

        lines = [
            Line(
                name=l["name"],
                p1=tuple(l["p1"]),
                p2=tuple(l["p2"]),
                in_direction=l.get("in_direction", "left_to_right"),
            )
            for l in raw.get("lines", [])
        ]
        zones = [
            Zone(
                name=z["name"],
                polygon=[tuple(pt) for pt in z["polygon"]],
                kind=z.get("kind", "generic"),
            )
            for z in raw.get("zones", [])
        ]

        d = raw.get("detection", {})
        b = raw.get("behaviour", {})
        s = raw.get("staff_exclusion", {})

        return Config(
            source=raw["source"],
            model_path=d.get("model_path", "yolov8n.pt"),
            confidence=float(d.get("confidence", 0.35)),
            device=d.get("device", "cpu"),
            person_class_id=int(d.get("person_class_id", 0)),
            lines=lines,
            zones=zones,
            dwell_alert_seconds=float(b.get("dwell_alert_seconds", 600)),
            queue_min_people=int(b.get("queue_min_people", 2)),
            queue_max_gap_px=float(b.get("queue_max_gap_px", 120)),
            group_max_distance_px=float(b.get("group_max_distance_px", 80)),
            safe_occupancy_limit=int(b.get("safe_occupancy_limit", 50)),
            staff_tag_hsv_lower=tuple(s.get("hsv_lower", [140, 80, 80])),
            staff_tag_hsv_upper=tuple(s.get("hsv_upper", [170, 255, 255])),
            night_mode=raw.get("night_mode", "auto"),
            heatmap_decay=float(raw.get("heatmap_decay", 0.0)),
            output_dir=raw.get("output_dir", "outputs"),
            fps_override=raw.get("fps_override"),
        )
