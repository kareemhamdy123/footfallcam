"""Typed configuration loaded from YAML.

`configs/default.yaml` is the single source of truth for every tunable (R3).
Nothing here hardcodes a coordinate, a threshold or a model path.

The dataclasses mirror the YAML structure one-for-one, so adding a knob means
editing the YAML and adding a field here - never editing logic.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

Point = tuple[float, float]


@dataclasses.dataclass
class Line:
    """A finite counting line segment.

    `p1`/`p2` stay normalised until `scaled()` resolves them against a frame
    size. `in_direction` names the movement axis whose sign decides whether a
    crossing counts as "in" or "out".
    """

    name: str
    p1: Point
    p2: Point
    in_direction: str = "left_to_right"

    def scaled(self, width: int, height: int) -> "Line":
        return Line(
            name=self.name,
            p1=_scale_point(self.p1, width, height),
            p2=_scale_point(self.p2, width, height),
            in_direction=self.in_direction,
        )


@dataclasses.dataclass
class Zone:
    """A polygon region of interest. `kind` selects the owning feature."""

    name: str
    polygon: list[Point]
    kind: str = "generic"  # generic | queue | sales | safe_occupancy | outside

    def scaled(self, width: int, height: int) -> "Zone":
        return Zone(
            name=self.name,
            polygon=[_scale_point(pt, width, height) for pt in self.polygon],
            kind=self.kind,
        )


@dataclasses.dataclass
class DetectionSettings:
    model_path: str = "yolov8n.onnx"
    confidence: float = 0.45
    device: str = "gpu"
    person_class_id: int = 0
    input_size: int = 640
    nms_iou: float = 0.45
    min_box_width_px: float = 12.0
    min_box_height_px: float = 16.0
    min_height_over_width: float = 0.30
    min_width_over_height: float = 0.15
    max_frame_fraction: float = 0.95


@dataclasses.dataclass
class TrackingSettings:
    iou_threshold: float = 0.30
    high_confidence_threshold: float = 0.45
    min_hits: int = 2
    max_missed: int = 30
    max_center_distance_px: float = 90.0
    max_occlusion_bridge: int = 2
    box_ema_alpha: float = 0.70
    velocity_ema_alpha: float = 0.60
    max_history: int = 300


@dataclasses.dataclass
class BehaviourSettings:
    dwell_alert_seconds: float = 60.0
    safe_occupancy_limit: int = 30
    queue_min_people: int = 2
    queue_max_gap_px: float = 120.0


@dataclasses.dataclass
class AutoModeSettings:
    """Zero-config boundary counting, used when no lines are configured."""

    missing_frame_debounce: int = 8
    passerby_margin_fraction: float = 0.10
    passerby_max_seconds: float = 3.2


@dataclasses.dataclass
class CountingSettings:
    group_max_distance_px: float = 90.0
    group_entry_window_s: float = 3.0
    auto_mode: AutoModeSettings = dataclasses.field(default_factory=AutoModeSettings)


@dataclasses.dataclass
class DemographicsSettings:
    """Brochure characteristic 6. Off by default - see configs/default.yaml."""

    enabled: bool = False
    model_path: str | None = None
    min_hits: int = 3
    heuristic_min_confidence: float = 0.60


@dataclasses.dataclass
class Config:
    source: str
    output_dir: str
    detection: DetectionSettings
    tracking: TrackingSettings
    behaviour: BehaviourSettings
    counting: CountingSettings
    demographics: DemographicsSettings
    lines: list[Line] = dataclasses.field(default_factory=list)

    def scaled_lines(self, width: int, height: int) -> list[Line]:
        """Resolve every configured line against a frame size (R3)."""
        return [line.scaled(width, height) for line in self.lines]

    @staticmethod
    def load(path: str | Path) -> "Config":
        raw: dict[str, Any] = yaml.safe_load(
            Path(path).read_text(encoding="utf-8")
        ) or {}
        return Config(
            source=str(raw["source"]),
            output_dir=str(raw.get("output_dir", "outputs")),
            detection=_detection_from(raw.get("detection") or {}),
            tracking=_tracking_from(raw.get("tracking") or {}),
            behaviour=_behaviour_from(raw.get("behaviour") or {}),
            counting=_counting_from(raw.get("counting") or {}),
            demographics=_demographics_from(raw.get("demographics") or {}),
            lines=[_line_from(entry) for entry in (raw.get("lines") or [])],
        )


def _line_from(entry: dict[str, Any]) -> Line:
    return Line(
        name=str(entry["name"]),
        p1=(float(entry["p1"][0]), float(entry["p1"][1])),
        p2=(float(entry["p2"][0]), float(entry["p2"][1])),
        in_direction=str(entry.get("in_direction", "left_to_right")),
    )


def _detection_from(d: dict[str, Any]) -> DetectionSettings:
    s = d.get("sanity") or {}
    return DetectionSettings(
        model_path=str(d.get("model_path", "yolov8n.onnx")),
        confidence=float(d.get("confidence", 0.45)),
        device=str(d.get("device", "gpu")),
        person_class_id=int(d.get("person_class_id", 0)),
        input_size=int(d.get("input_size", 640)),
        nms_iou=float(d.get("nms_iou", 0.45)),
        min_box_width_px=float(s.get("min_box_width_px", 12)),
        min_box_height_px=float(s.get("min_box_height_px", 16)),
        min_height_over_width=float(s.get("min_height_over_width", 0.30)),
        min_width_over_height=float(s.get("min_width_over_height", 0.15)),
        max_frame_fraction=float(s.get("max_frame_fraction", 0.95)),
    )


def _tracking_from(t: dict[str, Any]) -> TrackingSettings:
    return TrackingSettings(
        iou_threshold=float(t.get("iou_threshold", 0.30)),
        high_confidence_threshold=float(t.get("high_confidence_threshold", 0.45)),
        min_hits=int(t.get("min_hits", 2)),
        max_missed=int(t.get("max_missed", 30)),
        max_center_distance_px=float(t.get("max_center_distance_px", 90)),
        max_occlusion_bridge=int(t.get("max_occlusion_bridge", 2)),
        box_ema_alpha=float(t.get("box_ema_alpha", 0.70)),
        velocity_ema_alpha=float(t.get("velocity_ema_alpha", 0.60)),
        max_history=int(t.get("max_history", 300)),
    )


def _behaviour_from(b: dict[str, Any]) -> BehaviourSettings:
    return BehaviourSettings(
        dwell_alert_seconds=float(b.get("dwell_alert_seconds", 60)),
        safe_occupancy_limit=int(b.get("safe_occupancy_limit", 30)),
        queue_min_people=int(b.get("queue_min_people", 2)),
        queue_max_gap_px=float(b.get("queue_max_gap_px", 120)),
    )


def _counting_from(c: dict[str, Any]) -> CountingSettings:
    a = c.get("auto_mode") or {}
    return CountingSettings(
        group_max_distance_px=float(c.get("group_max_distance_px", 90)),
        group_entry_window_s=float(c.get("group_entry_window_s", 3.0)),
        auto_mode=AutoModeSettings(
            missing_frame_debounce=int(a.get("missing_frame_debounce", 8)),
            passerby_margin_fraction=float(a.get("passerby_margin_fraction", 0.10)),
            passerby_max_seconds=float(a.get("passerby_max_seconds", 3.2)),
        ),
    )


def _demographics_from(d: dict[str, Any]) -> DemographicsSettings:
    model_path = d.get("model_path")
    return DemographicsSettings(
        enabled=bool(d.get("enabled", False)),
        model_path=str(model_path) if model_path else None,
        min_hits=int(d.get("min_hits", 3)),
        heuristic_min_confidence=float(d.get("heuristic_min_confidence", 0.60)),
    )


def _scale_point(point: Point, width: int, height: int) -> tuple[int, int]:
    """Scale one point, treating any value in 0.0-1.0 as normalised."""
    x, y = point
    sx = int(x * width) if 0.0 <= x <= 1.0 else int(x)
    sy = int(y * height) if 0.0 <= y <= 1.0 else int(y)
    return (sx, sy)
