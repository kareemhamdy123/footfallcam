"""
Thin wrapper around Ultralytics YOLO for person detection.

Maps to brochure characteristics:
  1. Video Counting        -> per-frame person detections feed the counters
  11. Object clarification -> confidence score + class label returned per box
  19. Night vision mode    -> handled upstream in video_io.enhance_frame(),
                               detector just runs on whatever frame it is given
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - allows the repo to be inspected
    YOLO = None        # without the (heavy) dependency installed.


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def centroid(self) -> tuple:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def foot_point(self) -> tuple:
        """Bottom-center of the box - a better proxy for 'where a person is
        standing' than the box centroid, used for zone/line tests."""
        return ((self.x1 + self.x2) / 2.0, self.y2)


class PersonDetector:
    def __init__(self, model_path: str, confidence: float = 0.35,
                 device: str = "cpu", person_class_id: int = 0):
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. Run `pip install ultralytics` "
                "(see requirements.txt)."
            )
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device
        self.person_class_id = person_class_id

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            classes=[self.person_class_id],
            verbose=False,
        )
        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                detections.append(Detection(x1, y1, x2, y2, conf, cls))
        return detections
