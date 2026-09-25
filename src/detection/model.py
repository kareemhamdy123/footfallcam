"""PersonDetector - ONNX inference over a hardware provider chain.

Provider preference: DirectML -> CUDA -> CPU. DirectML is what makes an NVIDIA
GPU usable from plain ONNX Runtime on Windows with no CUDA toolkit. If the
configured .onnx is missing, fall back to the Ultralytics PyTorch model so the
pipeline still runs from a bare checkout.

Post-processing order:
    letterbox -> infer -> confidence gate -> geometric sanity -> NMS
    -> partial-body dedup

Every threshold arrives from `DetectionSettings` (R3); nothing is hardcoded.
"""
from __future__ import annotations

import os
from typing import Sequence

import cv2
import numpy as np

from ..config import DetectionSettings
from .dataclasses import Detection

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:  # pragma: no cover - depends on the install
    ort = None
    HAS_ORT = False

try:
    from ultralytics import YOLO

    HAS_ULTRALYTICS = True
except ImportError:  # pragma: no cover - depends on the install
    YOLO = None
    HAS_ULTRALYTICS = False

Box = tuple[float, float, float, float]


def letterbox(
    frame: np.ndarray, size: int = 640, pad_value: int = 114
) -> tuple[np.ndarray, float, float, float]:
    """Resize preserving aspect ratio into a `size` x `size` grey canvas.

    Returns `(canvas, ratio, pad_x, pad_y)`: the uniform scale factor plus the
    left/top borders in canvas pixels. Those three values are everything
    `unletterbox_box` needs to invert the mapping.
    """
    height, width = frame.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    pad_x = (size - new_w) / 2.0
    pad_y = (size - new_h) / 2.0

    top, left = int(round(pad_y)), int(round(pad_x))
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = cv2.resize(
        frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
    )
    return canvas, ratio, pad_x, pad_y


def unletterbox_box(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    ratio: float,
    pad_x: float,
    pad_y: float,
) -> Box:
    """Map a letterboxed centre/width/height box back to source pixels."""
    return (
        (cx - bw / 2.0 - pad_x) / ratio,
        (cy - bh / 2.0 - pad_y) / ratio,
        bw / ratio,
        bh / ratio,
    )


def suppress_part_duplicates(
    detections: Sequence[Detection],
    x_overlap_thresh: float = 0.60,
    y_overlap_thresh: float = 0.20,
) -> list[Detection]:
    """Drop torso/partial-body duplicates of a larger, more confident person box.

    YOLO sometimes fires on a visible arm or head of someone it also detects
    whole. The partial box is horizontally aligned with and vertically
    overlapping the full box, so it is redundant for counting.
    """
    if len(detections) <= 1:
        return list(detections)

    # Prefer big, confident boxes: sort by area * confidence, descending.
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.area * d.confidence, reverse=True):
        if any(
            _is_partial_of(det, k, x_overlap_thresh, y_overlap_thresh)
            for k in kept
        ):
            continue
        kept.append(det)
    return kept


def _is_partial_of(
    candidate: Detection,
    kept: Detection,
    x_overlap_thresh: float,
    y_overlap_thresh: float,
) -> bool:
    x_inter = max(0.0, min(candidate.x2, kept.x2) - max(candidate.x1, kept.x1))
    y_inter = max(0.0, min(candidate.y2, kept.y2) - max(candidate.y1, kept.y1))
    x_ratio = x_inter / max(1.0, min(candidate.width, kept.width))
    y_ratio = y_inter / max(1.0, min(candidate.height, kept.height))
    return x_ratio >= x_overlap_thresh and y_ratio >= y_overlap_thresh


def passes_sanity(
    box_w: float,
    box_h: float,
    frame_w: int,
    frame_h: int,
    cfg: DetectionSettings,
) -> bool:
    """Reject noise, poles and counter-edge strips (config-driven, R3).

    A pure function so the filter is unit-testable without loading a model.
    """
    if box_w < cfg.min_box_width_px or box_h < cfg.min_box_height_px:
        return False
    if (box_h / max(1.0, box_w)) < cfg.min_height_over_width:
        return False
    if (box_w / max(1.0, box_h)) < cfg.min_width_over_height:
        return False
    oversized = (
        box_w > frame_w * cfg.max_frame_fraction
        and box_h > frame_h * cfg.max_frame_fraction
    )
    return not oversized


def as_rows(
    output: np.ndarray, person_class_id: int
) -> list[tuple[float, float, float, float, float]] | None:
    """Normalise a raw YOLOv8 output tensor to per-box tuples.

    Accepts either (1, 4+nc, anchors) or an already-transposed
    (1, anchors, 4+nc), so either export layout survives. A pure function so
    the layout handling is unit-testable without a model.

    The layout is inferred from the axis lengths, which relies on anchors always
    outnumbering the 4+nc attribute columns - true of every YOLOv8 export.
    """
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T  # (4+nc, anchors) -> (anchors, 4+nc)

    return [
        (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s))
        for b, s in zip(arr[:, :4], arr[:, 4 + person_class_id])
    ]


def _nms(boxes: list[Box], scores: list[float], iou_threshold: float) -> list[int]:
    """Non-maximum suppression. Returns indices into the input lists."""
    xywh = [[int(x), int(y), int(w), int(h)] for x, y, w, h in boxes]
    raw = cv2.dnn.NMSBoxes(xywh, scores, float(min(scores)), iou_threshold)
    # OpenCV has returned both a flat array and a tuple of arrays across versions.
    return [int(i) for i in np.asarray(raw).reshape(-1)]


class PersonDetector:
    """Single-class (person) detector over a hardware provider chain."""

    def __init__(self, settings: DetectionSettings | None = None, **overrides):
        self.cfg = settings or DetectionSettings()
        for key, value in overrides.items():
            setattr(self.cfg, key, value)

        self.backend = "none"
        self.session = None
        self._input_name: str | None = None
        self._model = None

        if not self._init_onnx():
            self._init_ultralytics()

    # ------------------------------------------------------------------ setup

    def _init_onnx(self) -> bool:
        model_path = self.cfg.model_path
        if not HAS_ORT or not model_path.endswith(".onnx"):
            return False
        if not os.path.exists(model_path):
            print(f"[detection] ONNX model not found at '{model_path}'")
            return False

        providers: list[str] = []
        if any(
            token in str(self.cfg.device).lower()
            for token in ("gpu", "cuda", "dml", "directml")
        ):
            available = ort.get_available_providers()
            providers += [
                name
                for name in ("DmlExecutionProvider", "CUDAExecutionProvider")
                if name in available
            ]
        providers.append("CPUExecutionProvider")

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as exc:  # pragma: no cover - runtime specific
            print(f"[detection] ONNX session failed ({exc}); trying PyTorch")
            self.session = None
            return False

        self._input_name = self.session.get_inputs()[0].name
        self.backend = "onnx"
        print(
            f"[detection] ONNX ready on {self.session.get_providers()[0]}"
            f" (model={model_path}, input={self.cfg.input_size})"
        )
        return True

    def _init_ultralytics(self) -> None:
        if not HAS_ULTRALYTICS:
            raise ImportError(
                "No usable detector backend: install onnxruntime and export "
                f"'{self.cfg.model_path}', or install ultralytics."
            )
        self._model = YOLO(self.cfg.model_path)
        self.backend = "ultralytics"
        print(f"[detection] Ultralytics fallback (model={self.cfg.model_path})")

    # -------------------------------------------------------------- inference

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raw = (
            self._detect_onnx(frame)
            if self.backend == "onnx"
            else self._detect_ultralytics(frame)
        )
        return suppress_part_duplicates(raw)

    def _detect_onnx(self, frame: np.ndarray) -> list[Detection]:
        cfg = self.cfg
        frame_h, frame_w = frame.shape[:2]

        canvas, ratio, pad_x, pad_y = letterbox(frame, cfg.input_size)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis]

        output = self.session.run(None, {self._input_name: blob})[0]
        rows = as_rows(output, cfg.person_class_id)
        if rows is None:
            return []

        boxes: list[Box] = []
        scores: list[float] = []
        for cx, cy, bw, bh, score in rows:
            if score < cfg.confidence:
                continue
            x1, y1, box_w, box_h = unletterbox_box(
                cx, cy, bw, bh, ratio, pad_x, pad_y
            )
            if not passes_sanity(box_w, box_h, frame_w, frame_h, cfg):
                continue
            boxes.append((x1, y1, box_w, box_h))
            scores.append(score)

        if not boxes:
            return []

        detections = [
            Detection(
                x1=max(0.0, x),
                y1=max(0.0, y),
                x2=min(float(frame_w), x + w),
                y2=min(float(frame_h), y + h),
                confidence=conf,
                class_id=cfg.person_class_id,
            )
            for (x, y, w, h), conf in zip(boxes, scores)
        ]
        return [detections[i] for i in _nms(boxes, scores, cfg.nms_iou)]

    def _detect_ultralytics(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            source=frame,
            conf=self.cfg.confidence,
            classes=[self.cfg.person_class_id],
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=float(box.conf[0]),
                        class_id=int(box.cls[0]),
                    )
                )
        return detections
