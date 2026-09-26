"""Feature 6: Gender recognition.

Per-track classification plus the aggregate demographics that a retailer would
ask for. Brochure characteristic 6 ("Gender recognition").

READ THIS BEFORE ENABLING IT
---------------------------
Inferring gender from a low-resolution crop of a stranger is a weak, error-prone
inference with real consequences when it is wrong. The model-less fallback here
is a *multi-cue guess*, not evidence, and it is deliberately biased towards
answering "unknown" rather than guessing. That is why `demographics.enabled`
defaults to `false` in `configs/default.yaml`.

If you turn this on: supply a real trained model, make sure you have a lawful
basis for the processing, and validate accuracy on your own floor. Demographic
models degrade badly on under-represented groups, and a shop-floor camera sees
people at angles, distances and lighting conditions the training set never had.
An incorrect label attached to a tracked individual is a privacy problem, not
just an accuracy problem.

Backend order is PyTorch -> ONNX -> heuristic. The heuristic is the only one
that works out of the box, and it is the weakest of the three.

Why the fallback is set to abstain
----------------------------------
Measured on the sample store footage (3542 person crops, 634x360, retail
interior), sweeping the confidence threshold:

    threshold   male   female   unknown
        0.00    2999       543         0
        0.20    2344        20      1178
        0.40     175         0      3367
        0.60       0         0      3542

Two things to read out of that table. First, nothing is ever confidently
"female" - the female side is eliminated first as the threshold rises, because
the saturation cue votes female below its midpoint and most shop-floor crops sit
below it. Second, at the permissive end the split is roughly 85% male, which is
not a measurement of anything; it is the heuristic's own bias showing through.

So lowering the threshold does not reveal signal, it reveals bias. At the shipped
default of 0.60 the fallback abstains on every real crop, which is the only
defensible outcome for a guess about a stranger's gender made from a ~90x57
pixel crop. If you need demographics numbers, supply a trained model. Do not
"fix" this by rebalancing the midpoints below - that just moves the bias.
"""
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only (R1)
    from src.tracking.dataclasses import Track

UNKNOWN = "unknown"
LABELS = (UNKNOWN, "male", "female")

# A crop smaller than this carries too few pixels to say anything.
MIN_CROP_EDGE = 16

# Cue weights. Deliberately small and balanced: no single cue may dominate,
# because none of them is reliable on its own.
EDGE_WEIGHT = 0.34
SATURATION_WEIGHT = 0.33
ASPECT_WEIGHT = 0.33

# Each cue is squashed around these midpoints before being turned into a vote.
EDGE_MIDPOINT = 55.0
SATURATION_MIDPOINT = 90.0
ASPECT_MIDPOINT = 2.6


class GenderClassifier:
    """Classify one person crop into male / female / unknown."""

    def __init__(
        self,
        model_path: str | None = None,
        min_confidence: float = 0.60,
        torch_model=None,
    ):
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.backend = "heuristic"
        self._torch_model = torch_model
        self._onnx_session = None
        self._onnx_input = None

        if torch_model is not None:
            self.backend = "pytorch"
        elif model_path and model_path.endswith(".onnx") and os.path.exists(model_path):
            self._init_onnx(model_path)

    def _init_onnx(self, model_path: str) -> bool:
        try:
            import onnxruntime as ort

            self._onnx_session = ort.InferenceSession(
                model_path, providers=ort.get_available_providers()
            )
            self._onnx_input = self._onnx_session.get_inputs()[0].name
            inp_shape = self._onnx_session.get_inputs()[0].shape
            if len(inp_shape) >= 4 and isinstance(inp_shape[2], int) and isinstance(inp_shape[3], int):
                self._input_size = (inp_shape[3], inp_shape[2])
            else:
                self._input_size = (224, 224)
            self.backend = "onnx"
            return True
        except Exception as exc:  # pragma: no cover - environment specific
            print(f"[gender] ONNX session unavailable ({exc}); using the heuristic")
            self._onnx_session = None
            return False

    def classify(self, crop: np.ndarray | None) -> tuple[str, float]:
        """Return `(label, confidence)`. Label is one of `LABELS`.

        A crop too small to carry signal, or cues that disagree, both return
        `("unknown", confidence)` rather than a guess.
        """
        if not _is_usable(crop):
            return (UNKNOWN, 0.0)

        if self.backend == "pytorch":
            return self._classify_pytorch(crop)
        if self.backend == "onnx":
            return self._classify_onnx(crop)
        return self._classify_heuristic(crop)

    # ------------------------------------------------------------- backends

    def _classify_pytorch(self, crop: np.ndarray) -> tuple[str, float]:
        label, confidence = self._torch_model(crop)
        return _normalise(label, confidence)

    def _classify_onnx(self, crop: np.ndarray) -> tuple[str, float]:
        size = getattr(self, "_input_size", (224, 224))[0]
        blob = _preprocess_crop_for_onnx(crop, size=size)
        output = self._onnx_session.run(None, {self._onnx_input: blob})[0]
        scores = np.asarray(output).reshape(-1)
        index = int(scores.argmax())
        label, confidence = _from_scores(scores, index)
        if confidence < self.min_confidence:
            return (UNKNOWN, confidence)
        return (label, confidence)

    def _classify_heuristic(self, crop: np.ndarray) -> tuple[str, float]:
        cues = extract_cues(crop)
        return decide(cues, self.min_confidence)


def _is_usable(crop: np.ndarray | None) -> bool:
    if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
        return False
    if crop.ndim not in (2, 3):
        return False
    height, width = crop.shape[:2]
    return height >= MIN_CROP_EDGE and width >= MIN_CROP_EDGE


def extract_cues(crop: np.ndarray) -> dict[str, float]:
    """The three appearance cues the heuristic votes on.

    Split out as a pure function so the decision logic can be tested against
    known cue values without synthesising pixels.
    """
    bgr = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    height, width = bgr.shape[:2]

    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = float(np.mean(np.sqrt(grad_x**2 + grad_y**2)))

    saturation = float(np.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]))
    aspect = height / max(1, width)

    return {
        "edge_energy": edge_energy,
        "saturation": saturation,
        "aspect": aspect,
    }


def decide(cues: dict[str, float], min_confidence: float = 0.60) -> tuple[str, float]:
    """Turn cues into a label, or into `unknown` when they do not agree.

    Each cue votes for one label. The winner must beat the other by enough to
    clear `min_confidence`; below that the answer is `unknown`, because a
    low-confidence guess about a person is worse than no answer.
    """
    votes = {UNKNOWN: 0.0, "male": 0.0, "female": 0.0}

    edge = _squash(cues["edge_energy"], EDGE_MIDPOINT)
    saturation = _squash(cues["saturation"], SATURATION_MIDPOINT)
    aspect = _squash(cues["aspect"], ASPECT_MIDPOINT)

    for weight, cue in (
        (EDGE_WEIGHT, edge),
        (SATURATION_WEIGHT, saturation),
        (ASPECT_WEIGHT, aspect),
    ):
        # A cue sitting exactly on its midpoint is a coin flip: it votes for
        # neither label rather than arbitrarily picking one.
        magnitude = abs(cue)
        if magnitude < 1e-9:
            continue
        votes["male" if cue > 0 else "female"] += weight * magnitude

    male, female = votes["male"], votes["female"]
    if male == female:
        return (UNKNOWN, 0.0)

    label = "male" if male > female else "female"
    confidence = abs(male - female)
    if confidence < min_confidence:
        return (UNKNOWN, confidence)
    return (label, min(1.0, confidence))


def _squash(value: float, midpoint: float) -> float:
    """Map a raw cue to (-1, 1): -1 below the midpoint, +1 above it."""
    return math.tanh((value - midpoint) / max(1e-6, abs(midpoint)))


def _preprocess_crop_for_onnx(crop: np.ndarray, size: int = 224) -> np.ndarray:
    """Preprocess crop preserving aspect ratio then center-cropping to (size, size)."""
    h, w = crop.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((1, 3, size, size), dtype=np.float32)
    if h < w:
        new_h = size
        new_w = max(size, int(round(w * size / h)))
    else:
        new_w = size
        new_h = max(size, int(round(h * size / w)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    y0 = max(0, (new_h - size) // 2)
    x0 = max(0, (new_w - size) // 2)
    cropped = resized[y0:y0 + size, x0:x0 + size]
    if cropped.shape[:2] != (size, size):
        cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)
    if cropped.ndim == 2:
        rgb = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    blob = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis]
    return blob


def _from_scores(scores: np.ndarray, index: int) -> tuple[str, float]:
    """Map a model's class scores onto our labels using softmax."""
    scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    exp_scores = np.exp(scores_arr - np.max(scores_arr))
    total = float(np.sum(exp_scores))
    probabilities = exp_scores / total if total > 0 else scores_arr
    if len(scores_arr) == 2:
        # Standard binary classification: 0 = female, 1 = male (e.g. YOLOv8-cls, ImageNet-gender)
        binary_labels = ("female", "male")
        label = binary_labels[index] if index < len(binary_labels) else UNKNOWN
    elif index < len(LABELS):
        label = LABELS[index]
    else:
        label = UNKNOWN
    return (label, float(probabilities[index]))


def _normalise(label: str, confidence: float) -> tuple[str, float]:
    label = str(label).strip().lower()
    if label not in LABELS:
        return (UNKNOWN, 0.0)
    return (label, max(0.0, min(1.0, float(confidence))))


class DemographicsAggregator:
    """Aggregate classified tracks into the percentages a retailer asks for."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {label: 0 for label in LABELS}
        self.sales_conversations: dict[str, int] = {label: 0 for label in LABELS}
        self.track_labels: dict[int, str] = {}

    @property
    def seen_track_ids(self) -> set[int]:
        return set(self.track_labels.keys())

    def record_track(self, track: "Track") -> str | None:
        """Record one track's gender. A track is only ever counted once.

        Returns the label recorded, or None when the track has no gender yet.
        """
        gender = getattr(track, "gender", None)
        if gender not in LABELS or gender == UNKNOWN:
            return None
        track_id = getattr(track, "track_id", None)
        if track_id is not None:
            prev = self.track_labels.get(track_id)
            if prev == gender:
                return None  # idempotent per person
            if prev is not None:
                self.counts[prev] = max(0, self.counts[prev] - 1)
            self.track_labels[track_id] = gender
        self.counts[gender] += 1
        return gender

    def record_sales_conversation(self, gender: str) -> str | None:
        """Record that a sales conversation happened with someone of `gender`.

        The aggregator is only the sink here. Detecting the conversation itself
        is brochure characteristic 10, which arrives in Phase 10 and calls this.
        """
        gender = str(gender).strip().lower()
        if gender not in LABELS:
            return None
        self.sales_conversations[gender] += 1
        return gender

    def summary(self) -> dict:
        male = self.counts["male"]
        female = self.counts["female"]
        # Percentages describe the people we actually classified. "unknown" is
        # reported separately rather than folded in as a third share.
        total_classified = male + female
        return {
            "male_percentage": _percentage(male, total_classified),
            "female_percentage": _percentage(female, total_classified),
            "total_classified": total_classified,
            "counts": dict(self.counts),
            "sales_conversations": dict(self.sales_conversations),
        }

    def reset(self) -> None:
        for label in LABELS:
            self.counts[label] = 0
            self.sales_conversations[label] = 0
        self.track_labels.clear()


def _percentage(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0
