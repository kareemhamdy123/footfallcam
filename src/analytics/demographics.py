"""
Maps to brochure characteristic:
  6. Gender recognition -> OPTIONAL, OFF BY DEFAULT.

Important notes before enabling this module:
  - Automated gender/age classification from appearance is a sensitive
    biometric-adjacent feature. In many jurisdictions (e.g. EU/UK under
    GDPR, several US states) it triggers extra legal obligations: signage
    disclosing video analytics, a documented lawful basis, a privacy/DPIA
    assessment, and a defined retention policy.
  - These models are also well documented to have accuracy gaps across
    skin tones, ages, and gender-nonconforming appearances. Treat any
    output as a noisy, aggregate statistical signal for store analytics
    ("~45% of visitors this hour were classified female") - never as a
    factual label attached to an identifiable individual, and never persist
    it alongside anything that could re-identify a person.
  - This reference implementation does NOT ship a trained classifier. Wire
    in your own licensed/consented model (ONNX/TFLite) via
    `GenderClassifier.model_path`, and keep the output aggregated
    (see `summary()`), never per-frame face crops on disk.

If you don't need this feature, simply leave `enable_gender_recognition:
false` in the config (the default) and this module is never imported by
main.py.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


class GenderClassifier:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._session = None
        if model_path:
            try:
                import onnxruntime as ort  # optional dependency
                self._session = ort.InferenceSession(model_path)
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "onnxruntime is required for gender recognition. "
                    "Install it or leave this feature disabled."
                ) from exc

    def classify(self, face_crop: np.ndarray) -> str:
        """Returns 'female', 'male', or 'unknown'. Plug in your own
        preprocessing + inference here; left unimplemented in the reference
        repo since no model ships by default."""
        if self._session is None:
            return "unknown"
        raise NotImplementedError(
            "Wire up your ONNX model's input preprocessing/output mapping here."
        )


class DemographicsAggregator:
    """Keeps only running aggregate counts - never per-person records."""

    def __init__(self):
        self.counts: Counter = Counter()

    def add(self, label: str) -> None:
        self.counts[label] += 1

    def summary(self) -> dict:
        total = sum(self.counts.values())
        if total == 0:
            return {"total_classified": 0}
        return {
            "total_classified": total,
            **{k: round(100 * v / total, 1) for k, v in self.counts.items()},
        }
