"""Detection layer: raw person boxes for a single frame.

No tracking, no geometry, no features (AGENTS.md R1).
"""
from .dataclasses import Detection
from .model import PersonDetector

__all__ = ["Detection", "PersonDetector"]
