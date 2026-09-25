"""Tracking layer: persistent identities across frames.

No geometry, no features (AGENTS.md R1). A `Track` carries every attribute the
later phases need, so those fields are not bolted on retroactively.
"""
from .dataclasses import Track
from .tracker import SimpleTracker, iou

__all__ = ["SimpleTracker", "Track", "iou"]
