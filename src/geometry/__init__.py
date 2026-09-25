"""Geometry layer: the spatial predicates every feature reuses.

Pure functions over points and segments - no OpenCV, no NumPy, no I/O, so they
are cheap to unit test and cheap to reuse. Nothing here imports from
`features/` (AGENTS.md R1).
"""
from .zones import Band, auto_bands, crossed_line, point_in_polygon, zones_containing

__all__ = [
    "Band",
    "auto_bands",
    "crossed_line",
    "point_in_polygon",
    "zones_containing",
]
