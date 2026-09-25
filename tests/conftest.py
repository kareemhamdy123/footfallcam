"""Shared factory helpers for the test suite.

Logic tests must never need a video file (R4): these factories build the
`Track` and `Detection` value objects directly, so geometry and counting can be
exercised in isolation.
"""
import pytest

from src.config import TrackingSettings
from src.detection.dataclasses import Detection
from src.tracking.dataclasses import Track


@pytest.fixture
def tracking_cfg() -> TrackingSettings:
    """Default tracking settings, for tracker unit tests."""
    return TrackingSettings()


def make_detection(
    x1: float = 0.0,
    y1: float = 0.0,
    x2: float = 20.0,
    y2: float = 40.0,
    conf: float = 0.9,
    class_id: int = 0,
) -> Detection:
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_id=class_id)


def make_track(
    tid: int,
    x: float = 50.0,
    y: float = 50.0,
    is_staff: bool = False,
    width: float = 20.0,
    height: float = 40.0,
) -> Track:
    """A confirmed track whose foot point sits at (x, y).

    The foot point is the bottom-centre of the box, so the box is placed above
    (x, y) - that is the point every spatial feature measures from.
    """
    track = Track(
        track_id=tid,
        detection=make_detection(
            x1=x - width / 2.0,
            y1=y - height,
            x2=x + width / 2.0,
            y2=y,
            conf=0.9,
        ),
        hits=2,
        is_confirmed=True,
    )
    track.is_staff = is_staff
    track.history = [(0, x, y)]
    return track
