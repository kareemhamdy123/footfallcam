"""
Maps to brochure characteristic:
  12. Staff exclusion -> the real FootfallCam device clips a small
                          non-emitting tag onto staff clothing. We emulate
                          that with a simple, fast colour-based check: staff
                          wear a tag/lanyard in a distinct, uncommon colour
                          (default: magenta/pink, configurable in the YAML),
                          and any track whose upper-body crop contains a
                          strong patch of that colour is flagged `is_staff`
                          and excluded from footfall/queue/occupancy counts.

For higher accuracy in production, replace `looks_like_staff_tag` with a
small classifier trained on your actual staff badge/uniform, or an
apparel re-identification model - the rest of the pipeline only needs the
boolean `track.is_staff` flag.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..detector import Detection


def looks_like_staff_tag(
    frame: np.ndarray,
    detection: Detection,
    hsv_lower: tuple,
    hsv_upper: tuple,
    min_pixel_fraction: float = 0.02,
) -> bool:
    x1, y1, x2, y2 = map(int, (detection.x1, detection.y1, detection.x2, detection.y2))
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return False

    # Look only at the upper third of the box (chest/shoulder area, where a
    # tag or lanyard would be clipped).
    upper_third_y2 = y1 + max(1, (y2 - y1) // 3)
    crop = frame[y1:upper_third_y2, x1:x2]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    fraction = float(mask.mean()) / 255.0
    return fraction >= min_pixel_fraction


def tag_staff(frame: np.ndarray, tracks, detections_by_track_id: dict,
              hsv_lower: tuple, hsv_upper: tuple) -> None:
    for track in tracks:
        det = detections_by_track_id.get(track.track_id)
        if det is None:
            continue
        if looks_like_staff_tag(frame, det, hsv_lower, hsv_upper):
            track.is_staff = True
