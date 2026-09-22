"""
Video capture/writer helpers.

Maps to brochure characteristics:
  4.  Playback         -> annotated frames are written to an output video
  19. Night vision mode -> auto-detects low-light frames (via mean brightness)
                            and applies CLAHE + gamma correction before
                            detection, improving recall in dim scenes.
"""
from __future__ import annotations

import cv2
import numpy as np


def open_source(source: str) -> cv2.VideoCapture:
    """`source` may be a file path, an RTSP/HTTP URL, or a webcam index
    given as a string (e.g. "0")."""
    src: str | int = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {source}")
    return cap


def make_writer(path: str, fps: float, size: tuple) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, size)


def is_low_light(frame: np.ndarray, threshold: float = 70.0) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < threshold


def enhance_frame(frame: np.ndarray, mode: str = "auto") -> np.ndarray:
    """
    mode: "off" -> return frame unchanged
          "on"  -> always enhance
          "auto"-> enhance only if the frame is dim
    """
    if mode == "off":
        return frame
    if mode == "auto" and not is_low_light(frame):
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # mild gamma correction to lift shadows further
    gamma = 1.6
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(enhanced, table)
