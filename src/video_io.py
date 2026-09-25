"""Video capture, writer and browser-transcode helpers.

One capture path serves every source (R5): a file path, a webcam index given as
a string ("0"), and an RTSP/HTTP URL all go through `cv2.VideoCapture`, so the
pipeline never learns where frames came from.

`enhance_frame` is a Phase 0 stub. The real low-light implementation is a
feature module owned by Phase 8 and is deliberately not imported here (R1).
"""
from __future__ import annotations

import os
import shutil
import subprocess

import cv2
import numpy as np

# Safety net for streams that report no frame rate. Not a deployment tunable -
# a real deployment sets its own frame rate on the writer.
FALLBACK_FPS = 25.0

H264_ARGS = (
    "-c:v", "libx264",
    "-preset", "fast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",   # required for playback in browsers
    "-movflags", "+faststart",  # moov atom first, so the file can stream
)


def open_source(source: str | int) -> cv2.VideoCapture:
    """Open a video file, a webcam index, or an RTSP/HTTP URL."""
    src: str | int = int(source) if str(source).isdigit() else source
    capture = cv2.VideoCapture(src)
    if not capture.isOpened():
        raise IOError(f"Could not open video source: {source}")
    return capture


def read_frame(capture: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    """Read one frame. Returns `(False, None)` at end of stream."""
    return capture.read()


def get_stream_info(capture: cv2.VideoCapture) -> dict:
    """Frame rate, geometry and length of an open capture."""
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return {
        "fps": fps if fps > 0 else FALLBACK_FPS,
        "raw_fps": fps,
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }


def make_writer(
    path: str, fps: float, size: tuple[int, int]
) -> cv2.VideoWriter:
    """Open an mp4 writer. `size` is `(width, height)`."""
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, float(fps)), size
    )
    if not writer.isOpened():
        raise IOError(f"Could not open video writer: {path}")
    return writer


def convert_to_h264_web(video_path: str) -> bool:
    """Transcode in place to browser-playable H.264 + yuv420p + faststart (R6).

    Returns False when ffmpeg is missing or the transcode fails, leaving the
    original file untouched.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not os.path.exists(video_path):
        return False

    temp_path = f"{video_path}.h264.tmp.mp4"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", video_path, *H264_ARGS, temp_path],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"[video_io] H.264 transcode failed: {exc}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
        os.replace(temp_path, video_path)
        return True
    return False


def enhance_frame(frame: np.ndarray, mode: str = "off") -> np.ndarray:
    """Pre-detection frame enhancement.

    Phase 0 stub: only "off" is implemented, returning the frame unchanged.
    Night-vision enhancement (LAB CLAHE + gamma) is Phase 8 and arrives as its
    own feature module.
    """
    return frame
