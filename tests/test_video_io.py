"""Video I/O: capture paths, stream metadata and the browser transcode.

R5 requires one capture path for files, webcam indexes and network URLs, so the
pipeline never learns where frames came from. R6 requires the annotated output
to be browser-playable.

The file-backed tests need `data/sample.mp4`, which is gitignored, so they skip
cleanly on a fresh clone instead of failing.
"""
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from src.video_io import (
    convert_to_h264_web,
    enhance_frame,
    get_stream_info,
    make_writer,
    normalise_source,
    open_source,
    read_frame,
)

SAMPLE = "data/sample.mp4"
needs_video = pytest.mark.skipif(
    not shutil.os.path.exists(SAMPLE), reason="data/sample.mp4 not present"
)
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


# ----------------------------------------------------------------- open_source


@needs_video
def test_open_source_reads_a_video_file():
    capture = open_source(SAMPLE)
    try:
        assert capture.isOpened()
        ok, frame = read_frame(capture)
        assert ok and frame is not None
        assert frame.ndim == 3 and frame.shape[2] == 3
    finally:
        capture.release()


@needs_video
def test_open_source_raises_on_a_missing_file():
    with pytest.raises(IOError):
        open_source("data/definitely_not_here.mp4")


def test_open_source_treats_a_numeric_string_as_a_camera_index():
    """R5: "0" is a webcam index, not a filename."""
    capture = cv2.VideoCapture(0)
    opened = capture.isOpened()
    capture.release()

    if not opened:
        pytest.skip("no camera available on this machine")

    # The same cv2.VideoCapture path is used, so this must succeed.
    capture = open_source("0")
    try:
        assert capture.isOpened()
    finally:
        capture.release()


@pytest.mark.parametrize(
    "source, expected",
    [
        ("0", 0),          # webcam index
        ("2", 2),
        (1, 1),            # an int is already an index
        ("data/sample.mp4", "data/sample.mp4"),
        ("clip.mov", "clip.mov"),
        ("rtsp://cam/stream", "rtsp://cam/stream"),
        ("http://host/v.mp4", "http://host/v.mp4"),
        ("", ""),
    ],
)
def test_normalise_source_only_converts_numeric_strings(source, expected):
    """R5: indices become ints; paths and URLs must survive untouched.

    Tested as a pure function - opening a real RTSP URL would block on a TCP
    timeout, which says nothing about the argument handling.
    """
    result = normalise_source(source)
    assert result == expected
    assert type(result) is type(expected)


# --------------------------------------------------------------- stream info


@needs_video
def test_get_stream_info_reports_fps_and_size():
    capture = open_source(SAMPLE)
    try:
        info = get_stream_info(capture)
    finally:
        capture.release()

    assert info["width"] > 0
    assert info["height"] > 0
    assert info["fps"] > 0
    assert info["frame_count"] > 0
    assert (info["width"], info["height"]) == (634, 360)
    assert info["fps"] == pytest.approx(13.093, abs=0.01)


class _NoFpsCapture:
    """Stand-in for a capture that reports no frame rate, as some webcams do."""

    def __init__(self, fps, frame_count, width, height):
        self._values = {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_COUNT: frame_count,
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
        }

    def get(self, prop):
        return self._values[prop]


@pytest.mark.parametrize("reported", [0.0, -1.0, float("nan")])
def test_get_stream_info_falls_back_when_no_frame_rate_is_reported(reported):
    """A zero frame rate must not reach the writer and divide by zero later."""
    from src.video_io import FALLBACK_FPS

    info = get_stream_info(_NoFpsCapture(reported, 100, 640, 360))

    assert info["fps"] == FALLBACK_FPS
    assert info["raw_fps"] == pytest.approx(reported, nan_ok=True)
    assert (info["width"], info["height"], info["frame_count"]) == (640, 360, 100)


@needs_video
def test_get_stream_info_reports_the_real_frame_rate():
    capture = open_source(SAMPLE)
    try:
        info = get_stream_info(capture)
    finally:
        capture.release()

    assert info["fps"] == pytest.approx(13.093, abs=0.01)
    assert info["raw_fps"] > 0


# ------------------------------------------------------------------- writer


@needs_video
def test_make_writer_produces_a_readable_file(tmp_path):
    path = str(tmp_path / "out.mp4")
    writer = make_writer(path, fps=30.0, size=(320, 240))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    try:
        for _ in range(10):
            writer.write(frame)
    finally:
        writer.release()

    capture = cv2.VideoCapture(path)
    try:
        assert capture.isOpened()
        written = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    assert written == 10


def test_make_writer_raises_on_an_unwritable_path(tmp_path):
    with pytest.raises(IOError):
        make_writer(str(tmp_path / "no" / "such" / "dir" / "out.mp4"), 30.0, (320, 240))


# --------------------------------------------------------- h.264 transcode


@needs_video
@needs_ffmpeg
def test_convert_to_h264_web_yields_browser_playable_h264(tmp_path):
    """R6: H.264 + yuv420p so the artifact plays in a browser."""
    path = str(tmp_path / "annotated.mp4")
    writer = make_writer(path, fps=15.0, size=(320, 240))
    frame = np.full((240, 320, 3), 127, dtype=np.uint8)
    try:
        for _ in range(15):
            writer.write(frame)
    finally:
        writer.release()

    assert convert_to_h264_web(path) is True

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt",
            "-of", "default=noprint_wrappers=1", path,
        ],
        capture_output=True, text=True, check=True,
    )
    details = dict(
        line.split("=") for line in probe.stdout.strip().splitlines() if "=" in line
    )
    assert details["codec_name"] == "h264"
    assert details["pix_fmt"] == "yuv420p"


def test_convert_to_h264_web_returns_false_for_a_missing_file(tmp_path):
    assert convert_to_h264_web(str(tmp_path / "nope.mp4")) is False


# ------------------------------------------------------------- enhancement


def test_enhance_frame_is_a_passthrough_stub():
    """Phase 0 stub; the real implementation is the Phase 8 feature module."""
    frame = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
    assert enhance_frame(frame) is frame
    assert enhance_frame(frame, "off") is frame


def test_read_frame_signals_end_of_stream(tmp_path):
    path = str(tmp_path / "tiny.mp4")
    writer = make_writer(path, fps=10.0, size=(64, 48))
    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    capture = open_source(path)
    try:
        assert read_frame(capture)[0] is True
        assert read_frame(capture)[0] is False, "must report exhaustion"
    finally:
        capture.release()
