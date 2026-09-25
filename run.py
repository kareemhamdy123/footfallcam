"""CLI entry point.

    python run.py --video data/sample.mp4 [--show] [--max-frames N]

Examples:
    python run.py
    python run.py --video data/clip.mp4 --max-frames 300
    python run.py --video 0 --show           # webcam index, live preview
    python run.py --video rtsp://cam/stream  # network camera
"""
from __future__ import annotations

import argparse

from src.pipeline import Pipeline

DEFAULT_CONFIG = "configs/default.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="FootfallCam people counting over a video file or a camera.",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="path to the YAML config"
    )
    parser.add_argument(
        "--video",
        default=None,
        help="override the config source: file path, webcam index, or RTSP/HTTP URL",
    )
    parser.add_argument(
        "--show", action="store_true", help="open a live preview window (q quits)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="stop after N frames"
    )
    args = parser.parse_args()

    Pipeline.run(
        config_path=args.config,
        show=args.show,
        max_frames=args.max_frames,
        source_override=args.video,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
