"""Detection hardening: preprocessing, filtering and dedup.

These are the Phase 1 guarantees for brochure characteristic 1 (Video Counting)
and 11 (Object clarification): a box that reaches the tracker is a real person,
in the right place, with the right size.

No model file and no video are required: the tested units are pure functions, so
the suite runs anywhere (R4).
"""
import numpy as np
import pytest

from src.config import DetectionSettings
from src.detection.dataclasses import Detection
from src.detection.model import (
    as_rows,
    letterbox,
    passes_sanity,
    suppress_part_duplicates,
    unletterbox_box,
)


# ------------------------------------------------------------------- letterbox


@pytest.mark.parametrize(
    "frame_w, frame_h",
    [(640, 360), (360, 640), (634, 360), (1920, 1080)],
)
def test_letterbox_produces_a_square_canvas_and_preserves_aspect(frame_w, frame_h):
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    canvas, ratio, pad_x, pad_y = letterbox(frame, size=640)

    assert canvas.shape == (640, 640, 3)
    assert ratio == pytest.approx(640 / max(frame_w, frame_h))
    # Aspect preserved: the painted content is exactly the scaled frame.
    assert 640 - 2 * pad_x == pytest.approx(frame_w * ratio, abs=1)
    assert 640 - 2 * pad_y == pytest.approx(frame_h * ratio, abs=1)


def test_letterbox_of_a_square_frame_needs_no_padding():
    canvas, ratio, pad_x, pad_y = letterbox(np.zeros((100, 100, 3), np.uint8), size=640)
    assert canvas.shape == (640, 640, 3)
    assert ratio == pytest.approx(6.4)
    assert (pad_x, pad_y) == (0.0, 0.0)


def test_letterbox_roundtrips_a_box_back_to_source_pixels():
    """The whole point of letterboxing: coordinates must survive the trip."""
    frame_w, frame_h = 634, 360
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    # A box expressed in source pixels.
    src_x1, src_y1, src_x2, src_y2 = 100.0, 50.0, 180.0, 300.0
    src_cx, src_cy = (src_x1 + src_x2) / 2, (src_y1 + src_y2) / 2
    src_bw, src_bh = src_x2 - src_x1, src_y2 - src_y1

    _, ratio, pad_x, pad_y = letterbox(frame, size=640)

    # Source -> canvas, by inverting the forward mapping.
    c_cx = src_cx * ratio + pad_x
    c_cy = src_cy * ratio + pad_y
    c_bw, c_bh = src_bw * ratio, src_bh * ratio

    # Canvas -> source, via the function under test.
    x1, y1, bw, bh = unletterbox_box(c_cx, c_cy, c_bw, c_bh, ratio, pad_x, pad_y)

    assert (x1, y1) == pytest.approx((src_x1, src_y1))
    assert (x1 + bw, y1 + bh) == pytest.approx((src_x2, src_y2))


def test_unletterbox_returns_the_top_left_corner_and_size():
    """unletterbox_box emits (x1, y1, w, h); the centre form is its input."""
    frame = np.zeros((360, 634, 3), dtype=np.uint8)
    _, ratio, pad_x, pad_y = letterbox(frame, size=640)
    cx, cy, bw, bh = 100.0, 50.0, 80.0, 250.0  # centre form, source pixels

    canvas_box = (cx * ratio + pad_x, cy * ratio + pad_y, bw * ratio, bh * ratio)
    x1, y1, out_w, out_h = unletterbox_box(*canvas_box, ratio, pad_x, pad_y)

    assert (x1, y1) == pytest.approx((cx - bw / 2, cy - bh / 2))
    assert (out_w, out_h) == pytest.approx((bw, bh))


# ------------------------------------------------------------- sanity filters


def test_sanity_rejects_boxes_below_the_minimum_size():
    cfg = DetectionSettings()
    # 11x15 px: narrower and shorter than the configured 12x16 minimum.
    assert passes_sanity(11.0, 15.0, 640, 360, cfg) is False
    # Exactly at the minimum is accepted.
    assert passes_sanity(12.0, 16.0, 640, 360, cfg) is True


def test_sanity_rejects_flat_strips_and_thin_poles():
    cfg = DetectionSettings()
    # A counter edge: very wide, very short.
    assert passes_sanity(200.0, 30.0, 640, 360, cfg) is False
    # A pole or shelf upright: much taller than wide, below the aspect floor.
    assert passes_sanity(20.0, 400.0, 640, 360, cfg) is False
    # A normal standing person passes.
    assert passes_sanity(40.0, 120.0, 640, 360, cfg) is True


def test_sanity_rejects_a_box_filling_almost_the_whole_frame():
    cfg = DetectionSettings()  # max_frame_fraction 0.95 -> 608 x 342 on 640x360
    assert passes_sanity(620.0, 350.0, 640, 360, cfg) is False
    # Wide enough to break the width bound but too short to break the height
    # bound, so the oversize rule (which needs both) does not fire.
    assert passes_sanity(620.0, 300.0, 640, 360, cfg) is True
    # Comfortably inside the frame is fine.
    assert passes_sanity(400.0, 300.0, 640, 360, cfg) is True


def test_sanity_thresholds_come_from_config_not_hardcoded():
    """R3: a config change must change the filter's behaviour."""
    loose = DetectionSettings(min_box_width_px=1.0, min_box_height_px=1.0,
                              min_height_over_width=0.0, min_width_over_height=0.0)
    tight = DetectionSettings()
    assert passes_sanity(2.0, 2.0, 640, 360, loose) is True
    assert passes_sanity(2.0, 2.0, 640, 360, tight) is False


# ------------------------------------------------------------ output handling


def test_as_rows_accepts_both_yolo_output_layouts():
    """YOLOv8 exports (1, 4+nc, anchors); some toolchains emit it transposed.

    The layout is inferred from the axis lengths, so the tensor uses realistic
    proportions - anchors always outnumber the 4+nc attribute columns in a real
    export, and that is the assumption the inference relies on.
    """
    anchors, attributes = 100, 84  # 4 box values + 80 COCO classes
    canonical = np.zeros((1, attributes, anchors), dtype=np.float32)
    canonical[0, 0, :] = np.arange(anchors)          # cx
    canonical[0, 1, :] = np.arange(anchors) + 1000   # cy
    canonical[0, 2, :] = np.arange(anchors) + 2000   # bw
    canonical[0, 3, :] = np.arange(anchors) + 3000   # bh
    canonical[0, 4, :] = np.linspace(0.9, 0.5, anchors)  # person score

    upright = as_rows(canonical, 0)
    transposed = as_rows(canonical.transpose(0, 2, 1), 0)

    assert upright is not None
    assert len(upright) == anchors
    assert upright == transposed, "both layouts must normalise identically"
    assert upright[0] == pytest.approx((0.0, 1000.0, 2000.0, 3000.0, 0.9))
    assert upright[-1] == pytest.approx(
        (99.0, 1099.0, 2099.0, 3099.0, 0.5), abs=1e-3
    )


def test_as_rows_reads_the_requested_class_column():
    """A non-zero person_class_id must select that class's score column."""
    anchors, attributes = 100, 84
    tensor = np.zeros((1, attributes, anchors), dtype=np.float32)
    tensor[0, 4, :] = 0.10  # class 0
    tensor[0, 5, :] = 0.77  # class 1

    assert as_rows(tensor, 0)[0][4] == pytest.approx(0.10)
    assert as_rows(tensor, 1)[0][4] == pytest.approx(0.77)


def test_as_rows_returns_none_on_an_unusable_tensor():
    assert as_rows(np.zeros((1,), dtype=np.float32), 0) is None


# ----------------------------------------------------- partial-body dedup


def test_suppress_part_duplicates_drops_a_torso_inside_a_full_body():
    full_body = Detection(x1=100, y1=50, x2=200, y2=350, confidence=0.90)
    torso = Detection(x1=110, y1=150, x2=190, y2=300, confidence=0.70)

    kept = suppress_part_duplicates([torso, full_body])

    assert len(kept) == 1
    assert kept[0] is full_body, "the larger, more confident box must win"


def test_suppress_part_duplicates_keeps_distinct_people():
    left = Detection(x1=10, y1=50, x2=60, y2=300, confidence=0.9)
    right = Detection(x1=400, y1=60, x2=450, y2=310, confidence=0.9)

    assert len(suppress_part_duplicates([left, right])) == 2


def test_suppress_part_duplicates_respects_its_thresholds():
    """The overlap thresholds are parameters, not baked-in constants."""
    a = Detection(x1=0, y1=0, x2=100, y2=100, confidence=0.9)
    b = Detection(x1=0, y1=0, x2=100, y2=100, confidence=0.5)

    # Default thresholds: a fully contained duplicate is suppressed.
    assert len(suppress_part_duplicates([a, b])) == 1
    # Unreachable thresholds: nothing is treated as a duplicate.
    assert len(suppress_part_duplicates([a, b], 1.1, 1.1)) == 2


def test_suppress_part_duplicates_handles_trivial_input():
    single = Detection(x1=0, y1=0, x2=10, y2=10, confidence=0.5)
    assert suppress_part_duplicates([]) == []
    assert suppress_part_duplicates([single]) == [single]
