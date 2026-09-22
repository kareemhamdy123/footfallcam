import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Line
from src.zones import crossed_line, point_in_polygon


def test_point_in_polygon_inside():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), square) is True


def test_point_in_polygon_outside():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((15, 15), square) is False


def test_crossed_line_top_to_bottom_in():
    line = Line(name="entrance", p1=(0, 50), p2=(100, 50), in_direction="top_to_bottom")
    result = crossed_line((50, 40), (50, 60), line)
    assert result == "in"


def test_crossed_line_bottom_to_top_is_out():
    line = Line(name="entrance", p1=(0, 50), p2=(100, 50), in_direction="top_to_bottom")
    result = crossed_line((50, 60), (50, 40), line)
    assert result == "out"


def test_no_crossing_when_staying_on_one_side():
    line = Line(name="entrance", p1=(0, 50), p2=(100, 50), in_direction="top_to_bottom")
    result = crossed_line((50, 10), (50, 20), line)
    assert result is None


def test_left_to_right_direction():
    line = Line(name="gate", p1=(50, 0), p2=(50, 100), in_direction="left_to_right")
    result = crossed_line((40, 50), (60, 50), line)
    assert result == "in"
