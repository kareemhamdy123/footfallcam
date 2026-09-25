"""R7: VideoCounter delegates, it does not reimplement.

Delegation is asserted structurally rather than by reading the source: the
counter's own delegates are replaced with stubs, and the counter's output is
required to follow the stub. If the counter ever grew its own private tally or
its own clustering, these tests would fail.
"""
from features.counter import VideoCounter
from features.group_counting import GroupCounter
from features.multiple_counting_line import MultipleCountingLineManager
from src.config import Line
from tests.conftest import make_track

DOOR = Line("door", (32, 180), (608, 180), "top_to_bottom")


class StubLineManager:
    """Stands in for MultipleCountingLineManager with a known tallies."""

    def __init__(self, lines, **kwargs):
        self.lines = list(lines)
        self.counts = {"door": {"in": 7, "out": 3}}
        self.events = [{"line": "door", "direction": "in", "t": 0.0}]
        self.update_calls = 0
        self.last_seen_tracks = None

    def update(self, tracks, frame_idx, timestamp_s):
        self.update_calls += 1
        self.last_seen_tracks = list(tracks)
        return [{"line": "door", "direction": "in", "t": timestamp_s}]

    def summary(self):
        return {
            "per_line": self.counts,
            "total_in": 7,
            "total_out": 3,
            "net_inside": 4,
            "events_count": 1,
        }

    def reset(self):
        self.counts = {"door": {"in": 0, "out": 0}}
        self.events.clear()


class StubGroupCounter:
    def __init__(self):
        self.co_movement_seconds = {(1, 2): 9.0}
        self.summary_calls = 0
        self.cluster_calls = 0
        self.co_movement_calls = 0

    def frame_delta(self, timestamp_s):
        return 0.0

    def update_co_movement(self, tracks, timestamp_s, dt_s=0.0):
        self.co_movement_calls += 1

    def cluster_groups(self, in_events, window_s=None):
        self.cluster_calls += 1
        return [[{"track_id": 1}, {"track_id": 2}]]

    def summary(self, in_events=()):
        self.summary_calls += 1
        return {"groups": 1, "average_group_size": 2.0, "sentinel": True}

    def reset(self):
        self.co_movement_seconds = {}


# -------------------------------------------------------- construction


def test_counter_composes_both_delegates():
    c = VideoCounter(lines=[DOOR], group_max_distance_px=55.0)

    assert isinstance(c.line_manager, MultipleCountingLineManager)
    assert isinstance(c.group_counter, GroupCounter)
    assert c.group_counter.group_max_distance_px == 55.0


def test_counts_is_the_managers_tally_not_a_copy():
    """One tally, one source of truth - a copy would silently diverge."""
    c = VideoCounter(lines=[DOOR])

    assert c.counts is c.line_manager.counts
    c.line_manager.counts["door"]["in"] += 5
    assert c.counts["door"]["in"] == 5


def test_configured_lines_reach_the_manager():
    c = VideoCounter(lines=[DOOR, Line("side", (100, 0), (100, 360))])
    assert [line.name for line in c.line_manager.lines] == ["door", "side"]


# ------------------------------------------------------------ delegation


def test_totals_come_from_the_line_manager():
    c = VideoCounter(lines=[DOOR])
    c.line_manager = StubLineManager([DOOR])

    summary = c.summary()

    assert summary["total_in"] == 7
    assert summary["total_out"] == 3
    assert summary["net_inside"] == 4


def test_group_stats_come_from_the_group_counter():
    c = VideoCounter(lines=[DOOR])
    c.group_counter = StubGroupCounter()

    assert c.summary()["group_stats"]["sentinel"] is True


def test_update_calls_the_delegates_on_every_frame():
    c = VideoCounter(lines=[DOOR])
    c.line_manager = StubLineManager([DOOR])
    c.group_counter = StubGroupCounter()

    c.update([make_track(1)], 0, 0.0)
    c.update([make_track(1)], 1, 0.5)

    assert c.line_manager.update_calls == 2
    assert c.group_counter.co_movement_calls == 2


def test_co_movement_seconds_are_the_delegates_property():
    c = VideoCounter(lines=[DOOR])
    c.group_counter = StubGroupCounter()
    assert c.co_movement_seconds == {(1, 2): 9.0}


def test_grouped_entries_delegate_to_the_group_counter():
    c = VideoCounter(lines=[DOOR])
    c.group_counter = StubGroupCounter()

    assert len(c.grouped_entries()) == 1
    assert c.group_counter.cluster_calls >= 1


def test_reset_is_forwarded_to_both_delegates():
    c = VideoCounter(lines=[DOOR])
    c.line_manager = StubLineManager([DOOR])
    c.group_counter = StubGroupCounter()
    c.counts["door"]["in"] = 99

    c.reset()

    assert c.counts["door"]["in"] == 0
    assert c.group_counter.co_movement_seconds == {}


# -------------------------------------------------------------- policy


def test_staff_filtering_is_the_counters_policy_not_the_managers():
    """The counter decides who counts; the manager counts what it is given."""
    c = VideoCounter(lines=[DOOR])
    c.line_manager = StubLineManager([DOOR])

    c.update([make_track(1, is_staff=True), make_track(2)], 0, 0.0)

    forwarded = c.line_manager.last_seen_tracks
    assert [t.track_id for t in forwarded] == [2]
    assert all(not t.is_staff for t in forwarded)


def test_real_manager_would_have_counted_the_staff_track():
    """Confirms the filter lives above the manager, not inside it."""
    mgr = MultipleCountingLineManager([DOOR])
    mgr.update([make_track(1, x=300, y=100, is_staff=True)], 0, 0.0)
    mgr.update([make_track(1, x=300, y=250, is_staff=True)], 1, 0.0)
    assert mgr.counts["door"]["in"] == 1


# ------------------------------------------------------- no duplication


def test_counter_module_defines_no_crossing_or_clustering_logic():
    """The owned modules hold the maths; counter.py must not re-derive it."""
    import ast
    import inspect

    from features import counter as counter_module

    tree = ast.parse(inspect.getsource(counter_module))

    # It must not reach for the geometry primitive at runtime.
    imported = {
        alias.name.split(".")[-1]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "crossed_line" not in imported
    assert "point_in_polygon" not in imported

    # And it must not define a crossing test or clustering maths of its own.
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in defined:
        assert "cross" not in name.lower(), f"counter.py defines {name}"
        assert "cluster" not in name.lower(), f"counter.py defines {name}"
        assert "iou" not in name.lower(), f"counter.py defines {name}"

    # The two owners are the modules that do define that work.
    from features import group_counting, multiple_counting_line

    assert "crossed_line" in inspect.getsource(multiple_counting_line)
    assert "cluster_groups" in inspect.getsource(group_counting)
    # ... and the counter only exposes a thin pass-through.
    assert hasattr(counter_module.VideoCounter, "grouped_entries")
