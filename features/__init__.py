"""Feature modules.

One brochure characteristic = one module with one class, re-exported here
(AGENTS.md R2). The pipeline imports only what it uses.

Feature modules receive `list[Track]`, `dict[track_id -> Detection]`,
`frame_shape`, `timestamp_s` and `config` only. They must not import from
`detection/` or `tracking/` internals (AGENTS.md R1) - shared value types come
in under `if TYPE_CHECKING:` so the runtime dependency stays one-directional.

Implemented so far:
    characteristic  1  Video Counting        counter.VideoCounter
    characteristic 13  Multiple counting line counter.MultipleCountingLineManager
    characteristic 14  Group counting        group_counting.GroupCounter
                       (Phase 2 subset; completed in Phase 12)
"""
from .counter import LineCounter, VideoCounter
from .group_counting import GroupCounter
from .multiple_counting_line import MultipleCountingLineManager

__all__ = [
    "GroupCounter",
    "LineCounter",
    "MultipleCountingLineManager",
    "VideoCounter",
]
