"""Feature modules.

One brochure characteristic = one module with one class, re-exported here
(AGENTS.md R2). The pipeline imports only what it uses.

Feature modules receive `list[Track]`, `dict[track_id -> Detection]`,
`frame_shape`, `timestamp_s` and `config` only. They must not import from
`detection/` or `tracking/` internals (AGENTS.md R1) - shared value types come
in under `if TYPE_CHECKING:` so the runtime dependency stays one-directional.

Implemented so far:
    characteristic  1  Video Counting        counter.VideoCounter
    characteristic  4  Playback / video proof playback.PlaybackEngine
    characteristic  5  Area profiling       area_profiling.AreaProfiler
    characteristic  6  Gender recognition    gender.GenderClassifier
                                           gender.DemographicsAggregator
    characteristic 13  Multiple counting line counter.MultipleCountingLineManager
    characteristic 14  Group counting        group_counting.GroupCounter
                       (Phase 2 subset; completed in Phase 12)
    characteristic 17  Zone counting         zone_counting.ZoneCounter
"""
from .area_profiling import AreaProfiler
from .counter import LineCounter, VideoCounter
from .gender import DemographicsAggregator, GenderClassifier
from .group_counting import GroupCounter
from .multiple_counting_line import MultipleCountingLineManager
from .playback import PlaybackEngine
from .zone_counting import ZoneCounter

__all__ = [
    "AreaProfiler",
    "DemographicsAggregator",
    "GenderClassifier",
    "GroupCounter",
    "LineCounter",
    "MultipleCountingLineManager",
    "PlaybackEngine",
    "VideoCounter",
    "ZoneCounter",
]
