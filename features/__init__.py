"""Feature modules.

Empty in Phase 0: the first feature module lands in Phase 2. One brochure
characteristic = one module with one class, re-exported here (AGENTS.md R2).

Feature modules receive `list[Track]`, `dict[track_id -> Detection]`,
`frame_shape`, `timestamp_s` and `config` only. They must not import from
`detection/` or `tracking/` internals (AGENTS.md R1).
"""
