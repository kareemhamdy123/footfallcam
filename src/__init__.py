"""FootfallCam-CV: the AI/CV backbone for retail people counting.

Layering, with hard boundaries (AGENTS.md R1):

    detection/  ->  raw person boxes for one frame
    tracking/   ->  persistent Track identities across frames
    geometry/   ->  spatial predicates (finite segments, polygons)
    features/   ->  one brochure characteristic per module
    pipeline.py ->  the only place the layers are wired together

Nothing in `detection/`, `tracking/` or `geometry/` may import from `features/`.
Feature modules receive `list[Track]`, `dict[track_id -> Detection]`,
`frame_shape`, `timestamp_s` and `config`, and may not reach into the internals
of `detection/` or `tracking/`.
"""
