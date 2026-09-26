"""Feature 4: Playback / video proof.

The annotated output is the audit artifact behind the brochure's "most accurate
counting with video proof" claim: when a retailer disputes a number, this is
what they check. That framing drives the design. The frame shows the raw
detection the model actually produced rather than the tracker's smoothed box, so
what is on screen is what was measured, and every KPI that is not wired to a
feature yet renders as `n/a` rather than as a plausible-looking zero.

Brochure characteristic 4 ("Playback").
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from src.visualizer import annotate_frame

if TYPE_CHECKING:  # pragma: no cover - typing only (R1)
    import numpy as np

    from src.config import Line, Zone
    from src.detection.dataclasses import Detection
    from src.tracking.dataclasses import Track

# Sentinel for "this KPI has no feature behind it yet". The visualizer turns it
# into the literal text "n/a"; it must never become a 0.
UNWIRED = None


class PlaybackEngine:
    """Builds the annotated proof frames for a run."""

    def __init__(
        self,
        lines: Sequence["Line"] = (),
        zones: Sequence["Zone"] = (),
        source_label: str = "",
    ):
        self.lines = list(lines)
        self.zones = list(zones)
        self.source_label = source_label

    def render_proof_frame(
        self,
        frame: "np.ndarray",
        tracks: list["Track"],
        detections_by_id: dict[int, "Detection"] | None = None,
        counts_summary: dict | None = None,
        kpis: dict | None = None,
        queue_monitor=None,
        zone_counts: dict | None = None,
        zone_peaks: dict | None = None,
        dwell_percentages: dict | None = None,
        track_zones: dict[int, str] | None = None,
        fps: float = 13.0,
    ) -> "np.ndarray":
        """Return the annotated frame for one video frame.

        `queue_monitor` is accepted and forwarded for the dynamic queue
        overlay, but the feature that discovers queues is Phase 6, so nothing
        renders until then.
        """
        summary = counts_summary or {}
        line_counts = summary.get("per_line")
        footer_telemetry = self.build_footer_telemetry(
            summary, zone_counts, zone_peaks, dwell_percentages
        )
        return annotate_frame(
            frame,
            tracks=tracks,
            detections_by_id=detections_by_id,
            zones=self.zones,
            lines=self.lines,
            queue_polygons=self._queue_polygons(queue_monitor),
            kpis=self.build_kpis(counts_summary, kpis),
            audit_badge=self.audit_badge_text(),
            line_counts=line_counts,
            zone_counts=zone_counts,
            track_zones=track_zones,
            fps=fps,
            footer_telemetry=footer_telemetry,
        )

    def build_footer_telemetry(
        self,
        counts_summary: dict | None,
        zone_counts: dict | None,
        zone_peaks: dict | None,
        dwell_percentages: dict | None,
    ) -> str | None:
        """Format a clean, single-line telemetry string across finished features."""
        parts = []
        if zone_counts:
            z_strs = []
            for z_name, count in zone_counts.items():
                short = "Till" if "till" in z_name.lower() else ("Sales" if "sales" in z_name.lower() else z_name)
                pk = zone_peaks.get(z_name, count) if zone_peaks else count
                z_strs.append(f"{short} {count} (Pk {pk})")
            parts.append(f"ZONES: {' | '.join(z_strs)}")

        if dwell_percentages:
            d_strs = []
            for z_name, pct in dwell_percentages.items():
                short = "Till" if "till" in z_name.lower() else ("Sales" if "sales" in z_name.lower() else z_name)
                d_strs.append(f"{short} {pct:.0f}%")
            if d_strs:
                parts.append(f"DWELL: {' | '.join(d_strs)}")

        if counts_summary:
            groups = counts_summary.get("group_entries", 0)
            co_pairs = counts_summary.get("group_stats", {}).get("co_movement_pairs", 0)
            if groups or co_pairs:
                parts.append(f"GROUPS: {groups} ({co_pairs} co-move)")

        return "   |   ".join(parts) if parts else None

    def build_kpis(
        self, counts_summary: dict | None = None, extra: dict | None = None
    ) -> dict:
        """Assemble the HUD values.

        Only KPIs backed by a finished feature get a number. `queues` and
        `return_rate` stay `None` until Phases 6 and 7, and the HUD renders
        them as `n/a`.
        """
        summary = counts_summary or {}
        kpis = {
            "in_out": f"{summary.get('total_in', 0)} / {summary.get('total_out', 0)}",
            "inside": summary.get("net_inside", 0),
            "queues": UNWIRED,
            "return_rate": UNWIRED,
            "demo": UNWIRED,
        }
        if extra:
            kpis.update({k: v for k, v in extra.items() if v is not None})
        return kpis

    def audit_badge_text(self) -> str:
        return f"FootfallCam-CV video proof | {self.source_label}".strip(" |")

    @staticmethod
    def _queue_polygons(queue_monitor) -> dict | None:
        """Dynamic queue polygons, once the queue feature exists (Phase 6)."""
        if queue_monitor is None:
            return None
        auto_queues = getattr(queue_monitor, "auto_queues", None)
        if not auto_queues:
            return None
        return {
            name: state.polygon
            for name, state in auto_queues.items()
            if getattr(state, "polygon", None)
        }
