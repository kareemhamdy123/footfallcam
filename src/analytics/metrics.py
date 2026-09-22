"""
Pulls every module's `.summary()` into one JSON-serialisable report and maps
each field back to the brochure's numbered "Characteristics" list, so the
CLI output is traceable to the original spec sheet.
"""
from __future__ import annotations


CHARACTERISTIC_MAP = {
    1: "Video Counting -> counting.LineCounter (per-frame detections)",
    2: "Dynamic Queue Counting -> queue.QueueMonitor.summary()['current_length']",
    3: "Passenger Queue -> queue.QueueMonitor.summary()['wait_times_s']",
    4: "Playback -> annotated MP4 written by main.py",
    5: "Area Profiling -> heatmap.AreaProfiler.summary()",
    6: "Gender recognition -> analytics.demographics (opt-in, off by default)",
    7: "Metrics measures -> this report (metrics.build_report)",
    8: "Outside traffic -> counting on a Line/Zone with kind='outside'",
    9: "Turn in rate -> counting.TurnInRate.compute()",
    10: "Sales conversation -> dwell.SalesConversationMonitor.summary()",
    11: "Object clarification -> detector.Detection.confidence per box",
    12: "Staff exclusion -> staff.tag_staff() / track.is_staff",
    13: "Multiple counting line -> counting.LineCounter (list[Line])",
    14: "Group counting -> counting.LineCounter.grouped_entries()",
    15: "Safe occupancy -> occupancy.OccupancyMonitor.summary()",
    16: "Queue prediction -> queue.QueueMonitor.predict_next()",
    17: "Zone counting -> zones.Zone kind='generic' + zones_containing()",
    18: "Heatmap -> heatmap.HeatmapAccumulator.export_overlay()",
    19: "Night vision mode -> video_io.enhance_frame()",
    20: "Visitor in & out dwell time -> dwell.DwellTracker.summary()",
}


def build_report(*, line_summary, queue_summary, occupancy_summary,
                  dwell_summary, sales_summary, area_summary,
                  turn_in_rate, demographics_summary=None,
                  elapsed_seconds: float) -> dict:
    return {
        "elapsed_seconds": round(elapsed_seconds, 1),
        "counting": line_summary,
        "turn_in_rate": turn_in_rate,
        "queues": queue_summary,
        "safe_occupancy": occupancy_summary,
        "dwell_time": dwell_summary,
        "sales_conversations": sales_summary,
        "area_profiling_seconds": area_summary,
        "demographics": demographics_summary or {"enabled": False},
        "characteristic_reference": CHARACTERISTIC_MAP,
    }
