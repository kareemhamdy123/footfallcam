# FootfallCam-CV

An open-source, YOLO-based people-counting and retail-analytics pipeline that
re-implements the **20 "Characteristics"** advertised on the FootfallCam
`3D PRO 2` / `3D Extend` product brochure — using an off-the-shelf camera and
open computer-vision models instead of proprietary stereo-vision hardware.

> This is an independent reference implementation, not affiliated with
> FootfallCam or PAMAS Trading. It draws on their public product sheet as a
> functional spec for what a people-counting system should be able to do.

## What it does

A single pipeline (`src/main.py`) takes a video (file, RTSP stream, or
webcam), runs YOLO person detection + lightweight multi-object tracking, and
produces:

- `outputs/annotated.mp4` — playback video with boxes, IDs, counting lines,
  zone overlays and motion trails
- `outputs/heatmap.png` — an accumulated foot-traffic density heatmap
- `outputs/report.json` — full analytics report (counts, queues, dwell
  times, occupancy, etc.)

## Brochure characteristic → implementation map

| # | Characteristic | Implementation |
|---|---|---|
| 1 | Video Counting | `detector.PersonDetector` + `tracker.SimpleTracker` |
| 2 | Dynamic Queue Counting | `analytics/queue.py::QueueMonitor` |
| 3 | Passenger Queue (wait time) | `QueueMonitor.current_wait_times()` |
| 4 | Playback | annotated `.mp4` written every run |
| 5 | Area Profiling | `analytics/heatmap.py::AreaProfiler` |
| 6 | Gender recognition | `analytics/demographics.py` — **opt-in, off by default** (see [Ethics](#ethics--responsible-use)) |
| 7 | Metrics measures | `analytics/metrics.py::build_report` |
| 8 | Outside traffic | a `Line`/`Zone` outside the entrance (`kind: outside`) |
| 9 | Turn-in rate | `analytics/counting.py::TurnInRate` |
| 10 | Sales conversation | `analytics/dwell.py::SalesConversationMonitor` |
| 11 | Object clarification | per-box confidence in `detector.Detection` |
| 12 | Staff exclusion | `analytics/staff.py` (colour-tag detection → `track.is_staff`) |
| 13 | Multiple counting line | `Config.lines` is a list; each tallied independently |
| 14 | Group counting | `LineCounter.grouped_entries()` |
| 15 | Safe occupancy | `analytics/occupancy.py::OccupancyMonitor` |
| 16 | Queue prediction | `QueueMonitor.predict_next()` (linear trend) |
| 17 | Zone counting | `zones.py::zones_containing()` / `Zone(kind="generic")` |
| 18 | Heatmap | `analytics/heatmap.py::HeatmapAccumulator` |
| 19 | Night vision mode | `video_io.py::enhance_frame()` (CLAHE + gamma, auto-triggered on low light) |
| 20 | Visitor in & out dwell time | `analytics/dwell.py::DwellTracker` |

The same table is embedded in every `report.json` under
`characteristic_reference` so results are traceable back to this spec.

## Architecture

```
footfallcam-cv/
├── configs/
│   └── zones_example.yaml      # camera source, lines, zones, thresholds
├── scripts/
│   └── draw_zones.py           # click-to-draw helper -> prints YAML
├── src/
│   ├── config.py                # YAML -> dataclasses (Line, Zone, Config)
│   ├── detector.py               # Ultralytics YOLO wrapper (person only)
│   ├── tracker.py                # dependency-free IOU + centroid tracker
│   ├── zones.py                  # point-in-polygon, line-crossing geometry
│   ├── video_io.py               # capture/writer + low-light enhancement
│   ├── main.py                   # CLI orchestrator / annotation / report
│   └── analytics/
│       ├── counting.py           # in/out lines, turn-in rate, group counting
│       ├── queue.py              # dynamic queue length, wait time, prediction
│       ├── occupancy.py          # safe-occupancy limits & alerts
│       ├── dwell.py              # visit dwell time, sales-conversation proxy
│       ├── heatmap.py            # heatmap + per-zone area profiling
│       ├── staff.py              # colour-tag based staff exclusion
│       ├── demographics.py       # optional aggregate gender stats (off by default)
│       └── metrics.py            # combines every module into report.json
└── tests/
    └── test_zones.py             # geometry unit tests (no model/video needed)
```

Each analytics module only depends on `Track` objects and `Config`
dataclasses, so you can run/test/swap any single characteristic in
isolation — e.g. replace the tracker with ByteTrack/DeepSORT without
touching queueing, occupancy, or dwell-time logic.

## Setup

```bash
git clone <your-repo-url>
cd footfallcam-cv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

YOLO weights (`yolov8n.pt` by default) download automatically on first run
via `ultralytics`. Swap in `yolov8s/m/l.pt` for higher accuracy, or your own
fine-tuned checkpoint, by editing `detection.model_path` in the config.

## Configure your camera layout

1. Drop a sample clip at `data/sample.mp4` (or point `source:` at an RTSP
   URL / webcam index).
2. Draw your entrance line(s) and zones interactively:
   ```bash
   python scripts/draw_zones.py --source data/sample.mp4 \
       --kind line --name main_entrance --in-direction top_to_bottom

   python scripts/draw_zones.py --source data/sample.mp4 \
       --kind zone --name checkout_queue --zone-kind queue
   ```
   Click points on the popup window, press **u** to undo, **Enter/q** to
   print ready-to-paste YAML.
3. Paste the output into `configs/zones_example.yaml` (or your own copy).

## Run

```bash
python -m src.main --config configs/zones_example.yaml
# add --show for a live preview window, --max-frames N for a quick test
```

Output lands in `outputs/`: `annotated.mp4`, `heatmap.png`, `report.json`.

## Tests

Geometry logic (line-crossing, point-in-polygon) is unit-tested without
needing a GPU, video file, or the YOLO/OpenCV dependencies installed:

```bash
pip install pytest
pytest tests/
```

## Extending

- **Better tracking**: swap `tracker.SimpleTracker` for ByteTrack/DeepSORT
  (e.g. via the `supervision` package) for crowded/occluded scenes — the
  rest of the pipeline only needs a `Track` with `.history`, `.is_staff`,
  and `.counted_lines`.
- **Better staff exclusion**: replace the HSV colour check in
  `analytics/staff.py` with a trained badge/uniform classifier or apparel
  re-identification model.
- **Multi-camera**: run one pipeline instance per camera and merge
  `report.json` files downstream; track IDs are only unique per-camera.

## Ethics & responsible use

This system detects and tracks real people. Before deploying it:

- **Disclose video analytics** to visitors (signage is standard practice
  and legally required in many jurisdictions).
- **Gender recognition is disabled by default** (`analytics/demographics.py`).
  If you enable it: only store aggregate percentages, never per-person
  labels; these classifiers have known accuracy gaps across demographics
  and should be treated as a noisy aggregate signal, not a factual label.
- **Don't persist identifiable imagery** longer than necessary — dwell-time
  and counting logic only need track IDs and coordinates, not saved face
  crops.
- **Check local privacy law** (GDPR, CCPA, etc.) for camera analytics,
  retention limits, and any required DPIA/lawful-basis documentation.

## License

MIT — see [LICENSE](LICENSE).
