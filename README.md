# FootfallCam-CV

Retail people counting built on a layered AI/CV backbone: **detection →
tracking → geometry → features**, wired together by a single pipeline.

The target is the FootfallCam `3D PRO 2` / `3D Extend` brochure's 20
characteristics. The CV backbone that makes all 20 possible is complete; the
feature modules are being added one phase at a time. **5 characteristics are
done, 2 are partial, 13 have not started.**

| # | Characteristic | Phase | State |
|---|---|---|---|
| 1 | Video Counting | 2 | **done** — `features/counter.py` |
| 2 | Dynamic Queue Counting | 6 | not started |
| 3 | Passenger Queue | 6 | not started |
| 4 | Playback | 4 | **done** — `features/playback.py` |
| 5 | Area Profiling | 5 | **done** — `features/area_profiling.py` |
| 6 | Gender recognition | 3 | **partial** — complete, but opt-in and **off by default** |
| 7 | Metrics measures | 9 | not started |
| 8 | Outside traffic | 7 | not started |
| 9 | Turn in rate | 7 | not started |
| 10 | Sales conversation | 10 | not started |
| 11 | Object clarification | 11 | not started |
| 12 | Staff exclusion | 10 | not started — `Track.is_staff` exists, nothing sets it yet |
| 13 | Multiple counting line | 2 | **done** — `features/multiple_counting_line.py` |
| 14 | Group counting | 2, 12 | **partial** — Phase 2 subset; Phase 12 completes it |
| 15 | Safe occupancy | 13 | not started |
| 16 | Queue prediction | 6 | not started |
| 17 | Zone counting | 5 | **done** — `features/zone_counting.py` |
| 18 | Heatmap | 8 | not started |
| 19 | Night vision mode | 8 | not started |
| 20 | Visitor in & out dwell time | 14 | not started |

Phases 0–5 are complete and merged into `main`; Phase 6 (queues) is next.
`outputs/report.json` currently carries `meta`, `tracking`, `counting`,
`demographics`, `area_profiling` and `zone_counting`.

> This repo is mid-refactor. The previous 20-feature implementation is preserved
> on the `legacy/sprawling-implementation` branch; `main` is the rebuilt,
> layered version. See `REFACTOR_PLAN.md` for the phase breakdown and
> `AGENTS.md` for the engineering rules.

---

## Quick start

```bash
pip install -r requirements.txt

# Optional: the ONNX model is gitignored. Without it the detector falls back
# to Ultralytics, which downloads yolov8n.pt on first use.
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, opset=12, simplify=True)"

# The test video is gitignored too; drop any clip at this path.
python run.py --video data/sample.mp4
```

Writes two artifacts, both overwritten on every run:

| File | Contents |
|---|---|
| `outputs/annotated.mp4` | Person boxes, ID badges and foot markers, transcoded to H.264 + yuv420p + faststart for browser playback |
| `outputs/report.json` | Run metadata plus counting totals (`total_in`, `total_out`, `net_inside`) |

Useful flags:

```bash
python run.py --max-frames 300     # stop early
python run.py --show               # live preview window, q quits
python run.py --video 0            # webcam index
python run.py --video rtsp://...   # network camera
python run.py --config my.yaml     # alternate configuration
```

---

## Configuration

`configs/default.yaml` is the single source of truth for every tunable. No
coordinate, threshold or model path is hardcoded in code. All points are
normalised to `0.0–1.0` and scaled to the frame at runtime, so one config works
at any resolution.

```yaml
detection:
  model_path: "yolov8n.onnx"
  confidence: 0.45
  device: "gpu"          # DirectML on Windows, falls back to CUDA then CPU
  sanity: { ... }        # reject noise, poles and counter-edge strips
tracking:
  iou_threshold: 0.30
  min_hits: 2            # confirm a track on its second observation
  max_occlusion_bridge: 2
lines:
  - name: "main_entrance"
    p1: [0.05, 0.50]     # normalised: 5% in, 50% down
    p2: [0.95, 0.50]
    in_direction: "top_to_bottom"
```

---

## Architecture

```
configs/default.yaml    every tunable
        │
        ▼
   src/pipeline.py      the only module that wires the layers together
        │
   ┌────┴─────────────────────────────┐
   ▼                                  ▼
src/detection/                  src/tracking/
  PersonDetector                   SimpleTracker
  Detection                       Track
   ONNX on DML→CUDA→CPU           ByteTrack-style 2-stage association
   letterbox + NMS + dedup         + centroid-distance fallback
   geometric sanity filters        EMA smoothing, velocity prediction
   ▼                                ▼
   └──────────► features/ ◄────────┘
                  counter.VideoCounter      (characteristic 1)
                  playback.PlaybackEngine   (characteristic 4)
                  area_profiling.AreaProfiler  (characteristic 5)
                  zone_counting.ZoneCounter  (characteristic 17)
                        │
                        ▼
              src/visualizer.py  →  outputs/annotated.mp4
              (drawing toolkit: zones, lines, boxes, trails, HUD)
```

**Layer boundaries are hard.** `detection/`, `tracking/` and `geometry/` never
import from `features/`, and feature modules receive `list[Track]`,
`dict[track_id → Detection]`, `frame_shape`, `timestamp_s` and `config` only —
never the internals of detection or tracking.

`src/geometry/zones.py` holds the spatial primitives. `crossed_line` is
deliberately two-stage: a cross-product sign change proves the movement
straddled the line's *infinite* extension, and a second sign test on the line's
endpoints proves the intersection falls *between* them. Without that second
test, someone walking past the end of a short gate segment is still counted.

Each brochure characteristic gets one module in `features/` with one class,
re-exported from `features/__init__.py`. Feature modules receive `list[Track]`,
`dict[track_id → Detection]`, `frame_shape`, `timestamp_s` and `config` only —
they import `Track` and `Line` under `if TYPE_CHECKING:` so the runtime
dependency stays one-directional. `src/visualizer.py` sits below `features/` and
is a pure drawing toolkit: it renders what it is handed and decides nothing.

---

## Gender recognition is off by default

Characteristic 6 exists, but `demographics.enabled: false` in
`configs/default.yaml` and you should think hard before flipping it.

With no model supplied, the classifier falls back to a three-cue heuristic
(Sobel edge energy, HSV saturation, aspect ratio). Measured on the sample store
footage, sweeping its confidence threshold:

| threshold | male | female | unknown |
|---|---|---|---|
| 0.00 | 2999 | 543 | 0 |
| 0.20 | 2344 | 20 | 1178 |
| 0.40 | 175 | 0 | 3367 |
| **0.60** (default) | **0** | **0** | **3542** |

Nothing is ever confidently "female", and at the permissive end the split is
~85% male — which is the heuristic's own bias, not a measurement of anything.
Lowering the threshold reveals bias rather than signal.

So at the shipped default the fallback abstains on every real crop. That is the
intended behaviour, not a bug: a guess about a stranger's gender made from a
~90×57 pixel crop should not be asserted. If you need real demographics,
supply a trained model via `demographics.model_path` and validate it on your own
floor. Do not rebalance the heuristic's midpoints to force output.

---

## Tests

```bash
python -m pytest tests/ -v
```

249 tests, ~2.9 s. Logic tests need neither a video nor a model; the tests that
do need one skip cleanly when `data/sample.mp4` or `ffmpeg` is absent, and none
of them touch the network.

| File | Covers |
|---|---|
| `test_counting_and_geometry.py` | Crossing in/out, parallel motion, finite-segment rejection (R8), bi-directional tally, staff exclusion, polygon containment |
| `test_detection_model.py` | Letterbox round-trip, minimum-size and aspect filters, oversize rejection, both YOLO output layouts, partial-body dedup |
| `test_tracker_association.py` | IoU matching, low-confidence occlusion recovery, min-hits confirmation, max-missed deletion, EMA damping, nearest-track matching, no double assignment |
| `test_multiple_lines.py` | Independent per-gate tallies, one count per line per track, forgotten-and-returned tracks |
| `test_auto_mode_counting.py` | Entry on appearance, exit behind the debounce, dropout that is not an exit, passerby classification |
| `test_counter_delegation.py` | Delegation enforced by substituting stub delegates and requiring the counter's output to follow them, plus an AST check that it never imports the geometry primitive |
| `test_group_counting.py` | Co-movement accumulation, transitive entry clustering, configurable windows |
| `test_gender_classifier.py` | Label contract, unusable crops, cue extraction, cue agreement, backend precedence, determinism — mostly pinning the paths that answer "unknown" |
| `test_demographics_aggregator.py` | Idempotent per-person recording, percentages over the classified population, unknown handling |
| `test_playback_and_visualizer.py` | Frame is never mutated, role colours, trails, lines, zone tints, raw-detection preference, HUD fields, and that an unwired KPI renders as `n/a` rather than `0` |
| `test_area_profiling.py` | Dwell accumulation, foot-point attribution, unzoned time kept out of the denominator, engagement share, auto bands |
| `test_zone_counting.py` | Per-zone headcount, peak as a high-water mark, active track ids, auto bands |
| `test_video_io.py` | File/webcam/URL source rules, stream info and FPS fallback, writer round-trip, H.264 transcode verification |

Two infrastructure files are load-bearing: `pytest.ini` pins the project root
onto `sys.path`, and `tests/__init__.py` stops an unrelated `tests` package in
site-packages from shadowing the local one. Removing either breaks collection.

---

## Hardware notes

- **Windows + NVIDIA GPU** runs inference through ONNX Runtime's
  `DmlExecutionProvider`, which needs no CUDA toolkit.
- `onnxruntime-directml` and `onnxruntime-gpu` install into the same namespace —
  whichever is installed last wins. Installing or upgrading one silently changes
  the provider.
- `detection.input_size` must match the ONNX export's static input shape; the
  detector letterboxes to a fixed square.
- Measured throughput on the sample clip (634×360): ~37 fps end to end including
  tracking, counting, annotation and H.264 encoding.

---

## Ethics and responsible use

Video analytics of identifiable people carries real obligations.

- **Disclose it.** Visitors are entitled to know they are being counted; signage
  is standard practice.
- **Get consent or a lawful basis** before analysing footage of people, and
  check local privacy law (GDPR, CCPA and equivalents) for your deployment.
- **Minimise retention.** Aggregate counts, not raw video, wherever the task
  allows.
- **Bias is a real risk.** Detection and demographic models trained on broad
  populations degrade on under-represented groups, and the brochure's hardware
  assumptions (mount height, minimum illumination) narrow where this works at
  all. Validate against your own floor before trusting a number.
- **No covert identification.** Nothing here should be repurposed for
  individual identification or surveillance beyond the counting purpose it was
  built for.

---

## License

See `LICENSE`.
