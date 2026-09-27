# FootfallCam-CV: Enterprise Retail Video Analytics & People Counting Engine

A high-precision, production-grade computer vision pipeline for retail footfall counting, customer journey tracking, and spatial intelligence. Engineered with a modular, strictly layered architecture:
> **Video Ingestion** → **Detection** → **Tracking** → **Spatial Geometry** → **Feature Analytics** → **Audit Playback and Reporting**

Designed for retail stores, shopping centres, transport hubs, and commercial venues, FootfallCam-CV transforms standard overhead or angled camera feeds into actionable foot traffic analytics, zone dwell profiling, and live occupancy metrics.

---

## Key Capabilities & Requirements Delivered

The system implements the core analytical capabilities expected of enterprise 3D retail vision sensors, organized into production-ready modules and roadmap extensions:

### Analytics Capability Matrix

| # | Retail Intelligence Capability | Status | Implementation Module | Description |
|---|---|---|---|---|
| 1 | **Bidirectional Video Counting** | **Production Ready** | `features/counter.py` | High-precision entry/exit counting (`IN`, `OUT`, `net_inside`) across configurable doorway thresholds with finite-segment crossing geometry and boundary debounce. |
| 2 | **Multiple Counting Lines** | **Production Ready** | `features/multiple_counting_line.py` | Independent multi-gate and multi-entrance tallying with per-gate deduplication and individual line metrics. |
| 3 | **Group Counting & Buying Units** | **Production Ready** | `features/group_counting.py` | Spatial-temporal co-movement clustering to distinguish group visitors (families, couples) from individual shoppers for accurate conversion rate calculations. |
| 4 | **Audit Playback & Verification** | **Production Ready** | `features/playback.py` | Full visual audit trail with persistent track trails, ID badges, role color coding, HUD statistics, and automatic H.264 web-compatible video export. |
| 5 | **Area Profiling & Dwell Times** | **Production Ready** | `features/area_profiling.py` | Continuous customer dwell time accumulation, foot-point polygon containment, and engagement share distribution across store departments. |
| 6 | **Pedestrian Demographics** | **Production Ready** | `features/gender.py` | Deep learning gender classification via ONNX Runtime with aspect-preserving preprocessing, progressive confidence refinement, and heuristic fallback. |
| 7 | **Zone Counting & Peak Occupancy** | **Production Ready** | `features/zone_counting.py` | Real-time multi-zone customer headcount, high-water mark peak occupancy tracking, and automatic spatial band partitioning. |
| 8 | Dynamic Queue Management | *Roadmap* | `features/queue_counting.py` | Real-time queue length tracking and service wait duration monitoring. |
| 9 | Passenger Queue Tracking | *Roadmap* | `features/passenger_queue.py` | Transit hub passenger queue estimation and density monitoring. |
| 10 | Queue Wait-Time Prediction | *Roadmap* | `features/queue_prediction.py` | Predictive service time modeling based on service velocity and queue depth. |
| 11 | Outside Passerby Traffic | *Roadmap* | `features/outside_traffic.py` | Street and corridor footfall monitoring outside store frontages. |
| 12 | Turn-in Conversion Rate | *Roadmap* | `features/turn_in_rate.py` | Storefront window conversion efficiency: `Store Entries / Outside Footfall`. |
| 13 | Heatmap & Flow Density | *Roadmap* | `features/heatmap.py` | Accumulated 2D spatial occupancy density mapping. |
| 14 | Low-Light / Night Mode | *Roadmap* | `features/night_mode.py` | Adaptive contrast enhancement for low-light retail conditions. |
| 15 | Sales Conversion Analysis | *Roadmap* | `features/sales_conversion.py` | Correlation of POS transactional logs with physical footfall. |
| 16 | Staff Exclusion Filtering | *Roadmap* | `features/staff_exclusion.py` | Non-invasive, passive filtering of staff members from customer counts. |
| 17 | Object Clarification | *Roadmap* | `features/object_clarification.py` | Rejection of shopping carts, strollers, and non-pedestrian objects. |
| 18 | Safe Occupancy Limit Alerts | *Roadmap* | `features/safe_occupancy.py` | Real-time threshold monitoring and safety compliance alerts. |
| 19 | Visitor Dwell Duration | *Roadmap* | `features/visitor_dwell.py` | Store-wide total visit duration from physical entrance to exit. |
| 20 | Retail Performance KPIs | *Roadmap* | `features/metrics.py` | Aggregated executive KPI scorecards and trend summaries. |

---

## Architectural Design

The pipeline enforces strict separation of concerns with unidirectional dependency flows:

```
                  configs/default.yaml (Single Source of Truth)
                                │
                                ▼
                       src/pipeline.py (Orchestrator)
                                │
       ┌────────────────────────┴────────────────────────┐
       ▼                                                 ▼
src/detection/                                    src/tracking/
  PersonDetector                                    SimpleTracker
  Detection                                         Track
  • ONNX DirectML / CUDA / CPU                      • ByteTrack 2-stage association
  • Letterbox + NMS + dedup                         • Centroid distance fallback
  • Geometric sanity filters                        • EMA velocity & box smoothing
       │                                                 │
       └────────────────────────┬────────────────────────┘
                                ▼
                         src/geometry/
                           zones.py
                           • Point-in-polygon containment
                           • Finite-segment line crossing
                           • Relative spatial auto-bands
                                │
                                ▼
                            features/
                              ├── counter.VideoCounter
                              ├── multiple_counting_line.MultipleCountingLineManager
                              ├── group_counting.GroupCounter
                              ├── area_profiling.AreaProfiler
                              ├── zone_counting.ZoneCounter
                              ├── gender.GenderClassifier & DemographicsAggregator
                              └── playback.PlaybackEngine
                                │
                                ▼
                        src/visualizer.py (Stateless Drawing Toolkit)
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
        outputs/annotated.mp4          outputs/report.json
        (H.264 + Web Faststart)        (Structured Retail Analytics)
```

### Architectural Principles

1. **Strict Layer Decoupling**: Upstream perception layers (`detection`, `tracking`, `geometry`) have zero knowledge of downstream analytics modules (`features`).
2. **Pure Data Passing**: Feature classes accept standard Python structures (`list[Track]`, `dict[track_id -> Detection]`, `frame_shape`, `timestamp_s`, `config`). They never import detector or tracker internals.
3. **Finite-Segment Line Crossing**: Line crossings are verified using a rigorous two-stage vector cross-product test: first proving the movement vector straddled the infinite line, and second confirming the intersection lies strictly within the finite line segment.
4. **Single Source of Truth**: All operational parameters, spatial coordinates, confidence thresholds, and model configurations reside in `configs/default.yaml`.
5. **Resolution Independence**: Spatial coordinates (lines, polygons, zones) are normalized to `0.0–1.0` and scaled dynamically to native video dimensions at runtime.

---

## Quick Start

### 1. Prerequisites & Installation

Clone the repository and install dependencies in a Python 3.10+ environment:

```bash
git clone https://github.com/your-org/footfallcam.git
cd footfallcam

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Export Detection Model (Optional)

The detector runs directly on ONNX Runtime (`yolov8n.onnx`). If the ONNX file is not present locally, the pipeline automatically falls back to Ultralytics and downloads the base weights on first execution. To export a high-performance, letterboxed ONNX model:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=640, opset=12, simplify=True)"
```

### 3. Run Pipeline

Process a video file, live webcam, or network RTSP stream:

```bash
# Run on the included sample clip
python run.py --video data/sample.mp4

# Display real-time visualization window (press 'q' to exit)
python run.py --video data/sample.mp4 --show

# Process live webcam or IP/RTSP camera
python run.py --video 0 --show
python run.py --video rtsp://admin:password@192.168.1.100:554/stream1

# Limit processing frames for quick testing
python run.py --max-frames 300
```

---

## Configuration (`configs/default.yaml`)

All parameters are centrally governed by `configs/default.yaml`. No coordinates or thresholds are hardcoded in application logic.

```yaml
source: "data/sample.mp4"
output_dir: "outputs"

detection:
  model_path: "yolov8n.onnx"
  confidence: 0.45
  device: "gpu"          # DirectML on Windows, CUDA on Linux, or CPU
  person_class_id: 0
  input_size: 640
  nms_iou: 0.45
  sanity:                # Filter optical noise and false positives
    min_box_width_px: 12
    min_box_height_px: 16
    min_height_over_width: 0.30
    min_width_over_height: 0.15
    max_frame_fraction: 0.95

tracking:
  iou_threshold: 0.30
  high_confidence_threshold: 0.45
  min_hits: 2            # Confirm track on second observation
  max_missed: 30         # Maximum frame memory for occluded tracks
  max_center_distance_px: 90
  max_occlusion_bridge: 2
  box_ema_alpha: 0.70    # Bounding-box exponential smoothing
  velocity_ema_alpha: 0.60
  max_history: 300       # Retained trajectory trail length

lines:
  # Normalized coordinates [x, y] across the entrance threshold
  - name: "main_entrance"
    p1: [0.50, 0.86]
    p2: [0.94, 0.62]
    in_direction: "bottom_to_top"

zones:
  # Normalized polygon boundaries for spatial dwell & occupancy analysis
  - name: "checkout_till"
    kind: "queue"
    polygon: [[0.08, 0.40], [0.32, 0.40], [0.36, 0.72], [0.08, 0.72]]
  - name: "sales_floor"
    kind: "sales"
    polygon: [[0.56, 0.18], [0.78, 0.18], [0.88, 0.58], [0.52, 0.58]]

demographics:
  enabled: true
  model_path: "models/gender_yolov8n.onnx"
  min_hits: 3
  heuristic_min_confidence: 0.54
```

---

## Analytics Outputs & Audit Trail

Every execution produces two deterministic artifacts in the configured output directory:

### 1. Web-Optimized Audit Video (`outputs/annotated.mp4`)
- Real-time bounding boxes with raw detection preference for forensic auditing.
- Track ID badges, motion history trails, and foot-point contact markers.
- Highlighting for configured counting lines and perspective floor zones.
- Real-time Head-Up Display (HUD) displaying frame index, live entries (`IN`), exits (`OUT`), active occupancy (`INSIDE`), and demographic distribution.
- **Automated Transcoding**: Encoded to **H.264** with `yuv420p` pixel format and `faststart` MOOV atom placement, ensuring native playback in modern web browsers and web dashboards without buffering.

### 2. Standardized Analytics Payload (`outputs/report.json`)
A structured JSON payload ready for integration into retail BI platforms, data warehouses, or REST APIs:

```json
{
  "meta": {
    "source": "data/sample.mp4",
    "detector_backend": "onnx",
    "frames_processed": 1452,
    "fps": 25.0,
    "frame_size": [634, 360],
    "elapsed_seconds": 38.2,
    "generated_at": "2026-09-27T11:45:00.000000+00:00",
    "h264_web_transcode": true
  },
  "tracking": {
    "live_tracks": 2,
    "total_tracks_opened": 57
  },
  "counting": {
    "total_in": 6,
    "total_out": 4,
    "net_inside": 2,
    "lines": {
      "main_entrance": { "in": 6, "out": 4, "net": 2 }
    }
  },
  "demographics": {
    "male_count": 4,
    "female_count": 2,
    "unknown_count": 1,
    "total_classified": 6,
    "male_percentage": 66.7,
    "female_percentage": 33.3
  },
  "area_profiling": {
    "total_dwell_seconds": 182.4,
    "zones": {
      "sales_floor": { "dwell_seconds": 124.5, "percentage": 68.3 },
      "checkout_till": { "dwell_seconds": 57.9, "percentage": 31.7 }
    }
  },
  "zone_counting": {
    "zones": {
      "sales_floor": { "current_headcount": 1, "peak_occupancy": 3 },
      "checkout_till": { "current_headcount": 1, "peak_occupancy": 2 }
    }
  }
}
```

---

## Hardware Acceleration & Performance

The pipeline is engineered for low latency and high throughput across heterogeneous hardware environments:

- **Windows Environments**: Native execution via ONNX Runtime `DmlExecutionProvider` (DirectML), utilizing NVIDIA, AMD, or Intel discrete and integrated GPUs without requiring complex CUDA toolkit installations.
- **Linux / Cloud Environments**: Automatic acceleration via NVIDIA CUDA / TensorRT when available, with clean fallback to optimized multi-threaded CPU execution.
- **Throughput**: Processes ~37+ FPS end-to-end on retail video (634×360) on standard workstation GPUs—including full detection, ByteTrack tracking, multi-line counting, zone profiling, rendering, and H.264 web encoding.

---

## Quality Assurance & Automated Testing

The codebase includes an extensive automated test suite covering pure geometry, tracking edge cases, counter delegation, and video I/O:

```bash
python -m pytest tests/ -v
```

**250+ unit and integration tests passing in ~3 seconds.**

### Test Coverage Highlights

| Test Module | Scope & Validation |
|---|---|
| `test_counting_and_geometry.py` | Vector cross-product line crossing, finite-segment rejection, bidirectional tallying, and polygon containment. |
| `test_multiple_lines.py` | Independent gate tallies, per-line deduplication, and multi-entrance traversal. |
| `test_group_counting.py` | Co-movement distance accumulation and temporal entry window clustering. |
| `test_area_profiling.py` | Foot-point dwell time accumulation and unzoned baseline exclusion. |
| `test_zone_counting.py` | Multi-zone occupancy, peak high-water mark tracking, and active track attribution. |
| `test_gender_classifier.py` | Aspect-ratio preserving crops, multi-scale refinement, and deterministic fallback behaviour. |
| `test_tracker_association.py` | Two-stage IoU association, low-confidence occlusion recovery, EMA velocity damping, and track lifecycle management. |
| `test_detection_model.py` | Letterbox transformation round-trips, aspect-ratio filters, and non-maximum suppression. |
| `test_playback_and_visualizer.py` | Non-mutating visualizer drawing, role color mapping, HUD formatting, and trail stability. |
| `test_video_io.py` | Robust source handling (files, webcams, RTSP streams), FPS fallback, and H.264 web transcoding. |

---

## Data Privacy, Ethics & Responsible AI

Computer vision deployed in commercial environments carries significant privacy responsibilities:

- **Anonymized Analytics**: The system computes aggregate metrics (headcounts, dwell times, directional vectors). No facial recognition, individual identification, or persistent biometric signatures are stored.
- **Data Minimization**: Video frames are processed in-memory. Only structured analytical counts and aggregated dwell statistics are persisted in production deployments.
- **Transparency & Disclosure**: Store operators should provide clear signage informing visitors of optical footfall analytics in accordance with GDPR, CCPA, and applicable local privacy legislation.
- **Demographic Model Validation**: Gender and demographic classification should be deployed only where permitted by law and validated against specific store demographic profiles to prevent statistical bias.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
