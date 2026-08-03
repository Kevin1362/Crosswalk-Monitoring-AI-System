
# GuideKaro Enhanced

**GuideKaro** is an AI-powered crosswalk safety prototype that detects and tracks vehicles and pedestrians, estimates motion features, predicts short-term trajectories, calculates a feature-based risk score, logs events, and presents live and historical analytics.

## What changed in this enhanced version

The original prototype mainly counted detections. This version adds:

- YOLOv8 object detection
- ByteTrack tracking with persistent object IDs
- Vehicle speed estimation
- Distance-to-crosswalk estimation
- Pedestrian–vehicle separation
- Movement direction (`TOWARD_CROSSWALK`, `AWAY_FROM_CROSSWALK`, `STATIONARY`)
- Short-term trajectory prediction
- Approximate time-to-collision (TTC)
- Feature-based risk engine
- Track-level database records
- Expanded Streamlit dashboard
- FPS and latency metrics
- Evaluation script for precision / recall when labelled ground truth is available
- Failure-case testing plan
- YAML configuration
- Exception handling and application logging
- Pytest unit tests
- Docker packaging
- Grafana Cloud monitoring documentation
- Full project architecture diagram

## Project structure

```text
GuideKaro_Enhanced/
├── assets/
├── config/
│   └── settings.yaml
├── core/
│   ├── config.py
│   ├── geometry.py
│   ├── logging_setup.py
│   ├── risk_engine.py
│   └── trajectory.py
├── database/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASET_AND_TESTING.md
│   ├── EVALUATION_METRICS.md
│   ├── FAILURE_CASES.md
│   └── guidekaro_full_architecture.png
├── evaluation/
│   ├── evaluate_video.py
│   └── failure_case_template.csv
├── monitoring/grafana/
│   └── README.md
├── scripts/
├── tests/
├── videos/
├── dashboard.py
├── enhanced_video_runner.py
├── event_logger.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 1. Windows setup

Open PowerShell in the project folder.

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Run tracking on a real video

Put a video in `videos/`, for example:

```text
videos/real_crosswalk.mp4
```

Run:

```powershell
python enhanced_video_runner.py --source ".\videos\real_crosswalk.mp4" --display --loop
```

The default tracker is ByteTrack.

You can explicitly specify:

```powershell
python enhanced_video_runner.py --source ".\videos\real_crosswalk.mp4" --tracker bytetrack.yaml --display
```

Or try BoT-SORT:

```powershell
python enhanced_video_runner.py --source ".\videos\real_crosswalk.mp4" --tracker botsort.yaml --display
```

## 3. Run webcam

```powershell
python enhanced_video_runner.py --source 0 --display
```

## 4. Start dashboard

Use a second terminal:

```powershell
.venv\Scripts\activate
python -m streamlit run dashboard.py
```

Open:

```text
http://localhost:8501
```

## Dashboard sections

- **Live Monitor**: tracked frame, risk, track ID, speed, distance, pedestrian separation, TTC, FPS and latency.
- **Analytics**: risk trends, speed-distance behavior, movement directions, separation distribution and highest-risk tracks.
- **Performance**: average FPS, average/P95 latency, unique tracks.
- **Failure Cases**: validation matrix for poor lighting, rain, occlusion, overlapping vehicles, motion blur and camera shake.
- **Event Log**: complete feature-based event history.
- **Reports**: CSV export and summary.

## Risk score

The risk engine combines normalized features:

```text
confidence
+ vehicle speed
+ distance to crosswalk
+ pedestrian–vehicle separation
+ movement direction
+ predicted trajectory conflict
+ crosswalk occupancy
+ time-to-collision
+ environment flags
```

The weights and thresholds are in:

```text
config/settings.yaml
```

Important: the score is a **prototype safety indicator**, not a calibrated probability of collision.

## Speed and distance calibration

The default implementation converts pixel motion into metres using:

```yaml
tracking:
  pixels_per_meter: 22.0
```

This must be calibrated for the actual camera.

For stronger real-world accuracy, replace the simple scale with:
- a measured ground-plane reference,
- camera calibration,
- or a homography / perspective transform.

## Evaluation

Runtime-only evaluation:

```powershell
python evaluation\evaluate_video.py --video ".\videos\real_crosswalk.mp4"
```

This reports:
- processed frames
- average FPS
- average latency
- P95 latency

### Precision and recall

Precision and recall require labelled ground-truth frames.

Provide YOLO-format labels such as:

```text
labels/
├── frame_000001.txt
├── frame_000002.txt
└── ...
```

Then:

```powershell
python evaluation\evaluate_video.py `
  --video ".\videos\real_crosswalk.mp4" `
  --labels-dir ".\labels"
```

Do not claim precision/recall values when no ground-truth annotations were used.

## Automated tests

```powershell
python -m pytest -q
```

Tests currently validate:
- low-risk safe behavior
- high-risk conflict behavior
- trajectory prediction

## Docker

Build:

```powershell
docker build -t guidekaro-enhanced .
```

Run:

```powershell
docker run --rm -p 8501:8501 `
  -v "${PWD}\database:/app/database" `
  -v "${PWD}\assets:/app/assets" `
  guidekaro-enhanced
```

Or:

```powershell
docker compose up --build
```

## Grafana Cloud / observability

GuideKaro writes application logs to:

```text
logs/guidekaro.log
```

and also writes logs to the console, which makes them visible in Docker logs.

See:

```text
monitoring/grafana/README.md
```

for the recommended Grafana Cloud integration approach.

## Architecture

See:

- `docs/guidekaro_full_architecture.png`
- `docs/ARCHITECTURE.md`

The architecture includes:
- cameras and video sources
- AI/computer-vision processing
- tracking and risk engine
- backend/application layer
- dashboard
- SQLite storage
- alerts
- analytics
- Docker deployment
- logging and monitoring
- Grafana Cloud observability

## Failure-case evidence

A complete project should include evidence for:
- poor lighting
- heavy rain / snow
- occluded pedestrians
- multiple overlapping vehicles
- motion blur
- camera shake

Use:

```text
evaluation/failure_case_template.csv
```

to record test results.

## Reproducibility checklist

Before submission, record:
- Python version
- model file
- tracker
- confidence threshold
- camera/video resolution
- hardware
- pixels-per-meter calibration
- test video source/licence
- number of labelled frames
- precision/recall methodology
- FPS and latency
- failure cases
