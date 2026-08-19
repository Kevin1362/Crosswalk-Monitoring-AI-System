# GuideKaro Enhanced

**GuideKaro** is an AI-powered crosswalk safety prototype that detects and tracks vehicles and pedestrians, estimates motion features, predicts short-term trajectories, calculates a feature-based risk score, logs safety events, and presents live and historical analytics.

The enhanced version also includes a complete **MLOps service layer** using FastAPI, three ML use-case pipelines, Docker containerization, Docker Secrets Management, application/API/pipeline logging, and a Streamlit dashboard.

---

## Project Overview

GuideKaro is designed to support safer pedestrian crossings by processing live or recorded video and identifying potentially unsafe interactions between vehicles and pedestrians.

The system combines:

* Computer vision
* Object tracking
* Motion analysis
* Trajectory prediction
* Feature-based risk scoring
* Safe decision support
* Violation/event analytics
* Streamlit visualization
* FastAPI REST services
* MLOps pipelines
* Docker deployment
* Logging and monitoring

---

## ML Use Cases

GuideKaro currently implements three ML use cases.

### UC-01 — Early Warning / Conflict Prediction

Detects and tracks pedestrians and vehicles, analyzes their motion, predicts short-term trajectories, and determines whether their movement paths may conflict.

Pipeline stages include:

* Data Collection
* Preprocessing
* Object Detection
* Object Tracking
* Trajectory Prediction
* Conflict Prediction
* Pipeline Execution

---

### UC-02 — Safe Decision Support

Uses traffic and pedestrian features to calculate a risk score and recommend an appropriate safety state.

Input features include:

* Vehicle speed
* Pedestrian distance
* Crosswalk occupancy
* Blocked-zone status
* Trajectory conflict information

Possible safety states include:

```text
SAFE
WARNING
STOP
```

Pipeline stages include:

* Feature Collection
* Feature Preprocessing
* Risk Calculation
* Crosswalk Occupancy Check
* Decision Classification
* Validation
* Pipeline Execution

---

### UC-03 — Violation Detection and Analytics

Processes safety events, classifies high-risk conditions, logs events, aggregates results, and provides analytics for traffic authorities.

Pipeline stages include:

* Event Collection
* Event Preprocessing
* Violation Classification
* Event Logging
* Event Aggregation
* Analytics Generation
* Pipeline Execution

---

## MLOps Service Contracts

Each ML use case is exposed through REST API service contracts.

```text
UC-01: 7 service contracts
UC-02: 7 service contracts
UC-03: 7 service contracts

Total: 21 service contracts
```

All three complete MLOps pipelines can be executed through API calls.

The API service is implemented with **FastAPI**.

Swagger documentation is available at:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

---

## What Changed in This Enhanced Version

The original prototype mainly counted detections. The enhanced version adds:

* YOLOv8 object detection
* ByteTrack tracking with persistent object IDs
* Vehicle speed estimation
* Distance-to-crosswalk estimation
* Pedestrian-vehicle separation
* Movement direction:

  * `TOWARD_CROSSWALK`
  * `AWAY_FROM_CROSSWALK`
  * `STATIONARY`
* Short-term trajectory prediction
* Approximate time-to-collision (TTC)
* Feature-based risk engine
* Track-level database records
* Expanded Streamlit dashboard
* FPS and latency metrics
* Evaluation script for precision/recall when labelled ground truth is available
* Failure-case testing plan
* YAML configuration
* Exception handling
* Application logging
* API logging
* MLOps pipeline logging
* Pytest unit tests
* FastAPI REST service layer
* Three MLOps pipelines
* 21 API service contracts
* Docker packaging
* Docker Compose deployment
* Docker Secrets Management
* Grafana Cloud monitoring documentation
* Full project architecture diagram

---

# Project Structure

```text
GuideKaro_Enhanced/
├── api/
│   ├── __init__.py
│   └── main.py
│
├── pipelines/
│   ├── __init__.py
│   ├── uc1_pipeline.py
│   ├── uc2_pipeline.py
│   └── uc3_pipeline.py
│
├── services/
│   └── __init__.py
│
├── assets/
│
├── config/
│   └── settings.yaml
│
├── core/
│   ├── config.py
│   ├── geometry.py
│   ├── logging_setup.py
│   ├── risk_engine.py
│   └── trajectory.py
│
├── database/
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASET_AND_TESTING.md
│   ├── EVALUATION_METRICS.md
│   ├── FAILURE_CASES.md
│   └── guidekaro_full_architecture.png
│
├── evaluation/
│   ├── evaluate_video.py
│   └── failure_case_template.csv
│
├── logs/
│
├── monitoring/
│   └── grafana/
│       └── README.md
│
├── scripts/
├── secrets/
├── tests/
├── videos/
│
├── dashboard_latest.py
├── enhanced_video_runner.py
├── event_logger.py
├── Dockerfile
├── docker-compose.yml
├── start.sh
├── requirements.txt
└── README.md
```

---

# 1. Windows Setup

Open PowerShell in the project folder.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

# 2. Run Tracking on a Real Video

Put a video in the `videos/` folder.

Example:

```text
videos/real_crosswalk.mp4
```

Run:

```powershell
python enhanced_video_runner.py --source ".\videos\real_crosswalk.mp4" --display --loop
```

The default tracker is ByteTrack.

You can explicitly specify ByteTrack:

```powershell
python enhanced_video_runner.py --source ".\videos\real_crosswalk.mp4" --tracker bytetrack.yaml --display
```

Or use BoT-SORT:

```powershell
python enhanced_video_runner.py --source ".\videos\real_crosswalk.mp4" --tracker botsort.yaml --display
```

---

# 3. Run Webcam

```powershell
python enhanced_video_runner.py --source 0 --display
```

---

# 4. Start the FastAPI MLOps Service

Use a terminal with the virtual environment activated.

Run:

```powershell
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open the API root:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

Health check:

```text
http://localhost:8000/health
```

The API exposes the service contracts for:

```text
UC-01 Early Warning / Conflict Prediction
UC-02 Safe Decision Support
UC-03 Violation Detection and Analytics
```

---

# 5. Start the Streamlit Dashboard

Open a second terminal.

Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

Run the current dashboard:

```powershell
python -m streamlit run dashboard_latest.py
```

Open:

```text
http://localhost:8501
```

The dashboard communicates with the FastAPI MLOps service running on port `8000`.

The sidebar includes:

* MLOps API Online/Offline status
* UC-01 pipeline test
* UC-02 pipeline test
* UC-03 pipeline test

---

# Dashboard Sections

### Live Monitor

Displays:

* Tracked frame
* Risk score
* Track ID
* Vehicle speed
* Distance to crosswalk
* Pedestrian separation
* TTC
* Movement direction
* Trajectory conflict
* FPS
* Processing latency

### Traffic Analytics

Provides historical risk and traffic analytics including:

* Risk trends
* Speed-distance behaviour
* Movement directions
* Separation distribution
* Highest-risk tracked objects

### System Performance

Displays:

* Average FPS
* Average latency
* P95 latency
* Unique tracked objects

### Failure Cases

Supports validation for:

* Poor lighting
* Heavy rain/snow
* Occluded pedestrians
* Multiple overlapping vehicles
* Motion blur
* Camera shake

### Event Log

Displays feature-based event history.

### Reports

Provides:

* CSV export
* Event summary
* Safety analytics

---

# Risk Score

The GuideKaro risk engine combines normalized features including:

```text
confidence
+ vehicle speed
+ distance to crosswalk
+ pedestrian-vehicle separation
+ movement direction
+ predicted trajectory conflict
+ crosswalk occupancy
+ time-to-collision
+ environment flags
```

Weights and thresholds are configured in:

```text
config/settings.yaml
```

The risk score is a **prototype safety indicator** and should not be interpreted as a calibrated probability of collision.

---

# Speed and Distance Calibration

The default implementation converts pixel motion into metres using:

```yaml
tracking:
  pixels_per_meter: 22.0
```

This value must be calibrated for the actual camera configuration.

For stronger real-world accuracy, the system can use:

* A measured ground-plane reference
* Camera calibration
* Homography
* Perspective transformation

---

# Evaluation

Runtime evaluation:

```powershell
python evaluation\evaluate_video.py --video ".\videos\real_crosswalk.mp4"
```

This reports:

* Processed frames
* Average FPS
* Average latency
* P95 latency

---

## Precision and Recall

Precision and recall require labelled ground-truth data.

Example:

```text
labels/
├── frame_000001.txt
├── frame_000002.txt
└── ...
```

Then run:

```powershell
python evaluation\evaluate_video.py `
  --video ".\videos\real_crosswalk.mp4" `
  --labels-dir ".\labels"
```

Do not claim precision or recall results if labelled ground-truth annotations were not used.

---

# Automated Tests

Run:

```powershell
python -m pytest -q
```

Current tests include:

* Low-risk safe behaviour
* High-risk conflict behaviour
* Trajectory prediction

---

# Logging

GuideKaro includes logging for:

* Application startup
* API call start
* API call end
* MLOps pipeline start
* MLOps pipeline end
* Individual pipeline stages
* Application errors

Application logs are written to:

```text
logs/guidekaro.log
```

Logs are also written to the console, making them visible through Docker logs.

---

# Docker Deployment

The complete GuideKaro application is containerized.

The Docker container runs:

```text
FastAPI API       → Port 8000
Streamlit GUI     → Port 8501
```

Both interfaces bind to:

```text
0.0.0.0
```

so they can be exposed from the Docker host.

---

## Build Docker Image

```powershell
docker build -t guidekaro-mlops:latest .
```

---

## Run Docker Image

```powershell
docker run --rm `
  -p 8501:8501 `
  -p 8000:8000 `
  --name guidekaro-mlops-test `
  guidekaro-mlops:latest
```

Open:

```text
Streamlit:
http://localhost:8501

FastAPI Swagger:
http://localhost:8000/docs
```

---

# Docker Compose

Build and start the application:

```powershell
docker compose up -d --build
```

Check the running container:

```powershell
docker compose ps
```

Stop the application:

```powershell
docker compose down
```

The Compose configuration publishes:

```text
8501 → Streamlit
8000 → FastAPI
```

---

# Docker Secrets Management

GuideKaro uses Docker Secrets Management for sensitive configuration.

The secret is mounted inside the running container at:

```text
/run/secrets/guidekaro_secret
```

Verify that the secret is configured without displaying its value:

```powershell
docker compose exec guidekaro-dashboard sh -c "test -f /run/secrets/guidekaro_secret && echo 'GuideKaro Docker Secret: CONFIGURED'"
```

Expected output:

```text
GuideKaro Docker Secret: CONFIGURED
```

The actual secret value should never be printed in screenshots or committed to GitHub.

The following files/directories should remain excluded through `.gitignore` and `.dockerignore`:

```text
.env
.env.*
secrets/
```

---

# Docker Hub

Docker Hub repository:

```text
https://hub.docker.com/r/kevin1362/guidekaro-dashboard
```

Final MLOps image:

```text
kevin1362/guidekaro-dashboard:mlops-final
```

Pull the final image:

```powershell
docker pull kevin1362/guidekaro-dashboard:mlops-final
```

Run the published image:

```powershell
docker run --rm `
  -p 8501:8501 `
  -p 8000:8000 `
  kevin1362/guidekaro-dashboard:mlops-final
```

---

# MLOps API Service Contracts

## UC-01

Example endpoints:

```text
POST /api/uc1/collect
POST /api/uc1/preprocess
POST /api/uc1/detect
POST /api/uc1/track
POST /api/uc1/predict
POST /api/uc1/conflict
POST /api/uc1/pipeline/run
```

---

## UC-02

Example endpoints:

```text
POST /api/uc2/features
POST /api/uc2/preprocess
POST /api/uc2/risk
POST /api/uc2/occupancy
POST /api/uc2/decision
POST /api/uc2/validate
POST /api/uc2/pipeline/run
```

---

## UC-03

Example endpoints:

```text
POST /api/uc3/event
POST /api/uc3/preprocess
POST /api/uc3/classify
POST /api/uc3/log
GET  /api/uc3/aggregate
GET  /api/uc3/analytics
POST /api/uc3/pipeline/run
```

---

# Grafana Cloud / Observability

GuideKaro writes application logs to:

```text
logs/guidekaro.log
```

and also writes console logs that can be inspected using Docker.

See:

```text
monitoring/grafana/README.md
```

for the Grafana Cloud integration approach.

---

# Architecture

Architecture documentation is available at:

```text
docs/guidekaro_full_architecture.png
docs/ARCHITECTURE.md
```

The architecture includes:

* Cameras and video sources
* Computer-vision processing
* Object detection
* Object tracking
* Trajectory prediction
* Risk engine
* MLOps pipelines
* FastAPI service layer
* Streamlit dashboard
* SQLite storage
* Alerts
* Analytics
* Docker deployment
* Docker Secrets Management
* Logging
* Monitoring
* Grafana Cloud observability

---

# Failure-Case Evidence

A complete system evaluation should include evidence for:

* Poor lighting
* Heavy rain or snow
* Occluded pedestrians
* Multiple overlapping vehicles
* Motion blur
* Camera shake

Use:

```text
evaluation/failure_case_template.csv
```

to record results.

---

# Reproducibility Checklist

Before final deployment or evaluation, record:

* Python version
* Model file
* Tracker
* Confidence threshold
* Camera/video resolution
* Hardware
* Pixels-per-metre calibration
* Test video source/licence
* Number of labelled frames
* Precision/recall methodology
* FPS
* Processing latency
* Failure cases
* Docker image tag

---

# Sprint 6 / Project Management

Current Sprint 6 development branch:

```text
sprint-6-mlops
```

Sprint 6 work includes:

* MLOps pipeline implementation
* FastAPI service contracts
* Application/API/pipeline logging
* Streamlit-FastAPI integration
* Docker containerization
* Docker Secrets Management
* Local deployment testing
* Docker Hub publishing
* User Acceptance Testing activity
* Pull Request to `main`

---

# Project Links

### Docker Hub

```text
https://hub.docker.com/r/kevin1362/guidekaro-dashboard
```

### Azure DevOps

```text
https://dev.azure.com/Kpatel8612/AI-Based%20Crosswalk%20Monitoring%20and%20Driver%20Alert%20System/
```

### GitHub Repository

```text
https://github.com/Kevin1362/Crosswalk-Monitoring-AI-System
```

### Sprint 6 Branch

```text
https://github.com/Kevin1362/Crosswalk-Monitoring-AI-System/tree/sprint-6-mlops
```

### Pull Request

```text
https://github.com/Kevin1362/Crosswalk-Monitoring-AI-System/pull/7
```

---

# Final Deployment Interfaces

When the container is running:

```text
GuideKaro Streamlit Dashboard
http://localhost:8501
```

```text
GuideKaro FastAPI / Swagger
http://localhost:8000/docs
```

The application is configured so the GUI and API bind to `0.0.0.0`, allowing the interfaces to be published through Docker ports for remote deployment.
