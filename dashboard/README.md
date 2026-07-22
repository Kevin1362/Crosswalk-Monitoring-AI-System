
# GuideKaro Streamlit Dashboard

This dashboard is designed for the GuideKaro Intelligent Crosswalk Monitoring and Driver Alert System.

## What it shows

- Live GuideKaro state: `SAFE`, `WARNING`, or `VIOLATION`
- Latest YOLO/OpenCV frame and detection information
- Vehicle speed, distance, pedestrian count, confidence, risk score, and latency
- False-alarm control checks
- City safety KPI cards
- Crosswalk hotspot map and intersection ranking
- Weather, road-condition, hourly, and detection-performance analytics
- Searchable event log with CSV export
- Downloadable city safety report

## Project flow

Camera/video → YOLO/OpenCV detection → risk decision engine → alert → SQLite event log → Streamlit dashboard

## Recommended folder placement in your existing GuideKaro project

```text
GuideKaro/
├── dashboard/
│   ├── dashboard.py
│   └── assets/
│       └── latest_frame.jpg
├── database/
│   └── uc1_events.db
├── dev/
│   └── sprint1_uc1_runner.py
├── event_logger.py
└── requirements.txt
```

The dashboard searches both:

- `dashboard/database/uc1_events.db`
- `database/uc1_events.db` from the project root

You can also set an exact database path:

```powershell
$env:GUIDEKARO_DB_PATH="C:\full\path\to\GuideKaro\database\uc1_events.db"
```

## Windows / PyCharm setup

Open the terminal in the `GuideKaro_Dashboard` folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python create_sample_db.py
streamlit run dashboard.py
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
```

You can also run Streamlit without activating the environment:

```powershell
.venv\Scripts\python.exe -m streamlit run dashboard.py
```

## Connect it to the YOLO runner

Import and create the logger once:

```python
from event_logger import GuideKaroEventLogger

logger = GuideKaroEventLogger("database/uc1_events.db")
```

After the decision engine produces a result, log it:

```python
logger.log_event(
    intersection="King St & Victoria St",
    latitude=43.4516,
    longitude=-80.4925,
    weather="Snow",
    road_condition="Icy",
    visibility="Low",
    vehicle_count=3,
    pedestrian_count=1,
    vehicle_speed_kmh=46.0,
    distance_to_crosswalk_m=18.0,
    confidence=0.91,
    risk_score=84.0,
    risk_level="VIOLATION",
    status="VIOLATION",
    crosswalk_blocked=True,
    alert_channel="Audio + Visual",
    response_time_ms=98.0,
    frame_path="dashboard/assets/latest_frame.jpg",
    notes="Vehicle remained inside crosswalk ROI for 5 frames",
)
```

Save the latest annotated frame from OpenCV:

```python
cv2.imwrite("dashboard/assets/latest_frame.jpg", annotated_frame)
```

## False-alarm logic to keep in the detection service

The dashboard displays the outcome, but the decision engine should confirm:

1. The object overlaps the crosswalk region of interest.
2. Confidence is at least `0.45`.
3. The condition is present for `3–5` consecutive frames.
4. Object tracking confirms it is the same object.
5. A loud audio alert is not repeated continuously. The visual warning remains until the crosswalk clears.

A vehicle that is physically stuck on the crosswalk is not treated as a false detection. The visual warning stays active, while the repeated loud sound can stop after a few seconds to reduce driver frustration.

## Troubleshooting “Waiting for YOLO detections…”

1. Confirm the YOLO runner and dashboard use the same file: `database/uc1_events.db`.
2. Run:

```powershell
python event_logger.py
```

3. Refresh the dashboard. If the example appears, the database connection is working.
4. Make sure your runner inserts rows into an `events` table.
5. Make sure `dashboard/assets/latest_frame.jpg` is being updated if you want live visual evidence.

## Demo mode

If the database is missing or empty, the dashboard automatically shows realistic demonstration data. A warning appears in the sidebar so it is clear that the data is not live.
