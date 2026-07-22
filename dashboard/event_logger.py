
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    intersection TEXT,
    latitude REAL,
    longitude REAL,
    weather TEXT,
    road_condition TEXT,
    visibility TEXT,
    vehicle_count INTEGER,
    pedestrian_count INTEGER,
    vehicle_speed_kmh REAL,
    distance_to_crosswalk_m REAL,
    confidence REAL,
    risk_score REAL,
    risk_level TEXT,
    status TEXT,
    crosswalk_blocked INTEGER,
    alert_channel TEXT,
    response_time_ms REAL,
    frame_path TEXT,
    notes TEXT
)
"""


class GuideKaroEventLogger:
    """Small SQLite logger that can be called from the YOLO/OpenCV runner."""

    def __init__(self, db_path: str | Path = "database/uc1_events.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(SCHEMA)
            connection.commit()

    def log_event(self, **event: Any) -> int:
        record = {
            "timestamp": event.get("timestamp", datetime.now().isoformat(timespec="seconds")),
            "intersection": event.get("intersection", "Demo Crosswalk"),
            "latitude": event.get("latitude"),
            "longitude": event.get("longitude"),
            "weather": event.get("weather", "Unknown"),
            "road_condition": event.get("road_condition", "Unknown"),
            "visibility": event.get("visibility", "Unknown"),
            "vehicle_count": event.get("vehicle_count", 0),
            "pedestrian_count": event.get("pedestrian_count", 0),
            "vehicle_speed_kmh": event.get("vehicle_speed_kmh", 0.0),
            "distance_to_crosswalk_m": event.get("distance_to_crosswalk_m", 0.0),
            "confidence": event.get("confidence", 0.0),
            "risk_score": event.get("risk_score", 0.0),
            "risk_level": event.get("risk_level", event.get("status", "SAFE")),
            "status": event.get("status", "SAFE"),
            "crosswalk_blocked": int(bool(event.get("crosswalk_blocked", False))),
            "alert_channel": event.get("alert_channel", "None"),
            "response_time_ms": event.get("response_time_ms", 0.0),
            "frame_path": event.get("frame_path", ""),
            "notes": event.get("notes", ""),
        }

        columns = ", ".join(record.keys())
        placeholders = ", ".join("?" for _ in record)
        values = tuple(record.values())

        with sqlite3.connect(self.db_path) as connection:
            cursor = connection.execute(
                f"INSERT INTO events ({columns}) VALUES ({placeholders})",
                values,
            )
            connection.commit()
            return int(cursor.lastrowid)


if __name__ == "__main__":
    logger = GuideKaroEventLogger()
    event_id = logger.log_event(
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
        frame_path="assets/latest_frame.jpg",
        notes="Example event inserted by event_logger.py",
    )
    print(f"Inserted GuideKaro event #{event_id}")
