
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


BASE_COLUMNS = {
    "timestamp": "TEXT NOT NULL",
    "intersection": "TEXT",
    "latitude": "REAL",
    "longitude": "REAL",
    "weather": "TEXT",
    "road_condition": "TEXT",
    "visibility": "TEXT",
    "vehicle_count": "INTEGER",
    "pedestrian_count": "INTEGER",
    "vehicle_speed_kmh": "REAL",
    "distance_to_crosswalk_m": "REAL",
    "pedestrian_vehicle_distance_m": "REAL",
    "confidence": "REAL",
    "risk_score": "REAL",
    "risk_level": "TEXT",
    "status": "TEXT",
    "crosswalk_blocked": "INTEGER",
    "alert_channel": "TEXT",
    "response_time_ms": "REAL",
    "fps": "REAL",
    "track_id": "INTEGER",
    "object_class": "TEXT",
    "movement_direction": "TEXT",
    "trajectory_conflict": "INTEGER",
    "ttc_seconds": "REAL",
    "predicted_x": "REAL",
    "predicted_y": "REAL",
    "frame_path": "TEXT",
    "notes": "TEXT",
}


class GuideKaroEventLogger:
    def __init__(self, db_path: str | Path = "database/uc1_events.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            defs = ", ".join(f"{name} {dtype}" for name, dtype in BASE_COLUMNS.items())
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, {defs})"
            )

            existing = {
                row[1] for row in connection.execute("PRAGMA table_info(events)").fetchall()
            }
            for name, dtype in BASE_COLUMNS.items():
                if name not in existing:
                    connection.execute(f"ALTER TABLE events ADD COLUMN {name} {dtype}")

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS track_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    track_id INTEGER,
                    object_class TEXT,
                    x REAL,
                    y REAL,
                    speed_kmh REAL,
                    distance_to_crosswalk_m REAL,
                    movement_direction TEXT,
                    predicted_x REAL,
                    predicted_y REAL
                )
                """
            )
            connection.commit()

    def log_event(self, **event: Any) -> int:
        record = {
            "timestamp": event.get("timestamp", datetime.now().isoformat(timespec="seconds")),
            "intersection": event.get("intersection", "GuideKaro Intersection"),
            "latitude": event.get("latitude"),
            "longitude": event.get("longitude"),
            "weather": event.get("weather", "Unknown"),
            "road_condition": event.get("road_condition", "Unknown"),
            "visibility": event.get("visibility", "Unknown"),
            "vehicle_count": int(event.get("vehicle_count", 0)),
            "pedestrian_count": int(event.get("pedestrian_count", 0)),
            "vehicle_speed_kmh": float(event.get("vehicle_speed_kmh", 0.0)),
            "distance_to_crosswalk_m": float(event.get("distance_to_crosswalk_m", 0.0)),
            "pedestrian_vehicle_distance_m": float(event.get("pedestrian_vehicle_distance_m", 0.0)),
            "confidence": float(event.get("confidence", 0.0)),
            "risk_score": float(event.get("risk_score", 0.0)),
            "risk_level": event.get("risk_level", event.get("status", "SAFE")),
            "status": event.get("status", "SAFE"),
            "crosswalk_blocked": int(bool(event.get("crosswalk_blocked", False))),
            "alert_channel": event.get("alert_channel", "None"),
            "response_time_ms": float(event.get("response_time_ms", 0.0)),
            "fps": float(event.get("fps", 0.0)),
            "track_id": event.get("track_id"),
            "object_class": event.get("object_class"),
            "movement_direction": event.get("movement_direction", "UNKNOWN"),
            "trajectory_conflict": int(bool(event.get("trajectory_conflict", False))),
            "ttc_seconds": event.get("ttc_seconds"),
            "predicted_x": event.get("predicted_x"),
            "predicted_y": event.get("predicted_y"),
            "frame_path": event.get("frame_path", ""),
            "notes": event.get("notes", ""),
        }
        cols = ", ".join(record)
        vals = tuple(record.values())
        qs = ", ".join("?" for _ in vals)

        with sqlite3.connect(self.db_path) as connection:
            cur = connection.execute(
                f"INSERT INTO events ({cols}) VALUES ({qs})",
                vals,
            )
            connection.commit()
            return int(cur.lastrowid)

    def log_track_sample(self, **sample: Any) -> int:
        record = {
            "timestamp": sample.get("timestamp", datetime.now().isoformat(timespec="seconds")),
            "track_id": sample.get("track_id"),
            "object_class": sample.get("object_class"),
            "x": sample.get("x"),
            "y": sample.get("y"),
            "speed_kmh": sample.get("speed_kmh", 0.0),
            "distance_to_crosswalk_m": sample.get("distance_to_crosswalk_m", 0.0),
            "movement_direction": sample.get("movement_direction", "UNKNOWN"),
            "predicted_x": sample.get("predicted_x"),
            "predicted_y": sample.get("predicted_y"),
        }
        cols = ", ".join(record)
        vals = tuple(record.values())
        qs = ", ".join("?" for _ in vals)
        with sqlite3.connect(self.db_path) as connection:
            cur = connection.execute(
                f"INSERT INTO track_samples ({cols}) VALUES ({qs})",
                vals,
            )
            connection.commit()
            return int(cur.lastrowid)
