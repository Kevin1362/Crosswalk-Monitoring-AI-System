
from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

from event_logger import GuideKaroEventLogger


def main() -> None:
    random.seed(42)
    db_path = Path("database/uc1_events.db")
    if db_path.exists():
        db_path.unlink()

    logger = GuideKaroEventLogger(db_path)
    intersections = [
        ("King St & Victoria St", 43.4516, -80.4925),
        ("Homer Watson & Block Line", 43.4202, -80.4752),
        ("Fairway Rd & Wilson Ave", 43.4255, -80.4386),
        ("Ottawa St & Fischer-Hallman", 43.4292, -80.5074),
        ("University Ave & King St", 43.4724, -80.5367),
    ]
    weather_options = ["Clear", "Snow", "Light snow", "Rain", "Fog"]
    road_options = ["Dry", "Wet", "Snow-covered", "Icy"]

    for index in range(240):
        name, lat, lon = random.choice(intersections)
        weather = random.choice(weather_options)
        road = random.choice(road_options)
        pedestrians = random.randint(0, 4)
        speed = max(5.0, min(85.0, random.gauss(42, 13)))
        distance = max(2.0, min(120.0, random.gauss(38, 22)))
        winter_factor = 18 if road in {"Snow-covered", "Icy"} else 0
        risk = max(
            0.0,
            min(
                100.0,
                speed * 0.7
                + winter_factor
                + pedestrians * 7
                + max(0, 35 - distance) * 0.75
                - 18,
            ),
        )
        blocked = risk > 78 and random.random() < 0.55
        status = "VIOLATION" if blocked or risk >= 70 else "WARNING" if risk >= 40 else "SAFE"

        logger.log_event(
            timestamp=(datetime.now() - timedelta(minutes=(239 - index) * 15)).isoformat(timespec="seconds"),
            intersection=name,
            latitude=lat + random.uniform(-0.0004, 0.0004),
            longitude=lon + random.uniform(-0.0004, 0.0004),
            weather=weather,
            road_condition=road,
            visibility="Low" if weather in {"Snow", "Fog"} else "Normal",
            vehicle_count=random.randint(1, 12),
            pedestrian_count=pedestrians,
            vehicle_speed_kmh=round(speed, 1),
            distance_to_crosswalk_m=round(distance, 1),
            confidence=round(random.uniform(0.48, 0.97), 2),
            risk_score=round(risk, 1),
            risk_level=status,
            status=status,
            crosswalk_blocked=blocked,
            alert_channel="Audio + Visual" if status == "VIOLATION" else "Visual" if status == "WARNING" else "None",
            response_time_ms=round(max(35.0, min(280.0, random.gauss(105, 35))), 1),
            notes="Generated sample event",
        )

    print(f"Created {db_path} with 240 sample events.")


if __name__ == "__main__":
    main()
