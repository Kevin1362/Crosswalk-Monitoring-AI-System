
# Add this pattern to dev/sprint1_uc1_runner.py after your imports.

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from event_logger import GuideKaroEventLogger

logger = GuideKaroEventLogger(PROJECT_ROOT / "database" / "uc1_events.db")

# Call this after your detection and risk-decision logic.
# Replace the example variables with your existing variable names.

event_id = logger.log_event(
    intersection=intersection_name,
    latitude=latitude,
    longitude=longitude,
    weather=weather_label,
    road_condition=road_condition_label,
    visibility=visibility_label,
    vehicle_count=len(vehicle_detections),
    pedestrian_count=len(pedestrian_detections),
    vehicle_speed_kmh=estimated_speed_kmh,
    distance_to_crosswalk_m=estimated_distance_m,
    confidence=highest_detection_confidence,
    risk_score=risk_score,
    risk_level=status,                  # SAFE / WARNING / VIOLATION
    status=status,
    crosswalk_blocked=crosswalk_blocked,
    alert_channel=alert_channel,        # None / Visual / Audio + Visual
    response_time_ms=processing_ms,
    frame_path="dashboard/assets/latest_frame.jpg",
    notes=f"Confirmed for {consecutive_frames} frames",
)

# Save the annotated frame after bounding boxes and overlays are drawn.
cv2.imwrite(
    str(PROJECT_ROOT / "dashboard" / "assets" / "latest_frame.jpg"),
    annotated_frame,
)
