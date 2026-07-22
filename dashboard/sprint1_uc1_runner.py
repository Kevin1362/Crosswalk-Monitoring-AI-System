
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from event_logger import GuideKaroEventLogger


PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
LATEST_FRAME_PATH = ASSETS_DIR / "latest_frame.jpg"
DATABASE_PATH = PROJECT_DIR / "database" / "uc1_events.db"

PERSON_CLASS = 0
VEHICLE_CLASSES = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck


def parse_source(value: str):
    """Convert webcam numbers such as '0' to int; keep video paths as strings."""
    return int(value) if value.isdigit() else value


def point_inside_polygon(point: tuple[int, int], polygon) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def risk_status(
    pedestrian_in_crosswalk: bool,
    vehicle_in_crosswalk: bool,
    consecutive_risk_frames: int,
) -> tuple[str, float, bool]:
    """
    Simple GuideKaro demonstration decision logic.

    SAFE:
        No conflicting pedestrian/vehicle condition.
    WARNING:
        A pedestrian or vehicle is inside the crosswalk region.
    VIOLATION:
        A vehicle and pedestrian are both inside the region for at least
        three consecutive frames, or a vehicle remains in the crosswalk.
    """
    blocked = vehicle_in_crosswalk

    if pedestrian_in_crosswalk and vehicle_in_crosswalk and consecutive_risk_frames >= 3:
        return "VIOLATION", 90.0, blocked

    if vehicle_in_crosswalk and consecutive_risk_frames >= 5:
        return "VIOLATION", 78.0, blocked

    if pedestrian_in_crosswalk or vehicle_in_crosswalk:
        return "WARNING", 50.0, blocked

    return "SAFE", 10.0, False


def main() -> None:
    parser = argparse.ArgumentParser(description="GuideKaro YOLO camera runner")
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam number such as 0, or a path to a video file",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics model file, for example yolov8n.pt",
    )
    parser.add_argument(
        "--intersection",
        default="GuideKaro Demo Crosswalk",
        help="Intersection name stored in the event database",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="Minimum detection confidence",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Open an OpenCV preview window",
    )
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger = GuideKaroEventLogger(DATABASE_PATH)
    model = YOLO(args.model)

    source = parse_source(args.source)
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera/video source: {args.source}. "
            "Try another webcam number such as --source 1."
        )

    consecutive_risk_frames = 0
    last_logged_status = None
    last_log_time = 0.0

    print("GuideKaro YOLO runner started.")
    print(f"Saving latest frame to: {LATEST_FRAME_PATH}")
    print(f"Writing events to: {DATABASE_PATH}")
    print("Press Ctrl+C in the terminal to stop.")
    if args.display:
        print("Press Q in the preview window to stop.")

    try:
        while True:
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                print("The camera/video did not return another frame.")
                break

            height, width = frame.shape[:2]

            # Demonstration crosswalk region: lower-middle area of the frame.
            # Adjust these points later to match the real crosswalk in your video.
            crosswalk_polygon = np.array(
                [
                    [int(width * 0.20), int(height * 0.58)],
                    [int(width * 0.80), int(height * 0.58)],
                    [int(width * 0.95), int(height * 0.95)],
                    [int(width * 0.05), int(height * 0.95)],
                ],
                dtype=np.int32,
            )

            results = model.predict(
                frame,
                conf=args.confidence,
                verbose=False,
            )

            annotated = frame.copy()
            pedestrian_count = 0
            vehicle_count = 0
            pedestrian_in_crosswalk = False
            vehicle_in_crosswalk = False
            highest_confidence = 0.0

            result = results[0]
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    highest_confidence = max(highest_confidence, confidence)

                    if class_id != PERSON_CLASS and class_id not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    inside = point_inside_polygon(center, crosswalk_polygon)

                    if class_id == PERSON_CLASS:
                        pedestrian_count += 1
                        pedestrian_in_crosswalk = pedestrian_in_crosswalk or inside
                        label = f"Person {confidence:.2f}"
                    else:
                        vehicle_count += 1
                        vehicle_in_crosswalk = vehicle_in_crosswalk or inside
                        label = f"Vehicle {confidence:.2f}"

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(annotated, center, 4, (255, 255, 255), -1)
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            if pedestrian_in_crosswalk or vehicle_in_crosswalk:
                consecutive_risk_frames += 1
            else:
                consecutive_risk_frames = 0

            status, risk_score, blocked = risk_status(
                pedestrian_in_crosswalk,
                vehicle_in_crosswalk,
                consecutive_risk_frames,
            )

            status_colour = {
                "SAFE": (0, 180, 0),
                "WARNING": (0, 200, 255),
                "VIOLATION": (0, 0, 255),
            }[status]

            cv2.polylines(
                annotated,
                [crosswalk_polygon],
                isClosed=True,
                color=status_colour,
                thickness=3,
            )

            cv2.putText(
                annotated,
                f"GuideKaro Status: {status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_colour,
                3,
            )
            cv2.putText(
                annotated,
                f"Pedestrians: {pedestrian_count}  Vehicles: {vehicle_count}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated,
                f"Risk score: {risk_score:.0f}/100",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            processing_ms = (time.perf_counter() - started) * 1000

            # Save the latest annotated frame for Streamlit.
            cv2.imwrite(str(LATEST_FRAME_PATH), annotated)

            # Avoid inserting a database row for every video frame.
            # Log on status change or once every 10 seconds.
            now = time.time()
            should_log = status != last_logged_status or (now - last_log_time) >= 10

            if should_log:
                alert_channel = (
                    "Audio + Visual"
                    if status == "VIOLATION"
                    else "Visual"
                    if status == "WARNING"
                    else "None"
                )

                logger.log_event(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    intersection=args.intersection,
                    weather="Unknown",
                    road_condition="Unknown",
                    visibility="Unknown",
                    vehicle_count=vehicle_count,
                    pedestrian_count=pedestrian_count,
                    vehicle_speed_kmh=0.0,
                    distance_to_crosswalk_m=0.0,
                    confidence=round(highest_confidence, 3),
                    risk_score=risk_score,
                    risk_level=status,
                    status=status,
                    crosswalk_blocked=blocked,
                    alert_channel=alert_channel,
                    response_time_ms=round(processing_ms, 1),
                    frame_path="assets/latest_frame.jpg",
                    notes=f"Risk confirmed for {consecutive_risk_frames} consecutive frames",
                )
                last_logged_status = status
                last_log_time = now

            if args.display:
                cv2.imshow("GuideKaro YOLO Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopping GuideKaro runner.")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
