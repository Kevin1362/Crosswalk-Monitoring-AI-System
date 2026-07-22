
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from event_logger import GuideKaroEventLogger


PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
DATABASE_PATH = PROJECT_DIR / "database" / "uc1_events.db"
LATEST_FRAME_PATH = ASSETS_DIR / "latest_frame.jpg"

PERSON_CLASS = 0
VEHICLE_CLASSES = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def box_center(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def point_in_polygon(point: tuple[int, int], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def nearest_distance(
    people: list[tuple[int, int, int, int]],
    vehicles: list[tuple[int, int, int, int]],
) -> Optional[float]:
    if not people or not vehicles:
        return None

    smallest = float("inf")
    for person in people:
        px, py = box_center(person)
        for vehicle in vehicles:
            vx, vy = box_center(vehicle)
            smallest = min(smallest, math.hypot(px - vx, py - vy))
    return smallest


def calculate_risk(
    *,
    people: list[tuple[int, int, int, int]],
    vehicles: list[tuple[int, int, int, int]],
    person_in_zone: bool,
    vehicle_in_zone: bool,
    nearest_distance_px: Optional[float],
    frame_width: int,
    consecutive_conflict_frames: int,
    presentation_mode: bool,
) -> tuple[float, str, list[str]]:
    """
    Transparent demonstration risk model.

    This is not a calibrated collision-probability model. It combines visible
    conditions so the presentation can explain why risk rises or falls.
    """
    score = 5.0
    reasons: list[str] = []

    if people:
        score += 15
        reasons.append("pedestrian detected")

    if vehicles:
        score += 10
        reasons.append("vehicle detected")

    if people and vehicles:
        score += 15
        reasons.append("pedestrian and vehicle present together")

    if person_in_zone:
        score += 12
        reasons.append("pedestrian inside monitored zone")

    if vehicle_in_zone:
        score += 12
        reasons.append("vehicle inside monitored zone")

    if person_in_zone and vehicle_in_zone:
        score += 15
        reasons.append("both inside monitored zone")

    if nearest_distance_px is not None:
        normalized_distance = nearest_distance_px / max(frame_width, 1)

        if normalized_distance < 0.15:
            score += 25
            reasons.append("very small pedestrian–vehicle separation")
        elif normalized_distance < 0.30:
            score += 18
            reasons.append("small pedestrian–vehicle separation")
        elif normalized_distance < 0.50:
            score += 8
            reasons.append("moderate pedestrian–vehicle separation")

    if consecutive_conflict_frames >= 5:
        score += 8
        reasons.append("risk confirmed across consecutive frames")
    elif consecutive_conflict_frames >= 3:
        score += 4
        reasons.append("risk visible across multiple frames")

    # In presentation mode the whole frame is treated as the monitored approach.
    # This helps generic demonstration videos create meaningful risk changes.
    if presentation_mode and people and vehicles:
        score += 5
        reasons.append("presentation monitoring area active")

    score = min(100.0, score)

    if score >= 70:
        status = "VIOLATION"
    elif score >= 40:
        status = "WARNING"
    else:
        status = "SAFE"

    return score, status, reasons


def open_capture(source):
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video source: {source}. Check the path or webcam number."
        )
    return capture


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GuideKaro presentation video detector and risk-score demonstration"
    )
    parser.add_argument(
        "--source",
        default="videos/guidekaro_demo.mp4",
        help="Video path or webcam number such as 0",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics detection model",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Minimum YOLO confidence",
    )
    parser.add_argument(
        "--intersection",
        default="GuideKaro Presentation Crosswalk",
        help="Location stored in the event database",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Restart a video automatically when it ends",
    )
    parser.add_argument(
        "--presentation-mode",
        action="store_true",
        help="Treat the complete video as a monitored traffic approach",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Process without opening the OpenCV presentation window",
    )
    args = parser.parse_args()

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    source = parse_source(args.source)
    model = YOLO(args.model)
    logger = GuideKaroEventLogger(DATABASE_PATH)
    capture = open_capture(source)

    consecutive_conflict_frames = 0
    last_log_time = 0.0
    last_status = None

    print("GuideKaro presentation detector started.")
    print(f"Input: {args.source}")
    print(f"Dashboard image: {LATEST_FRAME_PATH}")
    print(f"Database: {DATABASE_PATH}")
    print("Press Q in the video window to stop.")

    try:
        while True:
            frame_started = time.perf_counter()
            success, frame = capture.read()

            if not success:
                if args.loop and not isinstance(source, int):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            height, width = frame.shape[:2]

            if args.presentation_mode:
                monitored_zone = np.array(
                    [
                        [int(width * 0.02), int(height * 0.12)],
                        [int(width * 0.98), int(height * 0.12)],
                        [int(width * 0.98), int(height * 0.98)],
                        [int(width * 0.02), int(height * 0.98)],
                    ],
                    dtype=np.int32,
                )
            else:
                monitored_zone = np.array(
                    [
                        [int(width * 0.20), int(height * 0.55)],
                        [int(width * 0.80), int(height * 0.55)],
                        [int(width * 0.96), int(height * 0.96)],
                        [int(width * 0.04), int(height * 0.96)],
                    ],
                    dtype=np.int32,
                )

            results = model.predict(
                source=frame,
                conf=args.confidence,
                verbose=False,
            )
            result = results[0]
            annotated = frame.copy()

            people_boxes: list[tuple[int, int, int, int]] = []
            vehicle_boxes: list[tuple[int, int, int, int]] = []
            person_in_zone = False
            vehicle_in_zone = False
            highest_confidence = 0.0

            if result.boxes is not None:
                for detected_box in result.boxes:
                    class_id = int(detected_box.cls[0].item())
                    confidence = float(detected_box.conf[0].item())

                    if class_id != PERSON_CLASS and class_id not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, detected_box.xyxy[0].tolist())
                    current_box = (x1, y1, x2, y2)
                    center = box_center(current_box)
                    inside_zone = point_in_polygon(center, monitored_zone)
                    class_name = str(model.names.get(class_id, class_id))
                    highest_confidence = max(highest_confidence, confidence)

                    if class_id == PERSON_CLASS:
                        people_boxes.append(current_box)
                        person_in_zone = person_in_zone or inside_zone
                        box_colour = (0, 255, 255)
                    else:
                        vehicle_boxes.append(current_box)
                        vehicle_in_zone = vehicle_in_zone or inside_zone
                        box_colour = (255, 170, 0)

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_colour, 2)
                    cv2.circle(annotated, center, 4, box_colour, -1)
                    cv2.putText(
                        annotated,
                        f"{class_name} {confidence:.2f}",
                        (x1, max(24, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        box_colour,
                        2,
                    )

            conflict_visible = bool(people_boxes and vehicle_boxes)
            if conflict_visible:
                consecutive_conflict_frames += 1
            else:
                consecutive_conflict_frames = 0

            distance_px = nearest_distance(people_boxes, vehicle_boxes)

            risk_score, status, reasons = calculate_risk(
                people=people_boxes,
                vehicles=vehicle_boxes,
                person_in_zone=person_in_zone,
                vehicle_in_zone=vehicle_in_zone,
                nearest_distance_px=distance_px,
                frame_width=width,
                consecutive_conflict_frames=consecutive_conflict_frames,
                presentation_mode=args.presentation_mode,
            )

            status_colour = {
                "SAFE": (0, 190, 0),
                "WARNING": (0, 200, 255),
                "VIOLATION": (0, 0, 255),
            }[status]

            cv2.polylines(
                annotated,
                [monitored_zone],
                isClosed=True,
                color=status_colour,
                thickness=3,
            )

            # Dark information panel for clear projection during a presentation.
            overlay = annotated.copy()
            panel_height = min(190, height)
            cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
            annotated = cv2.addWeighted(overlay, 0.64, annotated, 0.36, 0)

            cv2.putText(
                annotated,
                f"GuideKaro Status: {status}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_colour,
                3,
            )
            cv2.putText(
                annotated,
                f"Risk Score: {risk_score:.0f}/100",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                annotated,
                f"Pedestrians: {len(people_boxes)} | Vehicles: {len(vehicle_boxes)}",
                (20, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )

            distance_text = (
                f"Nearest separation: {distance_px:.0f} pixels"
                if distance_px is not None
                else "Nearest separation: unavailable"
            )
            cv2.putText(
                annotated,
                distance_text,
                (20, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )

            reason_text = ", ".join(reasons[:3]) if reasons else "no immediate conflict"
            cv2.putText(
                annotated,
                f"Why: {reason_text}",
                (20, 168),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 255),
                1,
            )

            processing_ms = (time.perf_counter() - frame_started) * 1000
            cv2.imwrite(str(LATEST_FRAME_PATH), annotated)

            now = time.time()
            should_log = status != last_status or now - last_log_time >= 5
            if should_log:
                logger.log_event(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    intersection=args.intersection,
                    weather="Presentation video",
                    road_condition="Video demonstration",
                    visibility="Normal",
                    vehicle_count=len(vehicle_boxes),
                    pedestrian_count=len(people_boxes),
                    vehicle_speed_kmh=0.0,
                    distance_to_crosswalk_m=0.0,
                    confidence=round(highest_confidence, 3),
                    risk_score=round(risk_score, 1),
                    risk_level=status,
                    status=status,
                    crosswalk_blocked=bool(vehicle_in_zone),
                    alert_channel=(
                        "Audio + Visual"
                        if status == "VIOLATION"
                        else "Visual"
                        if status == "WARNING"
                        else "None"
                    ),
                    response_time_ms=round(processing_ms, 1),
                    frame_path="assets/latest_frame.jpg",
                    notes="; ".join(reasons),
                )
                last_log_time = now
                last_status = status

            if not args.no_display:
                cv2.imshow("GuideKaro Presentation Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    cv2.waitKey(0)

    except KeyboardInterrupt:
        print("\nGuideKaro presentation stopped.")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
