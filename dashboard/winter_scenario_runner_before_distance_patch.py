from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from event_logger import GuideKaroEventLogger


PROJECT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_DIR / "assets"
DATABASE_PATH = PROJECT_DIR / "database" / "uc1_events.db"
LATEST_FRAME_PATH = ASSETS_DIR / "latest_frame.jpg"

CROSSWALK = (565, 365, 720, 650)  # x1, y1, x2, y2


def largest_box(mask: np.ndarray, min_area: int) -> tuple[int, int, int, int] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area >= min_area and area > best_area:
            best = (x, y, x + w, y + h)
            best_area = area
    return best


def intersects(a, b) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def center(box):
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def detect_simulated_objects(frame: np.ndarray):
    """
    Detect the controlled simulation objects by their stable presentation colors.

    This file is intentionally a controlled prototype detector, not a replacement
    for YOLO on real traffic-camera footage.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red car; limit to road area so the red traffic light is excluded.
    red1 = cv2.inRange(hsv, np.array([0, 130, 100]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 130, 100]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(red1, red2)
    red[:350, :] = 0
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    car_box = largest_box(red, 2500)

    # Blue pedestrian coat.
    blue = cv2.inRange(hsv, np.array([95, 100, 70]), np.array([130, 255, 255]))
    blue[:250, :] = 0
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    person_box = largest_box(blue, 350)

    return car_box, person_box


def calculate_risk(
    car_box,
    person_box,
    speed_kmh: float,
) -> tuple[float, str, list[str], float]:
    score = 10.0
    reasons = ["snow/ice road condition", "vehicle signal is red"]
    score += 20  # winter road
    score += 12  # red light

    crosswalk = CROSSWALK
    car_on_crosswalk = bool(car_box and intersects(car_box, crosswalk))
    person_on_crosswalk = bool(person_box and intersects(person_box, crosswalk))

    if person_box:
        score += 10
        reasons.append("pedestrian detected")
    if car_box:
        score += 8
        reasons.append("vehicle detected")
    if person_on_crosswalk:
        score += 15
        reasons.append("pedestrian inside crosswalk")
    if car_on_crosswalk:
        score += 22
        reasons.append("vehicle overlaps crosswalk")

    distance_px = 9999.0
    if car_box and person_box:
        cx, cy = center(car_box)
        px, py = center(person_box)
        distance_px = math.hypot(cx - px, cy - py)
        if distance_px < 170:
            score += 25
            reasons.append("critical separation")
        elif distance_px < 300:
            score += 16
            reasons.append("small separation")
        elif distance_px < 500:
            score += 8
            reasons.append("approaching conflict")

    if speed_kmh >= 35:
        score += 12
        reasons.append("high approach speed")
    elif speed_kmh >= 15:
        score += 6
        reasons.append("vehicle still moving")

    score = min(score, 100.0)
    status = "VIOLATION" if score >= 70 else "WARNING" if score >= 40 else "SAFE"
    return score, status, reasons, distance_px


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="videos/guidekaro_winter_crosswalk_scenario.mp4",
        help="Path to the controlled winter scenario video",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = PROJECT_DIR / source_path

    if not source_path.exists():
        raise FileNotFoundError(f"Video not found: {source_path}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {source_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24
    logger = GuideKaroEventLogger(DATABASE_PATH)

    previous_car_center_x = None
    previous_time = None
    last_logged_status = None
    last_log_time = 0.0

    print("GuideKaro controlled winter scenario started.")
    print("This runner uses color/position detection for the synthetic presentation video.")
    print("Press Q to stop, Space to pause.")

    while True:
        started = time.perf_counter()
        ok, frame = capture.read()
        if not ok:
            if args.loop:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                previous_car_center_x = None
                previous_time = None
                continue
            break

        video_time = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        car_box, person_box = detect_simulated_objects(frame)

        speed_kmh = 0.0
        if car_box:
            car_center_x = center(car_box)[0]
            if previous_car_center_x is not None and previous_time is not None:
                dt = max(video_time - previous_time, 1 / fps)
                px_per_sec = abs(car_center_x - previous_car_center_x) / dt
                # Presentation calibration for this controlled scene.
                speed_kmh = min(65.0, px_per_sec * 0.38)
            previous_car_center_x = car_center_x
            previous_time = video_time

        risk_score, status, reasons, distance_px = calculate_risk(
            car_box, person_box, speed_kmh
        )

        colours = {
            "SAFE": (0, 180, 0),
            "WARNING": (0, 190, 255),
            "VIOLATION": (0, 0, 255),
        }
        colour = colours[status]

        # Monitored crosswalk
        x1, y1, x2, y2 = CROSSWALK
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)
        cv2.putText(
            frame, "CROSSWALK ROI", (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2
        )

        if car_box:
            cv2.rectangle(frame, (car_box[0], car_box[1]), (car_box[2], car_box[3]), (255, 120, 30), 3)
            cv2.putText(
                frame, "VEHICLE DETECTED", (car_box[0], max(25, car_box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 120, 30), 2
            )

        if person_box:
            cv2.rectangle(frame, (person_box[0], person_box[1]), (person_box[2], person_box[3]), (0, 255, 255), 3)
            cv2.putText(
                frame, "PEDESTRIAN DETECTED", (person_box[0] - 40, max(25, person_box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2
            )

        # Dashboard-style panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 55), (510, 250), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.68, frame, 0.32, 0)

        cv2.putText(frame, f"GuideKaro: {status}", (35, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.92, colour, 3)
        cv2.putText(frame, f"Risk score: {risk_score:.0f}/100", (35, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255,255,255), 2)
        cv2.putText(frame, f"Estimated speed: {speed_kmh:.1f} km/h", (35, 165),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255,255,255), 2)
        separation = "N/A" if distance_px > 9000 else f"{distance_px:.0f} px"
        cv2.putText(frame, f"Pedestrian-vehicle separation: {separation}", (35, 198),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255,255,255), 2)
        cv2.putText(frame, f"Reason: {', '.join(reasons[-3:])}", (35, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,255), 1)

        processing_ms = (time.perf_counter() - started) * 1000
        cv2.imwrite(str(LATEST_FRAME_PATH), frame)

        now = time.time()
        if status != last_logged_status or now - last_log_time >= 3:
            car_on_crosswalk = bool(car_box and intersects(car_box, CROSSWALK))
            logger.log_event(
                timestamp=datetime.now().isoformat(timespec="seconds"),
                intersection="GuideKaro Winter Presentation Intersection",
                weather="Snow",
                road_condition="Icy / snow-covered",
                visibility="Reduced",
                vehicle_count=1 if car_box else 0,
                pedestrian_count=1 if person_box else 0,
                vehicle_speed_kmh=round(speed_kmh, 1),
                distance_to_crosswalk_m=0.0,
                confidence=1.0,
                risk_score=round(risk_score, 1),
                risk_level=status,
                status=status,
                crosswalk_blocked=car_on_crosswalk,
                alert_channel=(
                    "Audio + Visual" if status == "VIOLATION"
                    else "Visual" if status == "WARNING"
                    else "None"
                ),
                response_time_ms=round(processing_ms, 1),
                frame_path="assets/latest_frame.jpg",
                notes="Controlled simulation: " + "; ".join(reasons),
            )
            last_logged_status = status
            last_log_time = now

        if not args.no_display:
            cv2.imshow("GuideKaro Winter Scenario Detection", frame)
            key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.waitKey(0)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()