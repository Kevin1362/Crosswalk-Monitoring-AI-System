
from __future__ import annotations

import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from core.config import load_settings
from core.geometry import (
    box_center,
    distance_point_to_polygon_px,
    euclidean,
    pixels_to_meters,
    point_in_polygon,
)
from core.logging_setup import configure_logging
from core.risk_engine import RiskEngine, RiskFeatures
from core.trajectory import TrackHistory
from event_logger import GuideKaroEventLogger


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
LATEST_FRAME = ASSETS_DIR / "latest_frame.jpg"
DB_PATH = BASE_DIR / "database" / "uc1_events.db"


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def absolute_roi(relative_roi, width: int, height: int) -> np.ndarray:
    return np.asarray(
        [[int(x * width), int(y * height)] for x, y in relative_roi],
        dtype=np.int32,
    )


def atomic_jpeg_write(destination: Path, frame: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.stem + "_writing.jpg")
    if cv2.imwrite(str(temp), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        try:
            os.replace(temp, destination)
        except PermissionError:
            temp.unlink(missing_ok=True)


def direction_label(vx: float, vy: float, center: tuple[float, float], crosswalk_center: tuple[float, float]) -> str:
    if abs(vx) + abs(vy) < 1.0:
        return "STATIONARY"
    to_cw = (crosswalk_center[0] - center[0], crosswalk_center[1] - center[1])
    dot = vx * to_cw[0] + vy * to_cw[1]
    return "TOWARD_CROSSWALK" if dot > 0 else "AWAY_FROM_CROSSWALK"


def predicted_conflict(predicted: tuple[float, float], polygon: np.ndarray) -> bool:
    return point_in_polygon(predicted, polygon)


def main() -> None:
    parser = argparse.ArgumentParser(description="GuideKaro enhanced tracking and risk runner")
    parser.add_argument("--source", default="0", help="Webcam number or video file")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tracker", default=None, help="bytetrack.yaml, botsort.yaml, etc.")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    settings = load_settings(BASE_DIR / args.config)
    detector_cfg = settings["detection"]
    tracker_cfg = settings["tracking"]
    project_cfg = settings["project"]

    model_name = args.model or detector_cfg["model"]
    tracker_name = args.tracker or detector_cfg["tracker"]

    logger = configure_logging(BASE_DIR / "logs" / "guidekaro.log")
    event_logger = GuideKaroEventLogger(DB_PATH)
    risk_engine = RiskEngine(settings)
    history = TrackHistory(max_points=int(tracker_cfg["history_points"]))

    model = YOLO(model_name)
    source = parse_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    fps_nominal = capture.get(cv2.CAP_PROP_FPS) or 30.0
    pixels_per_meter = float(tracker_cfg["pixels_per_meter"])
    prediction_horizon = float(tracker_cfg["prediction_horizon_seconds"])
    person_classes = set(detector_cfg["classes"]["person"])
    vehicle_classes = set(detector_cfg["classes"]["vehicles"])

    last_event_time = 0.0
    last_status = None

    logger.info("Enhanced runner started: source=%s tracker=%s model=%s", args.source, tracker_name, model_name)

    try:
        while True:
            frame_started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                if args.loop and not isinstance(source, int):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            height, width = frame.shape[:2]
            roi = absolute_roi(settings["crosswalk"]["roi"], width, height)
            roi_center = tuple(np.mean(roi, axis=0).tolist())

            video_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_s = video_ms / 1000.0 if video_ms > 0 else time.time()

            # Ultralytics built-in ByteTrack/BoT-SORT tracking.
            result = model.track(
                frame,
                persist=True,
                tracker=tracker_name,
                conf=float(detector_cfg["confidence"]),
                verbose=False,
            )[0]

            annotated = frame.copy()
            cv2.polylines(annotated, [roi], True, (255, 200, 0), 3)
            cv2.putText(
                annotated,
                "Crosswalk ROI",
                tuple(roi[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            objects = []
            active_ids: set[int] = set()

            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()
                confs = result.boxes.conf.cpu().tolist()
                boxes = result.boxes.xyxy.cpu().tolist()

                for track_id, class_id, conf, raw_box in zip(ids, classes, confs, boxes):
                    if class_id not in person_classes and class_id not in vehicle_classes:
                        continue

                    box = tuple(map(float, raw_box))
                    cx, cy = box_center(box)
                    active_ids.add(track_id)
                    history.update(track_id, timestamp_s, cx, cy)
                    motion = history.estimate(track_id, horizon_s=prediction_horizon)
                    speed_kmh = pixels_to_meters(motion.speed_px_s, pixels_per_meter) * 3.6
                    distance_cw = pixels_to_meters(
                        distance_point_to_polygon_px((cx, cy), roi), pixels_per_meter
                    )
                    direction = direction_label(
                        motion.vx_px_s, motion.vy_px_s, (cx, cy), roi_center
                    )
                    pred = (motion.predicted_x, motion.predicted_y)
                    conflict = predicted_conflict(pred, roi)
                    inside = point_in_polygon((cx, cy), roi)
                    object_class = model.names.get(class_id, str(class_id))

                    objects.append(
                        {
                            "track_id": track_id,
                            "class_id": class_id,
                            "class_name": object_class,
                            "confidence": float(conf),
                            "box": box,
                            "center": (cx, cy),
                            "speed_kmh": speed_kmh,
                            "distance_to_crosswalk_m": distance_cw,
                            "direction": direction,
                            "trajectory_conflict": conflict,
                            "crosswalk_occupied": inside,
                            "predicted": pred,
                        }
                    )

                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 255) if class_id in person_classes else (255, 170, 0)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.line(
                        annotated,
                        (int(cx), int(cy)),
                        (int(pred[0]), int(pred[1])),
                        color,
                        2,
                    )
                    cv2.circle(annotated, (int(pred[0]), int(pred[1])), 5, color, -1)
                    cv2.putText(
                        annotated,
                        f"ID {track_id} {object_class} {conf:.2f}",
                        (x1, max(20, y1 - 28)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
                    cv2.putText(
                        annotated,
                        f"{speed_kmh:.1f} km/h | {distance_cw:.1f} m | {direction}",
                        (x1, max(38, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                    )

                    event_logger.log_track_sample(
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                        track_id=track_id,
                        object_class=object_class,
                        x=cx,
                        y=cy,
                        speed_kmh=round(speed_kmh, 2),
                        distance_to_crosswalk_m=round(distance_cw, 2),
                        movement_direction=direction,
                        predicted_x=pred[0],
                        predicted_y=pred[1],
                    )

            history.remove_stale(active_ids)

            people = [o for o in objects if o["class_id"] in person_classes]
            vehicles = [o for o in objects if o["class_id"] in vehicle_classes]

            # Evaluate each vehicle using nearest pedestrian and keep the highest risk.
            best = None
            for vehicle in vehicles:
                separation_m = 99.0
                nearest_person = None
                if people:
                    nearest_person = min(
                        people,
                        key=lambda p: euclidean(vehicle["center"], p["center"]),
                    )
                    separation_m = pixels_to_meters(
                        euclidean(vehicle["center"], nearest_person["center"]),
                        pixels_per_meter,
                    )

                relative_speed_ms = max(vehicle["speed_kmh"] / 3.6, 0.1)
                ttc = separation_m / relative_speed_ms if nearest_person else None

                features = RiskFeatures(
                    confidence=vehicle["confidence"],
                    speed_kmh=vehicle["speed_kmh"],
                    distance_to_crosswalk_m=vehicle["distance_to_crosswalk_m"],
                    pedestrian_vehicle_distance_m=separation_m,
                    moving_toward_crosswalk=vehicle["direction"] == "TOWARD_CROSSWALK",
                    trajectory_conflict=bool(vehicle["trajectory_conflict"]),
                    crosswalk_occupied=bool(vehicle["crosswalk_occupied"]),
                    ttc_seconds=ttc,
                    adverse_weather=project_cfg.get("road_condition", "").lower()
                    in {"wet", "icy", "snow", "snow-covered"},
                    low_visibility=project_cfg.get("visibility", "").lower()
                    in {"low", "reduced", "poor"},
                )
                decision = risk_engine.evaluate(features)

                candidate = {
                    "vehicle": vehicle,
                    "person": nearest_person,
                    "separation_m": separation_m,
                    "ttc": ttc,
                    "decision": decision,
                }
                if best is None or decision.score > best["decision"].score:
                    best = candidate

            if best is None:
                status = "SAFE"
                risk_score = 0.0
                reasons = ["no tracked vehicle risk"]
                top_speed = 0.0
                distance_cw = 0.0
                separation_m = 0.0
                movement = "UNKNOWN"
                conflict = False
                ttc = None
                track_id = None
                object_class = None
                pred = (None, None)
                confidence = max([o["confidence"] for o in objects], default=0.0)
            else:
                decision = best["decision"]
                vehicle = best["vehicle"]
                status = decision.status
                risk_score = decision.score
                reasons = decision.reasons
                top_speed = vehicle["speed_kmh"]
                distance_cw = vehicle["distance_to_crosswalk_m"]
                separation_m = best["separation_m"]
                movement = vehicle["direction"]
                conflict = vehicle["trajectory_conflict"]
                ttc = best["ttc"]
                track_id = vehicle["track_id"]
                object_class = vehicle["class_name"]
                pred = vehicle["predicted"]
                confidence = vehicle["confidence"]

            latency_ms = (time.perf_counter() - frame_started) * 1000.0
            fps_runtime = 1000.0 / latency_ms if latency_ms > 0 else 0.0

            # Overlay system summary.
            state_color = {"SAFE": (0, 190, 0), "WARNING": (0, 200, 255), "VIOLATION": (0, 0, 255)}[status]
            panel = annotated.copy()
            cv2.rectangle(panel, (10, 10), (610, 180), (0, 0, 0), -1)
            annotated = cv2.addWeighted(panel, 0.65, annotated, 0.35, 0)
            lines = [
                f"GuideKaro: {status} | Risk {risk_score:.1f}/100",
                f"Tracked: {len(vehicles)} vehicles, {len(people)} pedestrians",
                f"Highest-risk vehicle ID: {track_id if track_id is not None else 'N/A'}",
                f"Speed {top_speed:.1f} km/h | Crosswalk {distance_cw:.1f} m | Separation {separation_m:.1f} m",
                f"Direction: {movement} | TTC: {ttc:.1f}s" if ttc is not None else f"Direction: {movement} | TTC: N/A",
            ]
            for i, line in enumerate(lines):
                cv2.putText(
                    annotated, line, (25, 40 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.63, state_color if i == 0 else (255,255,255), 2
                )

            atomic_jpeg_write(LATEST_FRAME, annotated)

            now = time.time()
            if (
                status != last_status
                or now - last_event_time >= float(detector_cfg["log_interval_seconds"])
            ):
                event_logger.log_event(
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    intersection=project_cfg["intersection"],
                    weather=project_cfg.get("weather", "Unknown"),
                    road_condition=project_cfg.get("road_condition", "Unknown"),
                    visibility=project_cfg.get("visibility", "Unknown"),
                    vehicle_count=len(vehicles),
                    pedestrian_count=len(people),
                    vehicle_speed_kmh=round(top_speed, 2),
                    distance_to_crosswalk_m=round(distance_cw, 2),
                    pedestrian_vehicle_distance_m=round(separation_m, 2),
                    confidence=round(confidence, 3),
                    risk_score=risk_score,
                    risk_level=status,
                    status=status,
                    crosswalk_blocked=any(o["crosswalk_occupied"] for o in vehicles),
                    alert_channel="Audio + Visual" if status == "VIOLATION" else "Visual" if status == "WARNING" else "None",
                    response_time_ms=round(latency_ms, 2),
                    fps=round(fps_runtime, 2),
                    track_id=track_id,
                    object_class=object_class,
                    movement_direction=movement,
                    trajectory_conflict=conflict,
                    ttc_seconds=round(ttc, 2) if ttc is not None else None,
                    predicted_x=pred[0],
                    predicted_y=pred[1],
                    frame_path="assets/latest_frame.jpg",
                    notes="; ".join(reasons),
                )
                logger.info(
                    "status=%s risk=%.1f track=%s speed=%.1f distance=%.1f separation=%.1f fps=%.1f",
                    status, risk_score, track_id, top_speed, distance_cw, separation_m, fps_runtime
                )
                last_status = status
                last_event_time = now

            if args.display:
                cv2.imshow("GuideKaro Enhanced Tracking", annotated)
                if cv2.waitKey(max(1, int(1000 / fps_nominal))) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Runner interrupted by user")
    except Exception:
        logger.exception("Runner failed")
        raise
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
