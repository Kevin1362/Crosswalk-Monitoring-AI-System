
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def load_yolo_labels(path: Path, width: int, height: int):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, xc, yc, w, h = map(float, parts[:5])
        x1 = (xc - w / 2) * width
        y1 = (yc - h / 2) * height
        x2 = (xc + w / 2) * width
        y2 = (yc + h / 2) * height
        boxes.append((int(cls), (x1, y1, x2, y2)))
    return boxes


def match_detections(preds, truths, threshold: float = 0.5):
    used = set()
    tp = 0
    for p_cls, p_box in preds:
        best = None
        best_iou = 0.0
        for idx, (t_cls, t_box) in enumerate(truths):
            if idx in used or t_cls != p_cls:
                continue
            score = iou(p_box, t_box)
            if score > best_iou:
                best_iou, best = score, idx
        if best is not None and best_iou >= threshold:
            used.add(best)
            tp += 1
    fp = len(preds) - tp
    fn = len(truths) - tp
    return tp, fp, fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--labels-dir", default=None, help="YOLO labels named frame_000001.txt, frame_000002.txt, ...")
    parser.add_argument("--output", default="evaluation/results.csv")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")

    latencies = []
    frames = 0
    tp = fp = fn = 0
    labels_dir = Path(args.labels_dir) if args.labels_dir else None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        start = time.perf_counter()
        result = model.predict(frame, conf=args.confidence, verbose=False)[0]
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        if labels_dir:
            h, w = frame.shape[:2]
            preds = []
            if result.boxes is not None:
                for cls, box in zip(result.boxes.cls.int().cpu().tolist(), result.boxes.xyxy.cpu().tolist()):
                    preds.append((int(cls), tuple(map(float, box))))
            truths = load_yolo_labels(labels_dir / f"frame_{frames:06d}.txt", w, h)
            a, b, c = match_detections(preds, truths)
            tp += a
            fp += b
            fn += c

    cap.release()

    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.quantile(latencies, 0.95)) if latencies else 0.0
    avg_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    precision = tp / (tp + fp) if labels_dir and (tp + fp) else None
    recall = tp / (tp + fn) if labels_dir and (tp + fn) else None

    row = {
        "video": args.video,
        "frames": frames,
        "average_latency_ms": round(avg_latency, 3),
        "p95_latency_ms": round(p95_latency, 3),
        "average_fps": round(avg_fps, 3),
        "precision": "" if precision is None else round(precision, 4),
        "recall": "" if recall is None else round(recall, 4),
        "true_positives": "" if not labels_dir else tp,
        "false_positives": "" if not labels_dir else fp,
        "false_negatives": "" if not labels_dir else fn,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(row)
    if not labels_dir:
        print("Precision/recall not calculated because no labelled ground-truth directory was supplied.")


if __name__ == "__main__":
    main()
