
# Evaluation Metrics

## Detection metrics

When labelled ground truth is available:

- **Precision = TP / (TP + FP)**
- **Recall = TP / (TP + FN)**
- Optionally add F1 score and mAP using a standard object-detection evaluation workflow.

The included `evaluation/evaluate_video.py` computes simple frame-level precision and recall using YOLO-format label files and IoU matching.

## Runtime metrics

Always report:
- Average FPS
- Average latency per frame
- P95 latency
- Number of processed frames
- Hardware used
- Model name and confidence threshold

## Tracking metrics

Recommended for a more rigorous future evaluation:
- ID switches
- Track fragmentation
- MOTA / HOTA / IDF1 when labelled tracking ground truth is available

## Risk-engine validation

Create scenario tests where inputs are controlled:
- high speed + short distance
- low speed + large distance
- moving toward vs moving away
- predicted conflict vs no predicted conflict
- pedestrian inside crosswalk vs outside

The risk score is a prototype safety indicator and should not be described as a calibrated probability of collision.
