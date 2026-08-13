
# Dataset and Testing Documentation

## Dataset inventory

Record every video used for validation.

| ID | Source | Duration | Resolution | FPS | Conditions | Permission / licence | Ground truth |
|---|---|---:|---|---:|---|---|---|
| V01 | Add source | Add | Add | Add | Clear daylight | Add | Yes/No |
| V02 | Add source | Add | Add | Add | Snow / ice | Add | Yes/No |
| V03 | Add source | Add | Add | Add | Low light | Add | Yes/No |

Do not report precision or recall unless the tested frames have ground-truth annotations.

## Required test scenarios

1. Clear daylight
2. Snow / ice
3. Heavy rain
4. Poor lighting / night
5. Occluded pedestrian
6. Multiple overlapping vehicles
7. Motion blur
8. Camera shake
9. Dense pedestrian traffic
10. Vehicle moving away from crosswalk

## Functional tests

- Same object keeps the same track ID across consecutive frames.
- Speed is near zero for a stationary object.
- Movement direction changes when a vehicle moves away from the ROI.
- Risk rises when speed increases and distance decreases.
- Trajectory conflict increases risk.
- Database stores track ID and feature values.
- Dashboard updates without reading partial images.

## Failure cases

Document failures, not only successful examples. For every failure, include:
- Video/frame reference
- Condition
- Expected result
- Actual result
- Likely reason
- Mitigation or next improvement
