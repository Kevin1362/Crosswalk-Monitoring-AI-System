
# Failure Cases and Mitigations

| Failure case | What can happen | How to test | Mitigation |
|---|---|---|---|
| Poor lighting | missed pedestrian or low confidence | night / dark video | stronger model, night data, better camera |
| Heavy rain / snow | blur, occlusion, reflections | weather video | weather-aware thresholds, robust training data |
| Occluded pedestrians | late or broken detection | person behind vehicle | tracking keeps ID through temporary misses |
| Multiple overlapping vehicles | ID switches | dense traffic | tune ByteTrack, increase resolution |
| Motion blur | unstable boxes | fast traffic / low shutter | camera exposure and model improvements |
| Camera shake | false motion / bad speed | handheld or vibration | fixed mount, stabilization |
| Wrong pixel-to-meter scale | inaccurate speed/distance | calibration target | camera-specific calibration / homography |
| Perspective distortion | distance error changes by image region | near/far objects | perspective transform / calibrated ground plane |
