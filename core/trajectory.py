
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque
import numpy as np


@dataclass
class MotionEstimate:
    vx_px_s: float = 0.0
    vy_px_s: float = 0.0
    speed_px_s: float = 0.0
    predicted_x: float = 0.0
    predicted_y: float = 0.0


class TrackHistory:
    """Maintains recent object positions and predicts short-term linear motion."""

    def __init__(self, max_points: int = 12) -> None:
        self.history: dict[int, Deque[tuple[float, float, float]]] = defaultdict(
            lambda: deque(maxlen=max_points)
        )

    def update(self, track_id: int, timestamp_s: float, x: float, y: float) -> None:
        self.history[int(track_id)].append((float(timestamp_s), float(x), float(y)))

    def estimate(self, track_id: int, horizon_s: float = 1.5) -> MotionEstimate:
        points = list(self.history.get(int(track_id), []))
        if not points:
            return MotionEstimate()

        t0, x0, y0 = points[-1]
        if len(points) < 3:
            return MotionEstimate(predicted_x=x0, predicted_y=y0)

        arr = np.asarray(points, dtype=float)
        t = arr[:, 0] - arr[0, 0]
        if np.ptp(t) < 1e-6:
            return MotionEstimate(predicted_x=x0, predicted_y=y0)

        vx = float(np.polyfit(t, arr[:, 1], 1)[0])
        vy = float(np.polyfit(t, arr[:, 2], 1)[0])
        speed = float((vx * vx + vy * vy) ** 0.5)

        return MotionEstimate(
            vx_px_s=vx,
            vy_px_s=vy,
            speed_px_s=speed,
            predicted_x=x0 + vx * horizon_s,
            predicted_y=y0 + vy * horizon_s,
        )

    def remove_stale(self, active_ids: set[int]) -> None:
        stale = [track_id for track_id in self.history if track_id not in active_ids]
        for track_id in stale:
            if len(self.history[track_id]) == self.history[track_id].maxlen:
                self.history.pop(track_id, None)
