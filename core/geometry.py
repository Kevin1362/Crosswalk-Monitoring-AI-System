
from __future__ import annotations

import math
from typing import Iterable
import numpy as np


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_in_polygon(point: tuple[float, float], polygon: Iterable[Iterable[float]]) -> bool:
    pts = np.asarray(list(polygon), dtype=float)
    x, y = point
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def distance_point_to_polygon_px(point: tuple[float, float], polygon: Iterable[Iterable[float]]) -> float:
    """Return 0 when inside; otherwise nearest distance to a polygon edge."""
    pts = np.asarray(list(polygon), dtype=float)
    if point_in_polygon(point, pts):
        return 0.0

    px, py = point
    best = float("inf")
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        denom = vx * vx + vy * vy
        t = 0.0 if denom == 0 else max(0.0, min(1.0, (wx * vx + wy * vy) / denom))
        qx, qy = x1 + t * vx, y1 + t * vy
        best = min(best, math.hypot(px - qx, py - qy))
    return best


def pixels_to_meters(distance_px: float, pixels_per_meter: float) -> float:
    if pixels_per_meter <= 0:
        return 0.0
    return max(0.0, distance_px / pixels_per_meter)
