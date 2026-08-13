
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RiskFeatures:
    confidence: float = 0.0
    speed_kmh: float = 0.0
    distance_to_crosswalk_m: float = 99.0
    pedestrian_vehicle_distance_m: float = 99.0
    moving_toward_crosswalk: bool = False
    trajectory_conflict: bool = False
    crosswalk_occupied: bool = False
    ttc_seconds: float | None = None
    adverse_weather: bool = False
    low_visibility: bool = False


@dataclass
class RiskDecision:
    score: float
    status: str
    reasons: list[str]
    components: dict[str, float]


class RiskEngine:
    """Feature-based decision engine. The score is a prototype safety indicator, not collision probability."""

    def __init__(self, settings: dict[str, Any]) -> None:
        risk_cfg = settings.get("risk", {})
        self.weights = risk_cfg.get(
            "weights",
            {
                "confidence": 0.08,
                "speed": 0.18,
                "crosswalk_distance": 0.18,
                "separation": 0.20,
                "direction": 0.10,
                "trajectory": 0.12,
                "occupancy": 0.08,
                "ttc": 0.04,
                "environment": 0.02,
            },
        )
        self.safe_max = float(risk_cfg.get("safe_max", 39))
        self.warning_max = float(risk_cfg.get("warning_max", 69))
        self.reference_speed_kmh = float(risk_cfg.get("reference_speed_kmh", 60))
        self.reference_crosswalk_distance_m = float(risk_cfg.get("reference_crosswalk_distance_m", 20))
        self.reference_separation_m = float(risk_cfg.get("reference_separation_m", 12))
        self.reference_ttc_s = float(risk_cfg.get("reference_ttc_s", 5))

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def evaluate(self, f: RiskFeatures) -> RiskDecision:
        confidence = self._clamp01(f.confidence)
        speed = self._clamp01(f.speed_kmh / max(self.reference_speed_kmh, 1))
        crosswalk_distance = self._clamp01(
            1.0 - f.distance_to_crosswalk_m / max(self.reference_crosswalk_distance_m, 0.1)
        )
        separation = self._clamp01(
            1.0 - f.pedestrian_vehicle_distance_m / max(self.reference_separation_m, 0.1)
        )
        direction = 1.0 if f.moving_toward_crosswalk else 0.0
        trajectory = 1.0 if f.trajectory_conflict else 0.0
        occupancy = 1.0 if f.crosswalk_occupied else 0.0
        ttc = 0.0
        if f.ttc_seconds is not None and f.ttc_seconds >= 0:
            ttc = self._clamp01(1.0 - f.ttc_seconds / max(self.reference_ttc_s, 0.1))
        environment = 0.5 * float(f.adverse_weather) + 0.5 * float(f.low_visibility)

        components = {
            "confidence": confidence,
            "speed": speed,
            "crosswalk_distance": crosswalk_distance,
            "separation": separation,
            "direction": direction,
            "trajectory": trajectory,
            "occupancy": occupancy,
            "ttc": ttc,
            "environment": environment,
        }

        weighted = sum(
            components[name] * float(self.weights.get(name, 0.0))
            for name in components
        )
        weight_total = sum(float(self.weights.get(name, 0.0)) for name in components) or 1.0
        score = max(0.0, min(100.0, 100.0 * weighted / weight_total))

        reasons: list[str] = []
        if f.speed_kmh >= 35:
            reasons.append(f"vehicle speed {f.speed_kmh:.1f} km/h")
        if f.distance_to_crosswalk_m <= 8:
            reasons.append(f"crosswalk distance {f.distance_to_crosswalk_m:.1f} m")
        if f.pedestrian_vehicle_distance_m <= 8:
            reasons.append(f"pedestrian separation {f.pedestrian_vehicle_distance_m:.1f} m")
        if f.moving_toward_crosswalk:
            reasons.append("moving toward crosswalk")
        if f.trajectory_conflict:
            reasons.append("predicted path conflict")
        if f.crosswalk_occupied:
            reasons.append("crosswalk occupied")
        if f.ttc_seconds is not None and f.ttc_seconds <= 4:
            reasons.append(f"TTC {f.ttc_seconds:.1f} s")
        if f.adverse_weather:
            reasons.append("adverse road/weather condition")
        if f.low_visibility:
            reasons.append("reduced visibility")

        if score <= self.safe_max:
            status = "SAFE"
        elif score <= self.warning_max:
            status = "WARNING"
        else:
            status = "VIOLATION"

        return RiskDecision(score=round(score, 1), status=status, reasons=reasons, components=components)
