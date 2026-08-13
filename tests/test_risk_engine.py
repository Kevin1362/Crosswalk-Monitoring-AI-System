
from core.risk_engine import RiskEngine, RiskFeatures


def settings():
    return {
        "risk": {
            "safe_max": 39,
            "warning_max": 69,
            "reference_speed_kmh": 60,
            "reference_crosswalk_distance_m": 20,
            "reference_separation_m": 12,
            "reference_ttc_s": 5,
            "weights": {
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
        }
    }


def test_low_risk_is_safe():
    engine = RiskEngine(settings())
    d = engine.evaluate(
        RiskFeatures(
            confidence=0.8,
            speed_kmh=5,
            distance_to_crosswalk_m=30,
            pedestrian_vehicle_distance_m=30,
        )
    )
    assert d.status == "SAFE"


def test_conflict_becomes_high_risk():
    engine = RiskEngine(settings())
    d = engine.evaluate(
        RiskFeatures(
            confidence=0.95,
            speed_kmh=50,
            distance_to_crosswalk_m=2,
            pedestrian_vehicle_distance_m=2,
            moving_toward_crosswalk=True,
            trajectory_conflict=True,
            crosswalk_occupied=True,
            ttc_seconds=1.5,
            adverse_weather=True,
            low_visibility=True,
        )
    )
    assert d.status == "VIOLATION"
    assert d.score >= 70
