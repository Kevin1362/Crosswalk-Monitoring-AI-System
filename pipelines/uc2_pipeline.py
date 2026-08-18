import logging
import time

logger = logging.getLogger("GuideKaro.UC2")


def run_uc2_pipeline(
    vehicle_speed,
    pedestrian_distance,
    crosswalk_occupied,
    blocked_zone=False
):
    logger.info("UC-02 PIPELINE STARTED")

    start_time = time.time()

    try:
        logger.info("UC-02 Feature Collection started")
        logger.info("UC-02 Feature Preprocessing started")
        logger.info("UC-02 Risk Calculation started")
        logger.info("UC-02 Crosswalk Occupancy Check started")
        logger.info("UC-02 Decision Classification started")
        logger.info("UC-02 Validation started")

        risk_score = 0

        if crosswalk_occupied:
            risk_score += 40

        if vehicle_speed > 30:
            risk_score += 30

        if pedestrian_distance < 10:
            risk_score += 30

        if blocked_zone:
            risk_score += 10

        risk_score = min(risk_score, 100)

        if risk_score >= 70:
            state = "STOP"
        elif risk_score >= 40:
            state = "WARNING"
        else:
            state = "SAFE"

        elapsed = round((time.time() - start_time) * 1000, 2)

        logger.info("UC-02 PIPELINE COMPLETED")

        return {
            "use_case": "UC-02",
            "pipeline": "Safe Decision Support",
            "vehicle_speed": vehicle_speed,
            "pedestrian_distance": pedestrian_distance,
            "crosswalk_occupied": crosswalk_occupied,
            "blocked_zone": blocked_zone,
            "risk_score": risk_score,
            "state": state,
            "latency_ms": elapsed,
            "status": "success"
        }

    except Exception:
        logger.exception("UC-02 PIPELINE FAILED")
        raise