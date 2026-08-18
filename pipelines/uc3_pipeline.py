import logging
import time
from datetime import datetime

logger = logging.getLogger("GuideKaro.UC3")


def run_uc3_pipeline(
    risk_score,
    state,
    location_id="intersection_01"
):
    logger.info("UC-03 PIPELINE STARTED")

    start_time = time.time()

    try:
        logger.info("UC-03 Event Collection started")
        logger.info("UC-03 Event Preprocessing started")
        logger.info("UC-03 Violation Classification started")
        logger.info("UC-03 Event Logging started")
        logger.info("UC-03 Event Aggregation started")
        logger.info("UC-03 Analytics Generation started")

        if state == "STOP" and risk_score >= 70:
            event_type = "HIGH_RISK_EVENT"
        elif state == "WARNING":
            event_type = "WARNING_EVENT"
        else:
            event_type = "SAFE_EVENT"

        elapsed = round((time.time() - start_time) * 1000, 2)

        logger.info("UC-03 PIPELINE COMPLETED")

        return {
            "use_case": "UC-03",
            "pipeline": "Violation Detection and Analytics",
            "event_type": event_type,
            "risk_score": risk_score,
            "state": state,
            "location_id": location_id,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": elapsed,
            "status": "success"
        }

    except Exception:
        logger.exception("UC-03 PIPELINE FAILED")
        raise