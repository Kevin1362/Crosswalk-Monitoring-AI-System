from fastapi import FastAPI
from pydantic import BaseModel
import logging

from pipelines.uc1_pipeline import run_uc1_pipeline
from pipelines.uc2_pipeline import run_uc2_pipeline
from pipelines.uc3_pipeline import run_uc3_pipeline


# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger("GuideKaro")


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="GuideKaro MLOps API",
    description="MLOps API services for the GuideKaro Crosswalk Safety System",
    version="1.0.0"
)


# --------------------------------------------------
# Application Startup Logging
# --------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("GUIDEKARO APPLICATION STARTED")


# --------------------------------------------------
# Basic Endpoints
# --------------------------------------------------

@app.get("/")
def root():
    logger.info("API CALL START | GET /")

    return {
        "project": "GuideKaro",
        "application": "AI-Based Crosswalk Monitoring and Driver Alert System",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health():
    logger.info("API CALL START | GET /health")

    return {
        "status": "healthy",
        "service": "GuideKaro MLOps API"
    }

# --------------------------------------------------
# UC-01 Request Model
# --------------------------------------------------

class UC1Request(BaseModel):
    video_source: str
    location_id: str = "intersection_01"

# --------------------------------------------------
# UC-01: Early Warning / Conflict Prediction
# --------------------------------------------------

@app.post("/api/uc1/collect")
def uc1_collect(request: UC1Request):
    logger.info("API CALL START | POST /api/uc1/collect")

    result = {
        "use_case": "UC-01",
        "stage": "Data Collection",
        "video_source": request.video_source,
        "location_id": request.location_id,
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc1/collect")

    return result


@app.post("/api/uc1/preprocess")
def uc1_preprocess(request: UC1Request):
    logger.info("API CALL START | POST /api/uc1/preprocess")

    result = {
        "use_case": "UC-01",
        "stage": "Preprocessing",
        "video_source": request.video_source,
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc1/preprocess")

    return result


@app.post("/api/uc1/detect")
def uc1_detect(request: UC1Request):
    logger.info("API CALL START | POST /api/uc1/detect")

    result = {
        "use_case": "UC-01",
        "stage": "Object Detection",
        "objects": [
            "pedestrian",
            "vehicle"
        ],
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc1/detect")

    return result


@app.post("/api/uc1/track")
def uc1_track(request: UC1Request):
    logger.info("API CALL START | POST /api/uc1/track")

    result = {
        "use_case": "UC-01",
        "stage": "Object Tracking",
        "tracking": "active",
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc1/track")

    return result


@app.post("/api/uc1/predict")
def uc1_predict(request: UC1Request):
    logger.info("API CALL START | POST /api/uc1/predict")

    result = {
        "use_case": "UC-01",
        "stage": "Trajectory Prediction",
        "prediction": "paths evaluated",
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc1/predict")

    return result


@app.post("/api/uc1/conflict")
def uc1_conflict(request: UC1Request):
    logger.info("API CALL START | POST /api/uc1/conflict")

    result = {
        "use_case": "UC-01",
        "stage": "Conflict Prediction",
        "risk_detected": True,
        "risk_level": "WARNING",
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc1/conflict")

    return result


@app.post("/api/uc1/pipeline/run")
def uc1_pipeline_run(request: UC1Request):
    logger.info("API CALL START | POST /api/uc1/pipeline/run")

    result = run_uc1_pipeline(
        video_source=request.video_source,
        location_id=request.location_id
    )

    logger.info("API CALL END | POST /api/uc1/pipeline/run")

    return result

# --------------------------------------------------
# UC-02 Request Model
# --------------------------------------------------

class UC2Request(BaseModel):
    vehicle_speed: float
    pedestrian_distance: float
    crosswalk_occupied: bool
    blocked_zone: bool = False

# --------------------------------------------------
# UC-02: Safe Decision Support
# --------------------------------------------------

@app.post("/api/uc2/features")
def uc2_features(request: UC2Request):
    logger.info("API CALL START | POST /api/uc2/features")

    result = {
        "use_case": "UC-02",
        "stage": "Feature Collection",
        "vehicle_speed": request.vehicle_speed,
        "pedestrian_distance": request.pedestrian_distance,
        "crosswalk_occupied": request.crosswalk_occupied,
        "blocked_zone": request.blocked_zone,
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc2/features")
    return result


@app.post("/api/uc2/preprocess")
def uc2_preprocess(request: UC2Request):
    logger.info("API CALL START | POST /api/uc2/preprocess")

    result = {
        "use_case": "UC-02",
        "stage": "Feature Preprocessing",
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc2/preprocess")
    return result


@app.post("/api/uc2/risk")
def uc2_risk(request: UC2Request):
    logger.info("API CALL START | POST /api/uc2/risk")

    risk_score = 0

    if request.crosswalk_occupied:
        risk_score += 40

    if request.vehicle_speed > 30:
        risk_score += 30

    if request.pedestrian_distance < 10:
        risk_score += 30

    if request.blocked_zone:
        risk_score += 10

    risk_score = min(risk_score, 100)

    result = {
        "use_case": "UC-02",
        "stage": "Risk Calculation",
        "risk_score": risk_score,
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc2/risk")
    return result


@app.post("/api/uc2/occupancy")
def uc2_occupancy(request: UC2Request):
    logger.info("API CALL START | POST /api/uc2/occupancy")

    result = {
        "use_case": "UC-02",
        "stage": "Crosswalk Occupancy",
        "crosswalk_occupied": request.crosswalk_occupied,
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc2/occupancy")
    return result


@app.post("/api/uc2/decision")
def uc2_decision(request: UC2Request):
    logger.info("API CALL START | POST /api/uc2/decision")

    result = run_uc2_pipeline(
        vehicle_speed=request.vehicle_speed,
        pedestrian_distance=request.pedestrian_distance,
        crosswalk_occupied=request.crosswalk_occupied,
        blocked_zone=request.blocked_zone
    )

    logger.info("API CALL END | POST /api/uc2/decision")
    return result


@app.post("/api/uc2/validate")
def uc2_validate(request: UC2Request):
    logger.info("API CALL START | POST /api/uc2/validate")

    result = {
        "use_case": "UC-02",
        "stage": "Decision Validation",
        "validation": "completed",
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc2/validate")
    return result


@app.post("/api/uc2/pipeline/run")
def uc2_pipeline_run(request: UC2Request):
    logger.info("API CALL START | POST /api/uc2/pipeline/run")

    result = run_uc2_pipeline(
        vehicle_speed=request.vehicle_speed,
        pedestrian_distance=request.pedestrian_distance,
        crosswalk_occupied=request.crosswalk_occupied,
        blocked_zone=request.blocked_zone
    )

    logger.info("API CALL END | POST /api/uc2/pipeline/run")
    return result

# --------------------------------------------------
# UC-03 Request Model
# --------------------------------------------------

class UC3Request(BaseModel):
    risk_score: float
    state: str
    location_id: str = "intersection_01"

# --------------------------------------------------
# UC-03: Violation Detection and Analytics
# --------------------------------------------------

@app.post("/api/uc3/event")
def uc3_event(request: UC3Request):
    logger.info("API CALL START | POST /api/uc3/event")

    result = {
        "use_case": "UC-03",
        "stage": "Event Collection",
        "risk_score": request.risk_score,
        "state": request.state,
        "location_id": request.location_id,
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc3/event")
    return result


@app.post("/api/uc3/preprocess")
def uc3_preprocess(request: UC3Request):
    logger.info("API CALL START | POST /api/uc3/preprocess")

    result = {
        "use_case": "UC-03",
        "stage": "Event Preprocessing",
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc3/preprocess")
    return result


@app.post("/api/uc3/classify")
def uc3_classify(request: UC3Request):
    logger.info("API CALL START | POST /api/uc3/classify")

    result = run_uc3_pipeline(
        risk_score=request.risk_score,
        state=request.state,
        location_id=request.location_id
    )

    logger.info("API CALL END | POST /api/uc3/classify")
    return result


@app.post("/api/uc3/log")
def uc3_log(request: UC3Request):
    logger.info("API CALL START | POST /api/uc3/log")

    result = {
        "use_case": "UC-03",
        "stage": "Event Logging",
        "location_id": request.location_id,
        "status": "success"
    }

    logger.info("API CALL END | POST /api/uc3/log")
    return result


@app.get("/api/uc3/aggregate")
def uc3_aggregate():
    logger.info("API CALL START | GET /api/uc3/aggregate")

    result = {
        "use_case": "UC-03",
        "stage": "Event Aggregation",
        "safe_events": 25,
        "warning_events": 7,
        "high_risk_events": 3,
        "total_events": 35,
        "status": "success"
    }

    logger.info("API CALL END | GET /api/uc3/aggregate")
    return result


@app.get("/api/uc3/analytics")
def uc3_analytics():
    logger.info("API CALL START | GET /api/uc3/analytics")

    result = {
        "use_case": "UC-03",
        "stage": "Analytics Generation",
        "total_events": 35,
        "high_risk_events": 3,
        "warning_events": 7,
        "status": "success"
    }

    logger.info("API CALL END | GET /api/uc3/analytics")
    return result


@app.post("/api/uc3/pipeline/run")
def uc3_pipeline_run(request: UC3Request):
    logger.info("API CALL START | POST /api/uc3/pipeline/run")

    result = run_uc3_pipeline(
        risk_score=request.risk_score,
        state=request.state,
        location_id=request.location_id
    )

    logger.info("API CALL END | POST /api/uc3/pipeline/run")
    return result