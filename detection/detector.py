import os
import torch

from database import create_database, save_detection
from dev.sprint2_uc2_decision_engine import evaluate_risk
from logging_config import logger


# =====================================
# Initialize Database
# =====================================

create_database()

logger.info("Detection module initialized.")



# =====================================
# Load YOLOv5 Model
# =====================================

logger.info("Loading YOLOv5 model...")


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yolov5s.pt"
)


try:

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"YOLO model not found: {MODEL_PATH}"
        )


    model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=MODEL_PATH,
        force_reload=False
    )


    logger.info(
        "YOLOv5 model loaded successfully."
    )


except Exception as e:

    logger.error(
        f"Failed to load YOLO model: {e}"
    )

    raise



# =====================================
# Detection Configuration
# =====================================

CONF_THRESHOLD = 0.45


VEHICLES = [
    "car",
    "bus",
    "truck",
    "motorcycle"
]


PEDESTRIAN = "person"



# =====================================
# UC-1 Detection Function
# =====================================

def detect(frame):
    """
    UC-1:
    Detect pedestrians and vehicles using YOLOv5.

    Returns:
        vehicles:
            Number of vehicles detected

        pedestrians:
            Number of pedestrians detected

        status:
            Risk level from UC-2 decision engine

        detected_objects:
            Detected objects with confidence
    """


    logger.info(
        "Running object detection."
    )


    # Run YOLO inference
    results = model(frame)


    detections = results.pandas().xyxy[0]


    vehicles = 0
    pedestrians = 0


    detected_objects = []



    # =====================================
    # Process Detection Results
    # =====================================

    for _, row in detections.iterrows():

        confidence = float(
            row["confidence"]
        )

        object_name = row["name"]


        if confidence < CONF_THRESHOLD:
            continue



        detected_objects.append(
            (
                object_name,
                confidence
            )
        )



        if object_name in VEHICLES:

            vehicles += 1



        elif object_name == PEDESTRIAN:

            pedestrians += 1




    logger.info(
        f"Objects detected | "
        f"Vehicles={vehicles}, "
        f"Pedestrians={pedestrians}"
    )



    # =====================================
    # UC-2 Risk Evaluation
    # =====================================

    status = evaluate_risk(
        vehicles,
        pedestrians
    )


    logger.info(
        f"Risk evaluation completed | "
        f"Status={status}"
    )



    # =====================================
    # Save Detection Events
    # =====================================

    for obj, conf in detected_objects:

        save_detection(
            obj,
            conf,
            status
        )



    logger.info(
        f"Detection completed | "
        f"Vehicles={vehicles} | "
        f"Pedestrians={pedestrians} | "
        f"Status={status}"
    )



    return (
        vehicles,
        pedestrians,
        status,
        detected_objects
    )