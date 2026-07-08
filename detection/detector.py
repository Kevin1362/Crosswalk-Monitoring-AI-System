import torch
from database import create_database, save_detection
from decision_engine import evaluate_risk


# Initialize database
create_database()


# Load YOLOv5s model
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',
    pretrained=True
)


CONF_THRESHOLD = 0.45


VEHICLES = [
    "car",
    "bus",
    "truck",
    "motorcycle"
]


PEDESTRIAN = "person"



def detect(frame):

    """
    UC-1:
    YOLO object detection only
    """

    results = model(frame)

    detections = results.pandas().xyxy[0]


    vehicles = 0
    pedestrians = 0

    detected_objects = []


    for _, row in detections.iterrows():

        confidence = float(row["confidence"])
        object_name = row["name"]


        if confidence < CONF_THRESHOLD:
            continue


        detected_objects.append(
            (object_name, confidence)
        )


        if object_name in VEHICLES:
            vehicles += 1


        elif object_name == PEDESTRIAN:
            pedestrians += 1



    # UC-2 Decision Engine
    status = evaluate_risk(
        vehicles,
        pedestrians
    )


    # Save final system decision
    for obj, conf in detected_objects:

        save_detection(
            obj,
            conf,
            status
        )


    return vehicles, pedestrians, status