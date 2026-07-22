import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import cv2

from detection.detector import detect
from sprint1_uc1_alert_engine import evaluate_state
from sprint1_uc1_logger import log_event

from logging_config import logger


print("UC-1 Crosswalk System Running...")

logger.info("Crosswalk AI application started.")


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Camera could not be opened.")
        return

    logger.info("Camera initialized successfully.")


    while True:

        ret, frame = cap.read()

        if not ret:
            logger.warning("Unable to capture frame from camera.")
            break


        # Run YOLO detection
        vehicles, pedestrians, status, objects = detect(frame)


        # Run decision engine
        state = evaluate_state(
            vehicles,
            pedestrians
        )


        # Existing event logger
        log_event(state)


        # Assignment requirement:
        # INFO level log messages
        logger.info(
            f"Detection completed | "
            f"Vehicles={vehicles} | "
            f"Pedestrians={pedestrians} | "
            f"Status={status} | "
            f"Decision={state}"
        )


        # Display detected objects
        y = 80

        for obj, confidence in objects:

            cv2.putText(
                frame,
                f"{obj}: {confidence:.2f}",
                (50, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            y += 30


        # Display system state
        cv2.putText(
            frame,
            f"STATE: {state}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


        cv2.imshow(
            "AI Crosswalk Monitoring System",
            frame
        )


        # Press q to exit
        if cv2.waitKey(1000) & 0xFF == ord('q'):

            logger.info("Exit command received.")
            break



    logger.info("Crosswalk AI application closed.")

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()