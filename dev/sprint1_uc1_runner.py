import cv2

from sprint1_uc1_detection import detect
from sprint1_uc1_alert_engine import evaluate_state
from sprint1_uc1_logger import log_event

print("UC-1 Crosswalk System Running...")

def main():

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        vehicles, pedestrians = detect(frame)

        state = evaluate_state(vehicles, pedestrians)

        log_event(state)

        cv2.putText(
            frame,
            f"STATE: {state}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("UC-1 System", frame)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()