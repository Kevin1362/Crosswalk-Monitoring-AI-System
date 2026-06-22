import cv2
import numpy as np

print("UC-1 Crosswalk Detection System Starting...")

def detect_objects(frame):
    vehicles = []
    pedestrians = []
    return vehicles, pedestrians

def check_violation(vehicles, pedestrians):
    if len(vehicles) > 0 and len(pedestrians) > 0:
        return "VIOLATION"
    elif len(vehicles) > 0:
        return "WARNING"
    else:
        return "SAFE"

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vehicles, pedestrians = detect_objects(frame)
        status = check_violation(vehicles, pedestrians)

        cv2.putText(frame, status, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        cv2.imshow("UC-1 System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()