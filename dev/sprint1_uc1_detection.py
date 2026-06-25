import cv2
import numpy as np

# Background subtractor (learns scene over time)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

def detect(frame):
    """
    UC-1 improved detection (non-random, frame-based)
    """

    # Step 1: preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: foreground mask (motion detection)
    mask = bg_subtractor.apply(blur)

    # Step 3: clean noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Step 4: find contours (moving objects)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vehicles = 0
    pedestrians = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 500:
            continue  # ignore noise

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # crude classification logic
        if area > 1500 and aspect_ratio > 1.2:
            vehicles += 1
        else:
            pedestrians += 1

    return vehicles, pedestrians