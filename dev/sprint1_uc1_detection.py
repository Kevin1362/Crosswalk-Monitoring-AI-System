import cv2
import random

def detect(frame):
    """
    Simulated detection (UC-1 MVP version)
    """

    vehicles = random.randint(0, 2)
    pedestrians = random.randint(0, 2)

    return vehicles, pedestrians