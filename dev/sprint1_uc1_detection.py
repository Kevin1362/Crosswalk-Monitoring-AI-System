import cv2
import numpy as np

prev_frame = None

def detect(frame):
    global prev_frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    if prev_frame is None:
        prev_frame = gray
        return 0, 0

    # normalize difference (IMPORTANT FIX)
    diff = cv2.absdiff(prev_frame, gray)
    diff_sum = np.sum(diff)
    norm_diff = diff_sum / (gray.shape[0] * gray.shape[1])

    prev_frame = gray

    # 🎯 REALISTIC thresholds (normalized values)
    if norm_diff > 25:
        return 2, 2      # VIOLATION
    elif norm_diff > 10:
        return 1, 1      # WARNING
    elif norm_diff > 3:
        return 1, 0      # LOW activity
    else:
        return 0, 0      # SAFE