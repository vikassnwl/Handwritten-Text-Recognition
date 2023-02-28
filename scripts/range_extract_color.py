import numpy as np
import cv2


def extract_color_using_range(img_bgr, lower_range=np.array([111, 0, 0]), upper_range=np.array([168,255,211])):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower_range, upper_range)
    return mask