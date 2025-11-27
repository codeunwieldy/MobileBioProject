import cv2
import numpy as np
import os

def load_image(path, size=(128, 128)):
    """
    Loads and resizes an image from disk.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, size)
    return img
