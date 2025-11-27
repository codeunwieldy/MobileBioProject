import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from utils import load_image

def extract_lbp(image, P=8, R=1.0):
    lbp = local_binary_pattern(image, P, R, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, P + 3), density=True)
    return hist

def extract_dataset_lbp(X_paths, output_path):
    features = []
    for path in X_paths:
        img = load_image(path)
        if img is None:
            continue
        feat = extract_lbp(img)
        features.append(feat)

    np.save(output_path, np.array(features))
    print("[LBP] Saved:", output_path)

if __name__ == "__main__":
    young_X = np.load("data/processed/young_X.npy")
    old_X = np.load("data/processed/old_X.npy")

    extract_dataset_lbp(young_X, "features/young_lbp.npy")
    extract_dataset_lbp(old_X, "features/old_lbp.npy")
