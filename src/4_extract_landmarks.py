import numpy as np
import cv2
import mediapipe as mp
from utils import load_image

mp_face_mesh = mp.solutions.face_mesh

def extract_landmarks(img):
    """
    Uses Mediapipe Face Mesh to extract 468 facial landmarks.
    Returns a flattened vector of length 1404 (468 landmarks × 3 coordinates).
    """

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        points = []
        for lm in results.multi_face_landmarks[0].landmark:
            points.extend([lm.x, lm.y, lm.z])

        return np.array(points, dtype=np.float32)

def extract_dataset_landmarks(paths, output_path):
    feats = []

    for path in paths:
        img = load_image(path)

        if img is None:
            continue

        lm = extract_landmarks(img)

        if lm is None:
            continue

        feats.append(lm)

    feats = np.array(feats, dtype=np.float32)

    np.save(output_path, feats)
    print("[Landmarks] Saved:", output_path)

if __name__ == "__main__":
    young_X = np.load("data/processed/young_X.npy", allow_pickle=True)
    old_X = np.load("data/processed/old_X.npy", allow_pickle=True)

    extract_dataset_landmarks(young_X, "features/young_landmarks.npy")
    extract_dataset_landmarks(old_X, "features/old_landmarks.npy")

