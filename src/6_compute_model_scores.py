import numpy as np
from sklearn.svm import SVC
import joblib

def train_svm(X, y, save_path):
    model = SVC(probability=True, kernel='rbf')
    model.fit(X, y)
    joblib.dump(model, save_path)
    print("Saved:", save_path)

if __name__ == "__main__":
    train_svm(np.load("features/young_lbp.npy"),
              np.load("data/processed/young_y.npy"),
              "models/young_lbp_svm.pkl")

    train_svm(np.load("features/young_landmarks.npy"),
              np.load("data/processed/young_y.npy"),
              "models/young_landmark_svm.pkl")

    train_svm(np.load("features/old_lbp.npy"),
              np.load("data/processed/old_y.npy"),
              "models/old_lbp_svm.pkl")

    train_svm(np.load("features/old_landmarks.npy"),
              np.load("data/processed/old_y.npy"),
              "models/old_landmark_svm.pkl")
