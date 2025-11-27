import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

def cross_age_scores(model_path, young_X, young_y, old_X, old_y):
    model = joblib.load(model_path)

    young_embed = model.predict_proba(young_X)
    old_embed = model.predict_proba(old_X)

    genuine = []
    impostor = []

    for i in range(len(young_embed)):
        for j in range(len(old_embed)):
            sim = cosine_similarity([young_embed[i]], [old_embed[j]])[0][0]

            if young_y[i] == old_y[j]:
                genuine.append(sim)
            else:
                impostor.append(sim)

    return np.array(genuine), np.array(impostor)
