import numpy as np
import json

def fuse_scores(scores1, scores2, w1=0.5, w2=0.5):
    return w1*scores1 + w2*scores2

def save_weights(w1, w2):
    with open("models/fusion_weights.json", "w") as f:
        json.dump({"w1": w1, "w2": w2}, f)
