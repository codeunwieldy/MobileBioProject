import numpy as np
import json
import os

def split_data_by_age(filtered):
    young_X, young_y = [], []
    old_X, old_y = [], []

    with open("data/processed/identity_map.json") as f:
        id_map = json.load(f)

    for item in filtered:
        path = item["full_path"]
        age_group = item["age_group"]
        name = item["name"]
        label = id_map[name]

        if age_group == "young":
            young_X.append(path)
            young_y.append(label)
        else:
            old_X.append(path)
            old_y.append(label)

    np.save("data/processed/young_X.npy", np.array(young_X))
    np.save("data/processed/young_y.npy", np.array(young_y))
    np.save("data/processed/old_X.npy", np.array(old_X))
    np.save("data/processed/old_y.npy", np.array(old_y))

    print("Saved young/old splits!")

if __name__ == "__main__":
    filtered = np.load("data/wiki_filtered.npy", allow_pickle=True)
    split_data_by_age(filtered)
