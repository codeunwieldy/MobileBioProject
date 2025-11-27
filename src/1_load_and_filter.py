import numpy as np
import json
import os

DATA_PATH = "data/wiki_filtered.npy"

def load_filtered_wiki():
    data = np.load(DATA_PATH, allow_pickle=True)
    return data

def filter_identities(data, min_images=2):
    """
    Keeps only identities that appear at least twice AND have both young & old images.
    """
    name_to_ages = {}

    for entry in data:
        name = entry['name']
        age = entry['age_group']
        if name not in name_to_ages:
            name_to_ages[name] = set()
        name_to_ages[name].add(age)

    valid_names = [n for n, ages in name_to_ages.items() if len(ages) == 2]

    filtered = [d for d in data if d['name'] in valid_names]
    return filtered, valid_names

def save_identity_map(valid_names):
    mapping = {name: idx for idx, name in enumerate(valid_names)}
    with open("data/processed/identity_map.json", "w") as f:
        json.dump(mapping, f)

if __name__ == "__main__":
    data = load_filtered_wiki()
    filtered, names = filter_identities(data)
    save_identity_map(names)
    print("Filtered identities:", len(names))
