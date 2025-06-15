import ujson
import os
from collections import defaultdict
import numpy as np

root = "/cluster/work/andrebw/repos/temporal_garage/evaluation/bench2drive"
names = ["static-LB9s1", "static-LB9s2", "LB5s1", "static-LB9s1-notraj", "lidar-LB5s1", "static-LB5s1", "tfpp_default", "static-LB5s1-large", "static-LB5s1-noprune", "static-LB5s1x", "static-LB5s1-notraj"]

for name in names:
    model_path = os.path.join(root, f"{name}_e30")
    if not os.path.isdir(model_path):
        print(f"Evaluation for '{name}' not found.")
        continue
    paths = [os.path.join(model_path, d) for d in os.listdir(model_path) if "_r" in d]

    metrics = defaultdict(list)
    for path in paths:
        ability_json = os.path.join(path, "results", "bench2drive_split", "merged_ability.json")
        merged_json = os.path.join(path, "results", "bench2drive_split", "merged.json")

        if not os.path.isfile(ability_json) or not os.path.isfile(merged_json):
            print(f"Merged metrics not found for '{name}'.")
            continue

        with open(ability_json, "r") as f:
            f_json = ujson.load(f)
            [metrics[k].append(v) for k, v in f_json.items()]
        with open(merged_json) as f:
            f_json = ujson.load(f)
            f_json = {k: v for k ,v in f_json.items() if k != "_checkpoint"}
            [metrics[k].append(v) for k, v in f_json.items()]

    metrics = {k: v for k,v in metrics.items() if k != "crashed"}
    metrics_mean = {k: np.mean(v) for k, v in metrics.items()}
    metrics_stdv = {k: np.std(v) for k, v in metrics.items()}
    merged = {
        'mean': metrics_mean,
        'std': metrics_stdv
    }
    
    save_to = f"/cluster/work/andrebw/repos/temporal_garage/evaluation/merged_b2d/{name}.json"
    with open(save_to, "w") as f:
        ujson.dump(merged, f, indent=4)
        print(save_to)



