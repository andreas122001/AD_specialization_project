import glob
import gzip
import numpy as np
from tqdm import tqdm
import ujson

root = "/cluster/work/andrebw/repos/temporal_garage/results/data/garage_v2_2025_03_25/results"
root = "/cluster/work/andrebw/repos/temporal_garage/evaluation/tfpp_s2s1_old_routes_validation_model_0030/tfpp_s2s1_old_routes_validation_e0_model_0030/results"
#root = "/cluster/work/andrebw/repos/temporal_garage/evaluation/tfpp_static_s5s2_routes_validation_model_0030/tfpp_static_s5s2_routes_validation_e0_model_0030/results"
#root = "/cluster/work/andrebw/repos/temporal_garage/evaluation/tfpp_default_routes_validation_model_0030/tfpp_default_routes_validation_e0_model_0030/results"

result_files = glob.glob(f"{root}/**/*.json", recursive=True)

failed = 0
success = 0

# Scenarios
results = {}
for result_path in tqdm(result_files):
    scenario = result_path.split("/")[-2]
    if scenario not in results.keys():
        results[scenario] = []
    with open(result_path, "rt", encoding="utf-8") as f:
        try:
            results_route = ujson.load(f)
        except Exception as e:
            print(f"Failed to open {result_path}: {e}")
        if "scores_mean" not in results_route["_checkpoint"]["global_record"]:
            failed += 1
            continue
        if results_route["_checkpoint"]["global_record"]["status"] == "Failed":
            failed += 1
            continue
        success += 1
        results[scenario].append(results_route["_checkpoint"]["global_record"]["scores_mean"])

# Post-process 
for scenario, scenario_res in results.items():
    score_route = [res["score_route"] for res in scenario_res]
    score_penalty = [res["score_penalty"] for res in scenario_res]
    score_composed = [res["score_composed"] for res in scenario_res]
    # Just overwrite with the averages
    results[scenario] = {
        "scores_mean": {
            "score_route": np.mean(score_route),
            "score_penalty": np.mean(score_penalty),
            "score_composed": np.mean(score_composed)
        },
        "scores_std": {
            "score_route": np.std(score_route),
            "score_penalty": np.std(score_penalty),
            "score_composed": np.std(score_composed)
        }
    }

scores_route = [entry["scores_mean"]["score_route"] for entry in results.values()]
score_penalty = [entry["scores_mean"]["score_penalty"] for entry in results.values()]
score_composed = [entry["scores_mean"]["score_composed"] for entry in results.values()]

results['Total'] = {
    "scores_mean": {
        "score_route": np.mean(scores_route),
        "score_penalty": np.mean(score_penalty),
        "score_composed": np.mean(score_composed)
    },
    # Notice, the std here is across scenarios, not routes in total
    "scores_std": {
        "score_route": np.std(scores_route),
        "score_penalty": np.std(score_penalty),
        "score_composed": np.std(score_composed)
    },
}

print(f"Success: {success} / {success + failed}")
print()

from pathlib import Path
save_path = "/".join(root.split("/")[:-1]) + "/results.json"
print(Path(save_path))
with open(save_path, "w") as f:
    ujson.dump(results, f, indent=4)
