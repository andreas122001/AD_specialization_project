import os
import ujson
import subprocess
from time import sleep

WD = "."
HZ = 20

# TODO: make the script better? Also, it depends on:
# scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py
# to add self._collision_time to the printout. Maybe there is a better solution...?
root_folder = "evaluation/town13_withheld_routes_validation_model_0030_0/town13_withheld_routes_validation_e0_model_0030_0"
results_folder = os.path.join(root_folder, "results")
results = os.listdir(os.path.join(WD, results_folder))
save_folder = os.path.join(root_folder, "movies")
os.makedirs(save_folder, exist_ok=True)

subprocess.check_output("module load FFmpeg/6.0-GCCcore-12.3.0", shell=True)

for f_name in results:
    with open(os.path.join(results_folder, f_name)) as f:
        results_json = ujson.load(f)
        records = results_json["_checkpoint"]["records"]
        if len(records) == 0:
            continue
        record = records[0]
        timestamp = record["timestamp"]
        infractions = record["infractions"]["collisions_vehicle"]
        for infraction_type, infractions in record['infractions'].items(): 

            for i, infraction in enumerate(infractions):
                if " t=" not in infraction:
                    continue

                images_folder = os.path.join(root_folder, "logs", timestamp)
                start_number = int(infraction.split(" t=")[1].split(")")[0])  # not safe

                movie_name = f"{infraction_type}_{f_name.split('.')[0]}_{i}.mp4"
                cmd = f"ffmpeg -framerate 20 -start_number {(start_number * HZ)-40} -i {images_folder}/%04d.png -frames:v 80 -c:v libx264 -y -pix_fmt yuv420p {save_folder}/{movie_name}"
                print(cmd)
                os.system(cmd)
