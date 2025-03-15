#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --account=share-ie-idi
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=./gen_data_out.log
#SBATCH --error=./gen_data_out.log
#SBATCH --partition=CPUQ

# print info about current job
echo "START TIME: $(date)"
start=`date +%s`

for i in $(seq 1 1); do
  python -u collect_dataset_slurm.py &
done
wait

end=`date +%s`
runtime=$((end-start))
echo "END TIME: $(date)"
echo "Runtime: ${runtime}"
