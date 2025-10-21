#!/bin/bash
#SBATCH --job-name=run_evals
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --output=/cluster/work/andrebw/repos/temporal_garage/evaluation/v3/eval_server_%a_%A.out
#SBATCH --error=/cluster/work/andrebw/repos/temporal_garage/evaluation/v3/eval_server_%a_%A.out
#SBATCH --partition=CPUQ

version="v5"

runs=($(ls results/training/$version))

for f in ${runs[@]}; do
  echo $f
  sleep 1
  if ! compgen -G "results/training/$version/$f/model*0030.pth" >> /dev/null; then
    echo "Training not finished for '$f'." 1>&2
    continue
  fi
  python evaluate_routes_slurm_tfpp.py \
    --experiment $f --benchmark bench2drive --num_repetitions 3 \
    --model_dir /cluster/work/andrebw/repos/temporal_garage/results/training/$version
done | tqdm --total ${#runs[@]}

echo "Waiting for remaining evals..."
sleep 1560

echo
echo "================================================="
echo "Evaluations finished. Starting B2D aggregation..."
echo "================================================="
echo

sbatch slurm_merge_bd2.sh $version

