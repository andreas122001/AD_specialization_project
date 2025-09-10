#!/bin/sh

#SBATCH --partition=GPUQ
#SBATCH --job-name=merge
#SBATCH --output=/cluster/work/andrebw/repos/temporal_garage/merge.out
#SBATCH --error=/cluster/work/andrebw/repos/temporal_garage/merge.out
#SBATCH --account=share-ie-idi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=andreaswinje@hotmail.com
#SBATCH --mail-type=BEGIN

cd /cluster/work/andrebw/repos/temporal_garage
sh merge_b2d.sh

echo 
sleep 10
python tools/aggregate_b2d.py

