#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=tfpp_base
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH -o=/cluster/work/andrebw/repos/temporal_garage/results/logs/tfpp_010_0_%a_%A.out  # File to which STDOUT will be written
#SBATCH -e=/cluster/work/andrebw/repos/temporal_garage/results/logs/tfpp_010_0_%a_%A.out  # File to which STDERR will be written
#SBATCH --partition=GPUQ

# IMPORTANT: Start this script from within team_code folder, otherwise it will not work

# print info about current job
scontrol show job $SLURM_JOB_ID

pwd
export PROJECT_ROOT=/cluster/work/andrebw/repos/temporal_garage
export DATASET=garage_v2_2025_03_25
export CARLA_ROOT=$PROJECT_ROOT/carla
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/lustre/work/geiger/bjaeger25/miniconda3/lib

export OMP_NUM_THREADS=32  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=4 --max_restarts=0 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    train.py --id $SLURM_JOB_NAME \
    --use_disk_cache 1 \
    --crop_image 1 \
    --seed 0 \
    --epochs 31 \
    --batch_size 16 \
    --lr 3e-4 \
    --setting all \
    --root_dir $PROJECT_ROOT/results/data/$DATASET/data \
    --logdir $PROJECT_ROOT/results \
    --use_controller_input_prediction 1 \
    --continue_epoch 0 \
    --cpu_cores $OMP_NUM_THREADS \
    --num_repetitions 1 \
    --use_cosine_schedule 1 \
    --cosine_t0 1 \
    --image_architecture regnety_032 \
    --lidar_architecture regnety_032