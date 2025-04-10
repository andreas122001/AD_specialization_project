#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=tfpp_trajectory
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=32
#SBATCH -o /cluster/work/andrebw/repos/temporal_garage/results/logs/%x/tfpp_010_0_%a_%A.out  # File to which STDOUT will be written
#SBATCH -e /cluster/work/andrebw/repos/temporal_garage/results/logs/%x/tfpp_010_0_%a_%A.out  # File to which STDERR will be written
#SBATCH --partition=GPUQ
#SBATCH --constraint=(a100|h100)
# #SBATCH --nodelist=idun-01-[01-06],idun-06-[01-07],idun-07-[08-10],idun-08-01

# IMPORTANT: Start this script from within team_code folder, otherwise it will not work

# print info about current job
scontrol show job $SLURM_JOB_ID

echo SLURM_JOB_GPUS: $SLURM_JOB_GPUS
export NGPUS=$(echo $SLURM_JOB_GPUS | grep -oP [0-9]+ | wc -l)
echo NGPUS: $NGPUS
echo Batch size: $((32 / $NGPUS))

pwd
export PROJECT_ROOT=/cluster/work/andrebw/repos/temporal_garage
export DATASET=garage_v2
export CARLA_ROOT=$PROJECT_ROOT/carla
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}

export OMP_NUM_THREADS=32  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=$NGPUS --max_restarts=0 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    train.py --id $SLURM_JOB_NAME \
    --use_disk_cache 1 \
    --crop_image 1 \
    --seed 0 \
    --epochs 31 \
    --batch_size $((32 / $NGPUS)) \
    --use_temporal_fusion 1 \
    --seq_len 2 \
    --seq_step 1 \
    --lidar_seq_len 1 \
    --lidar_step_size 1 \
    --use_trajectory_prediction 1 \
    --trajectory_pred_len 10 \
    --use_semantic 0 \
    --use_bev_semantic 0 \
    --use_depth 0 \
    --detect_boxes 0 \
    --use_controller_input_prediction 0 \
    --use_wp_gru 0 \
    --continue_epoch 0 \
    --lr 3e-4 \
    --setting 13_withheld \
    --root_dir $PROJECT_ROOT/results/data/$DATASET/data \
    --logdir $PROJECT_ROOT/results \
    --cpu_cores $OMP_NUM_THREADS \
    --num_repetitions 1 \
    --use_cosine_schedule 1 \
    --cosine_t0 1 \
    --validation \
    --image_architecture regnety_032 \
    --lidar_architecture regnety_032
