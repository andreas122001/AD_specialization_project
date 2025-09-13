#!/bin/bash
#SBATCH --account=share-ie-idi
#SBATCH --job-name=v3/static-LB5s1-large-gating-seed32
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=16
#SBATCH -o /cluster/work/andrebw/repos/temporal_garage/results/logs/%x/tfpp_%a_%A.out  # File to which STDOUT will be written
#SBATCH -e /cluster/work/andrebw/repos/temporal_garage/results/logs/%x/tfpp_%a_%A.out  # File to which STDERR will be written
#SBATCH --partition=GPUQ
#SBATCH --mail-user=andreaswinje@hotmail.com
#SBATCH --mail-type=ALL
#SBATCH --constraint=(a100|h100)
# #SBATCH --nodelist=idun-01-[01-06],idun-06-[01-07],idun-07-[08-10],idun-08-01

# IMPORTANT: Start this script from within team_code folder, otherwise it will not work

# print info about current job
scontrol show job $SLURM_JOB_ID

SEED=$(echo $SLURM_JOB_NAME | grep -oE "seed[0-9]*" | grep -oE "[0-9]*")
if [ -z $SEED ]; then SEED=69; fi
echo SEED: $SEED

echo SLURM_JOB_GPUS: $SLURM_JOB_GPUS
export NGPUS=$(echo $SLURM_JOB_GPUS | grep -oP [0-9]+ | wc -l)
export BATCH_SIZE=$((32 / $NGPUS))
echo NGPUS: $NGPUS
echo Per-GPU batch size: $BATCH_SIZE

pwd
export PROJECT_ROOT=/cluster/work/andrebw/repos/temporal_garage
export DATASET=leaderboard_2
export CARLA_ROOT=$PROJECT_ROOT/carla
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}

# Extract temporal len/res
LB=$(echo $SLURM_JOB_NAME | grep -oP 'LB[0-9]*s[0-9]*')
SEQ_LEN=${LB:2:1}
SEQ_STEP=${LB:4:1}

if [[ $SLURM_JOB_NAME == *"static"* ]]; then RECURRENT=0; else RECURRENT=1; fi
if [[ $SLURM_JOB_NAME == *"self"* ]]; then SELF=1; else SELF=0; fi
if [[ $SLURM_JOB_NAME == *"XL"* ]]; then LAYERS=16; else LAYERS=8; fi
if [[ $SLURM_JOB_NAME == *"traj"* ]]; then TRAJ=1; else TRAJ=0; fi

# Architectures:
# resnet34, regnety_032, video_resnet18, video_swin_tiny

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=$NGPUS --max_restarts=0 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    train.py --id $SLURM_JOB_NAME \
    --use_disk_cache 1 \
    --crop_image 1 \
    --seed $SEED \
    --epochs 31 \
    --batch_size $BATCH_SIZE \
    --use_temporal_fusion 1 \
    --use_recurrent_training $RECURRENT \
    --use_temporal_self_attn $SELF \
    --use_memory_gating 1 \
    --temporal_fusion_layers $LAYERS \
    --temporal_fusion_heads 4 \
    --seq_len $SEQ_LEN \
    --seq_step $SEQ_STEP \
    --lidar_seq_len 1 \
    --lidar_step_size 1 \
    --use_trajectory_prediction $TRAJ \
    --trajectory_decoder_layers 2 \
    --trajectory_loss_type huber \
    --use_trajectory_target_speed_mask 1 \
    --trajectory_pred_len 6 \
    --trajectory_step_size 2 \
    --trajectory_modes 6 \
    --use_semantic 1 \
    --use_bev_semantic 1 \
    --use_depth 1 \
    --detect_boxes 1 \
    --use_controller_input_prediction 1 \
    --use_wp_gru 0 \
    --continue_epoch 1 \
    --lr 3e-4 \
    --setting 13_withheld \
    --root_dir $PROJECT_ROOT/results/data/$DATASET/data \
    --logdir $PROJECT_ROOT/results/training \
    --cpu_cores $OMP_NUM_THREADS \
    --num_repetitions 1 \
    --use_cosine_schedule 1 \
    --cosine_t0 1 \
    --validation \
    --image_architecture regnety_032 \
    --lidar_architecture regnety_032 \
    --load_file $PROJECT_ROOT/results/training/v3/static-LB5s1-large-gating-seed32/model_0029.pth
    #--load_file $PROJECT_ROOT/results/training/tfpp_base/model_0030.pth
    # --load_file $PROJECT_ROOT/results/training/v2/static-LB9s1-notraj/model_0029.pth
    # --load_file $PROJECT_ROOT/results/training/v2/lidar-LB5s1/model_0011.pth
    # --load_file $PROJECT_ROOT/results/training/v1/stg1-lidar-LB5s1/model_0030.pth
    # --load_file $PROJECT_ROOT/results/training/v1/stg1-lidar-LB2s1/model_0030.pth
