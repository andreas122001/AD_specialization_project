export SLURM_JOB_NAME="dev"

echo $SLURM_JOB_ID

pwd
export PROJECT_ROOT=/cluster/work/andrebw/repos/temporal_garage
export DATASET=garage_v2_2025_03_15
export CARLA_ROOT=$PROJECT_ROOT/carla
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}

export OMP_NUM_THREADS=4  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=1 --max_restarts=0 --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
    train.py --id $SLURM_JOB_NAME \
    --use_disk_cache 1 \
    --crop_image 1 \
    --seed 0 \
    --epochs 31 \
    --batch_size 8 \
    --use_recurrent_training 1 \
    --seq_len 4 \
    --lr 3e-4 \
    --setting 13_withheld \
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
