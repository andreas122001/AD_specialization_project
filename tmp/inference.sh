#!/bin/bash
#SBATCH --job-name=tfpp_default
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH -o %x/log.out
#SBATCH -e %x/log.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32gb
#SBATCH --time=00-00:10:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=(h100)

echo $SLURM_JOB_NAME

FREE_WORLD_PORT=`comm -23 <(seq 10000 10049 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`


echo 'World Port:' $FREE_WORLD_PORT


FREE_STREAMING_PORT=`comm -23 <(seq 20000 20049 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`


echo 'Streaming Port:' $FREE_STREAMING_PORT


/cluster/work/andrebw/repos/temporal_garage/carla/CarlaUE4.sh -carla-rpc-port=${FREE_WORLD_PORT} -nosound -RenderOffScreen -carla-primary-port=0 -graphicsadapter=0 -carla-streaming-port=${FREE_STREAMING_PORT} &


mkdir -p /cluster/work/andrebw/repos/temporal_garage/tmp/${SLURM_JOB_NAME}/res/logs

sleep 20


export CARLA_ROOT=/cluster/work/andrebw/repos/temporal_garage/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=/cluster/work/andrebw/repos/temporal_garage/scenario_runner
export LEADERBOARD_ROOT=/cluster/work/andrebw/repos/temporal_garage/leaderboard
export PYTHONPATH="${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export PORT=$FREE_WORLD_PORT
echo 'World Port:' $PORT
export TM_PORT=`comm -23 <(seq 43300 43349 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
echo 'TM Port:' $TM_PORT
export ROUTES=/cluster/work/andrebw/repos/temporal_garage/data/collection/50x36_Town13/HazardAtSideLane/1710_4.xml
export TEAM_AGENT=/cluster/work/andrebw/repos/temporal_garage/team_code/sensor_agent.py
export TEAM_CONFIG=/cluster/work/andrebw/repos/temporal_garage/team_code/checkpoints/${SLURM_JOB_NAME}_model_0030/
export CHALLENGE_TRACK_CODENAME=SENSORS
export REPETITIONS=1
export RESUME=0
export SEED=0
export CHECKPOINT_ENDPOINT=/cluster/work/andrebw/repos/temporal_garage/tmp/${SLURM_JOB_NAME}/res/1710_4.json
export DEBUG_ENV_AGENT=0
export DEBUG_CHALLENGE=0
export RECORD=1
export DIRECT=1
export COMPILE=0
export TOWN=eval
export REPETITION=0
export DATAGEN=0
export TUNED_AIM_DISTANCE=0
export SLOWER=0
export UNCERTAINTY_WEIGHT=1
export STOP_AFTER_METER=-1
export SAVE_PATH=/cluster/work/andrebw/repos/temporal_garage/tmp/${SLURM_JOB_NAME}/res/logs
export IS_BENCH2DRIVE=0
export WORK_DIR=/cluster/work/andrebw/repos/temporal_garage

module purge
module load Anaconda3/2024.02-1
module load libjpeg-turbo/2.1.5.1-GCCcore-12.3.0

conda activate lb2

nvidia-smi

python3 -u ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py --routes=${ROUTES} --repetitions=${REPETITIONS} --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} --agent-config=${TEAM_CONFIG} --debug=0 --traffic-manager-seed=${SEED} --record=${RECORD_PATH} --resume=${RESUME} --port=${PORT} --timeout=120 --traffic-manager-port=${TM_PORT}
