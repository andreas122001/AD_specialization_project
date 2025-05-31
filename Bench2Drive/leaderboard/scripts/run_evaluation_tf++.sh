#!/bin/bash
#SBATCH --job-name=b2d/tfpp_default
#SBATCH --account=share-ie-idi
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=0-02:30:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --output=/cluster/work/andrebw/repos/temporal_garage/evaluation/bench2drive/logs/b2d_009_%a_%A.out  # File to which STDOUT will be written
#SBATCH --error=/cluster/work/andrebw/repos/temporal_garage/evaluation/bench2drive/logs/b2d_009_%a_%A.out   # File to which STDERR will be written
#SBATCH --partition=GPUQ
#SBATCH --constraint=(v100|p100)

export CARLA_ROOT=/cluster/work/andrebw/repos/temporal_garage/carla
export WORK_DIR=/cluster/work/andrebw/repos/temporal_garage/Bench2Drive
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH=$PYTHONPATH:/cluster/work/andrebw/repos/temporal_garage/team_code
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export MODEL=$(echo $SLURM_JOB_NAME | cut -d'/' -f2)
export CKPT=$MODEL"_model_0030"

export NGPUS=$(echo $SLURM_JOB_GPUS | grep -oP [0-9]+ | wc -l)
echo Num GPUS: $NGPUS
gpu_list=(${SLURM_JOB_GPUS//,/ })
TASK_NUM=8 # $((2*$NGPUS))
echo GPU_LIST: $(echo $SLURM_JOB_GPUS | sed -e "s/,/ /g")
echo MODEL=$MODEL

#!/bin/bash
BASE_PORT=30000
BASE_TM_PORT=50000
IS_BENCH2DRIVE=True
BASE_ROUTES=${WORK_DIR}/leaderboard/data/bench2drive220
TEAM_AGENT=/cluster/work/andrebw/repos/temporal_garage/team_code/sensor_agent.py
TEAM_CONFIG=/cluster/work/andrebw/repos/temporal_garage/team_code/checkpoints/$CKPT
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220
PLANNER_TYPE=traj
ALGO=$MODEL
SAVE_PATH=${WORK_DIR}/leaderboard/data/eval_bench2drive220_${ALGO}_${PLANNER_TYPE}

if [ ! -d "${WORK_DIR}/../evaluation/bench2drive/${ALGO}_b2d_${PLANNER_TYPE}" ]; then
    mkdir "${WORK_DIR}/../evaluation/bench2drive/${ALGO}_b2d_${PLANNER_TYPE}"
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} created. \033[0m"
else
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} already exists. \033[0m"
fi

# Check if the split_xml script needs to be executed
if [ ! -f "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag" ]; then
    echo -e "****************************\033[33m Attention \033[0m ****************************"
    echo -e "\033[33m Running split_xml.py \033[0m"
    python -u ${WORK_DIR}/tools/split_xml.py $BASE_ROUTES $TASK_NUM $ALGO $PLANNER_TYPE
    touch "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
    echo -e "\033[32m Splitting complete. Flag file created. \033[0m"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi

echo -e "**************\033[36m Please Manually adjust GPU or TASK_ID \033[0m **************"
# Example, 8*H100, 1 task per gpu
# IFS=',' read -ra GPU_RANK_LIST <<< "$SLURM_JOB_GPUS"
# TASK_LIST=( $(seq 0 $(($TASK_NUM-1))) )
GPU_RANK_LIST=( 0 0 1 1 )
TASK_LIST=( 0 1 2 3 )
echo -e "\033[32m GPU_RANK_LIST: ${GPU_RANK_LIST[*]} \033[0m"
echo -e "\033[32m TASK_LIST: ${TASK_LIST[*]} \033[0m"
echo -e "***********************************************************************************"

# Load slurm modules
echo "Loading modules..."
module load Anaconda3/2024.02-1
module load libjpeg-turbo/2.1.5.1-GCCcore-12.3.0

echo "Activating conda env (lb2)..."
conda activate lb2

nvidia-smi

# bash $CARLA_SERVER -RenderOffScreen -nosound -carla-rpc-port=$PORT -graphicsadapter=$GPU_RANK

length=${#TASK_LIST[@]}
for ((i=0; i<$length; i++ )); do
    PORT=$((BASE_PORT + i * 150))
    TM_PORT=$((BASE_TM_PORT + i * 150))
    ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.xml"
    CHECKPOINT_ENDPOINT="${WORK_DIR}/../evaluation/bench2drive/${ALGO}_b2d_${PLANNER_TYPE}/${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
    mkdir -p "${WORK_DIR}/../evaluation/bench2drive/${ALGO}_b2d_${PLANNER_TYPE}"
    GPU_RANK=${GPU_RANK_LIST[$i]}
    echo -e "\033[32m ALGO: $ALGO \033[0m"
    echo -e "\033[32m PLANNER_TYPE: $PLANNER_TYPE \033[0m"
    echo -e "\033[32m TASK_ID: $i \033[0m"
    echo -e "\033[32m PORT: $PORT \033[0m"
    echo -e "\033[32m TM_PORT: $TM_PORT \033[0m"
    echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
    echo -e "\033[32m GPU_RANK: $GPU_RANK \033[0m"
    echo -e "\033[32m bash ${WORK_DIR}/leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \033[0m"
    echo -e "***********************************************************************************"
    bash -e ${WORK_DIR}/leaderboard/scripts/run_evaluation.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 > ${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.log &

    sleep 10
done
wait

echo Finished!
