#!/bin/bash
#SBATCH --partition=hhai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=1:30:00
#SBATCH --mem=200G
#SBATCH --job-name=grpo_combo
#SBATCH --output=grpo_%j.out
#SBATCH --error=grpo_%j.err

set -euo pipefail

# =========================
# Arg parsing
# =========================
CFG=""
WORK_DIR=""
RUNS_DIR=""
RUN_ID=""
PORT="8000"
SERVER_GPUS="2,3"
TRAIN_GPUS="0,1"
NUM_PROCESSES="2"

usage() {
  cat <<EOF
Usage (sbatch args after script name):
  grpo_job.sh --cfg PATH --work_dir PATH --runs_dir PATH --run_id ID
              [--port PORT] [--server_gpus LIST] [--train_gpus LIST] [--num_processes N]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg)            CFG="$2"; shift 2;;
    --work_dir)       WORK_DIR="$2"; shift 2;;
    --runs_dir)       RUNS_DIR="$2"; shift 2;;
    --run_id)         RUN_ID="$2"; shift 2;;
    --port)           PORT="$2"; shift 2;;
    --server_gpus)    SERVER_GPUS="$2"; shift 2;;
    --train_gpus)     TRAIN_GPUS="$2"; shift 2;;
    --num_processes)  NUM_PROCESSES="$2"; shift 2;;
    -h|--help)        usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "${CFG}" || -z "${WORK_DIR}" || -z "${RUNS_DIR}" || -z "${RUN_ID}" ]]; then
  echo "Missing required arguments."; usage; exit 1
fi

RUN_DIR="${RUNS_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

# =========================
# Environment
# =========================
module load cuda/12.2

source /work/HHRI-AI/anaconda/etc/profile.d/conda.sh
conda activate axolotl

export PYTHONPATH="${WORK_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export TRANSFORMERS_VERBOSITY=error

# Reward-model env
export RM_ID="/work/HHRI-AI/POC/public/pretraining_weights/Alibaba-Qwen/qwen3/Qwen-3-Nemotron-32B-Reward"
export RM_DEVICE_MAP="auto"
export RM_DTYPE="bf16"
export RM_BATCH_SIZE="4"
export RM_FORMAT="nothink"

# =========================
# Logging
# =========================
JOB_OUT="${RUN_DIR}/combo_${SLURM_JOB_ID}"
mkdir -p "$JOB_OUT"
RUN_LOG="$JOB_OUT/run.log"
ERR_LOG="$JOB_OUT/run.err"
exec > >(tee -a "$RUN_LOG") \
     2> >(grep -v "Caching is incompatible with gradient checkpointing" | tee -a "$ERR_LOG" >&2)

echo "=== Combo job $SLURM_JOB_ID starting at $(date) ==="
echo "Node(s): $SLURM_JOB_NODELIST"
echo "CFG     : $CFG"
echo "RUN_DIR : $RUN_DIR"
echo "PORT    : $PORT"
echo "SERVER_GPUS=${SERVER_GPUS}  TRAIN_GPUS=${TRAIN_GPUS}  NP=${NUM_PROCESSES}"
echo "axolotl  : $(command -v axolotl || echo 'not found')"

# =========================
# Start vLLM server (background) on SERVER_GPUS
# =========================
BASE_URL="http://$(hostname -s):${PORT}"
echo "[SERVER] Starting vLLM on ${BASE_URL} (CUDA_VISIBLE_DEVICES=${SERVER_GPUS})"

set +e
CUDA_VISIBLE_DEVICES="${SERVER_GPUS}" axolotl vllm-serve "$CFG" &
SERVER_PID=$!
AXO_RC=$?
if [[ $AXO_RC -ne 0 ]]; then
  echo "[SERVER] axolotl CLI failed (rc=$AXO_RC). Falling back to 'python -m axolotl.cli.vllm_serve'."
  CUDA_VISIBLE_DEVICES="${SERVER_GPUS}" python -m axolotl.cli.vllm_serve "$CFG" &
  SERVER_PID=$!
fi
set -e

trap "echo '[SERVER] Stopping (PID ${SERVER_PID})'; kill ${SERVER_PID}; wait ${SERVER_PID} 2>/dev/null || true" SIGINT SIGTERM EXIT

# =========================
# Wait 60 seconds (simple approach as requested)
# =========================
echo "[SERVER] Sleeping 60s before training..."
sleep 60

# =========================
# GRPO training on TRAIN_GPUS
# =========================
echo "[TRAIN] Launch on CUDA_VISIBLE_DEVICES=${TRAIN_GPUS}  --num-processes ${NUM_PROCESSES}"


echo "[TRAIN] axolotl train \"${CFG}\" --num-processes ${NUM_PROCESSES}"
set +e
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" axolotl train "${CFG}" --num-processes "${NUM_PROCESSES}"
AXO_RC=$?
if [[ $AXO_RC -ne 0 ]]; then
  echo "[TRAIN] axolotl CLI failed (rc=$AXO_RC). Falling back to 'python -m axolotl.cli.train'."
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" python -m axolotl.cli.train "${CFG}" --num-processes "${NUM_PROCESSES}"
  AXO_RC=$?
fi
set -e

# =========================
# Done
# =========================
echo "=== Combo job $SLURM_JOB_ID finished with rc=$AXO_RC at $(date) ==="
exit $AXO_RC