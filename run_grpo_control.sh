# run_grpo_control.sh
#!/bin/bash
set -euo pipefail

# =========================
# Emits merged GRPO config, then submits one sbatch that:
#   - starts vLLM server (bg)
#   - waits 60s
#   - runs GRPO training
# Supports --wait to block until the job finishes.
# =========================

# -------- Defaults (override via flags) --------
BASE_MODEL="/work/HHRI-AI/POC/public/pretraining_weights/Alibaba-Qwen/qwen3/Qwen3-4B-Instruct-2507"
GRPO_DATA="/work/HHRI-AI/POC/public/SFT_Data/mixture_of_thought/All_chunks_split_chunk1/cluster_0000_size_498.jsonl"
WORK_DIR="/work/HHRI-AI/POC/angela/post-training-framework"
RUNS_DIR="${WORK_DIR}/runs"
WANDB_PROJECT="hhri-foxbrain"
SEQUENCE_LEN=4096
MICRO_BSZ=1
GRAD_ACCUM=8
EPOCHS=1
TRL_NUM_GENS=8
TRL_MAX_LEN=16384
VLLM_PORT=8000
RUN_ID="$(date +%Y%m%d_%H%M%S)"
SERVER_GPUS="7"
TRAIN_GPUS="0,1,2,3,4,5,6"
NUM_PROCESSES=2
WAIT_FLAG=0

usage() {
  cat <<EOF
Usage:
  $(basename "$0") --base_model PATH --dataset PATH --run_id ID
                   [--work_dir PATH] [--runs_dir PATH] [--wandb_project NAME]
                   [--sequence_len N] [--micro_bsz N] [--grad_accum N]
                   [--epochs N] [--trl_num_gens N] [--trl_max_len N]
                   [--vllm_port PORT] [--server_gpus LIST] [--train_gpus LIST]
                   [--num_processes N] [--wait]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_model)         BASE_MODEL="$2"; shift 2;;
    --dataset|--dataset_path) GRPO_DATA="$2"; shift 2;;
    --run_id)             RUN_ID="$2"; shift 2;;
    --work_dir)           WORK_DIR="$2"; shift 2;;
    --runs_dir)           RUNS_DIR="$2"; shift 2;;
    --wandb_project)      WANDB_PROJECT="$2"; shift 2;;
    --sequence_len)       SEQUENCE_LEN="$2"; shift 2;;
    --micro_bsz)          MICRO_BSZ="$2"; shift 2;;
    --grad_accum)         GRAD_ACCUM="$2"; shift 2;;
    --epochs)             EPOCHS="$2"; shift 2;;
    --trl_num_gens)       TRL_NUM_GENS="$2"; shift 2;;
    --trl_max_len)        TRL_MAX_LEN="$2"; shift 2;;
    --vllm_port)          VLLM_PORT="$2"; shift 2;;
    --server_gpus)        SERVER_GPUS="$2"; shift 2;;
    --train_gpus)         TRAIN_GPUS="$2"; shift 2;;
    --num_processes)      NUM_PROCESSES="$2"; shift 2;;
    --wait)               WAIT_FLAG=1; shift 1;;
    -h|--help)            usage; exit 0;;
    *) echo "Unknown flag: $1"; usage; exit 1;;
  esac
done

if [[ -z "${BASE_MODEL}" || -z "${GRPO_DATA}" || -z "${RUN_ID}" ]]; then
  echo "Missing required arguments."; usage; exit 1
fi

# Derive TP from SERVER_GPUS (comma-separated list)
VLLM_TP=$(awk -F, '{print NF}' <<< "${SERVER_GPUS}")

# -------- Builder env --------
module load cuda/12.2 || true
source /work/HHRI-AI/anaconda/etc/profile.d/conda.sh
conda activate axolotl

export PYTHONPATH="${WORK_DIR}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# -------- Paths --------
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
MERGED_CFG="${RUN_DIR}/${RUN_ID}.yaml"

echo "=== [CONTROL] Run ID: ${RUN_ID} ==="
echo "BASE_MODEL: ${BASE_MODEL}"
echo "DATASET   : ${GRPO_DATA}"
echo "RUN_DIR   : ${RUN_DIR}"
echo "PORT      : ${VLLM_PORT}"
echo "SERVER_GPUS=${SERVER_GPUS}  TRAIN_GPUS=${TRAIN_GPUS}  NP=${NUM_PROCESSES}"

# -------- Emit merged config --------
  # --trl_use_vllm \
python "${WORK_DIR}/src/config_builders.py" grpo \
  --base_model "$BASE_MODEL" \
  --dataset_path "$GRPO_DATA" \
  --output_path "$MERGED_CFG" \
  --adapter full \
  --output_dir "$RUN_DIR" \
  --sequence_len "$SEQUENCE_LEN" \
  --micro_batch_size "$MICRO_BSZ" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --num_epochs "$EPOCHS" \
  --trl_num_generations "$TRL_NUM_GENS" \
  --trl_max_completion_length "$TRL_MAX_LEN" \
  --trl_reward_funcs rewards.model_helpfulness_reward,rewards.think_format_reward \
  --trl_reward_weights 1.0,0.2 \
  --vllm_tensor_parallel_size "$VLLM_TP" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "qwen3-grpo-${RUN_ID}"

echo "Merged cfg -> ${MERGED_CFG}"

# -------- Submit combined job --------
SBATCH_WAIT=()
[[ $WAIT_FLAG -eq 1 ]] && SBATCH_WAIT=(--wait)

JOB_ID=$(sbatch "${SBATCH_WAIT[@]}" --parsable \
  grpo_job.sh \
  --cfg "$MERGED_CFG" \
  --work_dir "$WORK_DIR" \
  --runs_dir "$RUNS_DIR" \
  --run_id "$RUN_ID" \
  --server_gpus "$SERVER_GPUS" \
  --train_gpus "$TRAIN_GPUS" \
  --num_processes "$NUM_PROCESSES")

echo "[CONTROL] Submitted combined job: ${JOB_ID}"
echo "Run dir: ${RUN_DIR}"
