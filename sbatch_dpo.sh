#!/bin/bash
#SBATCH --partition=hhai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=10:00:00
#SBATCH --mem=400G
#SBATCH --job-name=dpo_qwen3_full_single
#SBATCH --output=dpo_qwen3_full_%j.out
#SBATCH --error=dpo_qwen3_full_%j.err

set -euo pipefail

# =========================
# USER CONFIG
# =========================
SEQUENCE_LEN="8192"
MICRO_BSZ="2"
GRAD_ACCUM="8"
EPOCHS="3"
LEARNING_RATE="2e-5"
BASE_MODEL="${INIT_BASE_MODEL:-/work/HHRI-AI/POC/public/pretraining_weights/Alibaba-Qwen/qwen3/Qwen3-4B-Instruct-2507}"
DPO_DATA="/work/HHRI-AI/POC/angela/post-training-framework/data/reduced_data/dpo/chunk2-reduced.jsonl"
WORK_DIR="/work/HHRI-AI/POC/angela/post-training-framework"
RUNS_DIR="${WORK_DIR}/runs"
WANDB_PROJECT="hhri-foxbrain"
RUN_ID="dpo_qwen3_full_single"

usage() {
  cat <<EOF
Usage:
  sbatch DPO_JOB.sh --base_model PATH --dataset PATH --run_id ID
                    [--work_dir PATH] [--runs_dir PATH] [--wandb_project NAME]
                    [--sequence_len N] [--micro_bsz N] [--grad_accum N] [--epochs N]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_model)    BASE_MODEL="$2"; shift 2;;
    --dataset|--dataset_path) DPO_DATA="$2"; shift 2;;
    --run_id)        RUN_ID="$2"; shift 2;;
    --work_dir)      WORK_DIR="$2"; shift 2;;
    --runs_dir)      RUNS_DIR="$2"; shift 2;;
    --wandb_project) WANDB_PROJECT="$2"; shift 2;;
    --sequence_len)  SEQUENCE_LEN="$2"; shift 2;;
    --micro_bsz)     MICRO_BSZ="$2"; shift 2;;
    --grad_accum)    GRAD_ACCUM="$2"; shift 2;;
    --epochs)        EPOCHS="$2"; shift 2;;
    -h|--help)       usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

if [[ -z "${BASE_MODEL}" || -z "${DPO_DATA}" || -z "${RUN_ID}" ]]; then
  echo "Missing required arguments."; usage; exit 1
fi

# =========================
# Environment
# =========================
module load cuda/12.2

source /work/HHRI-AI/anaconda/etc/profile.d/conda.sh
conda activate axolotl

# Unique dataset and transformers cache per job
export HF_DATASETS_CACHE="${WORK_DIR}/.cache/hf_datasets_${SLURM_JOB_ID}"
export TRANSFORMERS_CACHE="${WORK_DIR}/.cache/hf_transformers_${SLURM_JOB_ID}"
export TORCH_DISABLE_ADDR2LINE=1
mkdir -p "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# Limit parallelism and improve I/O reliability
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export HF_DATASETS_NUM_PROC=24
export TORCH_SHOW_CPP_STACKTRACES=1

# =========================
# Paths and Logging
# =========================
JOB_OUT="${RUNS_DIR}/${RUN_ID}"
MERGED_CFG="${JOB_OUT}/${RUN_ID}.yaml"
DONE_FLAG="${JOB_OUT}/.done"
mkdir -p "${JOB_OUT}"

RUN_LOG="${JOB_OUT}/run.log"
ERR_LOG="${JOB_OUT}/run.err"
exec > >(stdbuf -oL tee -a "${RUN_LOG}") 2> >(stdbuf -oL tee -a "${ERR_LOG}" >&2)

echo "=== Job ${SLURM_JOB_ID} starting at $(date) ==="
echo "RUN_ID: ${RUN_ID}"
echo "BASE_MODEL: ${BASE_MODEL}"
echo "DPO_DATA: ${DPO_DATA}"

# =========================
# Safety: Clear corrupt dataset cache
# =========================
if [[ -d "${DPO_DATA}" ]]; then
  echo "[INFO] Checking and cleaning broken Arrow cache files in: ${DPO_DATA}"
  find "${DPO_DATA}" -name "*.arrow" -type f -size -1k -delete
fi

# =========================
# 1) Emit Axolotl config
# =========================
echo "[1/2] Generating DPO config..."
python "${WORK_DIR}/src/config_builders.py" dpo \
  --base_model "${BASE_MODEL}" \
  --dataset_path "${DPO_DATA}" \
  --output_path "${MERGED_CFG}" \
  --output_dir "${JOB_OUT}" \
  --adapter full \
  --sequence_len "${SEQUENCE_LEN}" \
  --micro_batch_size "${MICRO_BSZ}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --num_epochs "${EPOCHS}" \
  --chat_template "chatml" \
  --learning_rate "${LEARNING_RATE}" \
  --trl_beta 0.1 \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_name "dpo-${RUN_ID}"

if [[ ! -f "${MERGED_CFG}" ]]; then
  echo "[ERROR] Failed to generate Axolotl config. Exiting."
  exit 1
fi
echo "Merged config -> ${MERGED_CFG}"

# =========================
# 2) Launch Axolotl in its own process group
# =========================
echo "Launching Axolotl training..."

setsid axolotl train "${MERGED_CFG}" &
AXO_PID=$!

echo "Axolotl PID: $AXO_PID"

# Completion patterns (extended regex)
echo "Starting watcher"
(
  tail -n +1 -F "${RUN_LOG}" | while IFS= read -r line; do
    [[ "$line" == "[WATCHER]"* ]] && continue
    echo "[WATCHER] $line"
    if [[ "$line" == *"Model successfully saved to"* ]]; then
      echo "[WATCHER] Detected model save"

      # Mark done
      touch "${DONE_FLAG}"

      # Kill the Axolotl process group
      echo "[WATCHER] Killing Axolotl process group (PGID: $AXO_PID)"
      kill -- -"$AXO_PID" 2>/dev/null || true

      # Exit cleanly
      echo "=== Model save detected. SLURM job exiting early with code 0 ==="
      exit 0
    fi
  done
) &

WATCHER_PID=$!

wait -n "$AXO_PID" "$WATCHER_PID" || true

# If model was saved, exit cleanly
if [[ -f "${DONE_FLAG}" ]]; then
  echo "=== Training completed and model saved. Exiting with code 0 ==="
  exit 0
else
  echo "=== Axolotl training exited without model save. Exiting with code 1 ==="
  exit 1
fi
