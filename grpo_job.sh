#!/bin/bash
#SBATCH --partition=hhai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=1:30:00
#SBATCH --mem=200G
#SBATCH --job-name=grpo_qwen3_full_single
#SBATCH --output=grpo_qwen3_full_%j.out
#SBATCH --error=grpo_qwen3_full_%j.err

set -euo pipefail

# =========================
# USER CONFIG
# =========================
BASE_MODEL="/work/HHRI-AI/POC/public/pretraining_weights/Alibaba-Qwen/qwen3/Qwen3-4B-Instruct-2507"
GRPO_DATA="/work/HHRI-AI/POC/angela/post-training-framework/data/chunk1-merged.jsonl"
WORK_DIR="/work/HHRI-AI/POC/angela/post-training-framework"   # repo root containing src/
RUNS_DIR="${WORK_DIR}/runs"
WANDB_PROJECT="hhri-foxbrain"

# Reward model for model_helpfulness_reward (can be overridden per-job)
export RM_ID="nvidia/Qwen-3-Nemotron-32B-Reward"
export RM_DEVICE_MAP="auto"
export RM_DTYPE="bf16"    # or fp16/fp32
export RM_BATCH_SIZE="2"
export RM_FORMAT="nothink"

# =========================
# Environment
# =========================
module load cuda/12.2

# Conda
source /work/HHRI-AI/anaconda/etc/profile.d/conda.sh
conda activate axolotl-angela

# Make local src/ importable so "rewards.*" resolves
export PYTHONPATH="${WORK_DIR}/src:${PYTHONPATH:-}"

# Optional caches (safe to remove)
# export HF_HOME="/work/HHRI-AI/.hf"
# export TRANSFORMERS_CACHE="$HF_HOME/transformers"
# export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

# =========================
# Paths and logging
# =========================
JOB_OUT="${RUNS_DIR}/qwen3_grpo_${SLURM_JOB_ID}"
MERGED_CFG="${JOB_OUT}/${SLURM_JOB_ID}.yaml"
mkdir -p "$JOB_OUT"

# Save full logs inside JOB_OUT (in addition to Slurm logs)
RUN_LOG="$JOB_OUT/run.log"
ERR_LOG="$JOB_OUT/run.err"
exec > >(tee -a "$RUN_LOG") 2> >(tee -a "$ERR_LOG" >&2)

echo "=== Job $SLURM_JOB_ID starting at $(date) ==="
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Conda env: $(conda info --envs | grep '*' || true)"
echo "axolotl path: $(command -v axolotl || echo 'not found')"
echo "PYTHONPATH: $PYTHONPATH"
echo "Reward model: ${RM_ID}"

# =========================
# 1) Emit merged Axolotl YAML (GRPO)
# =========================
echo "[1/2] Emitting GRPO Axolotl config..."
python "${WORK_DIR}/src/config_builders.py" grpo \
  --base_model "$BASE_MODEL" \
  --dataset_path "$GRPO_DATA" \
  --output_path "$MERGED_CFG" \
  --adapter full \
  --output_dir "$JOB_OUT" \
  --sequence_len 4096 \
  --micro_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_epochs 3 \
  --trl_num_generations 8 \
  --trl_max_completion_length 256 \
  --trl_reward_funcs rewards.model_helpfulness_reward,rewards.think_format_reward \
  --trl_reward_weights 1.0,0.2 \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name qwen3-grpo-run1

echo "Merged config -> $MERGED_CFG"
echo "Outputs dir   -> $JOB_OUT"

# =========================
# 2) Train with Axolotl on the merged config
# =========================
echo "[2/2] Training: axolotl train \"$MERGED_CFG\""
set +e
axolotl train "$MERGED_CFG"
AXO_RC=$?

# Fallback to python module form if CLI entrypoint is problematic
if [[ $AXO_RC -ne 0 ]]; then
  echo "axolotl CLI failed (rc=$AXO_RC). Falling back to 'python -m axolotl.cli.train'."
  python -m axolotl.cli.train "$MERGED_CFG"
  AXO_RC=$?
fi
set -e

echo "=== Job $SLURM_JOB_ID finished with rc=$AXO_RC at $(date) ==="
exit $AXO_RC
