#!/bin/bash
#SBATCH --partition=hhai
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=1:00:00
#SBATCH --mem=200G
#SBATCH --job-name=sft_qwen3_lora_single
#SBATCH --output=sft_qwen3_lora_%j.out
#SBATCH --error=sft_qwen3_lora_%j.err

set -euo pipefail

# =========================
# USER CONFIG
# =========================
BASE_MODEL="/work/HHRI-AI/POC/public/pretraining_weights/Alibaba-Qwen/qwen3/Qwen3-4B-Instruct-2507"
SFT_DATA="/work/HHRI-AI/POC/angela/post-training-framework/data/chunk1-merged.jsonl"
WORK_DIR="/work/HHRI-AI/POC/angela/post-training-framework"   # repo root containing src/sft.py
RUNS_DIR="${WORK_DIR}/runs"
WANDB_PROJECT="qwen3-sft"

# =========================
# Environment
# =========================
module load cuda/12.2

# Conda
source /work/HHRI-AI/anaconda/etc/profile.d/conda.sh
conda activate axolotl

# Optional caches (safe to remove)
# export HF_HOME="/work/HHRI-AI/.hf"
# export TRANSFORMERS_CACHE="$HF_HOME/transformers"
# export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"

# =========================
# Paths and logging
# =========================
# BASE_CFG="${WORK_DIR}/src/configs/sft_config.yaml"
JOB_OUT="${RUNS_DIR}/qwen3_sft_${SLURM_JOB_ID}"
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

# =========================
# 1) Emit merged Axolotl YAML
# =========================
echo "[1/2] Emitting merged Axolotl config..."
python "${WORK_DIR}/src/config_builders.py" sft \
  --base_model "$BASE_MODEL" \
  --dataset_path "$SFT_DATA" \
  --output_path "$MERGED_CFG" \
  --adapter full \
  --output_dir "$JOB_OUT" \
  --sequence_len 8192 \
  --micro_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --num_epochs 3 \
  --wandb_project hhri-foxbrain \
  --wandb_name qwen3-sft-run1

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
