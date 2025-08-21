#!/bin/bash
set -eo pipefail

echo "Pipeline script started"
# =========================
# Fixed-order cyclic pipeline:
# For each 3-bit mask (SFT=0, GRPO=1), run 3 stages in sequence.
# Stage i uses DATASETS[i] and starts from the previous stage's output.
# All knobs can be overridden via environment variables.
# =========================

# -------- User knobs (override by exporting or prefixing at call time) --------
INIT_BASE_MODEL="${INIT_BASE_MODEL:-/work/HHRI-AI/POC/public/pretraining_weights/Alibaba-Qwen/qwen3/Qwen3-4B-Instruct-2507}"
WORK_DIR="${WORK_DIR:-/work/HHRI-AI/POC/angela/post-training-framework}"
RUNS_DIR="${RUNS_DIR:-$WORK_DIR/sft_dpo_runs_final}"
WANDB_PROJECT="${WANDB_PROJECT:-hhri-foxbrain}"

# GRPO combined-job resources
# SERVER_GPUS="${SERVER_GPUS:-4,5}"  # Unused variable in original script, kept for reference
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
VLLM_PORT="${VLLM_PORT:-8000}"

# Which masks to run: space-separated list of integers in [0..7].
# Default runs all 8 combinations.
export MASKS="0 6"
export PYTHONUNBUFFERED=1
#export MASKS="2 3 4 5"
echo $MASKS

# Optional prefix to make runs easier to group in RUNS_DIR / W&B
RUN_PREFIX="${RUN_PREFIX:-order}"

# -------- Datasets: fixed order --------
SFT_DATASETS=(
  "/work/HHRI-AI/POC/angela/post-training-framework/data/reduced_data/sft/chunk1-reduced.jsonl"
  "/work/HHRI-AI/POC/angela/post-training-framework/data/reduced_data/sft/chunk2-reduced.jsonl"
  "/work/HHRI-AI/POC/angela/post-training-framework/data/reduced_data/sft/chunk4-reduced.jsonl"
)
DPO_DATASETS=(
  "/work/HHRI-AI/POC/angela/post-training-framework/data/reduced_data/dpo/chunk2-reduced.jsonl"
  "/work/HHRI-AI/POC/angela/post-training-framework/data/reduced_data/dpo/chunk4-reduced.jsonl"
)

timestamp() {
  date +%Y%m%d_%H%M%S
}

echo "Pipeline started at: $(timestamp)"

# Validate that required files exist
if [[ ! -f "/work/HHRI-AI/POC/angela/post-training-framework/sft_job.sh" ]]; then
  echo "ERROR: SFT_JOB.sh not found in current directory"
  exit 1
fi

if [[ ! -f "/work/HHRI-AI/POC/angela/post-training-framework/sbatch_dpo.sh" ]]; then
  echo "ERROR: sbatch_dpo.sh not found in current directory"
  exit 1
fi

# Create runs directory if it doesn't exist
mkdir -p "${RUNS_DIR}"

series_counter=0

for mask in $MASKS; do
  ((series_counter+=1))
  series_timestamp=$(timestamp)
  series_tag="${RUN_PREFIX}_m${mask}_${series_timestamp}_s${series_counter}"
  
  echo "==== Starting Series ${series_tag} (mask=${mask}) ===="
  
  base_model="${INIT_BASE_MODEL}"
  
  for stage in 0 1 2; do
    # Determine algorithm and dataset based on stage and mask bits:
    # Stage 0: always SFT using SFT_DATASETS[0]
    # Stage 1 and 2: if corresponding mask bit is 1 => DPO, else SFT
    if (( stage == 0 )); then
      ALG="sft"
      sft_dataset="${SFT_DATASETS[$stage]}"
      dpo_dataset=""
    else
      # Extract bit at position 'stage' from mask
      bit=$(( (mask >> stage) & 1 ))
      if (( bit == 1 )); then
        ALG="dpo"
        sft_dataset=""
        # DPO datasets are indexed differently (stage-1 since stage 0 is always SFT)
        dpo_dataset="${DPO_DATASETS[$((stage - 1))]}"
      else
        ALG="sft"
        sft_dataset="${SFT_DATASETS[$stage]}"
        dpo_dataset=""
      fi
    fi
    
    # Create unique run ID for this stage
    stage_timestamp=$(timestamp)
    run_id="${series_tag}_stage${stage}_${ALG}_${stage_timestamp}"
    
    echo "[Stage ${stage}] ALG=${ALG}, SFT_DATASET=${sft_dataset}, DPO_DATASET=${dpo_dataset}, RUN_ID=${run_id}"
    
    # Create run directory
    run_dir="${RUNS_DIR}/${run_id}"
    mkdir -p "${run_dir}"
    
    # Validate dataset files exist
    if [[ "${ALG}" == "sft" && ! -f "${sft_dataset}" ]]; then
      echo "ERROR: SFT dataset not found: ${sft_dataset}"
      exit 1
    elif [[ "${ALG}" == "dpo" && ! -f "${dpo_dataset}" ]]; then
      echo "ERROR: DPO dataset not found: ${dpo_dataset}"
      exit 1
    fi
    
    if [[ "${ALG}" == "sft" ]]; then
      # --- SFT stage (blocking) ---
      echo "Submitting SFT job for ${run_id}..."


      if ! sbatch --wait --job-name "sft_${run_id}" /work/HHRI-AI/POC/angela/post-training-framework/sft_job.sh \
        --base_model "${base_model}" \
        --dataset "${sft_dataset}" \
        --run_id "${run_id}" \
        --work_dir "${WORK_DIR}" \
        --runs_dir "${RUNS_DIR}" \
        --wandb_project "${WANDB_PROJECT}"; then
        echo "ERROR: SFT job failed for ${run_id}"
        exit 1
      fi
      
      echo Update base model for next stage
      base_model="${run_dir}"
      
    else
      # --- DPO stage (blocking) ---
      echo "Submitting DPO job for ${run_id}..."
      if ! sbatch --wait --job-name "dpo_${run_id}" /work/HHRI-AI/POC/angela/post-training-framework/sbatch_dpo.sh \
        --base_model "${base_model}" \
        --dataset "${dpo_dataset}" \
        --run_id "${run_id}" \
        --work_dir "${WORK_DIR}" \
        --runs_dir "${RUNS_DIR}" \
        --wandb_project "${WANDB_PROJECT}"; then
        echo "ERROR: DPO job failed for ${run_id}"
        exit 1
      fi
      
      echo Update base model for next stage
      base_model="${run_dir}"
    fi
    
    echo "[Stage ${stage}] Completed successfully. Output model: ${base_model}"
  done
  
  echo "==== Completed series ${series_tag}; final model: ${base_model} ===="
  echo ""
done

echo "All requested masks finished successfully at $(timestamp)"