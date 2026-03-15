#!/bin/bash
# slurm_train_casehold.sh — Submit a CaseHOLD QLoRA experiment to the MLP cluster.
# Uses train_casehold_lora.py (SFTTrainer + 4-bit QLoRA).
#
# Usage:
#   sbatch scripts/slurm_train_casehold.sh qlora_casehold_qwen
#   sbatch scripts/slurm_train_casehold.sh qlora_casehold_llama

#SBATCH --job-name=casehold-qlora
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --time=12:00:00            # CaseHOLD QLoRA ~4-6h on 2080 Ti
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=14G
#SBATCH --cpus-per-task=2

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "./scripts/slurm_train_casehold.sh" ]]; then
    REPO_ROOT="$(pwd)"
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

DEBUG_LOG="${LOG_DIR}/debug-casehold-${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "${DEBUG_LOG}") 2>&1

trap 'echo "[ERROR] line ${LINENO}: command failed: ${BASH_COMMAND}"' ERR

CONFIG_NAME=${1:?"Usage: sbatch scripts/slurm_train_casehold.sh <config_name>"}
CONFIG_FILE="${REPO_ROOT}/configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config not found — $CONFIG_FILE"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-ft

cd "${REPO_ROOT}"

echo "=============================="
echo " Job ID   : ${SLURM_JOB_ID:-local}"
echo " Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo " Config   : $CONFIG_NAME"
echo " Repo     : ${REPO_ROOT}"
echo " Workdir  : $(pwd)"
echo " Submit   : ${SLURM_SUBMIT_DIR:-N/A}"
echo " DebugLog : ${DEBUG_LOG}"
echo " Started  : $(date)"
echo "=============================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export WANDB_RUN_NAME="${CONFIG_NAME}_job${SLURM_JOB_ID:-0}"

# ── Step 1: Train ─────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Training..."
python "${REPO_ROOT}/src/train/train_casehold_lora.py" --config "$CONFIG_FILE"

# ── Step 2: Inference ─────────────────────────────────────────────────────────
echo ""
echo "[2/3] Inference..."
python "${REPO_ROOT}/src/evaluate/inference.py" --config "$CONFIG_FILE" --split test

# ── Step 3: Evaluate ──────────────────────────────────────────────────────────
echo ""
echo "[3/3] Evaluating..."
OUTPUT_DIR=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['output']['dir'])")
OUTPUT_DIR_ABS="${REPO_ROOT}/${OUTPUT_DIR}"

python "${REPO_ROOT}/src/evaluate/eval_casehold.py" \
    --predictions "${OUTPUT_DIR_ABS}/predictions_test.jsonl" \
    --output      "${OUTPUT_DIR_ABS}/eval_test.json"

echo ""
echo "=============================="
echo " Finished : $(date)"
echo " Results  : ${OUTPUT_DIR_ABS}"
echo "=============================="
