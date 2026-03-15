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

set -e

CONFIG_NAME=${1:?"Usage: sbatch scripts/slurm_train_casehold.sh <config_name>"}
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config not found — $CONFIG_FILE"
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-ft

mkdir -p logs

echo "=============================="
echo " Job ID   : ${SLURM_JOB_ID:-local}"
echo " Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo " Config   : $CONFIG_NAME"
echo " Started  : $(date)"
echo "=============================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export WANDB_RUN_NAME="${CONFIG_NAME}_job${SLURM_JOB_ID:-0}"

# ── Step 1: Train ─────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Training..."
python src/train/train_casehold_lora.py --config "$CONFIG_FILE"

# ── Step 2: Inference ─────────────────────────────────────────────────────────
echo ""
echo "[2/3] Inference..."
python src/evaluate/inference.py --config "$CONFIG_FILE" --split test

# ── Step 3: Evaluate ──────────────────────────────────────────────────────────
echo ""
echo "[3/3] Evaluating..."
OUTPUT_DIR=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['output']['dir'])")

python src/evaluate/eval_casehold.py \
    --predictions "${OUTPUT_DIR}/predictions_test.jsonl" \
    --output      "${OUTPUT_DIR}/eval_test.json"

echo ""
echo "=============================="
echo " Finished : $(date)"
echo " Results  : $OUTPUT_DIR"
echo "=============================="
