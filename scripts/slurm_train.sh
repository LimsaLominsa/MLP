#!/bin/bash
# slurm_train.sh — Submit a single fine-tuning experiment to the MLP cluster.
#
# Usage:
#   sbatch scripts/slurm_train.sh lora_billsum_qwen
#   sbatch scripts/slurm_train.sh full_billsum_qwen
#   sbatch scripts/slurm_train.sh lora_casehold_llama
#
# After training, inference + evaluation are run automatically.

# ── SLURM resource requests ───────────────────────────────────────────────────
#SBATCH --job-name=llm-ft
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#SBATCH --time=24:00:00          # BillSum ~20h; CaseHOLD ~4h — adjust as needed
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=14G                # Teaching QOS: 14G per CPU; 2 CPUs = 28G total
#SBATCH --cpus-per-task=1       # Teaching QOS max: 2 CPUs per job

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/scripts" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
elif [[ -f "./scripts/slurm_train.sh" ]]; then
    REPO_ROOT="$(pwd)"
else
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

DEBUG_LOG="${LOG_DIR}/debug-train-${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "${DEBUG_LOG}") 2>&1

trap 'echo "[ERROR] line ${LINENO}: command failed: ${BASH_COMMAND}"' ERR

# ── Argument ──────────────────────────────────────────────────────────────────
CONFIG_NAME=${1:?"Usage: sbatch scripts/slurm_train.sh <config_name>"}
CONFIG_FILE="${REPO_ROOT}/configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config not found — $CONFIG_FILE"
    exit 1
fi

# ── Environment ───────────────────────────────────────────────────────────────
# Load CUDA module if your cluster requires it (adjust version as needed):
# module load cuda/12.1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-ft

cd "${REPO_ROOT}"

# ── Header ────────────────────────────────────────────────────────────────────
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
echo "Python   : $(which python)"
python --version

# ── W&B ───────────────────────────────────────────────────────────────────────
export WANDB_RUN_NAME="${CONFIG_NAME}_job${SLURM_JOB_ID:-0}"
# Uncomment below if the cluster nodes have no internet access:
# export WANDB_MODE=offline

# ── Step 1: Train ─────────────────────────────────────────────────────────────
echo ""
echo "[1/3] Training..."
python "${REPO_ROOT}/src/train/train.py" --config "$CONFIG_FILE"

# ── Step 2: Inference (generate test-set predictions) ─────────────────────────
echo ""
echo "[2/3] Inference..."

TASK=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['model']['task'])")

if [ "$TASK" = "summarization" ]; then
    python "${REPO_ROOT}/src/evaluate/inference.py" --config "$CONFIG_FILE" --split test_us
    python "${REPO_ROOT}/src/evaluate/inference.py" --config "$CONFIG_FILE" --split test_ca
else
    python "${REPO_ROOT}/src/evaluate/inference.py" --config "$CONFIG_FILE" --split test
fi

# ── Step 3: Evaluate ──────────────────────────────────────────────────────────
echo ""
echo "[3/3] Evaluating..."

OUTPUT_DIR=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG_FILE')); print(c['output']['dir'])")
OUTPUT_DIR_ABS="${REPO_ROOT}/${OUTPUT_DIR}"

if [ "$TASK" = "summarization" ]; then
    python "${REPO_ROOT}/src/evaluate/eval_billsum.py" \
        --predictions "${OUTPUT_DIR_ABS}/predictions_test_us.jsonl" \
        --output      "${OUTPUT_DIR_ABS}/eval_test_us.json"
    python "${REPO_ROOT}/src/evaluate/eval_billsum.py" \
        --predictions "${OUTPUT_DIR_ABS}/predictions_test_ca.jsonl" \
        --output      "${OUTPUT_DIR_ABS}/eval_test_ca.json"
else
    python "${REPO_ROOT}/src/evaluate/eval_casehold.py" \
        --predictions "${OUTPUT_DIR_ABS}/predictions_test.jsonl" \
        --output      "${OUTPUT_DIR_ABS}/eval_test.json"
fi

echo ""
echo "=============================="
echo " Finished : $(date)"
echo " Results  : ${OUTPUT_DIR_ABS}"
echo "=============================="
