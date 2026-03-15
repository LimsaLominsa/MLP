#!/bin/bash
# [MOVED] This script has been superseded by scripts/slurm_train_casehold.sh
# which uses YAML configs and is configured for the MLP Teaching partition.
# This file is kept for reference only.
#
# Usage:
#   sbatch scripts/slurm_train_casehold.sh configs/finetune/qwen2.5_1.5b_high_mem.json

#SBATCH --job-name=casehold-lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Missing config path."
  echo "Example: sbatch scripts/slurm_train_casehold.sh configs/finetune/qwen2.5_1.5b_high_mem.json"
  exit 1
fi

CONFIG_PATH="$1"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"

cd "$PROJECT_DIR"
mkdir -p logs

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python src/train_casehold_lora.py --config "$CONFIG_PATH"
