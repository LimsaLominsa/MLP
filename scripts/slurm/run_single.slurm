#!/usr/bin/env bash
#SBATCH --job-name=g104-one
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/slurm_%x_%j.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CFG="${CFG:-$ROOT/configs/experiment.yaml}"
EXP="${EXP:?Set EXP=pretrained|lora_sft|random_label}"
SEED="${SEED:?Set SEED=<int>}"

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

mkdir -p "$ROOT/outputs/logs"
source "$ROOT/scripts/project_env.sh"
bash "$ROOT/scripts/preflight_check.sh" "$CFG"
bash "$ROOT/scripts/run_one_experiment.sh" "$CFG" "$EXP" "$SEED"
