#!/usr/bin/env bash
#SBATCH --job-name=g104-finalize
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/slurm_%x_%j.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CFG="${CFG:-$ROOT/configs/experiment.yaml}"

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

mkdir -p "$ROOT/outputs/logs"
source "$ROOT/scripts/project_env.sh"
bash "$ROOT/scripts/finalize_pipeline.sh" "$CFG"
