#!/usr/bin/env bash
#SBATCH --job-name=billsum-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%x_%j.out

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="${CFG:?Set CFG=/abs/path/to/billsum_config.yaml}"
SPLIT="${SPLIT:-test_us}"
LIMIT="${LIMIT:-}"

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

source "$ROOT/scripts/project_env.sh"

CMD=(python -m g104_pipeline.billsum_baseline --config "$CFG" --split "$SPLIT")
if [[ -n "$LIMIT" ]]; then
  CMD+=(--limit "$LIMIT")
fi

echo "[billsum-baseline] ${CMD[*]}"
"${CMD[@]}"
