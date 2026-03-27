#!/usr/bin/env bash
#SBATCH --job-name=g104-matrix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=outputs/logs/slurm_%x_%A_%a.out

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CFG="${CFG:-$ROOT/configs/experiment.yaml}"

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

mkdir -p "$ROOT/outputs/logs"
source "$ROOT/scripts/project_env.sh"
mapfile -t COMBOS < <(python - <<PY
import yaml
cfg=yaml.safe_load(open("$CFG","r",encoding="utf-8"))
exps=cfg.get("experiments",["pretrained","lora_sft","random_label"])
seeds=cfg.get("seeds",[cfg.get("seed",42)])
for e in exps:
    for s in seeds:
        print(e, s)
PY
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "$IDX" -ge "${#COMBOS[@]}" ]]; then
  echo "Array index $IDX exceeds combinations ${#COMBOS[@]} (skip)"
  exit 0
fi

EXP=$(echo "${COMBOS[$IDX]}" | awk '{print $1}')
SEED=$(echo "${COMBOS[$IDX]}" | awk '{print $2}')

echo "[matrix] idx=$IDX exp=$EXP seed=$SEED"

bash "$ROOT/scripts/run_one_experiment.sh" "$CFG" "$EXP" "$SEED"
