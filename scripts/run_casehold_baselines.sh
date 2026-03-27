#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
ROOT="$PROJECT_ROOT"

CONFIGS=(
  "$ROOT/configs/casehold_qwen_1p5b.yaml"
  "$ROOT/configs/casehold_llama_1b.yaml"
)

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(42 43 44)
fi

for CFG in "${CONFIGS[@]}"; do
  echo "[baseline] config=$CFG"
  python -m g104_pipeline.data_prep --config "$CFG"

  for SEED in "${SEEDS[@]}"; do
    echo "[baseline] pretrained seed=$SEED"
    bash "$ROOT/scripts/run_one_experiment.sh" "$CFG" pretrained "$SEED"
  done

  echo "[baseline] aggregate stats for $CFG"
  python -m g104_pipeline.stats --config "$CFG"
done

echo "[baseline] completed all configured base models"
