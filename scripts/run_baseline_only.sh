#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
ROOT="$PROJECT_ROOT"
CFG="${1:-$ROOT/configs/experiment.yaml}"
shift || true

echo "[baseline] prepare data"
python -m g104_pipeline.data_prep --config "$CFG"

if [[ "$#" -gt 0 ]]; then
  SEEDS=("$@")
else
  mapfile -t SEEDS < <(python - <<PY
import yaml
cfg = yaml.safe_load(open("$CFG", "r", encoding="utf-8"))
for seed in cfg.get("seeds", [cfg.get("seed")]):
    print(seed)
PY
)
fi

for SEED in "${SEEDS[@]}"; do
  echo "[baseline] run pretrained seed=$SEED"
  bash "$ROOT/scripts/run_one_experiment.sh" "$CFG" pretrained "$SEED"
done

echo "[baseline] aggregate currently available stats"
python -m g104_pipeline.stats --config "$CFG"

echo "[baseline] done (full validation and cross-condition reports should run after LoRA-SFT and random-label are available)"
