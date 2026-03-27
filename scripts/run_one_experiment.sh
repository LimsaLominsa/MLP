#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
ROOT="$PROJECT_ROOT"
CFG="${1:-$ROOT/configs/experiment.yaml}"
EXP="${2:?experiment required: pretrained|lora_sft|random_label}"
SEED="${3:?seed required}"
FORCE="${FORCE:-0}"

RUN_DIR=$(python - <<PY
import yaml
from pathlib import Path
cfg=yaml.safe_load(open("$CFG","r",encoding="utf-8"))
out=Path(cfg["project"]["output_root"])/"artifacts"/"$EXP"/"$SEED"
print(out)
PY
)

mkdir -p "$RUN_DIR"

if [[ "$FORCE" != "1" ]] && [[ -f "$RUN_DIR/metrics.json" ]] && [[ -f "$RUN_DIR/signals.parquet" ]] && [[ -f "$RUN_DIR/faithfulness.json" ]]; then
  echo "[resume] skip completed run: exp=$EXP seed=$SEED"
  exit 0
fi

echo "[run] train exp=$EXP seed=$SEED"
python -m g104_pipeline.train --config "$CFG" --experiment "$EXP" --seed "$SEED"

echo "[run] evaluate exp=$EXP seed=$SEED"
python -m g104_pipeline.evaluate --config "$CFG" --experiment "$EXP" --seed "$SEED"

echo "[run] signals exp=$EXP seed=$SEED"
python -m g104_pipeline.signals --config "$CFG" --experiment "$EXP" --seed "$SEED"

echo "[run] faithfulness exp=$EXP seed=$SEED"
python -m g104_pipeline.faithfulness --config "$CFG" --experiment "$EXP" --seed "$SEED"
