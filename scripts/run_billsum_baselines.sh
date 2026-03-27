#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLIT="${1:-test_us}"
LIMIT="${2:-}"
RESUME="${RESUME:-0}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

CONFIGS=(
  "$ROOT/configs/billsum_qwen_1p5b.yaml"
  "$ROOT/configs/billsum_llama_1b.yaml"
)

for CFG in "${CONFIGS[@]}"; do
  echo "[billsum-baseline] config=$CFG split=$SPLIT"
  CMD=(python -m g104_pipeline.billsum_baseline --config "$CFG" --split "$SPLIT")
  if [[ -n "$LIMIT" ]]; then
    CMD+=(--limit "$LIMIT")
  fi
  if [[ "$RESUME" == "1" ]]; then
    CMD+=(--resume)
  fi
  "${CMD[@]}"
done

echo "[billsum-baseline] completed"
