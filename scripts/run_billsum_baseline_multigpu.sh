#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
ROOT="$PROJECT_ROOT"
CFG="${1:?Usage: run_billsum_baseline_multigpu.sh <config> [split] [output_tag] [gpu...]}";
SPLIT="${2:-test_us}"
OUTPUT_TAG="${3:-multigpu}"
shift $(( $# >= 3 ? 3 : $# ))
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-0}"

GPUS=("$@")
if [[ ${#GPUS[@]} -eq 0 ]]; then
  GPUS=(0 1)
fi

NUM_SHARDS="${#GPUS[@]}"
LOG_DIR="$ROOT/outputs/logs"
mkdir -p "$LOG_DIR"

PIDS=()
for SHARD_ID in "${!GPUS[@]}"; do
  GPU="${GPUS[$SHARD_ID]}"
  LOG_FILE="$LOG_DIR/$(basename "$CFG" .yaml)_${SPLIT}_${OUTPUT_TAG}_shard${SHARD_ID}.log"
  echo "[billsum-multigpu] shard=$SHARD_ID/$NUM_SHARDS gpu=$GPU log=$LOG_FILE"
  CMD=(
    python -m g104_pipeline.billsum_baseline
    --config "$CFG"
    --split "$SPLIT"
    --output-tag "$OUTPUT_TAG"
    --shard-id "$SHARD_ID"
    --num-shards "$NUM_SHARDS"
  )
  if [[ -n "$LIMIT" ]]; then
    CMD+=(--limit "$LIMIT")
  fi
  if [[ "$RESUME" == "1" ]]; then
    CMD+=(--resume)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  PIDS+=("$!")
done

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAIL=1
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "[billsum-multigpu] one or more shards failed"
  exit 1
fi

echo "[billsum-multigpu] merging shards"
python -m g104_pipeline.billsum_baseline \
  --config "$CFG" \
  --split "$SPLIT" \
  --output-tag "$OUTPUT_TAG" \
  --num-shards "$NUM_SHARDS" \
  --merge-shards

echo "[billsum-multigpu] completed"
