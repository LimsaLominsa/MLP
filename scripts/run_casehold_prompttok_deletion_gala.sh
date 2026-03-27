#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/project_env.sh

PY_BIN="${PY_BIN:-./.venv/bin/python}"
TAG="${TAG:-actual_task_perf_prompttok_v2}"
STEPS="${STEPS:-10}"

mkdir -p outputs/logs

run_job() {
  local gpu_id="$1"
  local log_path="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu_id" "$@" > "$log_path" 2>&1 &
}

run_job 2 outputs/logs/casehold_prompttok_qwen_pre.log \
  "$PY_BIN" -m g104_pipeline.casehold_task_perf_deletion \
  --config configs/casehold_qwen_1p5b.yaml \
  --experiment pretrained \
  --seed 42 \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 3 outputs/logs/casehold_prompttok_qwen_lora.log \
  "$PY_BIN" -m g104_pipeline.casehold_task_perf_deletion \
  --config configs/casehold_qwen_1p5b.yaml \
  --experiment lora_sft \
  --seed 42 \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 4 outputs/logs/casehold_prompttok_qwen_random.log \
  "$PY_BIN" -m g104_pipeline.casehold_task_perf_deletion \
  --config configs/casehold_qwen_1p5b.yaml \
  --experiment random_label \
  --seed 42 \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 5 outputs/logs/casehold_prompttok_llama_pre.log \
  "$PY_BIN" -m g104_pipeline.casehold_task_perf_deletion \
  --config configs/casehold_llama_1b.yaml \
  --experiment pretrained \
  --seed 42 \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 6 outputs/logs/casehold_prompttok_llama_lora.log \
  "$PY_BIN" -m g104_pipeline.casehold_task_perf_deletion \
  --config configs/casehold_llama_1b.yaml \
  --experiment lora_sft \
  --seed 42 \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 7 outputs/logs/casehold_prompttok_llama_random.log \
  "$PY_BIN" -m g104_pipeline.casehold_task_perf_deletion \
  --config configs/casehold_llama_1b.yaml \
  --experiment random_label \
  --seed 42 \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

wait

echo "All CaseHOLD prompt-token deletion runs complete."
