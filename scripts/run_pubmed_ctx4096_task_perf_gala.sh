#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/project_env.sh

PY_BIN="${PY_BIN:-./.venv/bin/python}"
TAG="${TAG:-sourceaware4096_actual_v1}"
SUBSET_FILE="${SUBSET_FILE:-outputs/pubmed_qwen_1p5b_pretrained/baseline/test/sourceaware4096_full_v1/faithfulness_subset.jsonl}"
STEPS="${STEPS:-10}"

mkdir -p outputs/logs

run_job() {
  local gpu_id="$1"
  local log_path="$2"
  shift 2
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    CUDA_VISIBLE_DEVICES="$gpu_id" "$@" > "$log_path" 2>&1 &
}

run_job 2 outputs/logs/pubmed4096_actual_qwen_pre.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/pubmed_qwen_1p5b_ctx4096_preview.yaml \
  --task pubmed \
  --split test \
  --subset-file "$SUBSET_FILE" \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 3 outputs/logs/pubmed4096_actual_qwen_lora.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/pubmed_qwen_1p5b_lora_ctx4096_preview.yaml \
  --task pubmed \
  --split test \
  --subset-file "$SUBSET_FILE" \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 4 outputs/logs/pubmed4096_actual_qwen_random.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/pubmed_qwen_1p5b_random_ctx4096_preview.yaml \
  --task pubmed \
  --split test \
  --subset-file "$SUBSET_FILE" \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 5 outputs/logs/pubmed4096_actual_llama_pre.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/pubmed_llama_1b_ctx4096_preview.yaml \
  --task pubmed \
  --split test \
  --subset-file "$SUBSET_FILE" \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 6 outputs/logs/pubmed4096_actual_llama_lora.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/pubmed_llama_1b_lora_ctx4096_preview.yaml \
  --task pubmed \
  --split test \
  --subset-file "$SUBSET_FILE" \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 7 outputs/logs/pubmed4096_actual_llama_random.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/pubmed_llama_1b_random_ctx4096_preview.yaml \
  --task pubmed \
  --split test \
  --subset-file "$SUBSET_FILE" \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

wait

echo "All PubMed 4096 actual-task-performance runs complete."
