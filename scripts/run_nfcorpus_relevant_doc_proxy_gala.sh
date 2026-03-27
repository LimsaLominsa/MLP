#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/project_env.sh

PY_BIN="${PY_BIN:-./.venv/bin/python}"
TAG="${TAG:-actual_relevant_doc_proxy_v1}"
STEPS="${STEPS:-10}"
SUBSET_SIZE="${SUBSET_SIZE:-323}"

mkdir -p outputs/logs

run_job() {
  local gpu_id="$1"
  local log_path="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu_id" "$@" > "$log_path" 2>&1 &
}

run_job 2 outputs/logs/nfcorpus_qwen_pre_relevant_proxy.log \
  "$PY_BIN" -m g104_pipeline.nfcorpus_relevant_doc_deletion \
  --config configs/nfcorpus_qwen_1p5b.yaml \
  --split test \
  --subset-size "$SUBSET_SIZE" \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 3 outputs/logs/nfcorpus_qwen_lora_relevant_proxy.log \
  "$PY_BIN" -m g104_pipeline.nfcorpus_relevant_doc_deletion \
  --config configs/nfcorpus_qwen_1p5b_lora.yaml \
  --split test \
  --subset-size "$SUBSET_SIZE" \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 4 outputs/logs/nfcorpus_qwen_random_relevant_proxy.log \
  "$PY_BIN" -m g104_pipeline.nfcorpus_relevant_doc_deletion \
  --config configs/nfcorpus_qwen_1p5b_random.yaml \
  --split test \
  --subset-size "$SUBSET_SIZE" \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 5 outputs/logs/nfcorpus_llama_pre_relevant_proxy.log \
  "$PY_BIN" -m g104_pipeline.nfcorpus_relevant_doc_deletion \
  --config configs/nfcorpus_llama_1b.yaml \
  --split test \
  --subset-size "$SUBSET_SIZE" \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 6 outputs/logs/nfcorpus_llama_lora_relevant_proxy.log \
  "$PY_BIN" -m g104_pipeline.nfcorpus_relevant_doc_deletion \
  --config configs/nfcorpus_llama_1b_lora.yaml \
  --split test \
  --subset-size "$SUBSET_SIZE" \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

run_job 7 outputs/logs/nfcorpus_llama_random_relevant_proxy.log \
  "$PY_BIN" -m g104_pipeline.nfcorpus_relevant_doc_deletion \
  --config configs/nfcorpus_llama_1b_random.yaml \
  --split test \
  --subset-size "$SUBSET_SIZE" \
  --deletion-steps "$STEPS" \
  --output-tag "$TAG"

wait

echo "All NFCorpus relevant-doc proxy runs complete."
