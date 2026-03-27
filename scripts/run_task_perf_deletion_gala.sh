#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/project_env.sh

PY_BIN="${PY_BIN:-./.venv/bin/python}"
TAG="${TAG:-actual_task_perf_v1}"
CASEHOLD_STEPS="${CASEHOLD_STEPS:-10}"
PUBMED_STEPS="${PUBMED_STEPS:-10}"
BILLSUM_STEPS="${BILLSUM_STEPS:-10}"

echo "Using python: $PY_BIN"
echo "Output tag: $TAG"

mkdir -p outputs/logs

wait_for_gpu() {
  local gpu_id="$1"
  while true; do
    local used_mem
    used_mem=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [[ -n "$used_mem" ]] && [[ "$used_mem" -lt 2000 ]]; then
      break
    fi
    sleep 15
  done
}

run_when_gpu_free() {
  local gpu_id="$1"
  local log_path="$2"
  shift 2
  (
    wait_for_gpu "$gpu_id"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$@" > "$log_path" 2>&1
  ) &
}

echo "=== Wave 1: CaseHOLD actual accuracy under deletion ==="
run_when_gpu_free 2 outputs/logs/casehold_qwen_pretrained_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.casehold_task_perf_deletion \
  --config configs/casehold_qwen_1p5b.yaml \
  --experiment pretrained \
  --seed 42 \
  --deletion-steps "$CASEHOLD_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 3 outputs/logs/casehold_qwen_lora_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.casehold_task_perf_deletion \
  --config configs/casehold_qwen_1p5b.yaml \
  --experiment lora_sft \
  --seed 42 \
  --deletion-steps "$CASEHOLD_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 4 outputs/logs/casehold_qwen_random_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.casehold_task_perf_deletion \
  --config configs/casehold_qwen_1p5b.yaml \
  --experiment random_label \
  --seed 42 \
  --deletion-steps "$CASEHOLD_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 5 outputs/logs/casehold_llama_pretrained_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.casehold_task_perf_deletion \
  --config configs/casehold_llama_1b.yaml \
  --experiment pretrained \
  --seed 42 \
  --deletion-steps "$CASEHOLD_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 6 outputs/logs/casehold_llama_lora_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.casehold_task_perf_deletion \
  --config configs/casehold_llama_1b.yaml \
  --experiment lora_sft \
  --seed 42 \
  --deletion-steps "$CASEHOLD_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 7 outputs/logs/casehold_llama_random_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.casehold_task_perf_deletion \
  --config configs/casehold_llama_1b.yaml \
  --experiment random_label \
  --seed 42 \
  --deletion-steps "$CASEHOLD_STEPS" \
  --output-tag "$TAG"

wait

echo "=== Wave 2: PubMed ROUGE under deletion ==="
PUBMED_SUBSET="outputs/pubmed_qwen_1p5b_pretrained/baseline/test/faithfulness_subset.jsonl"

run_when_gpu_free 2 outputs/logs/pubmed_qwen_pretrained_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/pubmed_qwen_1p5b.yaml \
  --task pubmed \
  --split test \
  --subset-file "$PUBMED_SUBSET" \
  --deletion-steps "$PUBMED_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 3 outputs/logs/pubmed_qwen_lora_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/pubmed_qwen_1p5b_lora.yaml \
  --task pubmed \
  --split test \
  --subset-file "$PUBMED_SUBSET" \
  --deletion-steps "$PUBMED_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 4 outputs/logs/pubmed_qwen_random_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/pubmed_qwen_1p5b_random.yaml \
  --task pubmed \
  --split test \
  --subset-file "$PUBMED_SUBSET" \
  --deletion-steps "$PUBMED_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 5 outputs/logs/pubmed_llama_pretrained_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/pubmed_llama_1b.yaml \
  --task pubmed \
  --split test \
  --subset-file "$PUBMED_SUBSET" \
  --deletion-steps "$PUBMED_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 6 outputs/logs/pubmed_llama_lora_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/pubmed_llama_1b_lora.yaml \
  --task pubmed \
  --split test \
  --subset-file "$PUBMED_SUBSET" \
  --deletion-steps "$PUBMED_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 7 outputs/logs/pubmed_llama_random_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/pubmed_llama_1b_random.yaml \
  --task pubmed \
  --split test \
  --subset-file "$PUBMED_SUBSET" \
  --deletion-steps "$PUBMED_STEPS" \
  --output-tag "$TAG"

wait

echo "=== Wave 3: BillSum ROUGE under deletion ==="
BILLSUM_SUBSET="outputs/billsum_qwen_1p5b/baseline/test_us/qwen_testus_final_3gpu/faithfulness_subset.jsonl"

run_when_gpu_free 2 outputs/logs/billsum_qwen_pretrained_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/billsum_qwen_1p5b.yaml \
  --task billsum \
  --split test_us \
  --subset-file "$BILLSUM_SUBSET" \
  --deletion-steps "$BILLSUM_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 3 outputs/logs/billsum_qwen_lora_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/billsum_qwen_1p5b_lora.yaml \
  --task billsum \
  --split test_us \
  --subset-file "$BILLSUM_SUBSET" \
  --deletion-steps "$BILLSUM_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 4 outputs/logs/billsum_qwen_random_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/billsum_qwen_1p5b_random.yaml \
  --task billsum \
  --split test_us \
  --subset-file "$BILLSUM_SUBSET" \
  --deletion-steps "$BILLSUM_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 5 outputs/logs/billsum_llama_pretrained_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/billsum_llama_1b.yaml \
  --task billsum \
  --split test_us \
  --subset-file "$BILLSUM_SUBSET" \
  --deletion-steps "$BILLSUM_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 6 outputs/logs/billsum_llama_lora_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/billsum_llama_1b_lora.yaml \
  --task billsum \
  --split test_us \
  --subset-file "$BILLSUM_SUBSET" \
  --deletion-steps "$BILLSUM_STEPS" \
  --output-tag "$TAG"

run_when_gpu_free 7 outputs/logs/billsum_llama_random_actual_task_perf.log \
  "$PY_BIN" -m g104_pipeline.summarization_task_perf_deletion \
  --config configs/billsum_llama_1b_random.yaml \
  --task billsum \
  --split test_us \
  --subset-file "$BILLSUM_SUBSET" \
  --deletion-steps "$BILLSUM_STEPS" \
  --output-tag "$TAG"

wait

echo "All actual-task-performance-under-deletion runs complete."
