#!/bin/bash
# submit_billsum.sh — Batch-submit all 4 BillSum experiments to SLURM.
#
# Submits:
#   lora_billsum_qwen    (LoRA  × Qwen2.5-1.5B)
#   lora_billsum_llama   (LoRA  × Llama-3.2-1B)
#   full_billsum_qwen    (Full FT × Qwen2.5-1.5B)
#   full_billsum_llama   (Full FT × Llama-3.2-1B)
#
# Usage:
#   bash scripts/submit_billsum.sh           # submit all 4
#   bash scripts/submit_billsum.sh lora      # submit LoRA only
#   bash scripts/submit_billsum.sh full      # submit Full FT only

set -e

FILTER=${1:-"all"}   # "all" | "lora" | "full"

mkdir -p logs

declare -A JOBS   # config_name → job_id

submit() {
    local CFG=$1
    if [[ "$FILTER" == "all" ]] || [[ "$CFG" == ${FILTER}* ]]; then
        JOB_ID=$(sbatch --parsable scripts/slurm_train.sh "$CFG")
        JOBS[$CFG]=$JOB_ID
        echo "  Submitted  $CFG  →  Job $JOB_ID"
    fi
}

echo "=============================="
echo " Submitting BillSum experiments"
echo " Filter: $FILTER"
echo "=============================="

submit lora_billsum_qwen
submit lora_billsum_llama
submit full_billsum_qwen
submit full_billsum_llama

echo ""
echo "All submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm-*.out"
