#!/bin/bash
# submit_casehold.sh — Batch-submit all CaseHOLD experiments to SLURM.
#
# Submits:
#   qlora_casehold_qwen    (QLoRA  × Qwen2.5-1.5B  — train_casehold_lora.py)
#   qlora_casehold_llama   (QLoRA  × Llama-3.2-1B   — train_casehold_lora.py)
#   lora_casehold_qwen     (LoRA   × Qwen2.5-1.5B   — train.py)
#   lora_casehold_llama    (LoRA   × Llama-3.2-1B    — train.py)
#
# Usage:
#   bash scripts/submit_casehold.sh           # submit all 4
#   bash scripts/submit_casehold.sh qlora     # submit QLoRA only
#   bash scripts/submit_casehold.sh lora      # submit standard LoRA only

set -e

FILTER=${1:-"all"}

mkdir -p logs

echo "=============================="
echo " Submitting CaseHOLD experiments"
echo " Filter: $FILTER"
echo "=============================="

# QLoRA jobs use slurm_train_casehold.sh (SFTTrainer)
# Standard LoRA jobs use slurm_train.sh (Trainer)
if [[ "$FILTER" == "all" ]] || [[ "$FILTER" == "qlora" ]]; then
    JOB1=$(sbatch --parsable scripts/slurm_train_casehold.sh qlora_casehold_qwen)
    echo "  Submitted  qlora_casehold_qwen   →  Job $JOB1"
    JOB2=$(sbatch --parsable scripts/slurm_train_casehold.sh qlora_casehold_llama)
    echo "  Submitted  qlora_casehold_llama  →  Job $JOB2"
fi

if [[ "$FILTER" == "all" ]] || [[ "$FILTER" == "lora" ]]; then
    JOB3=$(sbatch --parsable scripts/slurm_train.sh lora_casehold_qwen)
    echo "  Submitted  lora_casehold_qwen    →  Job $JOB3"
    JOB4=$(sbatch --parsable scripts/slurm_train.sh lora_casehold_llama)
    echo "  Submitted  lora_casehold_llama   →  Job $JOB4"
fi

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/slurm-*.out"
