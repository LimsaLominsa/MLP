#!/bin/bash
# run_all_autodl.sh — Run all experiments sequentially on AutoDL (no SLURM).
#
# Usage (run from repo root inside tmux):
#   tmux new -s train
#   bash scripts/run_all_autodl.sh          # all 8 experiments
#   bash scripts/run_all_autodl.sh billsum  # only BillSum experiments
#   bash scripts/run_all_autodl.sh casehold # only CaseHOLD experiments
#
# To detach from tmux: Ctrl+B then D
# To re-attach:        tmux attach -t train
# To check progress:   tail -f logs/run_all.log

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/run_all.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

trap 'echo "[ERROR] line ${LINENO}: ${BASH_COMMAND}" | tee -a "${RUN_LOG}"; exit 1' ERR

FILTER="${1:-all}"   # all | billsum | casehold

echo "========================================"
echo " AutoDL run_all.sh"
echo " Started  : $(date)"
echo " Filter   : ${FILTER}"
echo " Repo     : ${REPO_ROOT}"
echo " Log      : ${RUN_LOG}"
echo "========================================"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python --version

# ── Helpers ───────────────────────────────────────────────────────────────────

run_experiment() {
    local config_name="$1"
    local train_script="$2"   # "train" or "train_casehold_lora"
    local task="$3"           # "summarization" or "classification"

    echo ""
    echo "========================================"
    echo " EXPERIMENT : ${config_name}"
    echo " Started    : $(date)"
    echo "========================================"

    local config_file="${REPO_ROOT}/configs/${config_name}.yaml"
    local exp_log="${LOG_DIR}/${config_name}.log"

    # ── VRAM smoke check (5 steps) ────────────────────────────────────────────
    echo "[Pre] VRAM smoke check..."
    python "${REPO_ROOT}/src/train/${train_script}.py" \
        --config "${config_file}" --max_steps 5 2>&1 | tee -a "${exp_log}"
    echo "[Pre] PASSED"

    # ── Train ─────────────────────────────────────────────────────────────────
    echo "[1/3] Training ${config_name}..."
    python "${REPO_ROOT}/src/train/${train_script}.py" \
        --config "${config_file}" 2>&1 | tee -a "${exp_log}"

    # ── Inference ─────────────────────────────────────────────────────────────
    echo "[2/3] Inference ${config_name}..."
    if [ "${task}" = "summarization" ]; then
        python "${REPO_ROOT}/src/evaluate/inference.py" \
            --config "${config_file}" --split test_us 2>&1 | tee -a "${exp_log}"
        python "${REPO_ROOT}/src/evaluate/inference.py" \
            --config "${config_file}" --split test_ca 2>&1 | tee -a "${exp_log}"
    else
        python "${REPO_ROOT}/src/evaluate/inference.py" \
            --config "${config_file}" --split test 2>&1 | tee -a "${exp_log}"
    fi

    # ── Evaluate ──────────────────────────────────────────────────────────────
    echo "[3/3] Evaluating ${config_name}..."
    local output_dir
    output_dir=$(python -c "import yaml; c=yaml.safe_load(open('${config_file}')); print(c['output']['dir'])")
    local output_dir_abs="${REPO_ROOT}/${output_dir}"

    if [ "${task}" = "summarization" ]; then
        python "${REPO_ROOT}/src/evaluate/eval_billsum.py" \
            --predictions "${output_dir_abs}/predictions_test_us.jsonl" \
            --output      "${output_dir_abs}/eval_test_us.json" 2>&1 | tee -a "${exp_log}"
        python "${REPO_ROOT}/src/evaluate/eval_billsum.py" \
            --predictions "${output_dir_abs}/predictions_test_ca.jsonl" \
            --output      "${output_dir_abs}/eval_test_ca.json" 2>&1 | tee -a "${exp_log}"
    else
        python "${REPO_ROOT}/src/evaluate/eval_casehold.py" \
            --predictions "${output_dir_abs}/predictions_test.jsonl" \
            --output      "${output_dir_abs}/eval_test.json" 2>&1 | tee -a "${exp_log}"
    fi

    echo " DONE : ${config_name}  ($(date))"
    echo "========================================"
}

# ── W&B ───────────────────────────────────────────────────────────────────────
export WANDB_MODE="${WANDB_MODE:-offline}"
echo "WANDB_MODE=${WANDB_MODE}  (set WANDB_MODE=online before running to stream live)"

# ── Experiment list ───────────────────────────────────────────────────────────
# Format: run_experiment <config_name> <train_script> <task>

if [[ "${FILTER}" == "all" || "${FILTER}" == "billsum" ]]; then
    run_experiment "lora_billsum_qwen"   "train" "summarization"
    run_experiment "lora_billsum_llama"  "train" "summarization"
    run_experiment "full_billsum_qwen"   "train" "summarization"
    run_experiment "full_billsum_llama"  "train" "summarization"
fi

if [[ "${FILTER}" == "all" || "${FILTER}" == "casehold" ]]; then
    run_experiment "lora_casehold_qwen"    "train"               "classification"
    run_experiment "lora_casehold_llama"   "train"               "classification"
    run_experiment "qlora_casehold_qwen"   "train_casehold_lora" "classification"
    run_experiment "qlora_casehold_llama"  "train_casehold_lora" "classification"
fi

echo ""
echo "========================================"
echo " ALL EXPERIMENTS FINISHED"
echo " Finished : $(date)"
echo " Results  : ${REPO_ROOT}/outputs/"
echo " Full log : ${RUN_LOG}"
echo "========================================"
