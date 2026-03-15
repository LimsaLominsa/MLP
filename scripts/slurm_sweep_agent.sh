#!/bin/bash
# slurm_sweep_agent.sh — Run a W&B sweep agent as a SLURM job.
#
# Each agent picks up and runs experiments from the sweep queue automatically.
# Submit multiple agents in parallel to run experiments concurrently.
#
# Usage:
#   # 1. Register the sweep first (once):
#   wandb sweep configs/sweep_billsum_qwen.yaml
#   # → prints sweep ID, e.g. "myentity/myproject/abc12345"
#
#   # 2. Submit 3 parallel agents, each running up to 3 experiments:
#   sbatch --array=1-3 scripts/slurm_sweep_agent.sh myentity/myproject/abc12345 3
#
#   # Or submit a single agent:
#   sbatch scripts/slurm_sweep_agent.sh myentity/myproject/abc12345 5

#SBATCH --job-name=sweep-agent
#SBATCH --output=logs/sweep-%j.out
#SBATCH --error=logs/sweep-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=14G
#SBATCH --cpus-per-task=1

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "${LOG_DIR}"

DEBUG_LOG="${LOG_DIR}/debug-sweep-${SLURM_JOB_ID:-local}.log"
exec > >(tee -a "${DEBUG_LOG}") 2>&1

trap 'echo "[ERROR] line ${LINENO}: command failed: ${BASH_COMMAND}"' ERR

SWEEP_ID=${1:?"Usage: sbatch scripts/slurm_sweep_agent.sh <entity/project/sweep_id> [num_runs]"}
NUM_RUNS=${2:-3}    # max experiments this agent will run before exiting

# ── Environment ───────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-ft

cd "${REPO_ROOT}"

echo "=============================="
echo " Sweep agent starting"
echo " Sweep ID : $SWEEP_ID"
echo " Max runs : $NUM_RUNS"
echo " Job ID   : ${SLURM_JOB_ID:-local}"
echo " Node     : ${SLURMD_NODENAME:-$(hostname)}"
echo " Repo     : ${REPO_ROOT}"
echo " Workdir  : $(pwd)"
echo " Submit   : ${SLURM_SUBMIT_DIR:-N/A}"
echo " DebugLog : ${DEBUG_LOG}"
echo " Started  : $(date)"
echo "=============================="

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Python   : $(which python)"
python --version

# Each call to `wandb agent --count N` picks N experiments from the sweep queue
wandb agent --count "$NUM_RUNS" "$SWEEP_ID"

echo ""
echo "Agent finished at $(date)"
