#!/usr/bin/env bash
#SBATCH --job-name=med-faith
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/slurm_med_faith_%A_%a.out
#SBATCH --array=0-5

# 6 jobs: PubMed faithfulness × 2 models × 3 conditions
# Each job: 64 samples × 10 deletion steps = 640 forward passes (~1-2h)

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "/mnt/raid0sata2/jingci/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

mkdir -p "$ROOT/outputs/logs"
source "$ROOT/scripts/project_env.sh"

CONFIGS=(
  "configs/pubmed_qwen_1p5b.yaml"
  "configs/pubmed_qwen_1p5b_lora.yaml"
  "configs/pubmed_qwen_1p5b_random.yaml"
  "configs/pubmed_llama_1b.yaml"
  "configs/pubmed_llama_1b_lora.yaml"
  "configs/pubmed_llama_1b_random.yaml"
)

IDX=${SLURM_ARRAY_TASK_ID}
CFG="${CONFIGS[$IDX]}"

echo "=== Faithfulness job $IDX: $CFG ==="
cd "$ROOT"

python -m src.g104_pipeline.pubmed_faithfulness \
  --config "$CFG" --split test --subset-size 64

echo "=== Faithfulness job $IDX complete ==="
