#!/usr/bin/env bash
#SBATCH --job-name=med-sig
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/slurm_med_sig_%A_%a.out
#SBATCH --array=0-11

# 12 jobs: 2 tasks × 2 models × 3 conditions
# Each job extracts signals for one config (~20-40 min on A5000)

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "/mnt/raid0sata2/jingci/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

mkdir -p "$ROOT/outputs/logs"
source "$ROOT/scripts/project_env.sh"

# Map array index to (task, model, condition)
CONFIGS=(
  # PubMed (indices 0-5)
  "pubmed:configs/pubmed_qwen_1p5b.yaml:64"
  "pubmed:configs/pubmed_qwen_1p5b_lora.yaml:64"
  "pubmed:configs/pubmed_qwen_1p5b_random.yaml:64"
  "pubmed:configs/pubmed_llama_1b.yaml:64"
  "pubmed:configs/pubmed_llama_1b_lora.yaml:64"
  "pubmed:configs/pubmed_llama_1b_random.yaml:64"
  # NFCorpus (indices 6-11)
  "nfcorpus:configs/nfcorpus_qwen_1p5b.yaml:323"
  "nfcorpus:configs/nfcorpus_qwen_1p5b_lora.yaml:323"
  "nfcorpus:configs/nfcorpus_qwen_1p5b_random.yaml:323"
  "nfcorpus:configs/nfcorpus_llama_1b.yaml:323"
  "nfcorpus:configs/nfcorpus_llama_1b_lora.yaml:323"
  "nfcorpus:configs/nfcorpus_llama_1b_random.yaml:323"
)

IDX=${SLURM_ARRAY_TASK_ID}
IFS=':' read -r TASK CFG SUBSET_SIZE <<< "${CONFIGS[$IDX]}"

echo "=== Job $IDX: task=$TASK config=$CFG subset=$SUBSET_SIZE ==="
cd "$ROOT"

if [[ "$TASK" == "pubmed" ]]; then
  python -m src.g104_pipeline.pubmed_signals \
    --config "$CFG" --split test --subset-size "$SUBSET_SIZE"
elif [[ "$TASK" == "nfcorpus" ]]; then
  python -m src.g104_pipeline.nfcorpus_signals \
    --config "$CFG" --split test --subset-size "$SUBSET_SIZE"
fi

echo "=== Job $IDX complete ==="
