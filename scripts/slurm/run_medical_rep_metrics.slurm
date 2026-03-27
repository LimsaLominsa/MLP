#!/usr/bin/env bash
#SBATCH --job-name=med-repr
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/logs/slurm_med_repr_%j.out

# CPU-only: compute JS/entropy/CKA from pre-existing signals.parquet files.
# Run AFTER run_medical_signals.slurm completes.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "/mnt/raid0sata2/jingci/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

source "$ROOT/scripts/project_env.sh"
cd "$ROOT"

echo "=== PubMed rep metrics ==="
for model in qwen_1p5b llama_1b; do
  python -m src.g104_pipeline.medical_rep_metrics \
    --pretrained-config "configs/pubmed_${model}.yaml" \
    --target-config "configs/pubmed_${model}_lora.yaml" \
    --label "pubmed_${model}_lora"

  python -m src.g104_pipeline.medical_rep_metrics \
    --pretrained-config "configs/pubmed_${model}.yaml" \
    --target-config "configs/pubmed_${model}_random.yaml" \
    --label "pubmed_${model}_random"
done

echo "=== NFCorpus rep metrics ==="
for model in qwen_1p5b llama_1b; do
  python -m src.g104_pipeline.medical_rep_metrics \
    --pretrained-config "configs/nfcorpus_${model}.yaml" \
    --target-config "configs/nfcorpus_${model}_lora.yaml" \
    --label "nfcorpus_${model}_lora"

  python -m src.g104_pipeline.medical_rep_metrics \
    --pretrained-config "configs/nfcorpus_${model}.yaml" \
    --target-config "configs/nfcorpus_${model}_random.yaml" \
    --label "nfcorpus_${model}_random"
done

echo "=== All rep metrics complete ==="
