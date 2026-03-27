#!/bin/bash
# Run medical interpretability on Gala WITHOUT SLURM.
# Uses CUDA_VISIBLE_DEVICES to pin each job to a separate GPU.
# 8 GPUs available (0-7).

set -e
cd "$(dirname "$0")/.."
source scripts/project_env.sh

echo "=========================================="
echo "Phase 1: Signals Extraction (12 jobs on 8 GPUs)"
echo "=========================================="

# PubMed signals (6 jobs → GPUs 0-5)
CUDA_VISIBLE_DEVICES=0 python -m src.g104_pipeline.pubmed_signals --config configs/pubmed_qwen_1p5b.yaml --split test --subset-size 64 &
CUDA_VISIBLE_DEVICES=1 python -m src.g104_pipeline.pubmed_signals --config configs/pubmed_qwen_1p5b_lora.yaml --split test --subset-size 64 &
CUDA_VISIBLE_DEVICES=2 python -m src.g104_pipeline.pubmed_signals --config configs/pubmed_qwen_1p5b_random.yaml --split test --subset-size 64 &
CUDA_VISIBLE_DEVICES=3 python -m src.g104_pipeline.pubmed_signals --config configs/pubmed_llama_1b.yaml --split test --subset-size 64 &
CUDA_VISIBLE_DEVICES=4 python -m src.g104_pipeline.pubmed_signals --config configs/pubmed_llama_1b_lora.yaml --split test --subset-size 64 &
CUDA_VISIBLE_DEVICES=5 python -m src.g104_pipeline.pubmed_signals --config configs/pubmed_llama_1b_random.yaml --split test --subset-size 64 &

# NFCorpus signals (6 jobs → GPUs 6-7, then reuse 0-3 after they finish)
CUDA_VISIBLE_DEVICES=6 python -m src.g104_pipeline.nfcorpus_signals --config configs/nfcorpus_qwen_1p5b.yaml --split test --subset-size 323 &
CUDA_VISIBLE_DEVICES=7 python -m src.g104_pipeline.nfcorpus_signals --config configs/nfcorpus_qwen_1p5b_lora.yaml --split test --subset-size 323 &

echo "Waiting for first 8 GPU jobs..."
wait

# Second wave: remaining 4 NFCorpus jobs
CUDA_VISIBLE_DEVICES=0 python -m src.g104_pipeline.nfcorpus_signals --config configs/nfcorpus_qwen_1p5b_random.yaml --split test --subset-size 323 &
CUDA_VISIBLE_DEVICES=1 python -m src.g104_pipeline.nfcorpus_signals --config configs/nfcorpus_llama_1b.yaml --split test --subset-size 323 &
CUDA_VISIBLE_DEVICES=2 python -m src.g104_pipeline.nfcorpus_signals --config configs/nfcorpus_llama_1b_lora.yaml --split test --subset-size 323 &
CUDA_VISIBLE_DEVICES=3 python -m src.g104_pipeline.nfcorpus_signals --config configs/nfcorpus_llama_1b_random.yaml --split test --subset-size 323 &

echo "Waiting for remaining 4 NFCorpus signal jobs..."
wait

echo "=========================================="
echo "Phase 2: Rep Metrics (CPU, fast)"
echo "=========================================="

for model in qwen_1p5b llama_1b; do
  python -m src.g104_pipeline.medical_rep_metrics \
    --pretrained-config "configs/pubmed_${model}.yaml" \
    --target-config "configs/pubmed_${model}_lora.yaml" \
    --label "pubmed_${model}_lora"

  python -m src.g104_pipeline.medical_rep_metrics \
    --pretrained-config "configs/pubmed_${model}.yaml" \
    --target-config "configs/pubmed_${model}_random.yaml" \
    --label "pubmed_${model}_random"

  python -m src.g104_pipeline.medical_rep_metrics \
    --pretrained-config "configs/nfcorpus_${model}.yaml" \
    --target-config "configs/nfcorpus_${model}_lora.yaml" \
    --label "nfcorpus_${model}_lora"

  python -m src.g104_pipeline.medical_rep_metrics \
    --pretrained-config "configs/nfcorpus_${model}.yaml" \
    --target-config "configs/nfcorpus_${model}_random.yaml" \
    --label "nfcorpus_${model}_random"
done

echo "=========================================="
echo "Phase 3: Faithfulness (PubMed, 6 jobs on 6 GPUs)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python -m src.g104_pipeline.pubmed_faithfulness --config configs/pubmed_qwen_1p5b.yaml --split test --subset-size 64 &
CUDA_VISIBLE_DEVICES=1 python -m src.g104_pipeline.pubmed_faithfulness --config configs/pubmed_qwen_1p5b_lora.yaml --split test --subset-size 64 &
CUDA_VISIBLE_DEVICES=2 python -m src.g104_pipeline.pubmed_faithfulness --config configs/pubmed_qwen_1p5b_random.yaml --split test --subset-size 64 &
CUDA_VISIBLE_DEVICES=3 python -m src.g104_pipeline.pubmed_faithfulness --config configs/pubmed_llama_1b.yaml --split test --subset-size 64 &
CUDA_VISIBLE_DEVICES=4 python -m src.g104_pipeline.pubmed_faithfulness --config configs/pubmed_llama_1b_lora.yaml --split test --subset-size 64 &
CUDA_VISIBLE_DEVICES=5 python -m src.g104_pipeline.pubmed_faithfulness --config configs/pubmed_llama_1b_random.yaml --split test --subset-size 64 &

echo "Waiting for faithfulness jobs..."
wait

echo "=========================================="
echo "All done!"
echo "=========================================="
