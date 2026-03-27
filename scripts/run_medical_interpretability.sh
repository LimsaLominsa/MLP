#!/bin/bash
# Run all medical interpretability analyses on Gala.
# Prerequisites:
#   1. Data prepared: python MLP_组员微调_最新/src/data/pubmed/pubmed_formatting.py --output_dir data/pubmed
#                     python MLP_组员微调_最新/src/data/nfcorpus/nfcorpus_formatting.py --output_dir data/nfcorpus
#   2. LoRA adapters present in MLP_组员微调_最新/models/{pubmed,nfcorpus}/
#   3. Dependencies installed: torch, transformers, peft, datasets, pandas, pyarrow, numpy, pyyaml, tqdm

set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo "Phase 1: Signals Extraction"
echo "=========================================="

# --- PubMed signals (6 configs: pretrained/lora/random × qwen/llama) ---
for model in qwen_1p5b llama_1b; do
  for cond in "" "_lora" "_random"; do
    cfg="configs/pubmed_${model}${cond}.yaml"
    echo "[signals] $cfg"
    python -m src.g104_pipeline.pubmed_signals --config "$cfg" --split test --subset-size 64
  done
done

# --- NFCorpus signals (6 configs) ---
for model in qwen_1p5b llama_1b; do
  for cond in "" "_lora" "_random"; do
    cfg="configs/nfcorpus_${model}${cond}.yaml"
    echo "[signals] $cfg"
    python -m src.g104_pipeline.nfcorpus_signals --config "$cfg" --split test --subset-size 323
  done
done

echo "=========================================="
echo "Phase 2: Representation Metrics"
echo "=========================================="

# --- PubMed rep metrics ---
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

# --- NFCorpus rep metrics ---
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

echo "=========================================="
echo "Phase 3: Faithfulness (PubMed only)"
echo "=========================================="

for model in qwen_1p5b llama_1b; do
  for cond in "" "_lora" "_random"; do
    cfg="configs/pubmed_${model}${cond}.yaml"
    echo "[faithfulness] $cfg"
    python -m src.g104_pipeline.pubmed_faithfulness --config "$cfg" --split test --subset-size 64
  done
done

echo "=========================================="
echo "All medical interpretability runs complete."
echo "=========================================="
