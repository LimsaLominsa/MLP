#!/bin/bash
# setup_env.sh
# GCP server environment setup script.
# Run once after cloning the repo on the remote machine.
#
# Usage:
#   bash scripts/setup_env.sh

set -e

echo "=============================="
echo " legal-llm-finetuning: setup"
echo "=============================="

# 1. Create and activate conda environment
conda create -n llm-ft python=3.11 -y
conda activate llm-ft

# 2. Install PyTorch (CUDA 12.1 — adjust cuda version for your GCP instance)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install training dependencies
pip install \
    transformers \
    peft \
    accelerate \
    datasets \
    bitsandbytes \
    scipy \
    sentencepiece \
    pyyaml \
    wandb

# 4. Install evaluation dependencies
pip install rouge-score bert-score

# 5. Confirm GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ""
echo "Setup complete. Activate with: conda activate llm-ft"
