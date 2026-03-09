#!/bin/bash
# run_experiment.sh
# Launch a single training experiment by config name.
#
# Usage:
#   bash scripts/run_experiment.sh lora_billsum_qwen
#   bash scripts/run_experiment.sh lora_casehold_llama

set -e

CONFIG_NAME=${1:?"Usage: bash scripts/run_experiment.sh <config_name>"}
CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config not found — $CONFIG_FILE"
    exit 1
fi

echo "=============================="
echo " Experiment: $CONFIG_NAME"
echo " Config    : $CONFIG_FILE"
echo "=============================="

# Log to wandb with the config name as the run name
export WANDB_RUN_NAME="$CONFIG_NAME"

# Launch training
python src/train/train.py --config "$CONFIG_FILE"

echo ""
echo "Training complete — outputs/$(grep 'dir:' $CONFIG_FILE | awk '{print $2}')"
