#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
ROOT="$PROJECT_ROOT"
CFG="$ROOT/configs/experiment.yaml"

python -m g104_pipeline.orchestrate --config "$CFG"
