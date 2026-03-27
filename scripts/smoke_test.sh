#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$ROOT/configs/smoke.yaml"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

python -m g104_pipeline.orchestrate --config "$CFG"
python -m g104_pipeline.validate_outputs --config "$CFG"
