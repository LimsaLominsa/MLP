#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
ROOT="$PROJECT_ROOT"
CFG="${1:-$ROOT/configs/experiment.yaml}"

echo "[finalize] rep_metrics per seed"
python - <<PY
import yaml, subprocess, sys
cfg=yaml.safe_load(open("$CFG","r",encoding="utf-8"))
seeds=cfg.get("seeds", [cfg.get("seed")])
for s in seeds:
    cmd=[sys.executable, "-m", "g104_pipeline.rep_metrics", "--config", "$CFG", "--seed", str(s)]
    print(" ".join(cmd))
    subprocess.check_call(cmd)
PY

echo "[finalize] aggregate stats"
python -m g104_pipeline.stats --config "$CFG"

echo "[finalize] failure cases"
python -m g104_pipeline.failure_cases --config "$CFG" --max-cases 12

echo "[finalize] validate outputs"
python -m g104_pipeline.validate_outputs --config "$CFG"
