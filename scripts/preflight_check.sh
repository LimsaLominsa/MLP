#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/project_env.sh"
ROOT="$PROJECT_ROOT"
CFG="${1:-$ROOT/configs/experiment.yaml}"

python - <<PY
import yaml
from pathlib import Path
cfg=yaml.safe_load(open("$CFG","r",encoding="utf-8"))
print("model_name:",cfg.get("model_name"))
print("backend:",cfg.get("backend"))
print("seeds:",cfg.get("seeds"))
print("experiments:",cfg.get("experiments"))
for k in [
    "preprocessed_train_file",
    "preprocessed_valid_file",
    "preprocessed_test_file",
    "train_file",
    "valid_file",
    "test_file",
    "fixed_eval_subset_file",
]:
    print(k, cfg["data"].get(k))
PY

python - <<'PY'
mods=["torch","transformers","peft","datasets","pyarrow","numpy","pandas","scipy"]
for m in mods:
    try:
        __import__(m)
        print(f"[ok] {m}")
    except Exception as e:
        print(f"[missing] {m}: {e}")

try:
    import torch
    print("[cuda_available]", torch.cuda.is_available())
    print("[cuda_device_count]", torch.cuda.device_count())
except Exception as e:
    print(f"[cuda_check_failed] {e}")
PY
