#!/usr/bin/env bash

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# On gala1, the CUDA toolkit stub libcuda can be resolved before the real
# NVIDIA driver. Prepending the system driver path keeps PyTorch on GPU.
if [[ -d "/lib/x86_64-linux-gnu" ]]; then
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu"
  fi
fi
