from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import load_config
from .hf_backend import extract_hf_signals
from .io_utils import read_jsonl, write_parquet
from .modeling import PairwiseChoiceModel


def _stable_noise(key: str, seed: int) -> float:
    h = hashlib.md5(f"{key}-{seed}".encode("utf-8")).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return float(v * 2.0 - 1.0)


def extract_signals_one(config_path: str, experiment: str, seed: int) -> str:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    output_root = Path(cfg["project"]["output_root"])
    backend = cfg.get("backend", "mock")

    run_dir = output_root / "artifacts" / experiment / str(seed)
    eval_rows = read_jsonl(data_cfg["fixed_eval_subset_file"])

    if backend == "hf_lora":
        out_rows = extract_hf_signals(
            cfg=cfg,
            eval_rows=eval_rows,
            run_dir=run_dir,
            experiment=experiment,
            seed=seed,
        )
    else:
        model = PairwiseChoiceModel.load(str(run_dir / "model.pkl"))

        cka_layers = cfg.get("analysis", {}).get("cka_layers", [0, 4, 8, 12])
        max_layer = int(max(cka_layers))
        n_heads = 8 if max_layer <= 6 else 12

        out_rows: List[Dict] = []

        for rec in eval_rows:
            imp = model.token_importance(rec)
            imp_vals = np.array(list(imp.values()), dtype=float)
            if imp_vals.size == 0:
                imp_vals = np.array([1.0])

            imp_mean = float(np.mean(imp_vals))
            imp_std = float(np.std(imp_vals) + 1e-6)

            prompt_len = max(1, len(rec["prompt"].split()))

            for layer in range(max_layer + 1):
                for head in range(n_heads):
                    noise = _stable_noise(f"{rec['id']}-{layer}-{head}", seed)
                    attention = max(1e-6, imp_mean * (1.0 + 0.15 * noise) * (1.0 + layer / (max_layer + 1)))
                    activation = (prompt_len / 100.0) * (1.0 + 0.05 * head) * (1.0 + layer / (max_layer + 1))
                    attribution = max(1e-6, imp_std * (1.0 + 0.1 * noise))

                    out_rows.append(
                        {
                            "sample_id": rec["id"],
                            "layer": int(layer),
                            "head": int(head),
                            "attention": float(attention),
                            "activation": float(activation),
                            "attribution": float(attribution),
                            "experiment": experiment,
                            "seed": int(seed),
                        }
                    )

    df = pd.DataFrame(out_rows)
    out_path = run_dir / "signals.parquet"
    write_parquet(out_path, df)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract standardized signals.parquet")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True, choices=["pretrained", "lora_sft", "random_label"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    out = extract_signals_one(args.config, args.experiment, args.seed)
    print(out)


if __name__ == "__main__":
    main()
