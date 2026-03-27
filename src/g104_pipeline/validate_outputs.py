from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from .config import load_config
from .io_utils import read_json, read_jsonl


def _check_dataset_schema(path: Path) -> None:
    rows = read_jsonl(path)
    if not rows:
        raise RuntimeError(f"Dataset file is empty: {path}")

    req = {"id", "prompt", "options", "label", "split"}
    miss = req - set(rows[0].keys())
    if miss:
        raise RuntimeError(f"Missing dataset keys {miss} in {path}")


def validate(config_path: str) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    output_root = Path(cfg["project"]["output_root"])

    # Dataset schema checks
    for p in [data_cfg["train_file"], data_cfg["valid_file"], data_cfg["test_file"], data_cfg["fixed_eval_subset_file"]]:
        _check_dataset_schema(Path(p))

    exps = cfg.get("experiments", ["pretrained", "lora_sft", "random_label"])
    seeds = [int(s) for s in cfg.get("seeds", [cfg["seed"]])]

    # Artifact schema checks
    for exp in exps:
        for seed in seeds:
            run_dir = output_root / "artifacts" / exp / str(seed)
            for must in ["metrics.json", "predictions.jsonl", "signals.parquet", "faithfulness.json"]:
                p = run_dir / must
                if not p.exists():
                    raise RuntimeError(f"Missing artifact: {p}")

            metrics = read_json(run_dir / "metrics.json")
            for key in ["experiment", "seed", "num_samples", "accuracy"]:
                if key not in metrics:
                    raise RuntimeError(f"metrics.json missing key `{key}` in {run_dir}")

            preds = read_jsonl(run_dir / "predictions.jsonl")
            pred_req = {"id", "label", "prediction", "option_scores", "confidence"}
            if not preds:
                raise RuntimeError(f"No predictions in {run_dir}")
            miss = pred_req - set(preds[0].keys())
            if miss:
                raise RuntimeError(f"predictions.jsonl missing keys {miss} in {run_dir}")

            sig = pd.read_parquet(run_dir / "signals.parquet")
            sig_req = {"sample_id", "layer", "head", "attention", "activation", "attribution"}
            miss_sig = sig_req - set(sig.columns)
            if miss_sig:
                raise RuntimeError(f"signals.parquet missing columns {miss_sig} in {run_dir}")

    summary = output_root / "stats" / "summary_stats.csv"
    if not summary.exists():
        raise RuntimeError("Missing summary_stats.csv")

    df = pd.read_csv(summary)
    req_cols = {"experiment", "seed", "metric", "value", "ci_low", "ci_high"}
    miss = req_cols - set(df.columns)
    if miss:
        raise RuntimeError(f"summary_stats.csv missing columns {miss}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate output artifacts and schemas.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    validate(args.config)
    print("validation_ok")


if __name__ == "__main__":
    main()
