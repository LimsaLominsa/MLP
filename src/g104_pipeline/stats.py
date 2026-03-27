from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import load_config
from .io_utils import read_json


def _bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]

    arr = np.array(values, dtype=float)
    means = []
    rng = np.random.default_rng(123)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(np.mean(sample))

    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def aggregate_summary_stats(config_path: str) -> str:
    cfg = load_config(config_path)
    output_root = Path(cfg["project"]["output_root"])
    seeds = [int(s) for s in cfg.get("seeds", [cfg["seed"]])]
    exps = cfg.get("experiments", ["pretrained", "lora_sft", "random_label"])
    n_boot = int(cfg.get("metrics", {}).get("bootstrap_samples", 1000))

    raw_rows: List[Dict] = []

    for exp in exps:
        for seed in seeds:
            run_dir = output_root / "artifacts" / exp / str(seed)
            metric_path = run_dir / "metrics.json"
            faith_path = run_dir / "faithfulness.json"
            rep_path = run_dir / "rep_metrics.json"

            if metric_path.exists():
                metrics = read_json(metric_path)
                if metrics.get("accuracy") is not None:
                    raw_rows.append({
                        "experiment": exp,
                        "seed": seed,
                        "metric": "accuracy",
                        "value": float(metrics["accuracy"]),
                    })

            if faith_path.exists():
                faith = read_json(faith_path)
                if faith.get("mean_aopc") is not None:
                    raw_rows.append({
                        "experiment": exp,
                        "seed": seed,
                        "metric": "mean_aopc",
                        "value": float(faith["mean_aopc"]),
                    })

            if rep_path.exists():
                rep = read_json(rep_path)
                for m in ["attention_js_mean", "entropy_delta_mean", "cka_activation_mean"]:
                    if rep.get(m) is not None:
                        raw_rows.append({
                            "experiment": exp,
                            "seed": seed,
                            "metric": m,
                            "value": float(rep[m]),
                        })

    raw_df = pd.DataFrame(raw_rows)
    if raw_df.empty:
        raise RuntimeError("No metric artifacts found for aggregation.")

    out_rows: List[Dict] = []
    for (exp, metric), grp in raw_df.groupby(["experiment", "metric"]):
        vals = [float(x) for x in grp["value"].tolist()]
        mean_val = float(np.mean(vals))
        ci_low, ci_high = _bootstrap_ci(vals, n_boot=n_boot)

        # Required schema columns.
        out_rows.append(
            {
                "experiment": exp,
                "seed": "ALL",
                "metric": metric,
                "value": mean_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    out_df = pd.DataFrame(out_rows).sort_values(["metric", "experiment"]).reset_index(drop=True)
    out_path = output_root / "stats" / "summary_stats.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Also write per-seed rows (same required columns + concrete seed).
    per_seed_rows: List[Dict] = []
    for _, r in raw_df.iterrows():
        per_seed_rows.append(
            {
                "experiment": r["experiment"],
                "seed": int(r["seed"]),
                "metric": r["metric"],
                "value": float(r["value"]),
                "ci_low": float(r["value"]),
                "ci_high": float(r["value"]),
            }
        )

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = output_root / "stats" / "summary_stats_per_seed.csv"
    per_seed_df.to_csv(per_seed_path, index=False)

    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics and confidence intervals into summary_stats.csv")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    out = aggregate_summary_stats(args.config)
    print(out)


if __name__ == "__main__":
    main()
