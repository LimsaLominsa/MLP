from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .config import load_config
from .data_prep import prepare_data
from .evaluate import evaluate_one
from .faithfulness import compute_faithfulness_one
from .failure_cases import build_failure_cases
from .io_utils import ensure_dir, write_json
from .rep_metrics import compute_rep_metrics_for_seed
from .signals import extract_signals_one
from .stats import aggregate_summary_stats
from .train import train_one


def run_pipeline(config_path: str) -> Dict:
    cfg = load_config(config_path)
    output_root = Path(cfg["project"]["output_root"])
    seeds = [int(s) for s in cfg.get("seeds", [cfg["seed"]])]
    experiments = cfg.get("experiments", ["pretrained", "lora_sft", "random_label"])

    ensure_dir(output_root / "logs")

    run_log: Dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "seeds": seeds,
        "experiments": experiments,
        "steps": [],
    }

    # Step 1: data
    prepare_data(config_path)
    run_log["steps"].append({"step": "prepare_data", "status": "ok"})

    # Steps 2-12 per experiment/seed
    for exp in experiments:
        for seed in seeds:
            train_one(config_path, exp, seed)
            run_log["steps"].append({"step": "train", "experiment": exp, "seed": seed, "status": "ok"})

            evaluate_one(config_path, exp, seed)
            run_log["steps"].append({"step": "evaluate", "experiment": exp, "seed": seed, "status": "ok"})

            extract_signals_one(config_path, exp, seed)
            run_log["steps"].append({"step": "extract_signals", "experiment": exp, "seed": seed, "status": "ok"})

            compute_faithfulness_one(config_path, exp, seed)
            run_log["steps"].append({"step": "faithfulness", "experiment": exp, "seed": seed, "status": "ok"})

    for seed in seeds:
        compute_rep_metrics_for_seed(config_path, seed)
        run_log["steps"].append({"step": "rep_metrics", "seed": seed, "status": "ok"})

    aggregate_summary_stats(config_path)
    run_log["steps"].append({"step": "aggregate_stats", "status": "ok"})

    build_failure_cases(config_path, max_cases=12)
    run_log["steps"].append({"step": "failure_cases", "status": "ok"})

    run_log["finished_at"] = datetime.now(timezone.utc).isoformat()
    run_manifest = output_root / "logs" / "pipeline_manifest.json"
    write_json(run_manifest, run_log)
    return run_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end project pipeline.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    log = run_pipeline(args.config)
    print(f"Pipeline completed with {len(log['steps'])} steps.")


if __name__ == "__main__":
    main()
