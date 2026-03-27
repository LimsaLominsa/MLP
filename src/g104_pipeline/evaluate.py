from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import load_config
from .hf_backend import evaluate_hf
from .io_utils import read_jsonl, write_json, write_jsonl
from .modeling import PairwiseChoiceModel


def evaluate_one(config_path: str, experiment: str, seed: int) -> str:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    output_root = Path(cfg["project"]["output_root"])

    eval_rows = read_jsonl(data_cfg["test_file"])

    run_dir = output_root / "artifacts" / experiment / str(seed)
    backend = cfg.get("backend", "mock")
    if backend == "hf_lora":
        metrics, pred_rows = evaluate_hf(
            cfg=cfg,
            eval_rows=eval_rows,
            run_dir=run_dir,
            experiment=experiment,
            seed=seed,
        )
    else:
        model_path = run_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = PairwiseChoiceModel.load(str(model_path))

        pred_rows: List[Dict] = []
        correct = 0
        total_labeled = 0

        for rec in eval_rows:
            pred, scores = model.predict(rec)
            label = int(rec.get("label", -1))

            if label >= 0:
                total_labeled += 1
                correct += int(pred == label)

            pred_rows.append(
                {
                    "id": rec["id"],
                    "split": rec.get("split", "test"),
                    "label": label,
                    "prediction": pred,
                    "option_scores": [float(x) for x in scores.tolist()],
                    "confidence": float(np.max(scores)),
                    "is_correct": bool(pred == label) if label >= 0 else None,
                }
            )

        acc = float(correct / total_labeled) if total_labeled > 0 else None
        metrics = {
            "experiment": experiment,
            "seed": seed,
            "num_samples": len(eval_rows),
            "num_labeled": total_labeled,
            "accuracy": acc,
        }

    write_json(run_dir / "metrics.json", metrics)
    write_jsonl(run_dir / "predictions.jsonl", pred_rows)

    return str(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one experiment run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True, choices=["pretrained", "lora_sft", "random_label"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    out = evaluate_one(args.config, args.experiment, args.seed)
    print(out)


if __name__ == "__main__":
    main()
