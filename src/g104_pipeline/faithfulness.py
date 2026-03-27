from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import load_config
from .hf_backend import faithfulness_hf
from .io_utils import read_jsonl, write_json, write_jsonl
from .modeling import PairwiseChoiceModel


def compute_faithfulness_one(config_path: str, experiment: str, seed: int) -> str:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    metric_cfg = cfg.get("metrics", {})
    output_root = Path(cfg["project"]["output_root"])
    backend = cfg.get("backend", "mock")

    topk_ratio = float(metric_cfg.get("faithfulness_topk_ratio", 0.2))
    deletion_steps = int(metric_cfg.get("deletion_steps", 10))

    eval_rows = read_jsonl(data_cfg["fixed_eval_subset_file"])

    run_dir = output_root / "artifacts" / experiment / str(seed)
    if backend == "hf_lora":
        summary, sample_rows = faithfulness_hf(
            cfg=cfg,
            eval_rows=eval_rows,
            run_dir=run_dir,
            experiment=experiment,
            seed=seed,
        )
    else:
        model = PairwiseChoiceModel.load(str(run_dir / "model.pkl"))

        sample_rows: List[Dict] = []
        del_curves: List[List[float]] = []
        ins_curves: List[List[float]] = []
        aopcs: List[float] = []

        for rec in eval_rows:
            curve = model.deletion_insertion_curve(rec, topk_ratio=topk_ratio, steps=deletion_steps)
            del_curves.append(curve["deletion"])
            ins_curves.append(curve["insertion"])
            aopcs.append(float(curve["aopc"]))

            sample_rows.append(
                {
                    "id": rec["id"],
                    "experiment": experiment,
                    "seed": seed,
                    "aopc": float(curve["aopc"]),
                    "deletion_curve": curve["deletion"],
                    "insertion_curve": curve["insertion"],
                }
            )

        mean_del = np.mean(np.array(del_curves), axis=0).tolist() if del_curves else []
        mean_ins = np.mean(np.array(ins_curves), axis=0).tolist() if ins_curves else []

        summary = {
            "experiment": experiment,
            "seed": seed,
            "num_samples": len(eval_rows),
            "mean_aopc": float(np.mean(aopcs)) if aopcs else None,
            "std_aopc": float(np.std(aopcs)) if aopcs else None,
            "mean_deletion_curve": [float(x) for x in mean_del],
            "mean_insertion_curve": [float(x) for x in mean_ins],
            "topk_ratio": topk_ratio,
            "steps": deletion_steps,
        }

    write_json(run_dir / "faithfulness.json", summary)
    write_jsonl(run_dir / "faithfulness_samples.jsonl", sample_rows)
    return str(run_dir / "faithfulness.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute deletion/insertion/AOPC faithfulness metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True, choices=["pretrained", "lora_sft", "random_label"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    out = compute_faithfulness_one(args.config, args.experiment, args.seed)
    print(out)


if __name__ == "__main__":
    main()
