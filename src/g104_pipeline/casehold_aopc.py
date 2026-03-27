from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .casehold_task_perf_deletion import _window_casehold_prompt
from .config import load_config
from .hf_backend import HFInferenceRunner
from .io_utils import read_jsonl, write_json, write_jsonl


_EXPERIMENT_ALIASES = {
    "pretrained": "pretrained",
    "lora": "lora_sft",
    "lora_sft": "lora_sft",
    "random": "random_label",
    "random_label": "random_label",
}


def _canonical_experiment(name: str) -> str:
    key = name.strip().lower()
    if key not in _EXPERIMENT_ALIASES:
        raise ValueError(f"Unsupported experiment: {name}")
    return _EXPERIMENT_ALIASES[key]


def _default_deletion_samples_file(run_root: Path, experiment: str, output_tag: str | None) -> Path:
    path = run_root / "analysis" / "task_perf_deletion" / experiment
    if output_tag:
        path = path / output_tag
    return path / "task_perf_deletion_samples.jsonl"


def compute_casehold_aopc(
    config_path: str,
    experiment: str,
    seed: int,
    *,
    subset_file: str | None = None,
    run_root: str | None = None,
    deletion_samples_file: str | None = None,
    output_dir: str | None = None,
    output_tag: str | None = None,
) -> str:
    canonical_experiment = _canonical_experiment(experiment)
    cfg = load_config(config_path)

    root = Path(run_root) if run_root else Path(cfg["project"]["output_root"])
    run_dir = root / "artifacts" / canonical_experiment / str(seed)
    runner = HFInferenceRunner(cfg=cfg, run_dir=run_dir, experiment=canonical_experiment)

    if deletion_samples_file:
        deletion_path = Path(deletion_samples_file)
    else:
        deletion_path = _default_deletion_samples_file(root, canonical_experiment, output_tag)
    if not deletion_path.exists():
        raise FileNotFoundError(f"Deletion samples file not found: {deletion_path}")

    deletion_rows = read_jsonl(deletion_path)
    deletion_by_id = {str(row["id"]): row for row in deletion_rows if row.get("id") is not None}
    if not deletion_by_id:
        raise RuntimeError(f"No deletion sample rows found in {deletion_path}")

    data_cfg = cfg["data"]
    subset_path = Path(subset_file) if subset_file else Path(data_cfg["fixed_eval_subset_file"])
    eval_rows = read_jsonl(subset_path)
    if not eval_rows:
        raise RuntimeError(f"No rows found in {subset_path}")

    sample_rows: List[Dict[str, Any]] = []
    aopc_prob_values: List[float] = []
    aopc_logprob_values: List[float] = []
    aopc_accuracy_values: List[float] = []
    base_prob_values: List[float] = []
    base_logprob_values: List[float] = []
    base_acc_values: List[float] = []

    del_prob_curves: List[List[float]] = []
    del_logprob_curves: List[List[float]] = []
    del_acc_curves: List[List[float]] = []

    matched = 0
    for rec in eval_rows:
        rec_id = str(rec.get("id", ""))
        row = deletion_by_id.get(rec_id)
        if row is None:
            continue

        label = int(rec.get("label", -1))
        if label < 0:
            continue

        windowed_rec, window_info = _window_casehold_prompt(runner, rec)
        pred, probs = runner.predict(windowed_rec)
        base_acc = float(pred == label)
        base_gold_prob = float(probs[label])
        base_gold_logprob = math.log(max(base_gold_prob, 1e-12))

        del_acc_curve = [float(x) for x in row["deletion_accuracy_curve"]]
        del_prob_curve = [float(x) for x in row["deletion_gold_prob_curve"]]
        del_logprob_curve = [math.log(max(x, 1e-12)) for x in del_prob_curve]

        aopc_prob = float(np.mean([base_gold_prob - x for x in del_prob_curve]))
        aopc_logprob = float(np.mean([base_gold_logprob - x for x in del_logprob_curve]))
        aopc_accuracy = float(np.mean([base_acc - x for x in del_acc_curve]))

        sample_rows.append(
            {
                "id": rec_id,
                "label": label,
                "experiment": canonical_experiment,
                "seed": seed,
                **window_info,
                "base_accuracy": base_acc,
                "base_gold_prob": base_gold_prob,
                "base_gold_logprob": base_gold_logprob,
                "deletion_accuracy_curve": del_acc_curve,
                "deletion_gold_prob_curve": del_prob_curve,
                "deletion_gold_logprob_curve": del_logprob_curve,
                "aopc_gold_prob": aopc_prob,
                "aopc_gold_logprob": aopc_logprob,
                "aopc_accuracy": aopc_accuracy,
            }
        )

        matched += 1
        aopc_prob_values.append(aopc_prob)
        aopc_logprob_values.append(aopc_logprob)
        aopc_accuracy_values.append(aopc_accuracy)
        base_prob_values.append(base_gold_prob)
        base_logprob_values.append(base_gold_logprob)
        base_acc_values.append(base_acc)
        del_prob_curves.append(del_prob_curve)
        del_logprob_curves.append(del_logprob_curve)
        del_acc_curves.append(del_acc_curve)

    if not sample_rows:
        raise RuntimeError(
            "No overlapping rows between the evaluation subset and the deletion samples file."
        )

    out_dir = Path(output_dir) if output_dir else deletion_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_prob_curve = np.mean(np.array(del_prob_curves), axis=0).tolist()
    mean_logprob_curve = np.mean(np.array(del_logprob_curves), axis=0).tolist()
    mean_acc_curve = np.mean(np.array(del_acc_curves), axis=0).tolist()

    summary = {
        "task": "casehold",
        "experiment": canonical_experiment,
        "seed": seed,
        "model_name": cfg["model_name"],
        "subset_file": str(subset_path),
        "deletion_samples_file": str(deletion_path),
        "run_dir": str(run_dir),
        "num_subset_rows": len(eval_rows),
        "num_deletion_rows": len(deletion_rows),
        "num_matched_samples": matched,
        "deletion_unit": "prompt_tokens",
        "x_values_percent": [2 * (i + 1) for i in range(len(mean_prob_curve))],
        "x_values_percent_with_base": [0] + [2 * (i + 1) for i in range(len(mean_prob_curve))],
        "mean_base_accuracy": float(np.mean(base_acc_values)),
        "mean_base_gold_prob": float(np.mean(base_prob_values)),
        "mean_base_gold_logprob": float(np.mean(base_logprob_values)),
        "mean_deletion_accuracy_curve": [float(x) for x in mean_acc_curve],
        "mean_deletion_gold_prob_curve": [float(x) for x in mean_prob_curve],
        "mean_deletion_gold_logprob_curve": [float(x) for x in mean_logprob_curve],
        "mean_accuracy_curve_with_base": [float(np.mean(base_acc_values))] + [float(x) for x in mean_acc_curve],
        "mean_gold_prob_curve_with_base": [float(np.mean(base_prob_values))] + [float(x) for x in mean_prob_curve],
        "mean_gold_logprob_curve_with_base": [float(np.mean(base_logprob_values))] + [float(x) for x in mean_logprob_curve],
        "mean_aopc_accuracy": float(np.mean(aopc_accuracy_values)),
        "std_aopc_accuracy": float(np.std(aopc_accuracy_values)),
        "mean_aopc_gold_prob": float(np.mean(aopc_prob_values)),
        "std_aopc_gold_prob": float(np.std(aopc_prob_values)),
        "mean_aopc_gold_logprob": float(np.mean(aopc_logprob_values)),
        "std_aopc_gold_logprob": float(np.std(aopc_logprob_values)),
    }

    write_json(out_dir / "casehold_aopc.json", summary)
    write_jsonl(out_dir / "casehold_aopc_samples.jsonl", sample_rows)
    return str(out_dir / "casehold_aopc.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute standard CaseHOLD AOPC by adding an undeleted baseline to prompt-token deletion outputs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument(
        "--run-root",
        default=None,
        help="Root directory containing artifacts/<experiment>/<seed> for the target model.",
    )
    parser.add_argument(
        "--deletion-samples-file",
        default=None,
        help="Path to task_perf_deletion_samples.jsonl. If omitted, derive it from run-root.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for casehold_aopc.json and casehold_aopc_samples.jsonl. Defaults to the deletion file's directory.",
    )
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Optional subdirectory tag when deriving the deletion-samples path from run-root.",
    )
    args = parser.parse_args()

    out = compute_casehold_aopc(
        config_path=args.config,
        experiment=args.experiment,
        seed=args.seed,
        subset_file=args.subset_file,
        run_root=args.run_root,
        deletion_samples_file=args.deletion_samples_file,
        output_dir=args.output_dir,
        output_tag=args.output_tag,
    )
    print(out)


if __name__ == "__main__":
    main()
