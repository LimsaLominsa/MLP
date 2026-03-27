from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from rouge_score import rouge_scorer

from .billsum_baseline import (
    _device,
    _generate_one,
    _load_config,
    _load_model_and_tokenizer,
    _normalise_summary as _normalise_billsum_summary,
    _read_jsonl,
    _require_hf,
)
from .billsum_faithfulness import (
    _bill_token_importance,
    _perturb_bill_tokens,
    _score_reference_logprob as _score_billsum_logprob,
)
from .io_utils import write_json, write_jsonl
from .pubmed_faithfulness import (
    _article_token_importance,
    _normalise_summary as _normalise_pubmed_summary,
    _perturb_article_tokens,
    _score_reference_logprob as _score_pubmed_logprob,
    _window_pubmed_row,
)


def _load_subset_rows(data_file: Path, subset_file: str | None, subset_size: int, subset_seed: int) -> List[Dict[str, Any]]:
    if subset_file:
        rows = _read_jsonl(Path(subset_file))
        if rows:
            return rows

    rows = _read_jsonl(data_file)
    rng = random.Random(subset_seed)
    indexed = list(enumerate(rows))
    rng.shuffle(indexed)
    chosen = indexed[: min(subset_size, len(indexed))]
    chosen.sort(key=lambda item: item[0])

    out: List[Dict[str, Any]] = []
    for dataset_index, row in chosen:
        row = dict(row)
        row["dataset_index"] = dataset_index
        row["sample_id"] = row.get("sample_id") or f"{data_file.stem}-{dataset_index}"
        out.append(row)
    return out


def _score_prediction(prediction: str, reference: str, scorer: rouge_scorer.RougeScorer) -> Dict[str, float]:
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


def compute_summarization_task_perf_deletion(
    config_path: str,
    split: str,
    task: str,
    subset_file: str | None = None,
    subset_size: int = 64,
    subset_seed: int = 42,
    topk_ratio: float = 0.2,
    deletion_steps: int = 10,
    output_tag: str | None = None,
) -> str:
    cfg = _load_config(config_path)
    hf = _require_hf()
    torch = hf["torch"]
    model, tokenizer = _load_model_and_tokenizer(cfg=cfg, hf=hf, attn_implementation="eager")

    data_path = Path(cfg["data"][f"{split}_file"])
    rows = _load_subset_rows(data_path, subset_file, subset_size, subset_seed)
    if not rows:
        raise RuntimeError(f"No rows available for {task} deletion task-performance run.")

    max_input_tokens = int(cfg.get("generation", {}).get("max_input_tokens", 2048))
    max_new_tokens = int(cfg.get("generation", {}).get("max_new_tokens", 512))
    do_sample = bool(cfg.get("generation", {}).get("do_sample", False))
    num_beams = int(cfg.get("generation", {}).get("num_beams", 1))
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    if task == "billsum":
        normalise_summary: Callable[[str], str] = _normalise_billsum_summary
        importance_fn = _bill_token_importance
        perturb_fn = _perturb_bill_tokens
        score_fn = _score_billsum_logprob
        window_row_fn = None
    elif task == "pubmed":
        normalise_summary = _normalise_pubmed_summary
        importance_fn = _article_token_importance
        perturb_fn = _perturb_article_tokens
        score_fn = _score_pubmed_logprob
        window_row_fn = _window_pubmed_row
    else:
        raise ValueError(f"Unsupported summarization task: {task}")

    step_scores = {
        "rouge1": [[] for _ in range(deletion_steps)],
        "rouge2": [[] for _ in range(deletion_steps)],
        "rougeL": [[] for _ in range(deletion_steps)],
    }
    sample_rows: List[Dict[str, Any]] = []

    for row in rows:
        row = dict(row)
        if window_row_fn is not None:
            row, window_info = window_row_fn(
                row=row,
                tokenizer=tokenizer,
                max_input_tokens=max_input_tokens,
            )
        else:
            window_info = {}
        _, token_scores = importance_fn(
            model=model,
            tokenizer=tokenizer,
            torch_mod=torch,
            row=row,
            max_input_tokens=max_input_tokens,
        )
        selected = sorted(range(len(token_scores)), key=lambda i: float(token_scores[i]), reverse=True)
        topk = max(1, int(len(selected) * topk_ratio))
        selected = selected[:topk]

        reference = normalise_summary(row["output"])
        deletion_curve_rouge1: List[float] = []
        deletion_curve_rouge2: List[float] = []
        deletion_curve_rougeL: List[float] = []

        for step in range(1, deletion_steps + 1):
            rec_del = perturb_fn(
                row=row,
                tokenizer=tokenizer,
                selected_positions=selected,
                step=step,
                steps=deletion_steps,
                mode="deletion",
            )
            prediction, _, _ = _generate_one(
                model=model,
                tokenizer=tokenizer,
                torch_mod=torch,
                prompt=rec_del["input"],
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=num_beams,
            )
            rouges = _score_prediction(prediction, reference, scorer)
            deletion_curve_rouge1.append(rouges["rouge1"])
            deletion_curve_rouge2.append(rouges["rouge2"])
            deletion_curve_rougeL.append(rouges["rougeL"])
            step_scores["rouge1"][step - 1].append(rouges["rouge1"])
            step_scores["rouge2"][step - 1].append(rouges["rouge2"])
            step_scores["rougeL"][step - 1].append(rouges["rougeL"])

        base_score, _ = score_fn(
            model=model,
            tokenizer=tokenizer,
            torch_mod=torch,
            row=row,
            max_input_tokens=max_input_tokens,
        )

        sample_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "dataset_index": row.get("dataset_index"),
                "base_logprob_proxy": float(base_score),
                **window_info,
                "deletion_curve_rouge1": deletion_curve_rouge1,
                "deletion_curve_rouge2": deletion_curve_rouge2,
                "deletion_curve_rougeL": deletion_curve_rougeL,
            }
        )

    summary = {
        "task": task,
        "model_name": cfg["model_name"],
        "adapter_dir": cfg.get("adapter_dir"),
        "split": split,
        "subset_size": len(rows),
        "subset_seed": subset_seed,
        "metric_name": "rougeL",
        "x_values_percent": [2 * (i + 1) for i in range(deletion_steps)],
        "mean_deletion_curve_rouge1": [float(np.mean(xs)) if xs else None for xs in step_scores["rouge1"]],
        "mean_deletion_curve_rouge2": [float(np.mean(xs)) if xs else None for xs in step_scores["rouge2"]],
        "mean_deletion_curve_rougeL": [float(np.mean(xs)) if xs else None for xs in step_scores["rougeL"]],
        "topk_ratio": topk_ratio,
        "steps": deletion_steps,
        "generation": {
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
        },
    }

    out_root = Path(cfg["project"]["output_root"]) / "analysis" / split / "task_perf_deletion"
    if output_tag:
        out_root = out_root / output_tag
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "task_perf_deletion.json", summary)
    write_jsonl(out_root / "task_perf_deletion_samples.jsonl", sample_rows)
    return str(out_root / "task_perf_deletion.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute summarization task performance under deletion.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True, choices=["billsum", "pubmed"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--topk-ratio", type=float, default=0.2)
    parser.add_argument("--deletion-steps", type=int, default=10)
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    out = compute_summarization_task_perf_deletion(
        config_path=args.config,
        split=args.split,
        task=args.task,
        subset_file=args.subset_file,
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
        topk_ratio=args.topk_ratio,
        deletion_steps=args.deletion_steps,
        output_tag=args.output_tag,
    )
    print(out)


if __name__ == "__main__":
    main()
