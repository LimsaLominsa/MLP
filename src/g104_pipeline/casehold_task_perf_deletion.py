from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import load_config
from .hf_backend import HFInferenceRunner
from .io_utils import read_jsonl, write_json, write_jsonl
from .prompting import to_instruction_prompt


def _answer_prompt(record: Dict[str, Any]) -> str:
    return to_instruction_prompt(record) + "\nAnswer:"


def _find_subsequence(sequence: List[int], subsequence: List[int]) -> tuple[int, int] | None:
    if not subsequence or len(subsequence) > len(sequence):
        return None
    last = len(sequence) - len(subsequence) + 1
    head = subsequence[0]
    for start in range(last):
        if sequence[start] != head:
            continue
        if sequence[start : start + len(subsequence)] == subsequence:
            return start, start + len(subsequence)
    return None


def _common_prefix_len(left: List[int], right: List[int]) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def _common_suffix_len(left: List[int], right: List[int], *, prefix_len: int) -> int:
    left_limit = len(left) - prefix_len
    right_limit = len(right) - prefix_len
    limit = min(left_limit, right_limit)
    idx = 0
    while idx < limit and left[len(left) - 1 - idx] == right[len(right) - 1 - idx]:
        idx += 1
    return idx


def _window_casehold_prompt(
    runner: HFInferenceRunner, record: Dict[str, Any]
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    full_prompt = _answer_prompt(record)
    empty_record = dict(record)
    empty_record["prompt"] = ""
    empty_prompt = _answer_prompt(empty_record)

    full_ids = runner.tokenizer.encode(full_prompt, add_special_tokens=False)
    empty_ids = runner.tokenizer.encode(empty_prompt, add_special_tokens=False)

    prefix_len = _common_prefix_len(full_ids, empty_ids)
    suffix_len = _common_suffix_len(full_ids, empty_ids, prefix_len=prefix_len)
    prompt_start = prefix_len
    prompt_end = len(full_ids) - suffix_len

    trunc_start = max(0, len(full_ids) - runner.max_len)
    trunc_prompt_ids = full_ids[trunc_start:]
    visible_start = max(prompt_start, trunc_start) - trunc_start
    visible_end = max(min(prompt_end, len(full_ids)), trunc_start) - trunc_start
    visible_prompt_ids = trunc_prompt_ids[visible_start:visible_end]

    if not visible_prompt_ids:
        raw_prompt_ids = runner.tokenizer.encode(record["prompt"], add_special_tokens=False)
        visible_prompt_ids = raw_prompt_ids[-runner.max_len :]

    visible_prompt = runner.tokenizer.decode(visible_prompt_ids, skip_special_tokens=True).strip()
    if not visible_prompt:
        visible_prompt = record["prompt"][:64]

    windowed = dict(record)
    windowed["prompt"] = visible_prompt
    info = {
        "original_prompt_token_count": len(runner.tokenizer.encode(record["prompt"], add_special_tokens=False)),
        "window_prompt_token_count": len(visible_prompt_ids),
        "instruction_token_count_before_truncation": len(full_ids),
        "truncated_left_tokens": trunc_start,
        "used_windowing": trunc_start > 0,
    }
    return windowed, info


def _prompt_token_importance(
    runner: HFInferenceRunner, record: Dict[str, Any]
) -> tuple[List[int], List[float], Dict[str, Any]]:
    prompt = _answer_prompt(record)
    inputs = runner._encode_prompt(prompt)

    with runner.torch.no_grad():
        out = runner.model(**inputs, output_attentions=True)

    full_ids = inputs["input_ids"][0].detach().cpu().tolist()
    prompt_ids = runner.tokenizer.encode(record["prompt"], add_special_tokens=False)
    span = _find_subsequence(full_ids, prompt_ids)
    if span is None:
        start, end = 0, len(full_ids)
        span_found = False
    else:
        start, end = span
        span_found = True

    layer_tensors = out.attentions[-4:] if len(out.attentions) >= 4 else out.attentions
    token_scores = np.zeros(end - start, dtype=np.float64)
    for att in layer_tensors:
        arr = att[0].detach().float().cpu().numpy()
        key_imp = arr.mean(axis=0).mean(axis=0)[start:end]
        token_scores += key_imp
    token_scores /= max(1, len(layer_tensors))

    info = {
        "prompt_span_found": span_found,
        "prompt_visible_token_count": end - start,
    }
    return full_ids[start:end], token_scores.tolist(), info


def _selected_token_positions(scores: List[float], topk_ratio: float) -> List[int]:
    if not scores:
        return []
    topk = max(1, int(len(scores) * topk_ratio))
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    return order[:topk]


def _perturb_prompt_tokens(
    *,
    record: Dict[str, Any],
    tokenizer,
    prompt_token_ids: List[int],
    selected_positions: List[int],
    step: int,
    steps: int,
) -> Dict[str, Any]:
    cut = max(1, int(len(selected_positions) * (step / steps)))
    removed = set(selected_positions[:cut])
    kept_ids = [tok for idx, tok in enumerate(prompt_token_ids) if idx not in removed]
    if not kept_ids:
        kept_ids = prompt_token_ids[:1]

    new_prompt = tokenizer.decode(kept_ids, skip_special_tokens=True).strip()
    if not new_prompt:
        new_prompt = record["prompt"][:32]

    rec_del = dict(record)
    rec_del["prompt"] = new_prompt
    return rec_del


def compute_casehold_task_perf_deletion(
    config_path: str,
    experiment: str,
    seed: int,
    subset_file: str | None = None,
    topk_ratio: float = 0.2,
    deletion_steps: int = 10,
    output_tag: str | None = None,
) -> str:
    cfg = load_config(config_path)
    output_root = Path(cfg["project"]["output_root"])
    run_dir = output_root / "artifacts" / experiment / str(seed)
    runner = HFInferenceRunner(cfg=cfg, run_dir=run_dir, experiment=experiment)

    data_cfg = cfg["data"]
    subset_path = Path(subset_file) if subset_file else Path(data_cfg["fixed_eval_subset_file"])
    eval_rows = read_jsonl(subset_path)
    if not eval_rows:
        raise RuntimeError(f"No rows found in {subset_path}")

    sample_rows: List[Dict[str, Any]] = []
    step_accuracy: List[List[float]] = [[] for _ in range(deletion_steps)]
    step_gold_prob: List[List[float]] = [[] for _ in range(deletion_steps)]

    for rec in eval_rows:
        label = int(rec.get("label", -1))
        if label < 0:
            continue

        windowed_rec, window_info = _window_casehold_prompt(runner, rec)
        prompt_token_ids, token_scores, prompt_info = _prompt_token_importance(runner, windowed_rec)
        selected = _selected_token_positions(token_scores, topk_ratio=topk_ratio)
        if not selected or not prompt_token_ids:
            continue

        deletion_acc_curve: List[float] = []
        deletion_gold_prob_curve: List[float] = []

        for step in range(1, deletion_steps + 1):
            rec_del = _perturb_prompt_tokens(
                record=windowed_rec,
                tokenizer=runner.tokenizer,
                prompt_token_ids=prompt_token_ids,
                selected_positions=selected,
                step=step,
                steps=deletion_steps,
            )
            pred, probs = runner.predict(rec_del)
            acc = float(pred == label)
            gold_prob = float(probs[label])

            deletion_acc_curve.append(acc)
            deletion_gold_prob_curve.append(gold_prob)
            step_accuracy[step - 1].append(acc)
            step_gold_prob[step - 1].append(gold_prob)

        sample_rows.append(
            {
                "id": rec["id"],
                "label": label,
                "selected_prompt_token_count": len(selected),
                "prompt_token_count": len(prompt_token_ids),
                **window_info,
                **prompt_info,
                "deletion_accuracy_curve": deletion_acc_curve,
                "deletion_gold_prob_curve": deletion_gold_prob_curve,
            }
        )

    summary = {
        "task": "casehold",
        "model_name": cfg["model_name"],
        "experiment": experiment,
        "seed": seed,
        "subset_file": str(subset_path),
        "num_samples": len(sample_rows),
        "metric_name": "accuracy",
        "deletion_unit": "prompt_tokens",
        "x_values_percent": [2 * (i + 1) for i in range(deletion_steps)],
        "mean_deletion_accuracy_curve": [float(np.mean(xs)) if xs else None for xs in step_accuracy],
        "mean_deletion_gold_prob_curve": [float(np.mean(xs)) if xs else None for xs in step_gold_prob],
        "topk_ratio": topk_ratio,
        "steps": deletion_steps,
    }

    out_root = output_root / "analysis" / "task_perf_deletion" / experiment
    if output_tag:
        out_root = out_root / output_tag
    out_root.mkdir(parents=True, exist_ok=True)
    write_json(out_root / "task_perf_deletion.json", summary)
    write_jsonl(out_root / "task_perf_deletion_samples.jsonl", sample_rows)
    return str(out_root / "task_perf_deletion.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CaseHOLD actual task performance under deletion.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True, choices=["pretrained", "lora_sft", "random_label"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset-file", default=None)
    parser.add_argument("--topk-ratio", type=float, default=0.2)
    parser.add_argument("--deletion-steps", type=int, default=10)
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    out = compute_casehold_task_perf_deletion(
        config_path=args.config,
        experiment=args.experiment,
        seed=args.seed,
        subset_file=args.subset_file,
        topk_ratio=args.topk_ratio,
        deletion_steps=args.deletion_steps,
        output_tag=args.output_tag,
    )
    print(out)


if __name__ == "__main__":
    main()
