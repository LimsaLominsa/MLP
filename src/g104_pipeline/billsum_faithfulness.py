from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .billsum_baseline import (
    _build_prompt,
    _device,
    _extract_bill_text,
    _load_model_and_tokenizer,
    _load_config,
    _normalise_summary,
    _read_jsonl,
    _require_hf,
    _torch_dtype,
    _write_json,
    _write_jsonl,
)

_TOKEN_CLEAN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\\-_/]*")


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> Tuple[int, int] | None:
    if not subsequence or len(subsequence) > len(sequence):
        return None
    last = len(sequence) - len(subsequence) + 1
    head = subsequence[0]
    for start in range(last):
        if sequence[start] != head:
            continue
        if list(sequence[start : start + len(subsequence)]) == list(subsequence):
            return start, start + len(subsequence)
    return None


def _clean_token(token: str) -> str:
    token = token.replace("Ġ", "").replace("▁", "").strip().lower()
    if not token or token.startswith("<|"):
        return ""
    match = _TOKEN_CLEAN_RE.match(token)
    if not match:
        return ""
    return match.group(0).lower()


def _score_reference_logprob(
    *,
    model,
    tokenizer,
    torch_mod,
    row: Dict[str, Any],
    max_input_tokens: int,
) -> Tuple[float, int]:
    prompt = _build_prompt(tokenizer, row["input"])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) > max_input_tokens:
        prompt_ids = prompt_ids[-max_input_tokens:]

    reference = _normalise_summary(row["output"])
    ref_ids = tokenizer.encode(reference, add_special_tokens=False)
    if not ref_ids:
        return 0.0, 0

    input_ids = prompt_ids + ref_ids
    encoded = {
        "input_ids": torch_mod.tensor([input_ids], dtype=torch_mod.long, device=_device(torch_mod)),
        "attention_mask": torch_mod.ones((1, len(input_ids)), dtype=torch_mod.long, device=_device(torch_mod)),
    }

    with torch_mod.no_grad():
        out = model(**encoded)
        logits = out.logits[0]

    prompt_len = len(prompt_ids)
    total = 0.0
    for pos, tok_id in enumerate(ref_ids):
        step_logits = logits[prompt_len - 1 + pos].float()
        log_probs = torch_mod.log_softmax(step_logits, dim=-1)
        total += float(log_probs[int(tok_id)].item())

    return total / len(ref_ids), len(ref_ids)


def _bill_token_importance(
    *,
    model,
    tokenizer,
    torch_mod,
    row: Dict[str, Any],
    max_input_tokens: int,
) -> Tuple[List[int], List[float]]:
    prompt = _build_prompt(tokenizer, row["input"])
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded = {k: v.to(_device(torch_mod)) for k, v in encoded.items()}

    with torch_mod.no_grad():
        out = model(**encoded, output_attentions=True)

    full_ids = encoded["input_ids"][0].detach().cpu().tolist()
    bill_ids = tokenizer.encode(_extract_bill_text(row["input"]), add_special_tokens=False)
    span = _find_subsequence(full_ids, bill_ids)
    if span is None:
        start, end = 0, len(full_ids)
    else:
        start, end = span

    if not out.attentions:
        raise RuntimeError("Attention tensors were not returned for BillSum faithfulness scoring.")

    layer_tensors = out.attentions[-4:] if len(out.attentions) >= 4 else out.attentions
    token_scores = np.zeros(end - start, dtype=np.float64)
    for att in layer_tensors:
        arr = att[0].detach().float().cpu().numpy()  # [heads, q, k]
        key_imp = arr.mean(axis=0).mean(axis=0)[start:end]
        token_scores += key_imp

    token_scores /= max(1, len(layer_tensors))
    bill_token_ids = full_ids[start:end]
    return bill_token_ids, token_scores.tolist()


def _selected_token_positions(scores: Sequence[float], topk_ratio: float) -> List[int]:
    if not scores:
        return []
    topk = max(1, int(len(scores) * topk_ratio))
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    return order[:topk]


def _replace_bill_text(raw_input: str, new_bill_text: str) -> str:
    original_bill_text = _extract_bill_text(raw_input)
    if original_bill_text in raw_input:
        return raw_input.replace(original_bill_text, new_bill_text, 1)
    if "### Bill:" in raw_input:
        prefix, _, rest = raw_input.partition("### Bill:")
        if "### Summary:" in rest:
            _, _, tail = rest.partition("### Summary:")
            return f"{prefix}### Bill:\n{new_bill_text}\n\n### Summary:{tail}"
        return f"{prefix}### Bill:\n{new_bill_text}"
    return new_bill_text


def _perturb_bill_tokens(
    *,
    row: Dict[str, Any],
    tokenizer,
    selected_positions: Sequence[int],
    step: int,
    steps: int,
    mode: str,
) -> Dict[str, Any]:
    bill_token_ids, _ = tokenizer.encode(_extract_bill_text(row["input"]), add_special_tokens=False), None
    cut = max(1, int(len(selected_positions) * (step / steps)))
    chosen = set(selected_positions[:cut])

    if mode == "deletion":
        kept_ids = [tok for idx, tok in enumerate(bill_token_ids) if idx not in chosen]
    elif mode == "insertion":
        kept_ids = [tok for idx, tok in enumerate(bill_token_ids) if idx in chosen]
    else:
        raise ValueError(f"Unsupported perturbation mode: {mode}")

    if not kept_ids:
        kept_ids = bill_token_ids[:1]

    new_bill_text = tokenizer.decode(kept_ids, skip_special_tokens=True).strip()
    if not new_bill_text:
        new_bill_text = _extract_bill_text(row["input"])[:32]

    mutated = dict(row)
    mutated["input"] = _replace_bill_text(row["input"], new_bill_text)
    return mutated


def _faithfulness_for_row(
    *,
    model,
    tokenizer,
    torch_mod,
    row: Dict[str, Any],
    topk_ratio: float,
    steps: int,
    max_input_tokens: int,
) -> Dict[str, Any]:
    base_score, ref_token_count = _score_reference_logprob(
        model=model,
        tokenizer=tokenizer,
        torch_mod=torch_mod,
        row=row,
        max_input_tokens=max_input_tokens,
    )
    bill_token_ids, token_scores = _bill_token_importance(
        model=model,
        tokenizer=tokenizer,
        torch_mod=torch_mod,
        row=row,
        max_input_tokens=max_input_tokens,
    )
    selected = _selected_token_positions(token_scores, topk_ratio=topk_ratio)

    deletion_curve: List[float] = []
    insertion_curve: List[float] = []
    for step in range(1, steps + 1):
        rec_del = _perturb_bill_tokens(
            row=row,
            tokenizer=tokenizer,
            selected_positions=selected,
            step=step,
            steps=steps,
            mode="deletion",
        )
        del_score, _ = _score_reference_logprob(
            model=model,
            tokenizer=tokenizer,
            torch_mod=torch_mod,
            row=rec_del,
            max_input_tokens=max_input_tokens,
        )
        deletion_curve.append(float(del_score))

        rec_ins = _perturb_bill_tokens(
            row=row,
            tokenizer=tokenizer,
            selected_positions=selected,
            step=step,
            steps=steps,
            mode="insertion",
        )
        ins_score, _ = _score_reference_logprob(
            model=model,
            tokenizer=tokenizer,
            torch_mod=torch_mod,
            row=rec_ins,
            max_input_tokens=max_input_tokens,
        )
        insertion_curve.append(float(ins_score))

    aopc = float(np.mean([base_score - score for score in deletion_curve]))
    return {
        "sample_id": row.get("sample_id"),
        "base_score": float(base_score),
        "reference_token_count": int(ref_token_count),
        "selected_token_count": len(selected),
        "bill_token_count": len(bill_token_ids),
        "aopc": aopc,
        "deletion_curve": deletion_curve,
        "insertion_curve": insertion_curve,
    }


def compute_billsum_faithfulness(
    config_path: str,
    split: str = "test_us",
    output_tag: str | None = None,
    subset_size: int = 64,
    subset_seed: int = 42,
    topk_ratio: float = 0.2,
    deletion_steps: int = 10,
) -> str:
    cfg = _load_config(config_path)
    hf = _require_hf()
    torch = hf["torch"]
    data_cfg = cfg["data"]
    input_path = Path(data_cfg[f"{split}_file"])
    rows = _read_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"No rows found in {input_path}")

    rng = random.Random(subset_seed)
    indexed_rows = list(enumerate(rows))
    rng.shuffle(indexed_rows)
    chosen = indexed_rows[: min(subset_size, len(indexed_rows))]
    chosen.sort(key=lambda item: item[0])

    model, tokenizer = _load_model_and_tokenizer(
        cfg=cfg,
        hf=hf,
        attn_implementation="eager",
    )

    max_input_tokens = int(cfg.get("generation", {}).get("max_input_tokens", 2048))
    output_root = Path(cfg["project"]["output_root"]) / "baseline" / split
    if output_tag:
        output_root = output_root / output_tag
    output_root.mkdir(parents=True, exist_ok=True)

    subset_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    del_curves: List[List[float]] = []
    ins_curves: List[List[float]] = []
    aopcs: List[float] = []

    for dataset_index, row in chosen:
        row = dict(row)
        row["dataset_index"] = dataset_index
        row["sample_id"] = row.get("sample_id") or f"{split}-{dataset_index}"
        subset_rows.append(row)

        result = _faithfulness_for_row(
            model=model,
            tokenizer=tokenizer,
            torch_mod=torch,
            row=row,
            topk_ratio=topk_ratio,
            steps=deletion_steps,
            max_input_tokens=max_input_tokens,
        )
        result["dataset_index"] = dataset_index
        sample_rows.append(result)
        del_curves.append(result["deletion_curve"])
        ins_curves.append(result["insertion_curve"])
        aopcs.append(float(result["aopc"]))

    mean_del = np.mean(np.array(del_curves), axis=0).tolist() if del_curves else []
    mean_ins = np.mean(np.array(ins_curves), axis=0).tolist() if ins_curves else []
    summary = {
        "task": "billsum",
        "model_name": cfg["model_name"],
        "adapter_dir": cfg.get("adapter_dir"),
        "split": split,
        "num_samples": len(sample_rows),
        "subset_seed": subset_seed,
        "subset_size": subset_size,
        "score_type": "gold_summary_mean_log_prob",
        "mean_aopc": float(np.mean(aopcs)) if aopcs else None,
        "std_aopc": float(np.std(aopcs)) if aopcs else None,
        "mean_deletion_curve": [float(x) for x in mean_del],
        "mean_insertion_curve": [float(x) for x in mean_ins],
        "topk_ratio": topk_ratio,
        "steps": deletion_steps,
    }

    _write_json(output_root / "faithfulness.json", summary)
    _write_jsonl(output_root / "faithfulness_samples.jsonl", sample_rows)
    _write_jsonl(output_root / "faithfulness_subset.jsonl", subset_rows)
    return str(output_root / "faithfulness.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BillSum deletion/insertion faithfulness metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test_us", choices=["valid", "test_us", "test_ca"])
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--topk-ratio", type=float, default=0.2)
    parser.add_argument("--deletion-steps", type=int, default=10)
    args = parser.parse_args()

    out = compute_billsum_faithfulness(
        config_path=args.config,
        split=args.split,
        output_tag=args.output_tag,
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
        topk_ratio=args.topk_ratio,
        deletion_steps=args.deletion_steps,
    )
    print(out)


if __name__ == "__main__":
    main()
