from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from .billsum_baseline import (
    _device,
    _load_config,
    _load_model_and_tokenizer,
    _read_jsonl,
    _require_hf,
    _write_json,
    _write_jsonl,
)
from .nfcorpus_faithfulness import (
    _extract_passages_text,
    _passage_token_importance,
    _perturb_passage_tokens,
)
from .nfcorpus_signals import _build_nfcorpus_prompt


def _candidate_rank_token_ids(tokenizer, n_candidates: int) -> Dict[int, List[int]]:
    token_map: Dict[int, List[int]] = {}
    for idx in range(1, n_candidates + 1):
        token_map[idx] = tokenizer.encode(f" {idx}", add_special_tokens=False)
    return token_map


def _sequence_logprob(*, model, torch_mod, input_ids: List[int], continuation_ids: Sequence[int]) -> float:
    if not continuation_ids:
        return 0.0

    full_ids = input_ids + list(continuation_ids)
    encoded = {
        "input_ids": torch_mod.tensor([full_ids], dtype=torch_mod.long, device=_device(torch_mod)),
        "attention_mask": torch_mod.ones((1, len(full_ids)), dtype=torch_mod.long, device=_device(torch_mod)),
    }

    with torch_mod.no_grad():
        out = model(**encoded)
        logits = out.logits[0]

    prompt_len = len(input_ids)
    total = 0.0
    for pos, tok_id in enumerate(continuation_ids):
        step_logits = logits[prompt_len - 1 + pos].float()
        log_probs = torch_mod.log_softmax(step_logits, dim=-1)
        total += float(log_probs[int(tok_id)].item())
    return total


def _first_rank_logprob_by_doc(
    *, model, tokenizer, torch_mod, row: Dict[str, Any], max_input_tokens: int
) -> Dict[int, float]:
    prompt = _build_nfcorpus_prompt(tokenizer, row["input"])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) > max_input_tokens:
        prompt_ids = prompt_ids[-max_input_tokens:]

    relevance = list(row.get("relevance", []))
    if not relevance:
        return {}

    token_map = _candidate_rank_token_ids(tokenizer, len(relevance))
    return {
        doc_id: _sequence_logprob(
            model=model,
            torch_mod=torch_mod,
            input_ids=prompt_ids,
            continuation_ids=token_map[doc_id],
        )
        for doc_id in range(1, len(relevance) + 1)
    }


def _gold_relevant_margin_score(
    *, model, tokenizer, torch_mod, row: Dict[str, Any], max_input_tokens: int
) -> float:
    relevance = list(row.get("relevance", []))
    if not relevance:
        return float("-inf")

    doc_scores = _first_rank_logprob_by_doc(
        model=model,
        tokenizer=tokenizer,
        torch_mod=torch_mod,
        row=row,
        max_input_tokens=max_input_tokens,
    )

    positive_scores = [doc_scores[idx + 1] for idx, rel in enumerate(relevance) if float(rel) > 0]
    negative_scores = [doc_scores[idx + 1] for idx, rel in enumerate(relevance) if float(rel) <= 0]
    if not positive_scores or not negative_scores:
        return float("-inf")

    return float(np.mean(positive_scores) - np.mean(negative_scores))


def _selected_token_positions(scores: Sequence[float], topk_ratio: float) -> List[int]:
    if not scores:
        return []
    topk = max(1, int(len(scores) * topk_ratio))
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    return order[:topk]


def compute_nfcorpus_relevant_doc_deletion(
    config_path: str,
    split: str = "test",
    subset_size: int = 323,
    subset_seed: int = 42,
    topk_ratio: float = 0.2,
    deletion_steps: int = 10,
    output_tag: str | None = None,
) -> str:
    cfg = _load_config(config_path)
    hf = _require_hf()
    torch = hf["torch"]

    rows = _read_jsonl(Path(cfg["data"][f"{split}_file"]))
    if not rows:
        raise RuntimeError("No NFCorpus rows found.")

    rng = random.Random(subset_seed)
    indexed_rows = list(enumerate(rows))
    rng.shuffle(indexed_rows)
    chosen = indexed_rows[: min(subset_size, len(indexed_rows))]
    chosen.sort(key=lambda item: item[0])

    model, tokenizer = _load_model_and_tokenizer(cfg=cfg, hf=hf, attn_implementation="eager")
    max_input_tokens = int(cfg.get("generation", {}).get("max_input_tokens", 1024))

    sample_rows: List[Dict[str, Any]] = []
    curves: List[List[float]] = []
    aopcs: List[float] = []

    for dataset_index, row in chosen:
        row = dict(row)
        row["dataset_index"] = dataset_index
        sample_id = row.get("id") or row.get("sample_id") or f"{split}-{dataset_index}"

        base_score = _gold_relevant_margin_score(
            model=model,
            tokenizer=tokenizer,
            torch_mod=torch,
            row=row,
            max_input_tokens=max_input_tokens,
        )
        _, token_scores = _passage_token_importance(
            model=model,
            tokenizer=tokenizer,
            torch_mod=torch,
            row=row,
            max_input_tokens=max_input_tokens,
        )
        selected = _selected_token_positions(token_scores, topk_ratio=topk_ratio)

        deletion_curve: List[float] = []
        for step in range(1, deletion_steps + 1):
            rec_del = _perturb_passage_tokens(
                row=row,
                tokenizer=tokenizer,
                selected_positions=selected,
                step=step,
                steps=deletion_steps,
                mode="deletion",
            )
            del_score = _gold_relevant_margin_score(
                model=model,
                tokenizer=tokenizer,
                torch_mod=torch,
                row=rec_del,
                max_input_tokens=max_input_tokens,
            )
            deletion_curve.append(float(del_score))

        aopc = float(np.mean([base_score - score for score in deletion_curve]))
        sample_rows.append(
            {
                "sample_id": sample_id,
                "dataset_index": dataset_index,
                "base_score": float(base_score),
                "selected_token_count": len(selected),
                "deletion_curve": deletion_curve,
                "aopc": aopc,
            }
        )
        curves.append(deletion_curve)
        aopcs.append(aopc)

    mean_curve = np.mean(np.asarray(curves, dtype=float), axis=0).tolist() if curves else []
    summary = {
        "task": "nfcorpus",
        "model_name": cfg["model_name"],
        "adapter_dir": cfg.get("adapter_dir"),
        "split": split,
        "num_samples": len(sample_rows),
        "subset_seed": subset_seed,
        "subset_size": subset_size,
        "score_type": "relevant_vs_nonrelevant_first_rank_log_prob_margin",
        "mean_aopc": float(np.mean(aopcs)) if aopcs else None,
        "std_aopc": float(np.std(aopcs)) if aopcs else None,
        "mean_deletion_curve": [float(x) for x in mean_curve],
        "topk_ratio": topk_ratio,
        "steps": deletion_steps,
    }

    output_root = Path(cfg["project"]["output_root"]) / "baseline" / split
    if output_tag:
        output_root = output_root / output_tag
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "relevant_doc_faithfulness.json", summary)
    _write_jsonl(output_root / "relevant_doc_faithfulness_samples.jsonl", sample_rows)
    return str(output_root / "relevant_doc_faithfulness.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute NFCorpus gold relevant document score under deletion.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset-size", type=int, default=323)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--topk-ratio", type=float, default=0.2)
    parser.add_argument("--deletion-steps", type=int, default=10)
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    out = compute_nfcorpus_relevant_doc_deletion(
        config_path=args.config,
        split=args.split,
        subset_size=args.subset_size,
        subset_seed=args.subset_seed,
        topk_ratio=args.topk_ratio,
        deletion_steps=args.deletion_steps,
        output_tag=args.output_tag,
    )
    print(out)


if __name__ == "__main__":
    main()
