"""Deletion-based faithfulness for NFCorpus reranking models.

Scoring function: NDCG@5 degradation under passage-token deletion.
Token importance: attention-based, restricted to passage tokens.
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from .nfcorpus_signals import _build_nfcorpus_prompt


# ── Reranking evaluation helpers (from eval_rerank.py) ──────────────────────

def _parse_ranking(text: str, num_candidates: int = 5) -> List[int]:
    numbers = re.findall(r"\d+", text)
    ranking: List[int] = []
    seen: set = set()
    for n in numbers:
        idx = int(n)
        if 1 <= idx <= num_candidates and idx not in seen:
            ranking.append(idx)
            seen.add(idx)
    for i in range(1, num_candidates + 1):
        if i not in seen:
            ranking.append(i)
    return ranking[:num_candidates]


def _ndcg_at_k(predicted_ranking: List[int], relevance_scores: List[float], k: int = 5) -> float:
    pred_rels = [relevance_scores[idx - 1] for idx in predicted_ranking[:k]]
    ideal_rels = sorted(relevance_scores, reverse=True)[:k]
    def _dcg(rels):
        r = np.array(rels[:k], dtype=float)
        if len(r) == 0:
            return 0.0
        return float(np.sum(r / np.log2(np.arange(1, len(r) + 1) + 1)))
    idcg = _dcg(ideal_rels)
    return _dcg(pred_rels) / idcg if idcg > 0 else 0.0


# ── Text extraction helpers ─────────────────────────────────────────────────

def _extract_passages_text(raw_input: str) -> str:
    """Extract the passages block from an NFCorpus inference prompt."""
    text = raw_input.strip()
    if "### Passages:" in text:
        text = text.split("### Passages:", 1)[-1].strip()
    if "### Ranking:" in text:
        text = text.split("### Ranking:", 1)[0].strip()
    return text


def _replace_passages_text(raw_input: str, new_passages: str) -> str:
    original = _extract_passages_text(raw_input)
    if original in raw_input:
        return raw_input.replace(original, new_passages, 1)
    if "### Passages:" in raw_input:
        prefix, _, rest = raw_input.partition("### Passages:")
        if "### Ranking:" in rest:
            _, _, tail = rest.partition("### Ranking:")
            return f"{prefix}### Passages:\n{new_passages}\n\n### Ranking:{tail}"
        return f"{prefix}### Passages:\n{new_passages}"
    return new_passages


# ── Scoring: generate ranking and compute NDCG@5 ───────────────────────────

def _score_ndcg(
    *, model, tokenizer, torch_mod, row: Dict[str, Any], max_input_tokens: int, max_new_tokens: int,
) -> float:
    prompt = _build_nfcorpus_prompt(tokenizer, row["input"])
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens,
    )
    encoded = {k: v.to(_device(torch_mod)) for k, v in encoded.items()}

    with torch_mod.no_grad():
        out_ids = model.generate(
            **encoded, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        )

    generated = tokenizer.decode(out_ids[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
    relevance = row.get("relevance", [])
    if not relevance:
        return 0.0
    ranking = _parse_ranking(generated, num_candidates=len(relevance))
    return _ndcg_at_k(ranking, relevance, k=5)


# ── Token importance over passage tokens ────────────────────────────────────

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


def _passage_token_importance(
    *, model, tokenizer, torch_mod, row: Dict[str, Any], max_input_tokens: int,
) -> Tuple[List[int], List[float]]:
    prompt = _build_nfcorpus_prompt(tokenizer, row["input"])
    encoded = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens,
    )
    encoded = {k: v.to(_device(torch_mod)) for k, v in encoded.items()}

    with torch_mod.no_grad():
        out = model(**encoded, output_attentions=True)

    full_ids = encoded["input_ids"][0].detach().cpu().tolist()
    passage_ids = tokenizer.encode(_extract_passages_text(row["input"]), add_special_tokens=False)
    span = _find_subsequence(full_ids, passage_ids)
    if span is None:
        start, end = 0, len(full_ids)
    else:
        start, end = span

    if not out.attentions:
        raise RuntimeError("Attention tensors were not returned.")

    layer_tensors = out.attentions[-4:] if len(out.attentions) >= 4 else out.attentions
    token_scores = np.zeros(end - start, dtype=np.float64)
    for att in layer_tensors:
        arr = att[0].detach().float().cpu().numpy()
        key_imp = arr.mean(axis=0).mean(axis=0)[start:end]
        token_scores += key_imp

    token_scores /= max(1, len(layer_tensors))
    passage_token_ids = full_ids[start:end]
    return passage_token_ids, token_scores.tolist()


def _selected_token_positions(scores: Sequence[float], topk_ratio: float) -> List[int]:
    if not scores:
        return []
    topk = max(1, int(len(scores) * topk_ratio))
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    return order[:topk]


# ── Perturbation ────────────────────────────────────────────────────────────

def _perturb_passage_tokens(
    *, row: Dict[str, Any], tokenizer, selected_positions: Sequence[int],
    step: int, steps: int, mode: str,
) -> Dict[str, Any]:
    passage_token_ids = tokenizer.encode(_extract_passages_text(row["input"]), add_special_tokens=False)
    cut = max(1, int(len(selected_positions) * (step / steps)))
    chosen = set(selected_positions[:cut])

    if mode == "deletion":
        kept_ids = [tok for idx, tok in enumerate(passage_token_ids) if idx not in chosen]
    elif mode == "insertion":
        kept_ids = [tok for idx, tok in enumerate(passage_token_ids) if idx in chosen]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if not kept_ids:
        kept_ids = passage_token_ids[:1]

    new_text = tokenizer.decode(kept_ids, skip_special_tokens=True).strip()
    if not new_text:
        new_text = _extract_passages_text(row["input"])[:32]

    mutated = dict(row)
    mutated["input"] = _replace_passages_text(row["input"], new_text)
    return mutated


# ── Per-row faithfulness ────────────────────────────────────────────────────

def _faithfulness_for_row(
    *, model, tokenizer, torch_mod, row: Dict[str, Any],
    topk_ratio: float, steps: int, max_input_tokens: int, max_new_tokens: int,
) -> Dict[str, Any]:
    base_ndcg = _score_ndcg(
        model=model, tokenizer=tokenizer, torch_mod=torch_mod,
        row=row, max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
    )
    _, token_scores = _passage_token_importance(
        model=model, tokenizer=tokenizer, torch_mod=torch_mod,
        row=row, max_input_tokens=max_input_tokens,
    )
    selected = _selected_token_positions(token_scores, topk_ratio=topk_ratio)

    deletion_curve: List[float] = []
    for step in range(1, steps + 1):
        rec_del = _perturb_passage_tokens(
            row=row, tokenizer=tokenizer, selected_positions=selected,
            step=step, steps=steps, mode="deletion",
        )
        del_ndcg = _score_ndcg(
            model=model, tokenizer=tokenizer, torch_mod=torch_mod,
            row=rec_del, max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
        )
        deletion_curve.append(float(del_ndcg))

    aopc = float(np.mean([base_ndcg - score for score in deletion_curve]))
    return {
        "sample_id": row.get("id") or row.get("sample_id"),
        "base_ndcg": float(base_ndcg),
        "passage_token_count": len(token_scores),
        "selected_token_count": len(selected),
        "aopc": aopc,
        "deletion_curve": deletion_curve,
    }


# ── Main driver ─────────────────────────────────────────────────────────────

def compute_nfcorpus_faithfulness(
    config_path: str,
    split: str = "test",
    output_tag: str | None = None,
    subset_size: int = 323,
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
        cfg=cfg, hf=hf, attn_implementation="eager",
    )

    max_input_tokens = int(cfg.get("generation", {}).get("max_input_tokens", 1024))
    max_new_tokens = int(cfg.get("generation", {}).get("max_new_tokens", 64))
    output_root = Path(cfg["project"]["output_root"]) / "baseline" / split
    if output_tag:
        output_root = output_root / output_tag
    output_root.mkdir(parents=True, exist_ok=True)

    sample_rows: List[Dict[str, Any]] = []
    del_curves: List[List[float]] = []
    aopcs: List[float] = []

    for dataset_index, row in chosen:
        row = dict(row)
        row["dataset_index"] = dataset_index
        if "sample_id" not in row:
            row["sample_id"] = row.get("id") or f"{split}-{dataset_index}"

        result = _faithfulness_for_row(
            model=model, tokenizer=tokenizer, torch_mod=torch,
            row=row, topk_ratio=topk_ratio, steps=deletion_steps,
            max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens,
        )
        result["dataset_index"] = dataset_index
        sample_rows.append(result)
        del_curves.append(result["deletion_curve"])
        aopcs.append(float(result["aopc"]))

    mean_del = np.mean(np.array(del_curves), axis=0).tolist() if del_curves else []
    summary = {
        "task": "nfcorpus",
        "model_name": cfg["model_name"],
        "adapter_dir": cfg.get("adapter_dir"),
        "split": split,
        "num_samples": len(sample_rows),
        "subset_seed": subset_seed,
        "subset_size": subset_size,
        "score_type": "ndcg@5_degradation",
        "mean_aopc": float(np.mean(aopcs)) if aopcs else None,
        "std_aopc": float(np.std(aopcs)) if aopcs else None,
        "mean_deletion_curve": [float(x) for x in mean_del],
        "topk_ratio": topk_ratio,
        "steps": deletion_steps,
    }

    _write_json(output_root / "faithfulness.json", summary)
    _write_jsonl(output_root / "faithfulness_samples.jsonl", sample_rows)
    return str(output_root / "faithfulness.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute NFCorpus deletion-based faithfulness (NDCG@5).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--subset-size", type=int, default=323)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--topk-ratio", type=float, default=0.2)
    parser.add_argument("--deletion-steps", type=int, default=10)
    args = parser.parse_args()

    out = compute_nfcorpus_faithfulness(
        config_path=args.config, split=args.split, output_tag=args.output_tag,
        subset_size=args.subset_size, subset_seed=args.subset_seed,
        topk_ratio=args.topk_ratio, deletion_steps=args.deletion_steps,
    )
    print(out)


if __name__ == "__main__":
    main()
