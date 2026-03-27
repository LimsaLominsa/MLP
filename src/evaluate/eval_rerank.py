"""
eval_rerank.py
Evaluate passage reranking predictions using NDCG@k and MAP@k.

Predictions file format (one JSON per line):
  {"prediction": "2, 4, 1, 5, 3", "reference": "1, 2, 3, 4, 5",
   "relevance": [0, 2, 0, 1, 0], "id": "..."}

Usage:
  python src/evaluate/eval_rerank.py \
      --predictions outputs/lora_nfcorpus_qwen/predictions_test.jsonl \
      --output      results/nfcorpus/lora_qwen_test.json
"""

import json
import re
import argparse
import numpy as np
from pathlib import Path


def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def parse_ranking(text: str, num_candidates: int = 5) -> list:
    """Parse a ranking string like '2, 4, 1, 5, 3' into a list of ints."""
    numbers = re.findall(r"\d+", text)
    ranking = []
    seen = set()
    for n in numbers:
        idx = int(n)
        if 1 <= idx <= num_candidates and idx not in seen:
            ranking.append(idx)
            seen.add(idx)
    # Pad with missing numbers if model output is incomplete
    for i in range(1, num_candidates + 1):
        if i not in seen:
            ranking.append(i)
    return ranking[:num_candidates]


def dcg_at_k(relevances: list, k: int) -> float:
    """Discounted Cumulative Gain @ k."""
    rel = np.array(relevances[:k], dtype=float)
    if len(rel) == 0:
        return 0.0
    discounts = np.log2(np.arange(1, len(rel) + 1) + 1)
    return float(np.sum(rel / discounts))


def ndcg_at_k(predicted_ranking: list, relevance_scores: list, k: int = 5) -> float:
    """NDCG@k: predicted_ranking is 1-based indices, relevance_scores[i] = score for candidate i+1."""
    pred_rels = [relevance_scores[idx - 1] for idx in predicted_ranking[:k]]
    ideal_rels = sorted(relevance_scores, reverse=True)[:k]

    dcg = dcg_at_k(pred_rels, k)
    idcg = dcg_at_k(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(predicted_ranking: list, relevance_scores: list, k: int = 5) -> float:
    """AP@k with binary relevance (score > 0 = relevant)."""
    pred_rels = [relevance_scores[idx - 1] for idx in predicted_ranking[:k]]

    num_relevant = 0
    sum_precision = 0.0
    for i, rel in enumerate(pred_rels):
        if rel > 0:
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)

    total_relevant = sum(1 for r in relevance_scores if r > 0)
    if total_relevant == 0:
        return 0.0
    return sum_precision / min(total_relevant, k)


def evaluate(predictions_file: str, output_file: str = None, k: int = 5):
    records = load_jsonl(predictions_file)
    print(f"Evaluating {len(records)} reranking predictions...")

    ndcg_scores = []
    map_scores = []
    parse_failures = 0

    for rec in records:
        prediction = rec.get("prediction", "")
        relevance = rec.get("relevance", [])

        if not relevance:
            continue

        num_candidates = len(relevance)
        ranking = parse_ranking(prediction, num_candidates)

        if len(set(ranking)) < num_candidates:
            parse_failures += 1

        ndcg_scores.append(ndcg_at_k(ranking, relevance, k))
        map_scores.append(average_precision_at_k(ranking, relevance, k))

    results = {
        "ndcg@5": {
            "mean": float(np.mean(ndcg_scores)),
            "std":  float(np.std(ndcg_scores)),
        },
        "map@5": {
            "mean": float(np.mean(map_scores)),
            "std":  float(np.std(map_scores)),
        },
        "num_samples": len(ndcg_scores),
        "parse_failures": parse_failures,
    }

    print(f"\n{'='*50}")
    print("NFCorpus Reranking Evaluation Results")
    print("=" * 50)
    for metric in ["ndcg@5", "map@5"]:
        vals = results[metric]
        print(f"  {metric:<20} {vals['mean']:.4f} ± {vals['std']:.4f}")
    print(f"  Samples: {results['num_samples']}")
    if parse_failures:
        print(f"  Parse failures: {parse_failures}")

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    evaluate(args.predictions, args.output, args.k)
