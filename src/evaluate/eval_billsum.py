"""
eval_billsum.py
Evaluate model-generated summaries against ground truth using ROUGE and BERTScore.

Usage:
  python src/evaluate/eval_billsum.py \
      --predictions outputs/lora_billsum_qwen/predictions_test_us.jsonl \
      --output      outputs/lora_billsum_qwen/eval_test_us.json

Predictions file format (one JSON per line):
  {"bill_id": "...", "prediction": "generated summary...", "reference": "ground truth..."}
"""

import json
import argparse
import numpy as np
from pathlib import Path


def load_jsonl(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def compute_rouge(predictions: list, references: list) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L (F1) for each sample."""
    from rouge_score import rouge_scorer as rs
    scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    r1, r2, rL = [], [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rL.append(s['rougeL'].fmeasure)

    def stats(lst):
        a = np.array(lst)
        return {"mean": float(a.mean()), "std": float(a.std())}

    return {"rouge1": stats(r1), "rouge2": stats(r2), "rougeL": stats(rL)}


def compute_bertscore(predictions: list, references: list,
                      model_type: str = "roberta-large",
                      batch_size: int = 16) -> dict:
    """Compute BERTScore F1 (semantic similarity)."""
    import bert_score
    _, _, F1 = bert_score.score(
        predictions, references,
        model_type=model_type,
        batch_size=batch_size,
        lang="en",
        verbose=True,
    )
    f1 = F1.tolist()
    return {"bertscore_f1": {
        "mean": float(np.mean(f1)),
        "std":  float(np.std(f1)),
    }}


def evaluate(predictions_file: str,
             output_file: str = None,
             use_bertscore: bool = True):

    records     = load_jsonl(predictions_file)
    predictions = [r["prediction"] for r in records]
    references  = [r["reference"]  for r in records]

    print(f"Evaluating {len(predictions)} samples...")

    # ROUGE
    results = compute_rouge(predictions, references)

    # BERTScore
    if use_bertscore:
        results.update(compute_bertscore(predictions, references))

    # Print summary
    print("\n" + "=" * 50)
    print("BillSum Evaluation Results")
    print("=" * 50)
    for metric, vals in results.items():
        print(f"  {metric:<20} {vals['mean']:.4f} ± {vals['std']:.4f}")

    # Save
    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved → {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output",      default=None)
    parser.add_argument("--no_bertscore", action="store_true")
    args = parser.parse_args()

    evaluate(
        predictions_file = args.predictions,
        output_file      = args.output,
        use_bertscore    = not args.no_bertscore,
    )
