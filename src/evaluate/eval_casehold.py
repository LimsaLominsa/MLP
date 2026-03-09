"""
eval_casehold.py
Evaluate model predictions on the CaseHOLD multiple-choice task.

Usage:
  python src/evaluate/eval_casehold.py \
      --predictions outputs/lora_casehold_qwen/predictions_test.jsonl \
      --output      outputs/lora_casehold_qwen/eval_test.json

Predictions file format (one JSON per line):
  {"example_id": "...", "prediction": "C", "reference": "C", "label": 2}

'prediction' and 'reference' should be option letters (A-E)
or integer indices (0-4) — both formats are accepted.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter


LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
IDX_TO_LETTER = {v: k for k, v in LETTER_TO_IDX.items()}


def load_jsonl(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def normalize(value) -> int:
    """Convert a prediction (letter or int) to integer index."""
    if isinstance(value, int):
        return value
    s = str(value).strip().upper()
    if s in LETTER_TO_IDX:
        return LETTER_TO_IDX[s]
    try:
        return int(s)
    except ValueError:
        return -1   # invalid prediction


def evaluate(predictions_file: str, output_file: str = None) -> dict:
    records = load_jsonl(predictions_file)

    pred_labels = [normalize(r["prediction"]) for r in records]
    true_labels = [normalize(r["reference"])  for r in records]

    n = len(records)
    correct = sum(p == t for p, t in zip(pred_labels, true_labels))
    accuracy = correct / n

    # Per-class accuracy (option 0–4)
    per_class = {}
    for cls in range(5):
        cls_indices = [i for i, t in enumerate(true_labels) if t == cls]
        if cls_indices:
            cls_correct = sum(pred_labels[i] == cls for i in cls_indices)
            per_class[IDX_TO_LETTER[cls]] = {
                "n":        len(cls_indices),
                "correct":  cls_correct,
                "accuracy": cls_correct / len(cls_indices),
            }

    # Invalid prediction count
    invalid = sum(1 for p in pred_labels if p == -1)

    results = {
        "n_samples":   n,
        "accuracy":    accuracy,
        "n_correct":   correct,
        "n_invalid":   invalid,
        "per_class":   per_class,
    }

    # Print summary
    print("\n" + "=" * 50)
    print("CaseHOLD Evaluation Results")
    print("=" * 50)
    print(f"  Samples     : {n:,}")
    print(f"  Accuracy    : {accuracy:.4f}  ({correct}/{n})")
    if invalid > 0:
        print(f"  Invalid pred: {invalid}  (model output could not be parsed)")
    print(f"\n  Per-class accuracy:")
    for letter, d in per_class.items():
        print(f"    {letter}: {d['accuracy']:.4f}  ({d['correct']}/{d['n']})")

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
    args = parser.parse_args()

    evaluate(
        predictions_file = args.predictions,
        output_file      = args.output,
    )
