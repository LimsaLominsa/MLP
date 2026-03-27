"""
check_token_length.py
Compute token length statistics for CaseHOLD MC-format prompts
using both target model tokenizers.

Run locally on CPU — no GPU required.
Helps determine the appropriate max_length value for CaseHOLD training configs.

Usage:
  python src/evaluate/check_token_length.py
"""

import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer


MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]

# Paths to the processed MC-format files (relative to repo root)
REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
CASEHOLD_MC = REPO_ROOT / "data/casehold/output/train_mc.jsonl"
BILLSUM_SFT = REPO_ROOT / "data/billsum/output/train_sft.jsonl"


def compute_stats(jsonl_path: Path,
                  tokenizer,
                  field: str,
                  n_samples: int = 2000) -> dict:
    """
    Tokenize the specified field for up to n_samples records.
    Returns length percentile statistics.
    """
    lengths = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            text = json.loads(line).get(field, "")
            ids  = tokenizer(text, truncation=False)["input_ids"]
            lengths.append(len(ids))

    arr = np.array(lengths)
    return {
        "n":      int(len(arr)),
        "mean":   int(arr.mean()),
        "median": int(np.median(arr)),
        "p90":    int(np.percentile(arr, 90)),
        "p95":    int(np.percentile(arr, 95)),
        "p99":    int(np.percentile(arr, 99)),
        "max":    int(arr.max()),
    }


def recommend_max_length(p95: int) -> int:
    """Round p95 up to the nearest power of 2."""
    power = 64
    while power < p95:
        power *= 2
    return power


def main():
    print("=" * 65)
    print("Token Length Analysis")
    print("=" * 65)

    for model_name in MODELS:
        print(f"\nTokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # ── CaseHOLD ──
        print(f"\n  [CaseHOLD]  {CASEHOLD_MC}")
        stats = compute_stats(CASEHOLD_MC, tokenizer,
                              field="input", n_samples=2000)
        rec   = recommend_max_length(stats["p95"])
        print(f"  samples : {stats['n']:,}")
        print(f"  mean    : {stats['mean']:,}  |  median : {stats['median']:,}")
        print(f"  p90     : {stats['p90']:,}  |  p95    : {stats['p95']:,}")
        print(f"  p99     : {stats['p99']:,}  |  max    : {stats['max']:,}")
        print(f"  ★ Recommended max_length → {rec}")

        # ── BillSum input portion ──
        print(f"\n  [BillSum input]  {BILLSUM_SFT}")
        stats_b = compute_stats(BILLSUM_SFT, tokenizer,
                                field="input", n_samples=2000)
        rec_b   = recommend_max_length(stats_b["p95"])
        print(f"  samples : {stats_b['n']:,}")
        print(f"  mean    : {stats_b['mean']:,}  |  median : {stats_b['median']:,}")
        print(f"  p90     : {stats_b['p90']:,}  |  p95    : {stats_b['p95']:,}")
        print(f"  p99     : {stats_b['p99']:,}  |  max    : {stats_b['max']:,}")
        print(f"  ★ Recommended max_input_length → {rec_b}")

    print("\n" + "=" * 65)
    print("Fill in the recommended values in:")
    print("  configs/lora_casehold_qwen.yaml   → data.max_length")
    print("  configs/lora_casehold_llama.yaml  → data.max_length")
    print("=" * 65)


if __name__ == "__main__":
    main()
