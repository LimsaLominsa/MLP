"""
smoke_test.py
Local mini pipeline validation — runs entirely on CPU, no GPU needed.

Validates end-to-end:
  1. Data loading (both datasets)
  2. Tokenizer compatibility (both models)
  3. Token length statistics (print recommended max_length values)
  4. Prompt format sanity check

Usage:
  cd legal-llm-finetuning
  python src/train/smoke_test.py

Takes ~3-5 minutes on CPU.
"""

import json
import sys
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MODELS = {
    "Qwen":  "Qwen/Qwen2.5-1.5B-Instruct",
    "Llama": "meta-llama/Llama-3.2-1B-Instruct",
}

DATA = {
    "billsum_train":    REPO_ROOT / "data/billsum/train_sft.jsonl",
    "billsum_val":      REPO_ROOT / "data/billsum/val_sft.jsonl",
    "casehold_train":   REPO_ROOT / "data/casehold/train_mc.jsonl",
    "casehold_val":     REPO_ROOT / "data/casehold/validation_mc.jsonl",
    "casehold_test":    REPO_ROOT / "data/casehold/test_mc.jsonl",
}

N_SAMPLES = 200   # number of samples to tokenize for length stats


# ── helpers ───────────────────────────────────────────────────────────────────

def load_n(path: Path, n: int) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            records.append(json.loads(line))
    return records


def required_fields(records: list, fields: list, name: str):
    missing = [f for f in fields for r in records if f not in r]
    if missing:
        print(f"  [FAIL] {name} missing fields: {set(missing)}")
        return False
    print(f"  [OK]   {name} — all required fields present")
    return True


def token_stats(records: list, tokenizer, field: str) -> dict:
    lengths = [
        len(tokenizer(r[field], truncation=False)["input_ids"])
        for r in records
    ]
    arr = np.array(lengths)
    return {
        "mean":   int(arr.mean()),
        "p90":    int(np.percentile(arr, 90)),
        "p95":    int(np.percentile(arr, 95)),
        "max":    int(arr.max()),
    }


def nearest_power_of_2(n: int) -> int:
    p = 64
    while p < n:
        p *= 2
    return p


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    errors = 0

    print("=" * 65)
    print("  Smoke Test — legal-llm-finetuning")
    print("=" * 65)

    # ── 1. Data file existence ─────────────────────────────────────────────
    print("\n[1] Checking data files...")
    for name, path in DATA.items():
        if path.exists():
            n = sum(1 for _ in open(path, encoding="utf-8"))
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  [OK]   {name:<22} {n:>7,} records   {size_mb:6.1f} MB")
        else:
            print(f"  [FAIL] {name:<22} NOT FOUND — {path}")
            errors += 1

    if errors:
        print("\nAbort: fix missing data files before continuing.")
        sys.exit(1)

    # ── 2. Field structure ─────────────────────────────────────────────────
    print("\n[2] Checking record field structure...")
    bs_records = load_n(DATA["billsum_train"], 5)
    ch_records = load_n(DATA["casehold_train"], 5)

    required_fields(bs_records, ["text", "input", "output"], "billsum_train")
    required_fields(ch_records, ["text", "input", "output", "label"], "casehold_train")

    # CaseHOLD label range
    labels = [r["label"] for r in ch_records]
    if all(l in range(5) for l in labels):
        print("  [OK]   casehold labels in range 0-4")
    else:
        print(f"  [FAIL] casehold labels out of range: {labels}")
        errors += 1

    # ── 3. Tokenizer + length stats ────────────────────────────────────────
    print("\n[3] Tokenizer compatibility + token length statistics...")
    print(f"    (sampling {N_SAMPLES} records per dataset)")

    recommendations = {}

    for short_name, model_name in MODELS.items():
        print(f"\n  Model: {model_name}")
        try:
            tok = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
        except Exception as e:
            # Llama is gated — requires HuggingFace login (huggingface-cli login)
            # Treat as a warning, not a hard failure
            print(f"  [WARN]  Could not load tokenizer (gated or network error): {model_name}")
            print(f"          Run: huggingface-cli login")
            print(f"          Skipping token stats for this model.")
            continue

        # BillSum — input length
        bs_sample = load_n(DATA["billsum_train"], N_SAMPLES)
        bs_stats  = token_stats(bs_sample, tok, "input")
        bs_rec    = nearest_power_of_2(bs_stats["p95"])
        print(f"    BillSum  input  p95={bs_stats['p95']:,}  max={bs_stats['max']:,}"
              f"  → recommend max_input_length={bs_rec}")

        # CaseHOLD — input (prompt+options, no answer)
        ch_sample = load_n(DATA["casehold_train"], N_SAMPLES)
        ch_stats  = token_stats(ch_sample, tok, "input")
        ch_rec    = nearest_power_of_2(ch_stats["p95"])
        print(f"    CaseHOLD input  p95={ch_stats['p95']:,}  max={ch_stats['max']:,}"
              f"  → recommend max_length={ch_rec}")

        # CaseHOLD — output (single letter A-E, always tiny)
        ch_out    = token_stats(ch_sample, tok, "output")
        print(f"    CaseHOLD output p95={ch_out['p95']:,}  (answer letter only)")

        recommendations[short_name] = {
            "billsum_max_input":    bs_rec,
            "casehold_max_length":  ch_rec,
        }

    # ── 4. Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if errors:
        print(f"Smoke test FAILED — {errors} error(s) found.")
    else:
        print("Smoke test PASSED — all checks OK.")

    print("\n★ Recommended config values:")
    for model, vals in recommendations.items():
        print(f"\n  [{model}]")
        print(f"    configs/lora_billsum_{model.lower()}.yaml")
        print(f"      data.max_input_length:  {vals['billsum_max_input']}")
        print(f"    configs/lora_casehold_{model.lower()}.yaml")
        print(f"      data.max_length:        {vals['casehold_max_length']}")

    print("=" * 65)
    return errors


if __name__ == "__main__":
    sys.exit(main())
