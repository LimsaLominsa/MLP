"""
pubmed_formatting.py
Download PubMed Summarization from HuggingFace and format into SFT JSONL.

Dataset: ccdv/pubmed-summarization (~133k train / 6.6k val / 6.7k test)
Task:    article → abstract (abstractive summarization)

To control training time, we sample a subset of the training set (default 20k).
Validation and test sets are used in full.

Usage:
    python src/data/pubmed/pubmed_formatting.py [--train_samples 20000] [--output_dir data/pubmed]
"""

import json
import argparse
import random
from pathlib import Path
from datasets import load_dataset


# ==================== Prompt Templates ====================
TRAIN_TEMPLATE = (
    "Below is a biomedical research article. Write a concise summary.\n\n"
    "### Article:\n{article}\n\n"
    "### Summary:\n{abstract}"
)

INFERENCE_TEMPLATE = (
    "Below is a biomedical research article. Write a concise summary.\n\n"
    "### Article:\n{article}\n\n"
    "### Summary:\n"
)

MAX_ARTICLE_CHARS = 12000  # rough character limit before tokenization


def format_record(article: str, abstract: str) -> dict:
    """Format a single PubMed record into SFT structure."""
    article_trunc = article[:MAX_ARTICLE_CHARS]
    return {
        "text": TRAIN_TEMPLATE.format(article=article_trunc, abstract=abstract),
        "input": INFERENCE_TEMPLATE.format(article=article_trunc),
        "output": abstract,
    }


def save_jsonl(records: list, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {filepath}")


def main():
    parser = argparse.ArgumentParser(description="PubMed Summarization data formatting")
    parser.add_argument("--train_samples", type=int, default=20000,
                        help="Number of training samples to use (default: 20000)")
    parser.add_argument("--output_dir", type=str, default="data/pubmed",
                        help="Output directory for JSONL files")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)

    # ── Download from HuggingFace ──
    print("Downloading PubMed Summarization dataset...")
    ds = load_dataset("ccdv/pubmed-summarization")
    print(f"  train: {len(ds['train']):,}  |  val: {len(ds['validation']):,}  |  test: {len(ds['test']):,}")

    # ── Sample training set ──
    train_indices = list(range(len(ds["train"])))
    if args.train_samples < len(train_indices):
        random.shuffle(train_indices)
        train_indices = sorted(train_indices[:args.train_samples])
        print(f"  Sampled {args.train_samples:,} training examples (seed={args.seed})")

    # ── Format each split ──
    print("\nFormatting training set...")
    train_records = []
    for idx in train_indices:
        row = ds["train"][idx]
        train_records.append(format_record(row["article"], row["abstract"]))
    save_jsonl(train_records, output_dir / "train_sft.jsonl")

    print("Formatting validation set...")
    val_records = [
        format_record(row["article"], row["abstract"])
        for row in ds["validation"]
    ]
    save_jsonl(val_records, output_dir / "val_sft.jsonl")

    print("Formatting test set...")
    test_records = [
        format_record(row["article"], row["abstract"])
        for row in ds["test"]
    ]
    save_jsonl(test_records, output_dir / "test_sft.jsonl")

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"PubMed SFT data ready:")
    print(f"  Train: {len(train_records):,}  →  {output_dir / 'train_sft.jsonl'}")
    print(f"  Val:   {len(val_records):,}  →  {output_dir / 'val_sft.jsonl'}")
    print(f"  Test:  {len(test_records):,}  →  {output_dir / 'test_sft.jsonl'}")


if __name__ == "__main__":
    main()
