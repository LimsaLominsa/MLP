"""
run_casehold_preprocessing.py
Full preprocessing pipeline for the CaseHOLD dataset.

Usage:
  cd "casehold preprocessing"
  python src/run_casehold_preprocessing.py

Output layout (output/casehold/):
  ├── train_mc.jsonl          Multiple-choice SFT format — train
  ├── val_mc.jsonl            Multiple-choice SFT format — validation
  ├── test_mc.jsonl           Multiple-choice SFT format — test
  ├── train_cls.jsonl         Flat classification format — train
  ├── val_cls.jsonl           Flat classification format — validation
  ├── test_cls.jsonl          Flat classification format — test
  ├── train_cleaned.jsonl     Cleaned raw format — train
  ├── val_cleaned.jsonl       Cleaned raw format — validation
  ├── test_cleaned.jsonl      Cleaned raw format — test
  └── casehold_distribution.png
"""

import sys
from pathlib import Path

# Allow direct imports from src/
sys.path.insert(0, str(Path(__file__).parent))

from casehold_cleaning   import load_and_clean_casehold
from casehold_analysis   import print_stats, plot_casehold_stats
from casehold_formatting import (format_as_multiple_choice,
                                 format_as_classification,
                                 save_jsonl)


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output" / "casehold"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download and clean ────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1  Download and clean CaseHOLD")
    print("=" * 60)
    splits = load_and_clean_casehold()

    # ── Step 2: Statistics and visualization ──────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2  Descriptive statistics")
    print("=" * 60)
    for name, df in splits.items():
        print_stats(df, name.upper())

    print("\n" + "-" * 60)
    print("Generating distribution plots...")
    plot_casehold_stats(splits, OUTPUT_DIR)

    # ── Step 3: Format ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3  Format and save")
    print("=" * 60)

    for split_name, df in splits.items():
        print(f"\n  [{split_name}]")

        # Multiple-choice SFT
        mc = format_as_multiple_choice(df)
        save_jsonl(mc, OUTPUT_DIR / f"{split_name}_mc.jsonl")

        # Classification flat
        cls = format_as_classification(df)
        save_jsonl(cls, OUTPUT_DIR / f"{split_name}_cls.jsonl")

        # Cleaned raw
        out_raw = OUTPUT_DIR / f"{split_name}_cleaned.jsonl"
        df.to_json(out_raw, orient='records', lines=True, force_ascii=False)
        size_mb = out_raw.stat().st_size / 1024 / 1024
        print(f"  Saved cleaned raw → {out_raw.name}  ({size_mb:.1f} MB)")

    # ── Step 4: Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4  Complete — output file summary")
    print("=" * 60)
    print(f"  Directory: {OUTPUT_DIR}\n")
    files = sorted(OUTPUT_DIR.glob("*"))
    total_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
    for f in files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.name:<45} {size_mb:6.1f} MB")
    print(f"\n  Total: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
