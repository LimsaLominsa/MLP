"""
run_preprocessing.py
Main entry point for the BillSum data preprocessing pipeline.

Steps:
  1. Load raw JSONL data (US train, US test, CA test)
  2. Clean and filter all datasets
  3. Print descriptive statistics
  4. Generate length distribution plots
  5. Split a validation set from the training data
  6. Format all datasets for SFT (Supervised Fine-Tuning)
  7. Save both cleaned originals and SFT-formatted outputs

Usage:
  cd "billsum data cleaning"
  python src/run_preprocessing.py
"""

import sys
from pathlib import Path

# Add src/ to import path so modules can be imported directly
sys.path.insert(0, str(Path(__file__).parent))

from data_cleaning import load_jsonl, preprocess_dataframe
from data_analysis import print_stats, plot_length_distribution
from data_formatting import format_for_sft, save_jsonl


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "billsum_v4_1"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    # ========== Step 1: Load raw data ==========
    print("=" * 60)
    print("Step 1: Loading raw data...")
    train_df = load_jsonl(data_dir / "us_train_data_final_OFFICIAL.jsonl")
    test_df  = load_jsonl(data_dir / "us_test_data_final_OFFICIAL.jsonl")
    ca_df    = load_jsonl(data_dir / "ca_test_data_final_OFFICIAL.jsonl")
    print(f"  Loaded: train={len(train_df)}, test_us={len(test_df)}, test_ca={len(ca_df)}")

    # ========== Step 2: Clean & Filter ==========
    print("\n" + "=" * 60)
    print("Step 2: Cleaning and filtering...")
    train_df = preprocess_dataframe(train_df, "US Train")
    test_df  = preprocess_dataframe(test_df,  "US Test")
    ca_df    = preprocess_dataframe(ca_df,    "CA Test")

    # ========== Step 3: Descriptive Statistics ==========
    print("\n" + "=" * 60)
    print("Step 3: Descriptive statistics")
    print_stats(train_df, "US Train")
    print_stats(test_df,  "US Test")
    print_stats(ca_df,    "CA Test")

    # ========== Step 4: Length Distribution Plots ==========
    print("\n" + "=" * 60)
    print("Step 4: Generating length distribution plots...")
    plot_length_distribution(train_df, test_df, ca_df, output_dir)

    # ========== Step 5: Split validation set ==========
    print("\n" + "=" * 60)
    print("Step 5: Splitting validation set (5% of training data)...")
    val_df   = train_df.sample(frac=0.05, random_state=42)
    train_df = train_df.drop(val_df.index).reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    # ========== Step 6: Format for SFT ==========
    print("\n" + "=" * 60)
    print("Step 6: Formatting for SFT...")
    train_records = format_for_sft(train_df)
    val_records   = format_for_sft(val_df)
    test_records  = format_for_sft(test_df)
    ca_records    = format_for_sft(ca_df)

    # ========== Step 7: Save outputs ==========
    print("\n" + "=" * 60)
    print("Step 7: Saving outputs...")

    # SFT-formatted data (for model training/evaluation)
    save_jsonl(train_records, output_dir / "train_sft.jsonl")
    save_jsonl(val_records,   output_dir / "val_sft.jsonl")
    save_jsonl(test_records,  output_dir / "test_us_sft.jsonl")
    save_jsonl(ca_records,    output_dir / "test_ca_sft.jsonl")

    # Cleaned original format (for other analysis purposes)
    train_df.to_json(output_dir / "train_cleaned.jsonl",
                     orient='records', lines=True, force_ascii=False)
    test_df.to_json(output_dir / "test_us_cleaned.jsonl",
                    orient='records', lines=True, force_ascii=False)
    ca_df.to_json(output_dir / "test_ca_cleaned.jsonl",
                  orient='records', lines=True, force_ascii=False)
    val_df.to_json(output_dir / "val_cleaned.jsonl",
                   orient='records', lines=True, force_ascii=False)

    # ========== Done ==========
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Output directory: {output_dir}")
    print("Output files:")
    for f in sorted(output_dir.glob("*")):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:30s} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
