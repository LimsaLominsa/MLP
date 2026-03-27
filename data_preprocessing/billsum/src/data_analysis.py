"""
data_analysis.py
- Print descriptive statistics for cleaned datasets
- Generate length distribution histograms
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI window)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def print_stats(df: pd.DataFrame, name: str):
    """Print descriptive statistics for a cleaned DataFrame."""
    print(f"\n===== {name} (after cleaning) =====")
    print(f"Sample count: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    print(f"\nText length (chars):")
    print(df['text_len'].describe().to_string())

    print(f"\nSummary length (chars):")
    print(df['summary_len'].describe().to_string())

    # Compression ratio: summary / text
    ratio = df['summary_len'] / df['text_len']
    print(f"\nCompression ratio (summary/text):")
    print(f"  mean: {ratio.mean():.4f} | median: {ratio.median():.4f} | "
          f"min: {ratio.min():.4f} | max: {ratio.max():.4f}")


def plot_length_distribution(train_df, test_df, ca_df, output_dir="output"):
    """
    Generate a 2x3 grid of histograms showing:
    - Row 1: Text length distribution for each dataset
    - Row 2: Summary length distribution for each dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    datasets = [
        (train_df, "US Train"),
        (test_df,  "US Test"),
        (ca_df,    "CA Test"),
    ]

    for i, (df, name) in enumerate(datasets):
        # Text length histogram
        axes[0][i].hist(df['text_len'], bins=50, edgecolor='black', alpha=0.7,
                        color='steelblue')
        axes[0][i].set_title(f'{name} - Text Length (chars)')
        axes[0][i].set_xlabel('Character Count')
        axes[0][i].set_ylabel('Frequency')
        axes[0][i].axvline(df['text_len'].median(), color='red',
                           linestyle='--', label=f"median={df['text_len'].median():.0f}")
        axes[0][i].legend()

        # Summary length histogram
        axes[1][i].hist(df['summary_len'], bins=50, edgecolor='black', alpha=0.7,
                        color='darkorange')
        axes[1][i].set_title(f'{name} - Summary Length (chars)')
        axes[1][i].set_xlabel('Character Count')
        axes[1][i].set_ylabel('Frequency')
        axes[1][i].axvline(df['summary_len'].median(), color='red',
                           linestyle='--', label=f"median={df['summary_len'].median():.0f}")
        axes[1][i].legend()

    plt.tight_layout()
    save_path = output_dir / "length_distribution.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nLength distribution plot saved to: {save_path}")
