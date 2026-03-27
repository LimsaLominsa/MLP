"""
casehold_analysis.py
Descriptive statistics and visualization for the cleaned CaseHOLD dataset.
"""

import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — no GUI window

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def print_stats(df: pd.DataFrame, name: str) -> None:
    """Print key statistics for a single CaseHOLD split."""
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    print(f"  Samples : {len(df):,}")

    # Label distribution
    print(f"\n  Label distribution (correct option index):")
    counts = df['label'].value_counts().sort_index()
    for lbl, cnt in counts.items():
        bar = '█' * int(cnt / len(df) * 40)
        print(f"    [{int(lbl)}] {cnt:>6,}  ({cnt / len(df) * 100:5.1f}%)  {bar}")

    # Prompt length
    pl = df['prompt_len']
    print(f"\n  citing_prompt length (chars):")
    print(f"    mean   = {pl.mean():,.0f}")
    print(f"    median = {pl.median():,.0f}")
    print(f"    p90    = {np.percentile(pl, 90):,.0f}")
    print(f"    max    = {pl.max():,.0f}")

    # Correct holding length
    cl = df['correct_holding'].str.len()
    print(f"\n  correct holding length (chars):")
    print(f"    mean   = {cl.mean():,.0f}")
    print(f"    median = {cl.median():,.0f}")
    print(f"    max    = {cl.max():,.0f}")


def plot_casehold_stats(splits: dict,
                        output_dir) -> None:
    """
    Generate a 2-row visualization for all splits:
      Row 0: citing_prompt length histograms
      Row 1: label (correct option) distribution bar charts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_names = list(splits.keys())
    n = len(split_names)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 10))

    for i, name in enumerate(split_names):
        df = splits[name]

        # ── Prompt length histogram ──────────────────────────────────────────
        axes[0][i].hist(df['prompt_len'], bins=50,
                        edgecolor='black', alpha=0.75, color='steelblue')
        med = df['prompt_len'].median()
        axes[0][i].axvline(med, color='red', linestyle='--',
                           label=f'median={med:,.0f}')
        axes[0][i].set_title(f'{name.capitalize()} — Prompt Length (chars)')
        axes[0][i].set_xlabel('Character Count')
        axes[0][i].set_ylabel('Frequency')
        axes[0][i].legend()

        # ── Label distribution bar chart ────────────────────────────────────
        counts = df['label'].value_counts().sort_index()
        axes[1][i].bar([str(int(x)) for x in counts.index],
                       counts.values,
                       edgecolor='black', alpha=0.75, color='darkorange')
        axes[1][i].set_title(f'{name.capitalize()} — Label Distribution')
        axes[1][i].set_xlabel('Correct Option (0–4)')
        axes[1][i].set_ylabel('Count')

    plt.tight_layout()
    save_path = output_dir / 'casehold_distribution.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved → {save_path}")
