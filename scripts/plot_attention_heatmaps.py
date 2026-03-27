"""
Plot layer×head attention-shift heatmaps comparing LoRA vs Random conditions.
Inspired by Zhao & Bethard (2020) Figure 4.

For each task × model, produces a side-by-side heatmap:
  Left: |attention_lora - attention_pretrained| (per layer, per head, averaged over samples)
  Right: |attention_random - attention_pretrained| (same)

Usage:
  python scripts/plot_attention_heatmaps.py --output-dir overleaf_upload/figures/
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_attention_matrix(parquet_path: str) -> np.ndarray:
    """Load signals.parquet and return (layers × heads) mean attention matrix."""
    df = pd.read_parquet(parquet_path)
    pivot = df.pivot_table(index="layer", columns="head", values="attention", aggfunc="mean")
    return pivot.sort_index().sort_index(axis=1).to_numpy()


def compute_shift(base_path: str, target_path: str) -> np.ndarray:
    """Compute absolute attention shift between base and target."""
    base = load_attention_matrix(base_path)
    target = load_attention_matrix(target_path)
    # Align shapes (use min in case of mismatch)
    n_layers = min(base.shape[0], target.shape[0])
    n_heads = min(base.shape[1], target.shape[1])
    return np.abs(target[:n_layers, :n_heads] - base[:n_layers, :n_heads])


def plot_paired_heatmap(lora_shift, random_shift, title_left, title_right, save_path, model_name):
    """Plot side-by-side heatmaps with large fonts matching Fig 4 style."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    vmax = max(lora_shift.max(), random_shift.max())
    vmin = 0

    im1 = ax1.imshow(lora_shift, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax, origin="lower")
    ax1.set_title(title_left, fontsize=18, fontweight="bold")
    ax1.set_xlabel("Head", fontsize=16)
    ax1.set_ylabel("Layer", fontsize=16)
    ax1.set_xticks(range(lora_shift.shape[1]))
    ax1.set_xticklabels(range(1, lora_shift.shape[1] + 1), fontsize=14)
    n_layers = lora_shift.shape[0]
    yticks = list(range(0, n_layers, 5))
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticks, fontsize=14)

    im2 = ax2.imshow(random_shift, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax, origin="lower")
    ax2.set_title(title_right, fontsize=18, fontweight="bold")
    ax2.set_xlabel("Head", fontsize=16)
    ax2.set_xticks(range(random_shift.shape[1]))
    ax2.set_xticklabels(range(1, random_shift.shape[1] + 1), fontsize=14)
    ax2.tick_params(axis="y", labelsize=14)

    fig.suptitle(f"{model_name}: Attention Shift from Pretrained", fontsize=20, fontweight="bold", y=1.02)
    cbar = fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="|Δ attention|")
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("|Δ attention|", fontsize=15)
    fig.subplots_adjust(wspace=0.08)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="overleaf_upload/figures")
    parser.add_argument("--outputs-root", default="outputs")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    root = Path(args.outputs_root)

    # Each entry: (task, model_label, pretrained_parquet, lora_parquet, random_parquet)
    # Paths are relative to outputs_root
    configs = [
        # Medical (consistent layout)
        ("PubMed", "Qwen", "pubmed_qwen_1p5b_pretrained/analysis/test/signals.parquet",
         "pubmed_qwen_1p5b_lora/analysis/test/signals.parquet",
         "pubmed_qwen_1p5b_random/analysis/test/signals.parquet"),
        ("PubMed", "Llama", "pubmed_llama_1b_pretrained/analysis/test/signals.parquet",
         "pubmed_llama_1b_lora/analysis/test/signals.parquet",
         "pubmed_llama_1b_random/analysis/test/signals.parquet"),
        ("NFCorpus", "Qwen", "nfcorpus_qwen_1p5b_pretrained/analysis/test/signals.parquet",
         "nfcorpus_qwen_1p5b_lora/analysis/test/signals.parquet",
         "nfcorpus_qwen_1p5b_random/analysis/test/signals.parquet"),
        ("NFCorpus", "Llama", "nfcorpus_llama_1b_pretrained/analysis/test/signals.parquet",
         "nfcorpus_llama_1b_lora/analysis/test/signals.parquet",
         "nfcorpus_llama_1b_random/analysis/test/signals.parquet"),
        # Legal — BillSum (pretrained has no _pretrained suffix)
        ("BillSum", "Qwen", "billsum_qwen_1p5b/analysis/test_us/signals.parquet",
         "billsum_qwen_1p5b_lora/analysis/test_us/signals.parquet",
         "billsum_qwen_1p5b_random/analysis/test_us/signals.parquet"),
        ("BillSum", "Llama", "billsum_llama_1b/analysis/test_us/signals.parquet",
         "billsum_llama_1b_lora/analysis/test_us/signals.parquet",
         "billsum_llama_1b_random/analysis/test_us/signals.parquet"),
        # Legal — CaseHOLD (artifacts layout)
        ("CaseHOLD", "Qwen", "casehold_qwen_1p5b/artifacts/pretrained/42/signals.parquet",
         "casehold_qwen_1p5b/artifacts/lora_sft/42/signals.parquet",
         "casehold_qwen_1p5b/artifacts/random_label/42/signals.parquet"),
        ("CaseHOLD", "Llama", "casehold_llama_1b/artifacts/pretrained/42/signals.parquet",
         "casehold_llama_1b/artifacts/lora_sft/42/signals.parquet",
         "casehold_llama_1b/artifacts/random_label/42/signals.parquet"),
    ]

    for task, model, pre_rel, lora_rel, rand_rel in configs:
        pre_path = root / pre_rel
        lora_path = root / lora_rel
        rand_path = root / rand_rel
        if not (pre_path.exists() and lora_path.exists() and rand_path.exists()):
            print(f"Skipping {task} {model}: signals not found")
            continue

        lora_shift = compute_shift(str(pre_path), str(lora_path))
        random_shift = compute_shift(str(pre_path), str(rand_path))

        filename = f"{task.lower()}_{model.lower()}_attention_shift.png"
        plot_paired_heatmap(
            lora_shift, random_shift,
            "LoRA-SFT", "Random-Label",
            str(out_dir / filename),
            f"{task} — {model}",
        )


if __name__ == "__main__":
    main()
