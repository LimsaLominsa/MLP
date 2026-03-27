"""
Plot 4-panel attention-shift heatmap: one legal task + one medical task.
Each panel is a single heatmap (no sub-titles needed, labeled in caption).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_attention_matrix(parquet_path: str) -> np.ndarray:
    df = pd.read_parquet(parquet_path)
    pivot = df.pivot_table(index="layer", columns="head", values="attention", aggfunc="mean")
    return pivot.sort_index().sort_index(axis=1).to_numpy()


def compute_shift(base_path: str, target_path: str) -> np.ndarray:
    base = load_attention_matrix(base_path)
    target = load_attention_matrix(target_path)
    n_layers = min(base.shape[0], target.shape[0])
    n_heads = min(base.shape[1], target.shape[1])
    return np.abs(target[:n_layers, :n_heads] - base[:n_layers, :n_heads])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="overleaf_upload/figures")
    parser.add_argument("--outputs-root", default="outputs")
    args = parser.parse_args()

    root = Path(args.outputs_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # BillSum Qwen
    bill_pre = root / "billsum_qwen_1p5b/analysis/test_us/signals.parquet"
    bill_lora = root / "billsum_qwen_1p5b_lora/analysis/test_us/signals.parquet"
    bill_rand = root / "billsum_qwen_1p5b_random/analysis/test_us/signals.parquet"

    # NFCorpus Qwen
    nfc_pre = root / "nfcorpus_qwen_1p5b_pretrained/analysis/test/signals.parquet"
    nfc_lora = root / "nfcorpus_qwen_1p5b_lora/analysis/test/signals.parquet"
    nfc_rand = root / "nfcorpus_qwen_1p5b_random/analysis/test/signals.parquet"

    bill_lora_shift = compute_shift(str(bill_pre), str(bill_lora))
    bill_rand_shift = compute_shift(str(bill_pre), str(bill_rand))
    nfc_lora_shift = compute_shift(str(nfc_pre), str(nfc_lora))
    nfc_rand_shift = compute_shift(str(nfc_pre), str(nfc_rand))

    # Separate color scales for legal vs medical (different magnitude)
    bill_vmax = max(bill_lora_shift.max(), bill_rand_shift.max())
    nfc_vmax = max(nfc_lora_shift.max(), nfc_rand_shift.max())

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), sharey=True)

    # Legal pair (shared scale)
    for ax, data, title in [(axes[0], bill_lora_shift, "BillSum\nLoRA-SFT"),
                             (axes[1], bill_rand_shift, "BillSum\nRandom-Label")]:
        im_bill = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=bill_vmax, origin="lower")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("Head", fontsize=14)
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(range(1, data.shape[1] + 1), fontsize=11)
        ax.tick_params(axis="y", labelsize=12)

    # Medical pair (shared scale)
    for ax, data, title in [(axes[2], nfc_lora_shift, "NFCorpus\nLoRA-SFT"),
                             (axes[3], nfc_rand_shift, "NFCorpus\nRandom-Label")]:
        im_nfc = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=nfc_vmax, origin="lower")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("Head", fontsize=14)
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(range(1, data.shape[1] + 1), fontsize=11)
        ax.tick_params(axis="y", labelsize=12)

    axes[0].set_ylabel("Layer", fontsize=14)
    n_layers = bill_lora_shift.shape[0]
    yticks = list(range(0, n_layers, 5))
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(yticks, fontsize=12)

    # Two colorbars
    cbar1 = fig.colorbar(im_bill, ax=axes[:2], shrink=0.85, pad=0.02)
    cbar1.ax.tick_params(labelsize=11)
    cbar1.set_label("|Δ attn| (legal)", fontsize=12)
    cbar2 = fig.colorbar(im_nfc, ax=axes[2:], shrink=0.85, pad=0.02)
    cbar2.ax.tick_params(labelsize=11)
    cbar2.set_label("|Δ attn| (medical)", fontsize=12)

    fig.subplots_adjust(wspace=0.08)
    save_path = out_dir / "attention_shift_4panel.png"
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
