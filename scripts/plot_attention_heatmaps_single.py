"""Generate individual attention-shift heatmaps (one per PNG), matching Fig 4 style."""
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


def plot_single(data, title, save_path, vmax=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    if vmax is None:
        vmax = data.max()
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax, origin="lower")
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel("Head", fontsize=16)
    ax.set_ylabel("Layer", fontsize=16)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(range(data.shape[1]), fontsize=14)
    n_layers = data.shape[0]
    yticks = list(range(0, n_layers, 5))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks, fontsize=14)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label("|Δ attention|", fontsize=15)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="overleaf_upload/figures")
    parser.add_argument("--outputs-root", default="outputs")
    args = parser.parse_args()

    root = Path(args.outputs_root)
    out = Path(args.output_dir)

    jobs = [
        ("BillSum LoRA-SFT", "billsum_qwen_1p5b/analysis/test_us/signals.parquet",
         "billsum_qwen_1p5b_lora/analysis/test_us/signals.parquet", "billsum_lora_shift.png"),
        ("BillSum Random-Label", "billsum_qwen_1p5b/analysis/test_us/signals.parquet",
         "billsum_qwen_1p5b_random/analysis/test_us/signals.parquet", "billsum_random_shift.png"),
        ("NFCorpus LoRA-SFT", "nfcorpus_qwen_1p5b_pretrained/analysis/test/signals.parquet",
         "nfcorpus_qwen_1p5b_lora/analysis/test/signals.parquet", "nfcorpus_lora_shift.png"),
        ("NFCorpus Random-Label", "nfcorpus_qwen_1p5b_pretrained/analysis/test/signals.parquet",
         "nfcorpus_qwen_1p5b_random/analysis/test/signals.parquet", "nfcorpus_random_shift.png"),
    ]

    # Compute all shifts first to get per-domain vmax
    shifts = {}
    for title, pre_rel, tgt_rel, fname in jobs:
        pre_path = root / pre_rel
        tgt_path = root / tgt_rel
        if pre_path.exists() and tgt_path.exists():
            shifts[fname] = (title, compute_shift(str(pre_path), str(tgt_path)))

    bill_vmax = max(s[1].max() for k, s in shifts.items() if "billsum" in k)
    nfc_vmax = max(s[1].max() for k, s in shifts.items() if "nfcorpus" in k)

    for fname, (title, data) in shifts.items():
        vmax = bill_vmax if "billsum" in fname else nfc_vmax
        plot_single(data, title, str(out / fname), vmax=vmax)


if __name__ == "__main__":
    main()
