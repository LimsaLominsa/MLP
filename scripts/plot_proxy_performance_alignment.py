from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SYNC_ROOT = ROOT / "outputs/task_perf_deletion_sync"
CASEHOLD_SYNC_ROOT = os.environ.get(
    "CASEHOLD_SYNC_ROOT",
    str(ROOT / "outputs/casehold_prompttok_sync"),
)
PUBMED_PROXY_ROOT = os.environ.get(
    "PUBMED_PROXY_ROOT",
    str(ROOT / "outputs/pubmed_ctx4096_sync/flat"),
)
PUBMED_ACTUAL_ROOT = os.environ.get(
    "PUBMED_ACTUAL_ROOT",
    str(ROOT / "outputs/pubmed_actual4096_sync"),
)
NFCORPUS_PROXY_ROOT = Path(
    os.environ.get(
        "NFCORPUS_PROXY_ROOT",
        str(ROOT / "outputs/nfcorpus_relevant_margin_proxy_sync/flat"),
    )
)
ALIGNMENT_OUTPUT_DIR = Path(
    os.environ.get(
        "ALIGNMENT_OUTPUT_DIR",
        str(ROOT / "格式整理/MLP_2025_26_CW3_4_Template/figures_nfcorpus_margin_preview"),
    )
)

COLS = [
    ("qwen", "pretrained"),
    ("qwen", "lora"),
    ("qwen", "random"),
    ("llama", "pretrained"),
    ("llama", "lora"),
    ("llama", "random"),
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _norm_from_log_curve(curve: list[float]) -> list[float]:
    start = float(curve[0])
    return [math.exp(float(v) - start) for v in curve]


def _norm_from_positive_curve(curve: list[float]) -> list[float]:
    start = max(float(curve[0]), 1e-12)
    return [float(v) / start for v in curve]


def _alignment_metrics(proxy_curve: list[float], actual_curve: list[float]) -> tuple[float, float, float]:
    proxy = np.asarray(proxy_curve, dtype=float)
    actual = np.asarray(actual_curve, dtype=float)
    mae = float(np.mean(np.abs(proxy - actual)))
    alignment = max(0.0, 1.0 - mae)
    rho, _ = spearmanr(proxy, actual)
    if np.isnan(rho):
        rho = 1.0 if np.allclose(proxy, actual) else 0.0
    return alignment, mae, float(rho)


def _proxy_actual_for_billsum(model: str, cond: str) -> tuple[list[float], list[float]]:
    proxy_path = ROOT / "outputs/figure6_inputs/legal" / f"billsum_{model}_{cond}_faithfulness.json"
    msfx = "1p5b" if model == "qwen" else "1b"
    perf_dir = f"billsum_{model}_{msfx}" if cond == "pretrained" else f"billsum_{model}_{msfx}_{cond}"
    actual_path = SYNC_ROOT / perf_dir / "analysis" / "test_us" / "task_perf_deletion" / "actual_task_perf_v1" / "task_perf_deletion.json"

    proxy_curve = _norm_from_log_curve(_read_json(proxy_path)["mean_deletion_curve"])
    actual_curve = _norm_from_positive_curve(_read_json(actual_path)["mean_deletion_curve_rougeL"])
    return proxy_curve, actual_curve


def _proxy_actual_for_pubmed(model: str, cond: str) -> tuple[list[float], list[float]]:
    msfx = "1p5b" if model == "qwen" else "1b"
    if PUBMED_PROXY_ROOT:
        proxy_path = Path(PUBMED_PROXY_ROOT) / f"pubmed_{model}_{msfx}_{cond}_faithfulness.json"
    else:
        proxy_path = ROOT / "outputs/medical_results" / f"pubmed_{model}_{msfx}_{cond}_faithfulness.json"

    if PUBMED_ACTUAL_ROOT:
        cond_dir = {"pretrained": "pretrained", "lora": "lora", "random": "random"}[cond]
        actual_path = Path(PUBMED_ACTUAL_ROOT) / model / cond_dir / "task_perf_deletion.json"
    else:
        perf_dir = f"pubmed_{model}_{msfx}_{cond}"
        actual_path = SYNC_ROOT / perf_dir / "analysis" / "test" / "task_perf_deletion" / "actual_task_perf_v1" / "task_perf_deletion.json"

    proxy_curve = _norm_from_log_curve(_read_json(proxy_path)["mean_deletion_curve"])
    actual_curve = _norm_from_positive_curve(_read_json(actual_path)["mean_deletion_curve_rougeL"])
    return proxy_curve, actual_curve


def _proxy_actual_for_casehold(model: str, cond: str) -> tuple[list[float], list[float]]:
    if CASEHOLD_SYNC_ROOT:
        cond_dir = {"pretrained": "pretrained", "lora": "lora", "random": "random"}[cond]
        samples_path = Path(CASEHOLD_SYNC_ROOT) / model / cond_dir / "task_perf_deletion_samples.jsonl"
        actual_path = Path(CASEHOLD_SYNC_ROOT) / model / cond_dir / "task_perf_deletion.json"
    else:
        mdir = "casehold_qwen_1p5b" if model == "qwen" else "casehold_llama_1b"
        cond_dir = {"pretrained": "pretrained", "lora": "lora_sft", "random": "random_label"}[cond]
        samples_path = SYNC_ROOT / mdir / "analysis" / "task_perf_deletion" / cond_dir / "actual_task_perf_v1" / "task_perf_deletion_samples.jsonl"
        actual_path = SYNC_ROOT / mdir / "analysis" / "task_perf_deletion" / cond_dir / "actual_task_perf_v1" / "task_perf_deletion.json"

    rows = _read_jsonl(samples_path)
    curve_len = len(rows[0]["deletion_gold_prob_curve"])
    proxy_curve = []
    for idx in range(curve_len):
        vals = [max(float(r["deletion_gold_prob_curve"][idx]), 1e-12) for r in rows]
        proxy_curve.append(sum(math.log(v) for v in vals) / len(vals))

    proxy_curve = _norm_from_log_curve(proxy_curve)
    actual_curve = _norm_from_positive_curve(_read_json(actual_path)["mean_deletion_accuracy_curve"])
    return proxy_curve, actual_curve


def _proxy_actual_for_nfcorpus(model: str, cond: str) -> tuple[list[float], list[float]]:
    msfx = "1p5b" if model == "qwen" else "1b"
    proxy_path = NFCORPUS_PROXY_ROOT / f"nfcorpus_{model}_{msfx}_{cond}_relevant_doc_faithfulness.json"
    actual_path = ROOT / "outputs/medical_results" / f"nfcorpus_{model}_{msfx}_{cond}_faithfulness.json"
    proxy_curve = _norm_from_log_curve(_read_json(proxy_path)["mean_deletion_curve"])
    actual_curve = _norm_from_positive_curve(_read_json(actual_path)["mean_deletion_curve"])
    return proxy_curve, actual_curve


TASK_LOADERS: list[tuple[str, str, Callable[[str, str], tuple[list[float], list[float]]]]] = [
    ("BillSum", "BillSum", _proxy_actual_for_billsum),
    ("CaseHOLD", "CaseHOLD", _proxy_actual_for_casehold),
    ("PubMed", "PubMed", _proxy_actual_for_pubmed),
    ("NFCorpus", "NFCorpus", _proxy_actual_for_nfcorpus),
]


def main() -> None:
    heat = np.zeros((len(TASK_LOADERS), len(COLS)), dtype=float)
    rows_for_csv: list[dict] = []

    for row_idx, (display_name, task_key, loader) in enumerate(TASK_LOADERS):
        for col_idx, (model, cond) in enumerate(COLS):
            proxy_curve, actual_curve = loader(model, cond)
            alignment, mae, rho = _alignment_metrics(proxy_curve, actual_curve)
            heat[row_idx, col_idx] = rho
            rows_for_csv.append(
                {
                    "task": task_key,
                    "display_task": display_name,
                    "model": model,
                    "condition": cond,
                    "alignment_score": alignment,
                    "mae": mae,
                    "spearman_rho": rho,
                }
            )

    out_dir = ALIGNMENT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "proxy_performance_alignment.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "display_task", "model", "condition", "alignment_score", "mae", "spearman_rho"],
        )
        writer.writeheader()
        writer.writerows(rows_for_csv)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, 4.6),
        gridspec_kw={"wspace": 0.20},
        sharey=True,
    )
    qwen_heat = heat[:, :3]
    llama_heat = heat[:, 3:]
    panels = [("Qwen", qwen_heat, axes[0]), ("Llama", llama_heat, axes[1])]

    ims = []
    for title, panel_heat, ax in panels:
        im = ax.imshow(panel_heat, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
        ims.append(im)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(["Pre", "LoRA", "Rand"], fontsize=11)
        ax.set_title(title, fontsize=15, pad=10)
        ax.set_yticks(np.arange(len(TASK_LOADERS)))
        if ax is axes[0]:
            ax.set_yticklabels([display for display, _, _ in TASK_LOADERS], fontsize=12)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

        for i in range(panel_heat.shape[0]):
            for j in range(panel_heat.shape[1]):
                val = panel_heat[i, j]
                text_color = "white" if abs(val) > 0.45 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=10)

    fig.suptitle("Proxy-to-Performance Rank Alignment Across Tasks", fontsize=17, y=0.985)
    cax = fig.add_axes([0.935, 0.18, 0.018, 0.66])
    cbar = fig.colorbar(ims[-1], cax=cax)
    cbar.set_label("Spearman ρ", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=0.12, right=0.90, top=0.84, bottom=0.18, wspace=0.18)

    fig_path = out_dir / "proxy_performance_alignment_heatmap.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
