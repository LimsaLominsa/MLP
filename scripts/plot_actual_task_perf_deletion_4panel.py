from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
TAG = "actual_task_perf_v1"
SYNC_ROOT = ROOT / "outputs/task_perf_deletion_sync"
CASEHOLD_ACTUAL_ROOT = os.environ.get("CASEHOLD_ACTUAL_ROOT")
PUBMED_ACTUAL_ROOT = os.environ.get("PUBMED_ACTUAL_ROOT")
TASK_PERF_OUTPUT_DIR = Path(os.environ.get("TASK_PERF_OUTPUT_DIR", str(ROOT / "outputs")))

CONDITION_STYLES = {
    "pretrained": {"color": "#6B7280", "label": "Pretrained"},
    "lora": {"color": "#2563EB", "label": "LoRA-SFT"},
    "random": {"color": "#DC2626", "label": "Random-Label"},
}
MODEL_STYLES = {
    "qwen": {"linestyle": "-", "label": "Qwen-1.5B"},
    "llama": {"linestyle": "--", "label": "Llama-1B"},
}

TASKS = {
    "billsum": {
        "title": "BillSum",
        "ylabel": "ROUGE-L after deletion",
        "metric_key": "mean_deletion_curve_rougeL",
        "inputs": {
            ("qwen", "pretrained"): ROOT / "outputs/billsum_qwen_1p5b/analysis/test_us/task_perf_deletion" / TAG / "task_perf_deletion.json",
            ("qwen", "lora"): ROOT / "outputs/billsum_qwen_1p5b_lora/analysis/test_us/task_perf_deletion" / TAG / "task_perf_deletion.json",
            ("qwen", "random"): ROOT / "outputs/billsum_qwen_1p5b_random/analysis/test_us/task_perf_deletion" / TAG / "task_perf_deletion.json",
            ("llama", "pretrained"): ROOT / "outputs/billsum_llama_1b/analysis/test_us/task_perf_deletion" / TAG / "task_perf_deletion.json",
            ("llama", "lora"): ROOT / "outputs/billsum_llama_1b_lora/analysis/test_us/task_perf_deletion" / TAG / "task_perf_deletion.json",
            ("llama", "random"): ROOT / "outputs/billsum_llama_1b_random/analysis/test_us/task_perf_deletion" / TAG / "task_perf_deletion.json",
        },
    },
    "casehold": {
        "title": "CaseHOLD",
        "ylabel": "Accuracy after deletion",
        "metric_key": "mean_deletion_accuracy_curve",
        "inputs": (
            {
                ("qwen", "pretrained"): Path(CASEHOLD_ACTUAL_ROOT) / "qwen/pretrained/task_perf_deletion.json",
                ("qwen", "lora"): Path(CASEHOLD_ACTUAL_ROOT) / "qwen/lora/task_perf_deletion.json",
                ("qwen", "random"): Path(CASEHOLD_ACTUAL_ROOT) / "qwen/random/task_perf_deletion.json",
                ("llama", "pretrained"): Path(CASEHOLD_ACTUAL_ROOT) / "llama/pretrained/task_perf_deletion.json",
                ("llama", "lora"): Path(CASEHOLD_ACTUAL_ROOT) / "llama/lora/task_perf_deletion.json",
                ("llama", "random"): Path(CASEHOLD_ACTUAL_ROOT) / "llama/random/task_perf_deletion.json",
            }
            if CASEHOLD_ACTUAL_ROOT
            else {
                ("qwen", "pretrained"): ROOT / "outputs/casehold_qwen_1p5b/analysis/task_perf_deletion/pretrained" / TAG / "task_perf_deletion.json",
                ("qwen", "lora"): ROOT / "outputs/casehold_qwen_1p5b/analysis/task_perf_deletion/lora_sft" / TAG / "task_perf_deletion.json",
                ("qwen", "random"): ROOT / "outputs/casehold_qwen_1p5b/analysis/task_perf_deletion/random_label" / TAG / "task_perf_deletion.json",
                ("llama", "pretrained"): ROOT / "outputs/casehold_llama_1b/analysis/task_perf_deletion/pretrained" / TAG / "task_perf_deletion.json",
                ("llama", "lora"): ROOT / "outputs/casehold_llama_1b/analysis/task_perf_deletion/lora_sft" / TAG / "task_perf_deletion.json",
                ("llama", "random"): ROOT / "outputs/casehold_llama_1b/analysis/task_perf_deletion/random_label" / TAG / "task_perf_deletion.json",
            }
        ),
    },
    "pubmed": {
        "title": "PubMed",
        "ylabel": "ROUGE-L after deletion",
        "metric_key": "mean_deletion_curve_rougeL",
        "inputs": (
            {
                ("qwen", "pretrained"): Path(PUBMED_ACTUAL_ROOT) / "qwen/pretrained/task_perf_deletion.json",
                ("qwen", "lora"): Path(PUBMED_ACTUAL_ROOT) / "qwen/lora/task_perf_deletion.json",
                ("qwen", "random"): Path(PUBMED_ACTUAL_ROOT) / "qwen/random/task_perf_deletion.json",
                ("llama", "pretrained"): Path(PUBMED_ACTUAL_ROOT) / "llama/pretrained/task_perf_deletion.json",
                ("llama", "lora"): Path(PUBMED_ACTUAL_ROOT) / "llama/lora/task_perf_deletion.json",
                ("llama", "random"): Path(PUBMED_ACTUAL_ROOT) / "llama/random/task_perf_deletion.json",
            }
            if PUBMED_ACTUAL_ROOT
            else {
                ("qwen", "pretrained"): ROOT / "outputs/pubmed_qwen_1p5b/analysis/test/task_perf_deletion" / TAG / "task_perf_deletion.json",
                ("qwen", "lora"): ROOT / "outputs/pubmed_qwen_1p5b_lora/analysis/test/task_perf_deletion" / TAG / "task_perf_deletion.json",
                ("qwen", "random"): ROOT / "outputs/pubmed_qwen_1p5b_random/analysis/test/task_perf_deletion" / TAG / "task_perf_deletion.json",
                ("llama", "pretrained"): ROOT / "outputs/pubmed_llama_1b/analysis/test/task_perf_deletion" / TAG / "task_perf_deletion.json",
                ("llama", "lora"): ROOT / "outputs/pubmed_llama_1b_lora/analysis/test/task_perf_deletion" / TAG / "task_perf_deletion.json",
                ("llama", "random"): ROOT / "outputs/pubmed_llama_1b_random/analysis/test/task_perf_deletion" / TAG / "task_perf_deletion.json",
            }
        ),
    },
    "nfcorpus": {
        "title": "NFCorpus",
        "ylabel": "NDCG@5 after deletion",
        "metric_key": "mean_deletion_curve",
        "inputs": {
            ("qwen", "pretrained"): ROOT / "outputs/medical_results/nfcorpus_qwen_1p5b_pretrained_faithfulness.json",
            ("qwen", "lora"): ROOT / "outputs/medical_results/nfcorpus_qwen_1p5b_lora_faithfulness.json",
            ("qwen", "random"): ROOT / "outputs/medical_results/nfcorpus_qwen_1p5b_random_faithfulness.json",
            ("llama", "pretrained"): ROOT / "outputs/medical_results/nfcorpus_llama_1b_pretrained_faithfulness.json",
            ("llama", "lora"): ROOT / "outputs/medical_results/nfcorpus_llama_1b_lora_faithfulness.json",
            ("llama", "random"): ROOT / "outputs/medical_results/nfcorpus_llama_1b_random_faithfulness.json",
        },
    },
}


def _load_curve(path: Path, key: str) -> tuple[list[int], list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    x = payload.get("x_values_percent") or [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    curve = payload[key]
    return [int(v) for v in x], [float(v) for v in curve]


def _resolve(path: Path) -> Path:
    if path.exists():
        return path
    rel = path.relative_to(ROOT / "outputs")
    sync_path = SYNC_ROOT / rel
    return sync_path


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.8))
    axes = axes.flatten()

    for ax, (task_key, spec) in zip(axes, TASKS.items()):
        for (model_key, cond_key), path in spec["inputs"].items():
            path = _resolve(path)
            if not path.exists():
                continue
            x, curve = _load_curve(path, spec["metric_key"])
            cond_style = CONDITION_STYLES[cond_key]
            model_style = MODEL_STYLES[model_key]
            ax.plot(
                x,
                curve,
                color=cond_style["color"],
                linestyle=model_style["linestyle"],
                linewidth=2.5,
            )
        ax.set_title(spec["title"], fontsize=17, pad=10)
        ax.set_xlabel("Top-ranked tokens removed (%)", fontsize=12)
        ax.set_ylabel(spec["ylabel"], fontsize=12)
        ax.grid(alpha=0.25, linewidth=0.8)
        ax.tick_params(axis="both", labelsize=10)

    handles = []
    for cond_key in ["pretrained", "lora", "random"]:
        for model_key in ["qwen", "llama"]:
            cond_style = CONDITION_STYLES[cond_key]
            model_style = MODEL_STYLES[model_key]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=cond_style["color"],
                    linestyle=model_style["linestyle"],
                    linewidth=3,
                    label=f"{cond_style['label']} / {model_style['label']}",
                )
            )

    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=11, frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_dir = TASK_PERF_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "task_perf_deletion_4panel.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
