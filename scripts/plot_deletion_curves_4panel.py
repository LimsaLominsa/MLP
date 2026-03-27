from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
NFCORPUS_PROXY_ROOT = Path(
    os.environ.get(
        "NFCORPUS_PROXY_ROOT",
        str(ROOT / "outputs/nfcorpus_relevant_margin_proxy_sync/flat"),
    )
)
PUBMED_PROXY_ROOT = Path(
    os.environ.get(
        "PUBMED_PROXY_ROOT",
        str(ROOT / "outputs/pubmed_ctx4096_sync/flat"),
    )
)
CASEHOLD_TASK_ROOT = Path(
    os.environ.get(
        "CASEHOLD_TASK_ROOT",
        str(ROOT / "outputs/casehold_prompttok_sync"),
    )
)
X_VALUES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

CONDITION_STYLES = {
    "pretrained": {"color": "#6B7280", "label": "Pretrained"},
    "lora": {"color": "#2563EB", "label": "LoRA-SFT"},
    "random": {"color": "#DC2626", "label": "Random-Label"},
}

MODEL_STYLES = {
    "qwen": {"linestyle": "-", "label": "Qwen-1.5B"},
    "llama": {"linestyle": "--", "label": "Llama-1B"},
}

TASK_SPECS = {
    "billsum": {
        "title": "BillSum",
        "raw_ylabel": "Score after deletion",
        "normalized_ylabel": "Relative score retained",
        "inputs": {
            ("qwen", "pretrained"): ROOT / "outputs/figure6_inputs/samples/billsum_qwen_pretrained_faithfulness_samples.jsonl",
            ("qwen", "lora"): ROOT / "outputs/figure6_inputs/samples/billsum_qwen_lora_faithfulness_samples.jsonl",
            ("qwen", "random"): ROOT / "outputs/figure6_inputs/samples/billsum_qwen_random_faithfulness_samples.jsonl",
            ("llama", "pretrained"): ROOT / "outputs/figure6_inputs/samples/billsum_llama_pretrained_faithfulness_samples.jsonl",
            ("llama", "lora"): ROOT / "outputs/figure6_inputs/samples/billsum_llama_lora_faithfulness_samples.jsonl",
            ("llama", "random"): ROOT / "outputs/figure6_inputs/samples/billsum_llama_random_faithfulness_samples.jsonl",
        },
    },
    "casehold": {
        "title": "CaseHOLD",
        "raw_ylabel": "Score after deletion",
        "normalized_ylabel": "Relative score retained",
        "inputs": {
            ("qwen", "pretrained"): CASEHOLD_TASK_ROOT / "qwen/pretrained/task_perf_deletion_samples.jsonl",
            ("qwen", "lora"): CASEHOLD_TASK_ROOT / "qwen/lora/task_perf_deletion_samples.jsonl",
            ("qwen", "random"): CASEHOLD_TASK_ROOT / "qwen/random/task_perf_deletion_samples.jsonl",
            ("llama", "pretrained"): CASEHOLD_TASK_ROOT / "llama/pretrained/task_perf_deletion_samples.jsonl",
            ("llama", "lora"): CASEHOLD_TASK_ROOT / "llama/lora/task_perf_deletion_samples.jsonl",
            ("llama", "random"): CASEHOLD_TASK_ROOT / "llama/random/task_perf_deletion_samples.jsonl",
        },
    },
    "pubmed": {
        "title": "PubMed",
        "raw_ylabel": "Score after deletion",
        "normalized_ylabel": "Relative score retained",
        "inputs": {
            ("qwen", "pretrained"): PUBMED_PROXY_ROOT / "pubmed_qwen_1p5b_pretrained_faithfulness.json",
            ("qwen", "lora"): PUBMED_PROXY_ROOT / "pubmed_qwen_1p5b_lora_faithfulness.json",
            ("qwen", "random"): PUBMED_PROXY_ROOT / "pubmed_qwen_1p5b_random_faithfulness.json",
            ("llama", "pretrained"): PUBMED_PROXY_ROOT / "pubmed_llama_1b_pretrained_faithfulness.json",
            ("llama", "lora"): PUBMED_PROXY_ROOT / "pubmed_llama_1b_lora_faithfulness.json",
            ("llama", "random"): PUBMED_PROXY_ROOT / "pubmed_llama_1b_random_faithfulness.json",
        },
    },
    "nfcorpus": {
        "title": "NFCorpus",
        "raw_ylabel": "Score after deletion",
        "normalized_ylabel": "Relative score retained",
        "inputs": {
            ("qwen", "pretrained"): NFCORPUS_PROXY_ROOT / "nfcorpus_qwen_1p5b_pretrained_relevant_doc_faithfulness.json",
            ("qwen", "lora"): NFCORPUS_PROXY_ROOT / "nfcorpus_qwen_1p5b_lora_relevant_doc_faithfulness.json",
            ("qwen", "random"): NFCORPUS_PROXY_ROOT / "nfcorpus_qwen_1p5b_random_relevant_doc_faithfulness.json",
            ("llama", "pretrained"): NFCORPUS_PROXY_ROOT / "nfcorpus_llama_1b_pretrained_relevant_doc_faithfulness.json",
            ("llama", "lora"): NFCORPUS_PROXY_ROOT / "nfcorpus_llama_1b_lora_relevant_doc_faithfulness.json",
            ("llama", "random"): NFCORPUS_PROXY_ROOT / "nfcorpus_llama_1b_random_relevant_doc_faithfulness.json",
        },
    },
}

_output_dirs_env = os.environ.get("DELETION_FIGURE_OUTPUT_DIRS")
if _output_dirs_env:
    FIGURE_OUTPUT_DIRS = [Path(part) for part in _output_dirs_env.split(":") if part.strip()]
else:
    FIGURE_OUTPUT_DIRS = [
        ROOT / "格式整理/MLP_2025_26_CW3_4_Template/figures",
        ROOT / "G104_overleaf/figures",
        ROOT / "overleaf_upload/figures",
        ROOT / "outputs",
    ]


def _sample_base(row: dict) -> float:
    if "base_score" in row:
        return float(row["base_score"])
    if "base_ndcg" in row:
        return float(row["base_ndcg"])
    if "deletion_curve" in row and "aopc" in row:
        curve = [float(x) for x in row["deletion_curve"]]
        return float(sum(curve) / len(curve) + float(row["aopc"]))
    raise ValueError(f"Unable to infer undeleted base score from row keys: {sorted(row.keys())}")


def _normalize_row(row: dict) -> list[float]:
    curve = [float(x) for x in row["deletion_curve"]]

    if "base_score" in row:
        base = float(row["base_score"])
        return [math.exp(score - base) for score in curve]

    base = _sample_base(row)
    if base <= 0:
        raise ValueError(f"Non-positive base score encountered: {base}")
    return [score / base for score in curve]


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"No sample rows found in {path}")
    return rows


def _load_curve(path: Path, *, normalized: bool) -> list[float]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        curve = [float(x) for x in payload["mean_deletion_curve"]]
        if normalized:
            if "log" in str(payload.get("score_type", "")).lower():
                start = float(curve[0])
                return [math.exp(v - start) for v in curve]
            start = max(float(curve[0]), 1e-12)
            return [v / start for v in curve]
        return curve

    rows = _load_rows(path)
    path_str = str(path)

    if "casehold_" in path_str and "task_perf_deletion_samples" in path_str:
        curve_len = len(rows[0]["deletion_gold_prob_curve"])
        mean_log_curve = [
            sum(math.log(max(float(row["deletion_gold_prob_curve"][idx]), 1e-12)) for row in rows) / len(rows)
            for idx in range(curve_len)
        ]
        if normalized:
            start = mean_log_curve[0]
            return [math.exp(v - start) for v in mean_log_curve]
        return mean_log_curve

    if normalized:
        curve_rows = [_normalize_row(row) for row in rows]
    else:
        curve_rows = [[float(x) for x in row["deletion_curve"]] for row in rows]

    curve_len = len(curve_rows[0])
    return [
        sum(row[idx] for row in curve_rows) / len(curve_rows)
        for idx in range(curve_len)
    ]


def _plot_task(task_key: str, out_name: str, *, normalized: bool) -> None:
    task = TASK_SPECS[task_key]
    fig, ax = plt.subplots(figsize=(6.1, 4.25))
    input_map = task.get("normalized_inputs", task["inputs"]) if normalized else task["inputs"]

    for (model_key, cond_key), path in input_map.items():
        curve = _load_curve(path, normalized=normalized)
        cond_style = CONDITION_STYLES[cond_key]
        model_style = MODEL_STYLES[model_key]
        ax.plot(
            X_VALUES,
            curve,
            color=cond_style["color"],
            linestyle=model_style["linestyle"],
            linewidth=2.5,
            alpha=0.97,
        )

    ax.set_title(task["title"], fontsize=16, pad=10)
    ax.set_xlabel("Top-ranked tokens removed (%)", fontsize=13)
    ax.set_ylabel(task["normalized_ylabel"] if normalized else task["raw_ylabel"], fontsize=13)
    ax.set_xticks(X_VALUES)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.25, linewidth=0.8)
    if normalized:
        ax.axhline(1.0, color="black", linewidth=0.9, alpha=0.4)
    fig.tight_layout()

    for out_dir in FIGURE_OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / out_name, dpi=220, bbox_inches="tight")

    plt.close(fig)


def _plot_legend() -> None:
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

    fig, ax = plt.subplots(figsize=(14.8, 1.05))
    ax.axis("off")
    ax.legend(
        handles=handles,
        ncol=3,
        loc="center",
        frameon=True,
        fontsize=12,
        handlelength=3.0,
        columnspacing=1.8,
    )
    fig.tight_layout(pad=0.15)

    for out_dir in FIGURE_OUTPUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "deletion_legend.png", dpi=220, bbox_inches="tight", pad_inches=0.02)

    plt.close(fig)


def main() -> None:
    _plot_task("billsum", "billsum_deletion_curve.png", normalized=False)
    _plot_task("casehold", "casehold_deletion_curve.png", normalized=False)
    _plot_task("pubmed", "pubmed_deletion_curve.png", normalized=False)
    _plot_task("nfcorpus", "nfcorpus_deletion_curve.png", normalized=False)
    _plot_task("billsum", "billsum_deletion_curve_relative.png", normalized=True)
    _plot_task("casehold", "casehold_deletion_curve_relative.png", normalized=True)
    _plot_task("pubmed", "pubmed_deletion_curve_relative.png", normalized=True)
    _plot_task("nfcorpus", "nfcorpus_deletion_curve_relative.png", normalized=True)
    _plot_legend()


if __name__ == "__main__":
    main()
