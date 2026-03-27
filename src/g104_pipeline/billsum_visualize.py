from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .billsum_baseline import (
    _build_prompt,
    _device,
    _extract_bill_text,
    _load_config,
    _read_jsonl,
    _require_hf,
    _torch_dtype,
)

_TOKEN_CLEAN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\\-_/]*")
_TOKEN_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "under",
    "with",
    "within",
}


def _normalised_entropy(attn: np.ndarray) -> float:
    probs = np.clip(attn, 1e-12, 1.0)
    ent = -(probs * np.log(probs)).sum(axis=-1)
    denom = max(1.0, math.log(attn.shape[-1]))
    return float(ent.mean() / denom)


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> Tuple[int, int] | None:
    if not subsequence or len(subsequence) > len(sequence):
        return None
    last = len(sequence) - len(subsequence) + 1
    head = subsequence[0]
    for start in range(last):
        if sequence[start] != head:
            continue
        if list(sequence[start : start + len(subsequence)]) == list(subsequence):
            return start, start + len(subsequence)
    return None


def _token_importance_for_sample(
    tokenizer,
    full_ids: List[int],
    attentions,
    bill_text: str,
) -> List[Tuple[str, float]]:
    bill_ids = tokenizer.encode(bill_text, add_special_tokens=False)
    span = _find_subsequence(full_ids, bill_ids)
    if span is None:
        start, end = 0, len(full_ids)
    else:
        start, end = span

    layer_tensors = attentions[-4:] if len(attentions) >= 4 else attentions
    token_scores = np.zeros(end - start, dtype=np.float64)

    for att in layer_tensors:
        arr = att[0].detach().float().cpu().numpy()
        key_imp = arr.mean(axis=0).mean(axis=0)[start:end]
        token_scores += key_imp

    token_scores /= max(1, len(layer_tensors))
    token_map: Dict[str, float] = defaultdict(float)
    for token_id, score in zip(full_ids[start:end], token_scores):
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        token = token.replace("Ġ", "").replace("▁", "").strip()
        if not token or token.startswith("<|"):
            continue
        m = _TOKEN_CLEAN_RE.match(token)
        if not m:
            continue
        clean = m.group(0).lower()
        if clean in _TOKEN_STOPWORDS or clean.isdigit() or len(clean) <= 2:
            continue
        token_map[clean] += float(score)

    ranked = sorted(token_map.items(), key=lambda item: item[1], reverse=True)
    return ranked[:20]


def _plot_attention_entropy(heatmap: np.ndarray, out_path: Path, model_name: str, split: str, sample_count: int) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 9.0))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Head", fontsize=27, labelpad=10)
    ax.set_ylabel("Layer", fontsize=29, labelpad=12)
    ax.set_title("Attention entropy", fontsize=27, pad=16)
    ax.set_xticks(range(heatmap.shape[1]))
    layer_ticks = np.arange(0, heatmap.shape[0], 5, dtype=int)
    ax.set_yticks(layer_ticks)
    ax.tick_params(axis="x", labelsize=23, pad=5)
    ax.tick_params(axis="y", labelsize=30, pad=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalised entropy", fontsize=25)
    cbar.ax.tick_params(labelsize=21)
    fig.tight_layout(pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _plot_hidden_norm(norms: np.ndarray, out_path: Path, model_name: str, split: str, sample_count: int) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 7.2))
    layers = np.arange(len(norms))
    ax.plot(layers, norms, marker="o", linewidth=2.9, markersize=8.5)
    ax.set_xlabel("Layer", fontsize=25, labelpad=9)
    ax.set_ylabel("Mean hidden-state L2 norm", fontsize=25, labelpad=11)
    ax.set_title("Hidden-state norms", fontsize=26, pad=16)
    ax.tick_params(axis="both", labelsize=21)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout(pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _plot_token_importance(ranked_tokens: List[Tuple[str, float]], out_path: Path, model_name: str, split: str) -> None:
    labels = [token for token, _ in ranked_tokens][::-1]
    values = [score for _, score in ranked_tokens][::-1]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(labels, values, color="#2f6db3")
    ax.set_xlabel("Aggregated attention score")
    ax.set_title(f"BillSum pretrained token importance example\\n{model_name} | {split}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_plot_cache(
    *,
    cache_path: Path,
    attn_mean: np.ndarray,
    hidden_mean: np.ndarray,
    prompt_lengths: Sequence[int],
    token_ranking: List[Tuple[str, float]] | None,
    model_name: str,
    split: str,
    sample_count: int,
    sample_index: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        attn_mean=attn_mean,
        hidden_mean=hidden_mean,
        prompt_lengths=np.asarray(prompt_lengths, dtype=np.int32),
        model_name=np.asarray(model_name),
        split=np.asarray(split),
        sample_count=np.asarray(sample_count, dtype=np.int32),
        sample_index=np.asarray(sample_index, dtype=np.int32),
    )
    ranking_path = cache_path.with_name("token_importance_sample.json")
    ranking_payload = {
        "model_name": model_name,
        "split": split,
        "sample_count": sample_count,
        "sample_index": sample_index,
        "token_ranking": [
            {"token": token, "score": float(score)}
            for token, score in (token_ranking or [])
        ],
    }
    ranking_path.write_text(json.dumps(ranking_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_plot_cache(cache_path: Path) -> Dict[str, object]:
    if not cache_path.exists():
        raise FileNotFoundError(f"Plot cache not found: {cache_path}")
    payload = np.load(cache_path, allow_pickle=True)
    ranking_path = cache_path.with_name("token_importance_sample.json")
    ranking_rows: List[Tuple[str, float]] = []
    if ranking_path.exists():
        ranking_payload = json.loads(ranking_path.read_text(encoding="utf-8"))
        ranking_rows = [
            (row["token"], float(row["score"]))
            for row in ranking_payload.get("token_ranking", [])
        ]
    return {
        "attn_mean": payload["attn_mean"],
        "hidden_mean": payload["hidden_mean"],
        "prompt_lengths": payload["prompt_lengths"].astype(np.int32).tolist(),
        "model_name": str(payload["model_name"].item()),
        "split": str(payload["split"].item()),
        "sample_count": int(payload["sample_count"].item()),
        "sample_index": int(payload["sample_index"].item()),
        "token_ranking": ranking_rows,
    }


def generate_billsum_visuals(
    config_path: str,
    split: str = "test_us",
    limit: int = 16,
    output_tag: str = "pretrained_visuals",
    sample_index: int = 0,
    reuse_cache: bool = False,
) -> Dict[str, str]:
    cfg = _load_config(config_path)
    output_root = Path(cfg["project"]["output_root"]) / "visuals" / split / output_tag
    output_root.mkdir(parents=True, exist_ok=True)
    cache_path = output_root / "figure_data.npz"
    attention_path = output_root / "attention_entropy_heatmap.png"
    hidden_path = output_root / "hidden_state_norms.png"
    token_path = output_root / "token_importance_sample.png"
    summary_path = output_root / "summary.json"
    ranking_cache_path = output_root / "token_importance_sample.json"

    if reuse_cache and cache_path.exists():
        cached = _load_plot_cache(cache_path)
        attn_mean = cached["attn_mean"]
        hidden_mean = cached["hidden_mean"]
        prompt_lengths = cached["prompt_lengths"]
        token_ranking = cached["token_ranking"]
        model_name = cached["model_name"]
        sample_count = cached["sample_count"]
    else:
        hf = _require_hf()
        torch = hf["torch"]
        AutoTokenizer = hf["AutoTokenizer"]
        AutoModelForCausalLM = hf["AutoModelForCausalLM"]

        split_key = f"{split}_file"
        input_path = Path(cfg["data"][split_key])
        rows = _read_jsonl(input_path, limit=limit)
        if not rows:
            raise RuntimeError(f"No rows found in {input_path}")

        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            torch_dtype=_torch_dtype(torch),
            attn_implementation="eager",
            trust_remote_code=True,
        )
        model.to(_device(torch))
        model.eval()

        max_input_tokens = int(cfg["generation"].get("max_input_tokens", 2048))
        attn_sum = None
        hidden_sum = None
        prompt_lengths: List[int] = []
        token_ranking: List[Tuple[str, float]] | None = None

        for idx, row in enumerate(rows):
            prompt = _build_prompt(tokenizer, row["input"])
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
            )
            prompt_lengths.append(int(encoded["input_ids"].shape[1]))
            encoded = {k: v.to(_device(torch)) for k, v in encoded.items()}

            with torch.no_grad():
                out = model(**encoded, output_attentions=True, output_hidden_states=True)

            attentions = out.attentions
            hidden_states = out.hidden_states[1:]
            if not attentions:
                raise RuntimeError(
                    "Attention tensors were not returned. Check the attention implementation for this model."
                )

            layer_head = []
            for att in attentions:
                arr = att[0].detach().float().cpu().numpy()
                layer_head.append([_normalised_entropy(arr[head]) for head in range(arr.shape[0])])
            layer_head_arr = np.asarray(layer_head, dtype=np.float64)
            hidden_arr = np.asarray(
                [float(torch.norm(hs[0], dim=-1).mean().item()) for hs in hidden_states],
                dtype=np.float64,
            )

            if attn_sum is None:
                attn_sum = np.zeros_like(layer_head_arr)
                hidden_sum = np.zeros_like(hidden_arr)
            attn_sum += layer_head_arr
            hidden_sum += hidden_arr

            if idx == sample_index:
                full_ids = encoded["input_ids"][0].detach().cpu().tolist()
                token_ranking = _token_importance_for_sample(
                    tokenizer=tokenizer,
                    full_ids=full_ids,
                    attentions=attentions,
                    bill_text=_extract_bill_text(row["input"]),
                )

        attn_mean = attn_sum / len(rows)
        hidden_mean = hidden_sum / len(rows)
        model_name = cfg["model_name"]
        sample_count = len(rows)
        _save_plot_cache(
            cache_path=cache_path,
            attn_mean=attn_mean,
            hidden_mean=hidden_mean,
            prompt_lengths=prompt_lengths,
            token_ranking=token_ranking,
            model_name=model_name,
            split=split,
            sample_count=sample_count,
            sample_index=sample_index,
        )

    _plot_attention_entropy(attn_mean, attention_path, model_name, split, sample_count)
    _plot_hidden_norm(hidden_mean, hidden_path, model_name, split, sample_count)
    if token_ranking:
        _plot_token_importance(token_ranking, token_path, model_name, split)

    summary = {
        "model_name": model_name,
        "split": split,
        "num_samples": sample_count,
        "avg_prompt_tokens_used": sum(prompt_lengths) / len(prompt_lengths),
        "attention_entropy_heatmap": str(attention_path),
        "hidden_state_norms": str(hidden_path),
        "token_importance_sample": str(token_path),
        "figure_data_cache": str(cache_path),
        "token_importance_cache": str(ranking_cache_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "attention_entropy_heatmap": str(attention_path),
        "hidden_state_norms": str(hidden_path),
        "token_importance_sample": str(token_path),
        "figure_data_cache": str(cache_path),
        "token_importance_cache": str(ranking_cache_path),
        "summary": str(summary_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BillSum internal representation visualisations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test_us", choices=["valid", "test_us", "test_ca"])
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--output-tag", default="pretrained_visuals")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    out = generate_billsum_visuals(
        config_path=args.config,
        split=args.split,
        limit=args.limit,
        output_tag=args.output_tag,
        sample_index=args.sample_index,
        reuse_cache=args.reuse_cache,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
