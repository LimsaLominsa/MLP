"""Extract internal signals (attention + activation) from PubMed summarization models."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .billsum_baseline import (
    _device,
    _load_config,
    _load_model_and_tokenizer,
    _read_jsonl,
    _require_hf,
)
from .io_utils import write_json, write_parquet

# ── PubMed-specific prompt builder ──────────────────────────────────────────

_PUBMED_SYSTEM_PROMPT = (
    "You are a biomedical summarisation assistant. Write a concise summary of the "
    "research article. Return only a single-paragraph summary."
)


def _extract_article_text(raw_input: str) -> str:
    """Extract the article body from a PubMed inference prompt."""
    text = raw_input.strip()
    if "### Article:" in text:
        text = text.split("### Article:", 1)[-1].strip()
    if "### Summary:" in text:
        text = text.split("### Summary:", 1)[0].strip()
    return text


def _build_pubmed_prompt(tokenizer, raw_input: str) -> str:
    article_text = _extract_article_text(raw_input)
    user_prompt = (
        "Below is a biomedical research article. Write a concise summary.\n\n"
        f"{article_text}"
    )
    messages = [
        {"role": "system", "content": _PUBMED_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {_PUBMED_SYSTEM_PROMPT}\n\nUser: {user_prompt}\n\nAssistant:"


# ── Signal extraction ───────────────────────────────────────────────────────

def extract_pubmed_signals(
    config_path: str,
    split: str = "test",
    subset_size: int = 64,
    subset_seed: int = 42,
    max_input_tokens: int | None = None,
) -> str:
    cfg = _load_config(config_path)
    hf = _require_hf()
    torch = hf["torch"]

    input_path = Path(cfg["data"][f"{split}_file"])
    rows = _read_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"No rows found in {input_path}")

    rng = random.Random(subset_seed)
    indexed_rows = list(enumerate(rows))
    rng.shuffle(indexed_rows)
    chosen = indexed_rows[: min(subset_size, len(indexed_rows))]
    chosen.sort(key=lambda item: item[0])

    model, tokenizer = _load_model_and_tokenizer(
        cfg=cfg, hf=hf, attn_implementation="eager",
    )

    out_rows: List[Dict[str, Any]] = []
    token_limit = int(max_input_tokens or cfg["generation"].get("max_input_tokens", 2048))

    for dataset_index, row in chosen:
        prompt = _build_pubmed_prompt(tokenizer, row["input"])
        encoded = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=token_limit,
        )
        encoded = {k: v.to(_device(torch)) for k, v in encoded.items()}

        with torch.no_grad():
            out = model(**encoded, output_attentions=True, output_hidden_states=True)

        attentions = out.attentions
        hidden_states = out.hidden_states[1:]
        sample_id = row.get("sample_id") or f"{split}-{dataset_index}"

        for layer, att in enumerate(attentions):
            att_layer = att[0].float()
            hs = hidden_states[layer][0].float()
            act_scalar = float(torch.norm(hs, dim=-1).mean().item())
            n_heads = int(att_layer.shape[0])

            for head in range(n_heads):
                att_scalar = float(att_layer[head].mean().item())
                out_rows.append({
                    "sample_id": sample_id,
                    "dataset_index": dataset_index,
                    "layer": layer,
                    "head": head,
                    "attention": att_scalar,
                    "activation": act_scalar,
                    "attribution": float(abs(att_scalar) * act_scalar),
                })

    output_root = Path(cfg["project"]["output_root"]) / "analysis" / split
    output_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    out_path = output_root / "signals.parquet"
    write_parquet(out_path, df)
    write_json(output_root / "signals_summary.json", {
        "task": "pubmed",
        "model_name": cfg["model_name"],
        "adapter_dir": cfg.get("adapter_dir"),
        "split": split,
        "subset_size": subset_size,
        "subset_seed": subset_seed,
        "max_input_tokens": token_limit,
        "num_rows": len(df),
        "num_samples": len({r["sample_id"] for r in out_rows}),
        "signals_path": str(out_path),
    })
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PubMed internal signals.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--subset-seed", type=int, default=42)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    args = parser.parse_args()

    out = extract_pubmed_signals(
        config_path=args.config, split=args.split,
        subset_size=args.subset_size, subset_seed=args.subset_seed,
        max_input_tokens=args.max_input_tokens,
    )
    print(out)


if __name__ == "__main__":
    main()
