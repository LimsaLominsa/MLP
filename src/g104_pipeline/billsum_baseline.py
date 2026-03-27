from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from rouge_score import rouge_scorer
from tqdm import tqdm


def _require_hf():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "BillSum baseline requires torch, transformers, and peft. Install requirements first."
        ) from e

    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "PeftModel": PeftModel,
    }


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required = ["project", "data", "model_name", "generation", "evaluation"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    base_dir = cfg_path.parent
    project = cfg["project"]
    data = cfg["data"]

    project["output_root"] = str((base_dir / project["output_root"]).resolve())
    for key, value in list(data.items()):
        if key.endswith("_file"):
            data[key] = str((base_dir / value).resolve())
    if cfg.get("adapter_dir"):
        cfg["adapter_dir"] = str((base_dir / cfg["adapter_dir"]).resolve())
    return cfg


def _read_jsonl(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _torch_dtype(torch_mod):
    if torch_mod.cuda.is_available():
        return torch_mod.bfloat16
    return torch_mod.float32


def _device(torch_mod) -> str:
    return "cuda" if torch_mod.cuda.is_available() else "cpu"


def _load_model_and_tokenizer(
    *,
    cfg: Dict[str, Any],
    hf: Dict[str, Any],
    attn_implementation: str | None = None,
):
    torch = hf["torch"]
    AutoTokenizer = hf["AutoTokenizer"]
    AutoModelForCausalLM = hf["AutoModelForCausalLM"]
    PeftModel = hf["PeftModel"]

    model_name = cfg["model_name"]
    adapter_dir = cfg.get("adapter_dir")
    # Always load tokenizer from base model to avoid incompatible
    # tokenizer_config.json in older adapter checkpoints.
    tokenizer_source = model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": _torch_dtype(torch),
        "trust_remote_code": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    if adapter_dir:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    model.to(_device(torch))
    model.eval()
    return model, tokenizer


def _sample_id(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]


_WHITESPACE_RE = re.compile(r"\s+")
_STOP_MARKERS = ("Human:", "Assistant:", "User:", "###")
_SYSTEM_PROMPT = (
    "You are a legal summarisation assistant. Write a concise, faithful summary of the "
    "bill. Return only a single-paragraph summary. Do not add headings, dialogue, "
    "questions, or extra commentary."
)


def _normalise_summary(text: str) -> str:
    text = text.strip()
    if "### Summary:" in text:
        text = text.split("### Summary:", 1)[-1].strip()
    cut_points = [text.find(marker) for marker in _STOP_MARKERS if text.find(marker) != -1]
    if cut_points:
        text = text[: min(cut_points)].strip()
    return _WHITESPACE_RE.sub(" ", text).strip()


def _extract_bill_text(prompt: str) -> str:
    text = prompt.strip()
    if "### Bill:" in text:
        text = text.split("### Bill:", 1)[-1].strip()
    if "### Summary:" in text:
        text = text.split("### Summary:", 1)[0].strip()
    return text


def _build_prompt(tokenizer, raw_prompt: str) -> str:
    bill_text = _extract_bill_text(raw_prompt)
    user_prompt = (
        "Summarise the following US legislative bill.\n\n"
        "Return only a single-paragraph summary. Do not add headings, dialogue, "
        "questions, or extra commentary.\n\n"
        f"{bill_text}"
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return (
        f"System: {_SYSTEM_PROMPT}\n\n"
        f"User: {user_prompt}\n\n"
        "Assistant:"
    )


def _terminator_ids(tokenizer) -> int | List[int] | None:
    ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))

    for token in ("<|eot_id|>", "<|eom_id|>", "<|im_end|>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except KeyError:
            token_id = None
        if token_id is None or token_id == tokenizer.unk_token_id or token_id < 0:
            continue
        ids.append(int(token_id))

    ids = list(dict.fromkeys(ids))
    if not ids:
        return None
    if len(ids) == 1:
        return ids[0]
    return ids


def _generate_one(
    *,
    model,
    tokenizer,
    torch_mod,
    prompt: str,
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    num_beams: int,
) -> Tuple[str, int, int]:
    prompt = _build_prompt(tokenizer, prompt)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    encoded = {k: v.to(_device(torch_mod)) for k, v in encoded.items()}
    prompt_len = int(encoded["input_ids"].shape[1])
    eos_token_id = _terminator_ids(tokenizer)

    with torch_mod.no_grad():
        generate_kwargs = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id
        out = model.generate(**generate_kwargs)

    gen_ids = out[0][prompt_len:]
    pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
    pred = _normalise_summary(pred)
    return pred, prompt_len, int(gen_ids.shape[0])


def _score_rouge(rows: List[Tuple[str, str]], use_stemmer: bool) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    for pred, ref in rows:
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += float(scores[key].fmeasure)

    n = max(1, len(rows))
    return {key: value / n for key, value in totals.items()}


def _slice_rows(
    rows: List[Dict[str, Any]],
    shard_id: int,
    num_shards: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    indexed_rows = list(enumerate(rows))
    if num_shards == 1:
        return indexed_rows
    return [(idx, row) for idx, row in indexed_rows if idx % num_shards == shard_id]


def _output_root_for_run(
    cfg: Dict[str, Any],
    split: str,
    output_tag: str | None,
    shard_id: int | None = None,
    num_shards: int = 1,
) -> Path:
    output_root = Path(cfg["project"]["output_root"]) / "baseline" / split
    if output_tag:
        output_root = output_root / output_tag
    if shard_id is not None and num_shards > 1:
        output_root = output_root / f"shard_{shard_id:02d}_of_{num_shards:02d}"
    return output_root


def _collect_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _load_existing_predictions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return _collect_prediction_rows(path)


def run_billsum_baseline(
    config_path: str,
    split: str,
    limit: int | None = None,
    output_tag: str | None = None,
    shard_id: int = 0,
    num_shards: int = 1,
    resume: bool = False,
) -> Dict[str, Any]:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards})")

    cfg = _load_config(config_path)
    hf = _require_hf()
    torch = hf["torch"]
    split_key = f"{split}_file"
    data_cfg = cfg["data"]
    if split_key not in data_cfg:
        raise ValueError(f"Unsupported split `{split}`. Expected one of: test_us, test_ca, valid")

    input_path = Path(data_cfg[split_key])
    rows = _read_jsonl(input_path, limit=limit)
    shard_rows = _slice_rows(rows, shard_id=shard_id, num_shards=num_shards)
    if not shard_rows:
        raise RuntimeError(f"No rows found in {input_path}")

    gen_cfg = cfg["generation"]
    eval_cfg = cfg["evaluation"]

    model, tokenizer = _load_model_and_tokenizer(cfg=cfg, hf=hf)

    output_root = _output_root_for_run(
        cfg,
        split,
        output_tag=output_tag,
        shard_id=shard_id if num_shards > 1 else None,
        num_shards=num_shards,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    pred_path = output_root / "predictions.jsonl"

    existing_rows = _load_existing_predictions(pred_path) if resume else []
    completed_indices = {
        int(row["dataset_index"])
        for row in existing_rows
        if "dataset_index" in row
    }
    if resume and existing_rows and not completed_indices:
        raise RuntimeError(
            f"Resume requested but {pred_path} has no dataset_index fields. "
            "Use a fresh output tag or regenerate this shard with the current script."
        )

    pred_rows: List[Dict[str, Any]] = list(existing_rows)
    rouge_pairs: List[Tuple[str, str]] = []
    prompt_token_lengths: List[int] = []
    generated_token_lengths: List[int] = []

    for row in existing_rows:
        rouge_pairs.append((row["prediction"], row["reference"]))
        prompt_token_lengths.append(int(row["prompt_tokens_used"]))
        generated_token_lengths.append(int(row["generated_tokens"]))

    pending_rows = [(idx, row) for idx, row in shard_rows if idx not in completed_indices]

    desc = f"{Path(config_path).stem}:{split}"
    if num_shards > 1:
        desc = f"{desc}:shard{shard_id + 1}/{num_shards}"
    if resume and completed_indices:
        desc = f"{desc}:resume"

    for dataset_index, row in tqdm(pending_rows, desc=desc, unit="sample"):
        pred, prompt_tok_len, gen_tok_len = _generate_one(
            model=model,
            tokenizer=tokenizer,
            torch_mod=torch,
            prompt=row["input"],
            max_input_tokens=int(gen_cfg.get("max_input_tokens", 2048)),
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 192)),
            do_sample=bool(gen_cfg.get("do_sample", False)),
            num_beams=int(gen_cfg.get("num_beams", 1)),
        )
        ref = _normalise_summary(row["output"])
        rouge_pairs.append((pred, ref))
        prompt_token_lengths.append(prompt_tok_len)
        generated_token_lengths.append(gen_tok_len)

        pred_rows.append(
            {
                "dataset_index": dataset_index,
                "sample_id": _sample_id(row["input"]),
                "prediction": pred,
                "reference": ref,
                "prompt_chars": len(row["input"]),
                "reference_chars": len(ref),
                "prediction_chars": len(pred),
                "prompt_tokens_used": prompt_tok_len,
                "generated_tokens": gen_tok_len,
            }
        )

    rouge = _score_rouge(rouge_pairs, use_stemmer=bool(eval_cfg.get("rouge_use_stemmer", True)))
    metrics = {
        "task": "billsum",
        "model_name": cfg["model_name"],
        "adapter_dir": cfg.get("adapter_dir"),
        "split": split,
        "num_samples": len(pred_rows),
        "shard_id": shard_id,
        "num_shards": num_shards,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "max_input_tokens": int(gen_cfg.get("max_input_tokens", 2048)),
        "max_new_tokens": int(gen_cfg.get("max_new_tokens", 192)),
        "avg_prompt_tokens_used": sum(prompt_token_lengths) / len(prompt_token_lengths),
        "avg_generated_tokens": sum(generated_token_lengths) / len(generated_token_lengths),
    }

    pred_rows.sort(key=lambda row: int(row.get("dataset_index", 0)))
    _write_json(output_root / "metrics.json", metrics)
    _write_jsonl(pred_path, pred_rows)
    return metrics


def merge_billsum_shards(
    config_path: str,
    split: str,
    num_shards: int,
    output_tag: str | None = None,
) -> Dict[str, Any]:
    if num_shards < 2:
        raise ValueError("merge_billsum_shards requires num_shards >= 2")

    cfg = _load_config(config_path)
    eval_cfg = cfg["evaluation"]
    shard_rows: List[Dict[str, Any]] = []
    for shard_id in range(num_shards):
        shard_root = _output_root_for_run(
            cfg,
            split,
            output_tag=output_tag,
            shard_id=shard_id,
            num_shards=num_shards,
        )
        pred_path = shard_root / "predictions.jsonl"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing shard predictions: {pred_path}")
        shard_rows.extend(_collect_prediction_rows(pred_path))

    shard_rows.sort(key=lambda row: int(row.get("dataset_index", 0)))
    rouge_pairs = [(row["prediction"], row["reference"]) for row in shard_rows]
    rouge = _score_rouge(rouge_pairs, use_stemmer=bool(eval_cfg.get("rouge_use_stemmer", True)))

    metrics = {
        "task": "billsum",
        "model_name": cfg["model_name"],
        "adapter_dir": cfg.get("adapter_dir"),
        "split": split,
        "num_samples": len(shard_rows),
        "num_shards": num_shards,
        "merged_from_shards": True,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "avg_prompt_tokens_used": sum(float(row["prompt_tokens_used"]) for row in shard_rows) / len(shard_rows),
        "avg_generated_tokens": sum(float(row["generated_tokens"]) for row in shard_rows) / len(shard_rows),
    }

    output_root = _output_root_for_run(cfg, split, output_tag=output_tag)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "metrics.json", metrics)
    _write_jsonl(output_root / "predictions.jsonl", shard_rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pretrained BillSum summarisation baseline and compute ROUGE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test_us", choices=["valid", "test_us", "test_ca"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.merge_shards:
        metrics = merge_billsum_shards(
            args.config,
            args.split,
            num_shards=args.num_shards,
            output_tag=args.output_tag,
        )
    else:
        metrics = run_billsum_baseline(
            args.config,
            args.split,
            limit=args.limit,
            output_tag=args.output_tag,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            resume=args.resume,
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
