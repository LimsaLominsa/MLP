"""
inference.py
Run test-set inference using a fine-tuned model (LoRA adapter or full model).
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_model(cfg: dict, output_dir: Path):
    model_name = cfg["model"]["name"]
    is_lora    = "lora" in cfg

    if cfg["training"].get("bf16", False) and torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    elif cfg["training"].get("fp16", False) and torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if is_lora:
        from peft import PeftModel
        adapter_path = output_dir / "final_adapter"
        print(f"Loading base model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_path), trust_remote_code=True
        )
    else:
        model_path = output_dir / "final_model"
        print(f"Loading full fine-tuned model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Causal LM batched inference requires left-padding
    tokenizer.padding_side = "left"

    model.eval()
    return model, tokenizer


def generate_summaries_batch(model, tokenizer, prompts: list,
                              max_new_tokens: int = 256) -> list:
    """Batched summary generation for BillSum."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Each sample: strip the prompt tokens (input length may differ due to padding)
    input_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(len(prompts)):
        new_ids = output_ids[i][input_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        results.append(text)
    return results


def generate_choices_batch(model, tokenizer, prompts: list) -> list:
    """Batched choice generation for CaseHOLD."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    results = []
    for i in range(len(prompts)):
        new_ids = output_ids[i][input_len:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        pred = raw
        for ch in raw.upper():
            if ch in "ABCDE":
                pred = ch
                break
        results.append(pred)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Inference batch size (default: 8)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root  = Path(__file__).resolve().parent.parent.parent
    output_dir = repo_root / cfg["output"]["dir"]
    data_cfg   = cfg["data"]
    task       = cfg["model"]["task"]

    split_key = args.split
    if split_key not in data_cfg:
        raise ValueError(
            f"Split '{split_key}' not found in config. "
            f"Available: {list(data_cfg.keys())}"
        )
    test_path = repo_root / data_cfg[split_key]

    print(f"Config     : {args.config}")
    print(f"Task       : {task}")
    print(f"Split      : {split_key}  →  {test_path}")
    print(f"Output     : {output_dir}")
    print(f"Batch size : {args.batch_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(cfg, output_dir)

    records = load_jsonl(str(test_path))
    print(f"Loaded {len(records):,} test records")

    predictions = []
    batch_size = args.batch_size

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start: batch_start + batch_size]
        prompts = [r["input"] for r in batch]

        if task == "summarization":
            preds = generate_summaries_batch(
                model, tokenizer, prompts, max_new_tokens=args.max_new_tokens
            )
            for rec, pred in zip(batch, preds):
                entry = {
                    "prediction": pred,
                    "reference":  rec.get("output", rec.get("summary", "")),
                }
                for key in ("bill_id", "id"):
                    if key in rec:
                        entry[key] = rec[key]
                        break
                predictions.append(entry)
        else:
            preds = generate_choices_batch(model, tokenizer, prompts)
            for rec, pred in zip(batch, preds):
                entry = {
                    "prediction": pred,
                    "reference":  rec.get("output", rec.get("label", "")),
                }
                for key in ("example_id", "id"):
                    if key in rec:
                        entry[key] = rec[key]
                        break
                predictions.append(entry)

        done = min(batch_start + batch_size, len(records))
        if done % 200 == 0 or done == len(records):
            print(f"  [{done}/{len(records)}] done")

    out_file = output_dir / f"predictions_{split_key}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in predictions:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nPredictions saved → {out_file}")
    if task == "summarization":
        print(f"Next: python src/evaluate/eval_billsum.py "
              f"--predictions {out_file} "
              f"--output {output_dir}/eval_{split_key}.json")
    else:
        print(f"Next: python src/evaluate/eval_casehold.py "
              f"--predictions {out_file} "
              f"--output {output_dir}/eval_{split_key}.json")


if __name__ == "__main__":
    main()