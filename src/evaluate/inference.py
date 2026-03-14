"""
inference.py
Run test-set inference using a fine-tuned model (LoRA adapter or full model).

Outputs a predictions JSONL file that eval_billsum.py / eval_casehold.py can consume.

Usage — BillSum:
  python src/evaluate/inference.py \\
      --config configs/lora_billsum_qwen.yaml \\
      --split  test_us

  python src/evaluate/inference.py \\
      --config configs/lora_billsum_qwen.yaml \\
      --split  test_ca

Usage — CaseHOLD:
  python src/evaluate/inference.py \\
      --config configs/lora_casehold_qwen.yaml \\
      --split  test

Output file is written to:
  outputs/<experiment>/predictions_<split>.jsonl

Then run evaluation:
  python src/evaluate/eval_billsum.py \\
      --predictions outputs/lora_billsum_qwen/predictions_test_us.jsonl \\
      --output      outputs/lora_billsum_qwen/eval_test_us.json

  python src/evaluate/eval_casehold.py \\
      --predictions outputs/lora_casehold_qwen/predictions_test.jsonl \\
      --output      outputs/lora_casehold_qwen/eval_test.json
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_model(cfg: dict, output_dir: Path):
    """
    Load the fine-tuned model for inference.
    - LoRA config  → load base model + merge adapter from final_adapter/
    - Full FT config → load directly from final_model/
    """
    model_name = cfg["model"]["name"]
    is_lora    = "lora" in cfg

    if cfg["training"]["bf16"] and torch.cuda.is_available():
        torch_dtype = torch.bfloat16
    elif cfg["training"]["fp16"] and torch.cuda.is_available():
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
        # Merge adapter weights into base model for faster inference
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

    model.eval()
    return model, tokenizer


# ── Generation helpers ────────────────────────────────────────────────────────

def generate_summary(model, tokenizer, prompt: str,
                     max_new_tokens: int = 256,
                     device: str = "cuda") -> str:
    """Generate a free-form summary (BillSum)."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy for reproducibility
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens (exclude the prompt)
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def generate_choice(model, tokenizer, prompt: str,
                    device: str = "cuda") -> str:
    """
    Generate a multiple-choice answer (CaseHOLD).
    Returns the first token of the generation as the predicted letter (A–E).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=5,         # only need the answer letter
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    # Extract the first meaningful character (A–E)
    for ch in raw.upper():
        if ch in "ABCDE":
            return ch
    return raw  # return as-is; eval_casehold.py will flag as invalid


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to YAML config (same config used for training)")
    parser.add_argument("--split", required=True,
                        help="Data split key: test_us | test_ca | test")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Inference batch size (default: 1)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max tokens to generate (BillSum only; default: 256)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_root  = Path(__file__).resolve().parent.parent.parent
    output_dir = repo_root / cfg["output"]["dir"]
    data_cfg   = cfg["data"]
    task       = cfg["model"]["task"]  # "summarization" or "classification"

    # Resolve the test file path
    split_key = args.split  # e.g. "test_us", "test_ca", "test"
    if split_key not in data_cfg:
        raise ValueError(
            f"Split '{split_key}' not found in config data section. "
            f"Available keys: {list(data_cfg.keys())}"
        )
    test_path = repo_root / data_cfg[split_key]

    print(f"Config  : {args.config}")
    print(f"Task    : {task}")
    print(f"Split   : {split_key}  →  {test_path}")
    print(f"Output  : {output_dir}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(cfg, output_dir)

    # Load test records
    records = load_jsonl(str(test_path))
    print(f"Loaded {len(records):,} test records")

    # Run inference
    predictions = []
    for i, rec in enumerate(records):
        prompt = rec["input"]

        if task == "summarization":
            pred = generate_summary(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                device=device,
            )
            entry = {
                "prediction": pred,
                "reference":  rec.get("output", rec.get("summary", "")),
            }
            # Carry over optional identifier fields
            for key in ("bill_id", "id"):
                if key in rec:
                    entry[key] = rec[key]
                    break
        else:
            # classification (CaseHOLD)
            pred = generate_choice(model, tokenizer, prompt, device=device)
            entry = {
                "prediction": pred,
                "reference":  rec.get("output", rec.get("label", "")),
            }
            for key in ("example_id", "id"):
                if key in rec:
                    entry[key] = rec[key]
                    break

        predictions.append(entry)

        if (i + 1) % 100 == 0 or (i + 1) == len(records):
            print(f"  [{i + 1}/{len(records)}] done")

    # Save predictions
    out_file = output_dir / f"predictions_{split_key}.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in predictions:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nPredictions saved → {out_file}")
    print(f"Next step:")
    if task == "summarization":
        print(f"  python src/evaluate/eval_billsum.py \\")
        print(f"      --predictions {out_file} \\")
        print(f"      --output {output_dir}/eval_{split_key}.json")
    else:
        print(f"  python src/evaluate/eval_casehold.py \\")
        print(f"      --predictions {out_file} \\")
        print(f"      --output {output_dir}/eval_{split_key}.json")


if __name__ == "__main__":
    main()
