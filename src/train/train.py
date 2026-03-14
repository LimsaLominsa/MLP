"""
train.py
LoRA fine-tuning script for BillSum (summarization) and CaseHOLD (classification).

Usage:
  python src/train/train.py --config configs/lora_billsum_qwen.yaml
  python src/train/train.py --config configs/lora_casehold_llama.yaml

Supports:
  - Qwen2.5-1.5B-Instruct
  - Llama-3.2-1B-Instruct
  - Any causal LM supported by transformers + peft
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Dataset preparation ───────────────────────────────────────────────────────

def build_dataset(records: list,
                  tokenizer,
                  max_input_length: int,
                  max_output_length: int) -> Dataset:
    """
    Tokenize SFT records.

    The 'input' field is the prompt; 'output' is the target.
    Labels for the input portion are set to -100 (masked from loss).
    """
    def tokenize(example):
        # Tokenize input (prompt only, no padding)
        input_ids = tokenizer(
            example["input"],
            truncation=True,
            max_length=max_input_length,
            add_special_tokens=True,
        )["input_ids"]

        # Tokenize output (target, no padding)
        output_ids = tokenizer(
            example["output"],
            truncation=True,
            max_length=max_output_length,
            add_special_tokens=False,
        )["input_ids"]

        # Append EOS token to output
        if tokenizer.eos_token_id is not None:
            output_ids = output_ids + [tokenizer.eos_token_id]

        full_ids  = input_ids + output_ids
        # Mask input portion from loss; only compute loss on output tokens
        labels    = [-100] * len(input_ids) + output_ids

        return {
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels,
        }

    dataset = Dataset.from_list(records)
    dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    return dataset


# ── Model & LoRA setup ────────────────────────────────────────────────────────

def build_model(cfg: dict):
    """
    Load base model.
    - If config contains a 'lora' section: apply LoRA adapter (PEFT).
    - Otherwise: full fine-tuning, all parameters trainable.
    Returns (model, tokenizer, is_lora).
    """
    model_name = cfg["model"]["name"]
    is_lora    = "lora" in cfg

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg["training"]["bf16"]:
        torch_dtype = torch.bfloat16
    elif cfg["training"]["fp16"]:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32   # CPU-safe fallback

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False   # required for gradient checkpointing

    if is_lora:
        # Parameter-efficient fine-tuning via LoRA
        lora_cfg = cfg["lora"]
        lora_config = LoraConfig(
            r                = lora_cfg["r"],
            lora_alpha       = lora_cfg["lora_alpha"],
            lora_dropout     = lora_cfg["lora_dropout"],
            target_modules   = lora_cfg["target_modules"],
            bias             = lora_cfg["bias"],
            task_type        = TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # Full fine-tuning — all parameters are trainable
        for param in model.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in model.parameters())
        print(f"Full fine-tuning: {total / 1e9:.2f}B trainable parameters")

    return model, tokenizer, is_lora


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    # Resolve data paths relative to repo root
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_cfg  = cfg["data"]
    train_path = repo_root / data_cfg["train"]
    val_path   = repo_root / data_cfg["val"]

    max_input  = data_cfg.get("max_input_length") or data_cfg.get("max_length") or 2048
    max_output = data_cfg.get("max_output_length", 256)

    print(f"Config  : {args.config}")
    print(f"Model   : {cfg['model']['name']}")
    print(f"Train   : {train_path}  ({sum(1 for _ in open(train_path)):,} records)")
    print(f"Val     : {val_path}")
    print(f"Max input length : {max_input}")
    print(f"Max output length: {max_output}")

    # Build model + tokenizer
    model, tokenizer, is_lora = build_model(cfg)

    # Load and tokenize data
    print("\nPreparing datasets...")
    train_records = load_jsonl(str(train_path))
    val_records   = load_jsonl(str(val_path))

    train_dataset = build_dataset(train_records, tokenizer, max_input, max_output)
    val_dataset   = build_dataset(val_records,   tokenizer, max_input, max_output)

    print(f"Train tokens — first sample length: "
          f"{len(train_dataset[0]['input_ids'])}")

    # Training arguments
    t = cfg["training"]
    output_dir = repo_root / cfg["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = str(output_dir),
        num_train_epochs            = t["num_train_epochs"],
        max_steps                   = t.get("max_steps", -1),  # -1 = use num_epochs
        per_device_train_batch_size = t["per_device_train_batch_size"],
        per_device_eval_batch_size  = t["per_device_eval_batch_size"],
        gradient_accumulation_steps = t["gradient_accumulation_steps"],
        gradient_checkpointing      = t["gradient_checkpointing"],
        learning_rate               = t["learning_rate"],
        lr_scheduler_type           = t["lr_scheduler_type"],
        warmup_ratio                = t["warmup_ratio"],
        bf16                        = t["bf16"],
        fp16                        = t["fp16"],
        optim                       = t["optim"],
        weight_decay                = t["weight_decay"],
        max_grad_norm               = t["max_grad_norm"],
        eval_strategy               = t["evaluation_strategy"],
        eval_steps                  = t["eval_steps"],
        save_strategy               = t["save_strategy"],
        save_steps                  = t["save_steps"],
        save_total_limit            = t["save_total_limit"],
        load_best_model_at_end      = t["load_best_model_at_end"],
        metric_for_best_model       = t["metric_for_best_model"],
        report_to                   = t["report_to"],
        logging_steps               = t["logging_steps"],
        seed                        = t["seed"],
    )

    # Data collator — handles dynamic padding within a batch
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # Trainer
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        data_collator   = data_collator,
    )

    print("\nStarting training...")
    trainer.train()

    # Save final model / adapter
    save_name = "final_adapter" if is_lora else "final_model"
    final_dir = output_dir / save_name
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nModel saved to: {final_dir}")


if __name__ == "__main__":
    main()
