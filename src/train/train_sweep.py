"""
train_sweep.py
W&B Sweep-compatible training script for BillSum LoRA hyperparameter search.

Loads a base YAML config, then overrides hyperparameters injected by the W&B
sweep agent (learning_rate, lora_r, lora_dropout, num_train_epochs).

Called automatically by `wandb agent` — do NOT run directly unless debugging.

Usage (via sweep agent — normal path):
  wandb sweep configs/sweep_billsum_qwen.yaml        # register sweep once
  sbatch scripts/slurm_sweep_agent.sh <sweep_id>    # run agents on cluster

Usage (manual debug run):
  python src/train/train_sweep.py --config configs/lora_billsum_qwen.yaml
"""

import os
import sys
import copy
import argparse
from pathlib import Path

import torch
import wandb
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
)

# Reuse all helpers from train.py (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_config, load_jsonl, build_dataset, build_model


# ── Override helpers ──────────────────────────────────────────────────────────

def apply_sweep_overrides(cfg: dict, sweep_cfg) -> dict:
    """
    Deep-copy the base config and patch any parameters that the W&B sweep
    agent has set. Missing keys are silently ignored (base config value kept).
    """
    cfg = copy.deepcopy(cfg)

    if hasattr(sweep_cfg, "learning_rate"):
        cfg["training"]["learning_rate"] = float(sweep_cfg.learning_rate)

    if hasattr(sweep_cfg, "lora_r"):
        r = int(sweep_cfg.lora_r)
        cfg["lora"]["r"] = r
        cfg["lora"]["lora_alpha"] = r * 2   # standard: alpha = 2 × rank

    if hasattr(sweep_cfg, "lora_dropout"):
        cfg["lora"]["lora_dropout"] = float(sweep_cfg.lora_dropout)

    if hasattr(sweep_cfg, "num_train_epochs"):
        cfg["training"]["num_train_epochs"] = int(sweep_cfg.num_train_epochs)

    return cfg


# ── Main training function ────────────────────────────────────────────────────

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to base YAML config (e.g. configs/lora_billsum_qwen.yaml)")
    args = parser.parse_args()

    # W&B init — sweep agent injects hyperparameters into wandb.config BEFORE
    # this process gets to use them. Must happen before we read wandb.config.
    run = wandb.init()

    # Load base config and apply sweep overrides
    base_cfg = load_config(args.config)
    cfg      = apply_sweep_overrides(base_cfg, wandb.config)

    # Each sweep run writes to its own output directory to avoid collisions
    cfg["output"]["dir"] = cfg["output"]["dir"] + f"_sweep_{run.id}"

    # Log the resolved config back to W&B for full reproducibility
    wandb.config.update({
        "resolved_lr":          cfg["training"]["learning_rate"],
        "resolved_lora_r":      cfg["lora"].get("r"),
        "resolved_lora_alpha":  cfg["lora"].get("lora_alpha"),
        "resolved_lora_dropout":cfg["lora"].get("lora_dropout"),
        "resolved_epochs":      cfg["training"]["num_train_epochs"],
    }, allow_val_change=True)

    set_seed(cfg["training"]["seed"])

    # ── Data ──────────────────────────────────────────────────────────────────
    repo_root  = Path(__file__).resolve().parent.parent.parent
    data_cfg   = cfg["data"]
    train_path = repo_root / data_cfg["train"]
    val_path   = repo_root / data_cfg["val"]

    max_input  = data_cfg.get("max_input_length") or data_cfg.get("max_length") or 2048
    max_output = data_cfg.get("max_output_length", 256)

    print(f"\nConfig  : {args.config}")
    print(f"Model   : {cfg['model']['name']}")
    print(f"LR      : {cfg['training']['learning_rate']}")
    print(f"LoRA r  : {cfg['lora'].get('r', 'N/A')} | alpha: {cfg['lora'].get('lora_alpha', 'N/A')}")
    print(f"Dropout : {cfg['lora'].get('lora_dropout', 'N/A')}")
    print(f"Epochs  : {cfg['training']['num_train_epochs']}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer, is_lora = build_model(cfg)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_records = load_jsonl(str(train_path))
    val_records   = load_jsonl(str(val_path))
    train_dataset = build_dataset(train_records, tokenizer, max_input, max_output)
    val_dataset   = build_dataset(val_records,   tokenizer, max_input, max_output)

    # ── TrainingArguments ─────────────────────────────────────────────────────
    t          = cfg["training"]
    output_dir = repo_root / cfg["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir                  = str(output_dir),
        num_train_epochs            = t["num_train_epochs"],
        max_steps                   = t.get("max_steps", -1),
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
        # Don't save model checkpoints during sweep — saves disk / time.
        # Only the final best model will be saved after Phase 1 is complete.
        save_strategy               = "no",
        load_best_model_at_end      = False,
        report_to                   = "wandb",
        logging_steps               = t["logging_steps"],
        seed                        = t["seed"],
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        data_collator = data_collator,
    )

    trainer.train()

    # Log final eval loss explicitly so W&B sweep can rank this run
    final_metrics = trainer.evaluate()
    wandb.log({"final_eval_loss": final_metrics.get("eval_loss", float("inf"))})

    wandb.finish()
    print(f"\nSweep run complete. Output: {output_dir}")


if __name__ == "__main__":
    train()
