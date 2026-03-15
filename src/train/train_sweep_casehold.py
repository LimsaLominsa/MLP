"""
train_sweep_casehold.py
W&B Sweep-compatible training script for CaseHOLD QLoRA hyperparameter search.

Loads a base YAML config (qlora_casehold_*.yaml), then overrides hyperparameters
injected by the W&B sweep agent (learning_rate, lora_r, lora_dropout, num_train_epochs).

Called automatically by `wandb agent` — do NOT run directly unless debugging.

Usage (via sweep agent — normal path):
  wandb sweep configs/sweep_casehold_qwen.yaml      # register sweep once
  sbatch --array=1-3 scripts/slurm_sweep_agent.sh <sweep_id> 3

Usage (manual debug):
  python src/train/train_sweep_casehold.py --config configs/qlora_casehold_qwen.yaml
"""

import sys
import argparse
from pathlib import Path

import wandb

# Import helpers from the QLoRA training script in the same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_casehold_lora import TrainConfig, run_train, _load_config_defaults

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to base YAML config (e.g. configs/qlora_casehold_qwen.yaml)")
    args = parser.parse_args()

    # W&B init — sweep agent injects hyperparameters into wandb.config before this runs
    run = wandb.init()

    # Load and flatten base config
    defaults = _load_config_defaults(args.config)

    # ── Apply sweep overrides ────────────────────────────────────────────────
    sweep_cfg = wandb.config

    if "learning_rate" in sweep_cfg:
        defaults["learning_rate"] = float(sweep_cfg["learning_rate"])
    if "lora_r" in sweep_cfg:
        r = int(sweep_cfg["lora_r"])
        defaults["lora_r"] = r
        defaults["lora_alpha"] = r * 2  # standard: alpha = 2 × rank
    if "lora_dropout" in sweep_cfg:
        defaults["lora_dropout"] = float(sweep_cfg["lora_dropout"])
    if "num_train_epochs" in sweep_cfg:
        defaults["num_train_epochs"] = float(sweep_cfg["num_train_epochs"])

    # Each sweep run gets its own output directory
    base_dir = defaults.get("output_dir", "outputs/qlora_casehold_sweep")
    defaults["output_dir"] = f"{base_dir}_sweep_{run.id}"
    defaults["report_to"] = "wandb"

    # Log resolved hyperparameters for full reproducibility in W&B
    wandb.config.update({
        "resolved_lr":           defaults.get("learning_rate"),
        "resolved_lora_r":       defaults.get("lora_r"),
        "resolved_lora_alpha":   defaults.get("lora_alpha"),
        "resolved_lora_dropout": defaults.get("lora_dropout"),
        "resolved_epochs":       defaults.get("num_train_epochs"),
    }, allow_val_change=True)

    # Validate required fields
    missing = [k for k in ["model_name_or_path", "train_file", "validation_file", "output_dir"]
               if not defaults.get(k)]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    # Build target_modules (may be list or comma-separated string)
    target_modules = defaults.get("target_modules", DEFAULT_TARGET_MODULES)
    if isinstance(target_modules, str):
        target_modules = [m.strip() for m in target_modules.split(",") if m.strip()]

    cfg = TrainConfig(
        model_name_or_path          = defaults["model_name_or_path"],
        train_file                  = defaults["train_file"],
        validation_file             = defaults["validation_file"],
        output_dir                  = defaults["output_dir"],
        seed                        = defaults.get("seed", 42),
        num_train_epochs            = defaults.get("num_train_epochs", 1.0),
        learning_rate               = defaults.get("learning_rate", 2e-4),
        max_seq_length              = defaults.get("max_seq_length", 1024),
        per_device_train_batch_size = defaults.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size  = defaults.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps = defaults.get("gradient_accumulation_steps", 8),
        warmup_ratio                = defaults.get("warmup_ratio", 0.03),
        weight_decay                = defaults.get("weight_decay", 0.0),
        logging_steps               = defaults.get("logging_steps", 20),
        eval_steps                  = defaults.get("eval_steps", 400),
        save_steps                  = defaults.get("save_steps", 9999),  # no mid-sweep checkpoints
        save_total_limit            = 1,
        gradient_checkpointing      = defaults.get("gradient_checkpointing", True),
        load_in_4bit                = defaults.get("load_in_4bit", True),
        bnb_4bit_quant_type         = defaults.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant   = defaults.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype      = defaults.get("bnb_4bit_compute_dtype", "float16"),
        lora_r                      = defaults.get("lora_r", 16),
        lora_alpha                  = defaults.get("lora_alpha", 32),
        lora_dropout                = defaults.get("lora_dropout", 0.05),
        target_modules              = target_modules,
        trust_remote_code           = defaults.get("trust_remote_code", False),
        report_to                   = "wandb",
    )

    run_train(cfg)
    wandb.finish()


if __name__ == "__main__":
    train()
