"""
train_casehold_lora.py
QLoRA (4-bit) SFT training for CaseHOLD multiple-choice task.

Uses TRL SFTTrainer + bitsandbytes 4-bit quantisation — fits on 11GB GPUs (RTX 2080 Ti).
Reads the same nested YAML config format as the rest of this project.

Training data: *_mc.jsonl files produced by casehold_formatting.py.
The `text` field in each record contains the full prompt + gold answer letter.

Usage:
  python src/train/train_casehold_lora.py --config configs/qlora_casehold_qwen.yaml
  python src/train/train_casehold_lora.py --config configs/qlora_casehold_llama.yaml

After training, run inference then evaluation:
  python src/evaluate/inference.py --config configs/qlora_casehold_qwen.yaml --split test
  python src/evaluate/eval_casehold.py \\
      --predictions outputs/qlora_casehold_qwen/predictions_test.jsonl \\
      --output      outputs/qlora_casehold_qwen/eval_test.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None  # type: ignore[assignment]

try:
    from trl import DataCollatorForCompletionOnlyLM
except ImportError:
    DataCollatorForCompletionOnlyLM = None  # type: ignore[assignment]


DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


@dataclass
class TrainConfig:
    model_name_or_path: str
    train_file: str
    validation_file: str
    output_dir: str
    seed: int = 42
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 20
    eval_steps: int = 400
    save_steps: int = 400
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    max_steps: int = -1          # -1 = use num_train_epochs; set to 5 for VRAM smoke check
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    trust_remote_code: bool = False
    report_to: str = "none"


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _load_config_defaults(config_path: str | None) -> dict[str, Any]:
    """
    Load a config file (YAML or JSON) and return a flat dict matching TrainConfig fields.

    Supports two formats:
    - Nested YAML (project standard): sections model/lora/qlora/training/data/output
    - Flat JSON/YAML (legacy): all keys at top level
    """
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError("Config file must be a YAML/JSON mapping.")

    # Detect format: nested if any top-level value is a dict
    if not any(isinstance(v, dict) for v in loaded.values()):
        return loaded  # flat (old JSON format) — return as-is

    # ── Flatten nested YAML ───────────────────────────────────────────────────
    flat: dict[str, Any] = {}

    # model section
    model = loaded.get("model", {})
    if "name" in model:
        flat["model_name_or_path"] = model["name"]
    if "trust_remote_code" in model:
        flat["trust_remote_code"] = model["trust_remote_code"]

    # data section
    data = loaded.get("data", {})
    if "train" in data:
        flat["train_file"] = data["train"]
    if "val" in data:
        flat["validation_file"] = data["val"]
    if "max_length" in data:
        flat["max_seq_length"] = data["max_length"]

    # output section
    output = loaded.get("output", {})
    if "dir" in output:
        flat["output_dir"] = output["dir"]

    # training section — direct key mapping
    for key in [
        "seed", "num_train_epochs", "learning_rate",
        "per_device_train_batch_size", "per_device_eval_batch_size",
        "gradient_accumulation_steps", "warmup_ratio", "weight_decay",
        "logging_steps", "eval_steps", "save_steps", "save_total_limit",
        "gradient_checkpointing", "report_to",
    ]:
        if key in loaded.get("training", {}):
            flat[key] = loaded["training"][key]

    # lora section
    lora = loaded.get("lora", {})
    if "r" in lora:
        flat["lora_r"] = lora["r"]
    for key in ["lora_alpha", "lora_dropout", "target_modules"]:
        if key in lora:
            flat[key] = lora[key]

    # qlora section (4-bit quantisation)
    qlora = loaded.get("qlora", {})
    for key in ["load_in_4bit", "bnb_4bit_quant_type",
                "bnb_4bit_use_double_quant", "bnb_4bit_compute_dtype"]:
        if key in qlora:
            flat[key] = qlora[key]

    return flat


def parse_args() -> TrainConfig:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()
    defaults = _load_config_defaults(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="CaseHOLD QLoRA SFT trainer")
    parser.add_argument("--config", type=str, default=bootstrap_args.config)
    parser.add_argument("--model_name_or_path", type=str, default=defaults.get("model_name_or_path"))
    parser.add_argument("--train_file", type=str, default=defaults.get("train_file"))
    parser.add_argument("--validation_file", type=str, default=defaults.get("validation_file"))
    parser.add_argument("--output_dir", type=str, default=defaults.get("output_dir"))
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))
    parser.add_argument("--num_train_epochs", type=float, default=defaults.get("num_train_epochs", 1.0))
    parser.add_argument("--learning_rate", type=float, default=defaults.get("learning_rate", 2e-4))
    parser.add_argument("--max_seq_length", type=int, default=defaults.get("max_seq_length", 1024))
    parser.add_argument("--per_device_train_batch_size", type=int,
                        default=defaults.get("per_device_train_batch_size", 1))
    parser.add_argument("--per_device_eval_batch_size", type=int,
                        default=defaults.get("per_device_eval_batch_size", 1))
    parser.add_argument("--gradient_accumulation_steps", type=int,
                        default=defaults.get("gradient_accumulation_steps", 16))
    parser.add_argument("--warmup_ratio", type=float, default=defaults.get("warmup_ratio", 0.03))
    parser.add_argument("--weight_decay", type=float, default=defaults.get("weight_decay", 0.0))
    parser.add_argument("--logging_steps", type=int, default=defaults.get("logging_steps", 20))
    parser.add_argument("--eval_steps", type=int, default=defaults.get("eval_steps", 400))
    parser.add_argument("--save_steps", type=int, default=defaults.get("save_steps", 400))
    parser.add_argument("--save_total_limit", type=int, default=defaults.get("save_total_limit", 2))
    parser.add_argument("--gradient_checkpointing", type=_str2bool,
                        default=defaults.get("gradient_checkpointing", True))
    parser.add_argument("--load_in_4bit", type=_str2bool, default=defaults.get("load_in_4bit", True))
    parser.add_argument("--bnb_4bit_quant_type", type=str,
                        default=defaults.get("bnb_4bit_quant_type", "nf4"))
    parser.add_argument("--bnb_4bit_use_double_quant", type=_str2bool,
                        default=defaults.get("bnb_4bit_use_double_quant", True))
    parser.add_argument("--bnb_4bit_compute_dtype", type=str,
                        choices=["float16", "bfloat16", "float32"],
                        default=defaults.get("bnb_4bit_compute_dtype", "float16"))
    parser.add_argument("--lora_r", type=int, default=defaults.get("lora_r", 16))
    parser.add_argument("--lora_alpha", type=int, default=defaults.get("lora_alpha", 32))
    parser.add_argument("--lora_dropout", type=float, default=defaults.get("lora_dropout", 0.05))
    parser.add_argument("--target_modules", type=str,
                        default=",".join(defaults.get("target_modules", DEFAULT_TARGET_MODULES)),
                        help="Comma-separated module names")
    parser.add_argument("--trust_remote_code", type=_str2bool,
                        default=defaults.get("trust_remote_code", False))
    parser.add_argument("--report_to", type=str, default=defaults.get("report_to", "none"))
    parser.add_argument("--max_steps", type=int, default=defaults.get("max_steps", -1),
                        help="Override max_steps (e.g. 5 for VRAM smoke check; -1 = use num_epochs)")

    args = parser.parse_args()

    missing = [
        name for name in ["model_name_or_path", "train_file", "validation_file", "output_dir"]
        if getattr(args, name) in (None, "")
    ]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    if not target_modules:
        target_modules = DEFAULT_TARGET_MODULES

    return TrainConfig(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        validation_file=args.validation_file,
        output_dir=args.output_dir,
        seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        trust_remote_code=args.trust_remote_code,
        report_to=args.report_to,
        max_steps=args.max_steps,
    )


def _get_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _select_train_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def run_train(config: TrainConfig) -> None:
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # Resolve paths relative to repo root (script is at src/train/, root is ../../)
    repo_root = Path(__file__).resolve().parent.parent.parent
    train_path = repo_root / config.train_file
    val_path   = repo_root / config.validation_file

    data_files = {
        "train":      str(train_path),
        "validation": str(val_path),
    }
    dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Only compute loss on the answer token (mirrors train.py label-only supervision)
    RESPONSE_TEMPLATE = "\n### Answer:\n"
    if DataCollatorForCompletionOnlyLM is not None:
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=RESPONSE_TEMPLATE,
            tokenizer=tokenizer,
        )
    else:
        data_collator = None

    quant_config = None
    if config.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=_get_torch_dtype(config.bnb_4bit_compute_dtype),
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        quantization_config=quant_config,
        torch_dtype=_select_train_dtype(),
        device_map="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules or DEFAULT_TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    args_cls: Any = SFTConfig if SFTConfig is not None else TrainingArguments
    args_params = inspect.signature(args_cls.__init__).parameters
    strategy_key = "evaluation_strategy" if "evaluation_strategy" in args_params else "eval_strategy"

    args_kwargs: dict[str, Any] = {
        "output_dir":                   config.output_dir,
        "num_train_epochs":             config.num_train_epochs,
        "max_steps":                    config.max_steps,
        "learning_rate":                config.learning_rate,
        "per_device_train_batch_size":  config.per_device_train_batch_size,
        "per_device_eval_batch_size":   config.per_device_eval_batch_size,
        "gradient_accumulation_steps":  config.gradient_accumulation_steps,
        "warmup_ratio":                 config.warmup_ratio,
        "weight_decay":                 config.weight_decay,
        "logging_steps":                config.logging_steps,
        "eval_steps":                   config.eval_steps,
        "save_steps":                   config.save_steps,
        "save_total_limit":             config.save_total_limit,
        "seed":                         config.seed,
        "bf16":                         bf16,
        "fp16":                         fp16,
        strategy_key:                   "steps",
        "save_strategy":                "steps",
        "gradient_checkpointing":       config.gradient_checkpointing,
        "lr_scheduler_type":            "cosine",
        "report_to":                    config.report_to,
        "load_best_model_at_end":       True,
        "metric_for_best_model":        "eval_loss",
    }
    if "dataset_text_field" in args_params:
        args_kwargs["dataset_text_field"] = "text"
    if "max_length" in args_params:
        args_kwargs["max_length"] = config.max_seq_length
    if "max_seq_length" in args_params:
        args_kwargs["max_seq_length"] = config.max_seq_length
    if "packing" in args_params:
        args_kwargs["packing"] = False
    if "do_train" in args_params:
        args_kwargs["do_train"] = True
    if "do_eval" in args_params:
        args_kwargs["do_eval"] = True

    training_args = args_cls(**args_kwargs)

    trainer_kwargs: dict[str, Any] = {
        "model":        model,
        "args":         training_args,
        "train_dataset": dataset["train"],
        "eval_dataset":  dataset["validation"],
        "peft_config":   lora_config,
    }
    if data_collator is not None:
        trainer_kwargs["data_collator"] = data_collator

    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = config.max_seq_length
    if "packing" in trainer_params:
        trainer_kwargs["packing"] = False

    trainer = SFTTrainer(**trainer_kwargs)

    train_result = trainer.train()

    # Save adapter to final_adapter/ — compatible with inference.py (PeftModel.from_pretrained)
    final_adapter_dir = Path(config.output_dir) / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))
    print(f"Adapter saved to: {final_adapter_dir}")

    # Save run summary and resolved config to output root
    run_summary = {
        "train_runtime":            train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "train_loss":               train_result.metrics.get("train_loss"),
        "seed":                     config.seed,
        "max_seq_length":           config.max_seq_length,
        "learning_rate":            config.learning_rate,
        "num_train_epochs":         config.num_train_epochs,
        "lora": {
            "r":              config.lora_r,
            "alpha":          config.lora_alpha,
            "dropout":        config.lora_dropout,
            "target_modules": config.target_modules,
        },
    }
    with open(Path(config.output_dir) / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)
    with open(Path(config.output_dir) / "resolved_train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"Training complete. Results saved to: {config.output_dir}")


if __name__ == "__main__":
    cfg = parse_args()
    run_train(cfg)
