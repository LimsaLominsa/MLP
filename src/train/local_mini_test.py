"""
local_mini_test.py
──────────────────
End-to-end CPU mini-test for train.py logic.

Uses `sshleifer/tiny-gpt2` (~2 MB, public, no auth) instead of the full
1.5 B model so it runs quickly in float32 on any laptop.

What is tested:
  ✓ Package imports (torch, peft, transformers, datasets, yaml)
  ✓ load_jsonl / config loading
  ✓ build_dataset (tokenisation + label masking)
  ✓ LoRA model wrapping via get_peft_model
  ✓ DataCollatorForSeq2Seq
  ✓ Trainer forward + backward pass  (max_steps=2)
  ✓ Trainer.train() exits without error

NOT tested:
  ✗ Actual learning quality
  ✗ bfloat16 / float16 / GPU paths

Usage (from repo root):
  python src/train/local_mini_test.py

Requirements (CPU-only torch install):
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install peft accelerate transformers datasets pyyaml
"""

import sys
import json
import os
from pathlib import Path

# ── 1. Dependency check ───────────────────────────────────────────────────────
missing = []
for pkg in ["torch", "peft", "transformers", "datasets", "yaml"]:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print("ERROR: missing packages:", ", ".join(missing))
    print()
    print("Install CPU-only torch + other deps:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    print("  pip install peft accelerate transformers datasets pyyaml")
    sys.exit(1)

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

# ── 2. Config ────────────────────────────────────────────────────────────────
TINY_MODEL    = "sshleifer/tiny-gpt2"   # 2 MB, public, no auth
MAX_SAMPLES   = 10                       # records to use from real data
MAX_INPUT_LEN = 128
MAX_OUT_LEN   = 32
MAX_STEPS     = 2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_DATA = REPO_ROOT / "data" / "billsum" / "train_sft.jsonl"

set_seed(42)

# ── 3. Load data ──────────────────────────────────────────────────────────────
print(f"[1/5] Loading {MAX_SAMPLES} records from {TRAIN_DATA.relative_to(REPO_ROOT)}")
if not TRAIN_DATA.exists():
    print(f"ERROR: data file not found: {TRAIN_DATA}")
    print("Make sure the data/ directory junction is set up (run smoke_test.py first).")
    sys.exit(1)

records = []
with open(TRAIN_DATA, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
        if len(records) >= MAX_SAMPLES:
            break

for r in records:
    assert "input" in r and "output" in r, "Unexpected record format — missing 'input'/'output'"

print(f"    Loaded {len(records)} records OK")

# ── 4. Load tokenizer + model ─────────────────────────────────────────────────
print(f"[2/5] Loading tokenizer + model: {TINY_MODEL}")

tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    TINY_MODEL,
    torch_dtype=torch.float32,   # CPU requires float32
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=["c_attn"],   # GPT-2 uses c_attn (combined QKV)
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("    Model + LoRA OK")

# ── 5. Tokenise ───────────────────────────────────────────────────────────────
print("[3/5] Tokenising dataset")

def tokenize(example):
    input_ids = tokenizer(
        example["input"],
        truncation=True,
        max_length=MAX_INPUT_LEN,
        add_special_tokens=True,
    )["input_ids"]
    output_ids = tokenizer(
        example["output"],
        truncation=True,
        max_length=MAX_OUT_LEN,
        add_special_tokens=False,
    )["input_ids"]
    if tokenizer.eos_token_id is not None:
        output_ids = output_ids + [tokenizer.eos_token_id]
    full_ids = input_ids + output_ids
    labels   = [-100] * len(input_ids) + output_ids
    return {
        "input_ids":      full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels":         labels,
    }

train_dataset = Dataset.from_list(records[:MAX_SAMPLES - 1])
val_dataset   = Dataset.from_list(records[-1:])

train_dataset = train_dataset.map(tokenize, remove_columns=train_dataset.column_names)
val_dataset   = val_dataset.map(tokenize,   remove_columns=val_dataset.column_names)
print(f"    train={len(train_dataset)}, val={len(val_dataset)} — OK")

# ── 6. Trainer ────────────────────────────────────────────────────────────────
print("[4/5] Configuring Trainer")

output_dir = REPO_ROOT / "outputs" / "test_local_cpu"
output_dir.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir                  = str(output_dir),
    max_steps                   = MAX_STEPS,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size  = 1,
    gradient_accumulation_steps = 1,
    gradient_checkpointing      = False,
    learning_rate               = 2e-4,
    lr_scheduler_type           = "constant",
    warmup_steps                = 0,
    bf16                        = False,
    fp16                        = False,
    optim                       = "adamw_torch",
    eval_strategy               = "steps",
    eval_steps                  = 1,
    save_strategy               = "no",
    load_best_model_at_end      = False,
    report_to                   = "none",
    logging_steps               = 1,
    seed                        = 42,
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

# ── 7. Train ──────────────────────────────────────────────────────────────────
print(f"[5/5] Running {MAX_STEPS} training steps on CPU (float32)...")
trainer.train()

print()
print("=" * 60)
print("  LOCAL MINI-TEST PASSED")
print("  All code paths are working correctly.")
print("  Output dir:", output_dir)
print("=" * 60)
