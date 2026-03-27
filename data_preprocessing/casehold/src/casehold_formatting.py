"""
casehold_formatting.py
Format cleaned CaseHOLD data for two use cases:

  1. Multiple-Choice SFT  — instruction-tuned LLM fine-tuning / prompting
     Fields: text (full prompt+answer), input (prompt only),
             output (correct letter A–E), label (int 0–4)

  2. Classification flat  — encoder / custom training loop
     Fields: example_id, citing_prompt, holding_0~4, label, correct_holding
"""

import json
import pandas as pd
from pathlib import Path


HOLDING_COLS  = ['holding_0', 'holding_1', 'holding_2',
                 'holding_3', 'holding_4']
OPTION_LABELS = ['A', 'B', 'C', 'D', 'E']

# ── Prompt templates ──────────────────────────────────────────────────────────
_MC_BODY = (
    "You are a legal expert. The following excerpt is from a court opinion.\n"
    "The token <HOLDING> marks a legal holding that has been removed.\n"
    "Choose the correct holding from the five options below.\n\n"
    "### Context:\n{context}\n\n"
    "### Options:\n"
    "A) {holding_0}\n"
    "B) {holding_1}\n"
    "C) {holding_2}\n"
    "D) {holding_3}\n"
    "E) {holding_4}\n\n"
    "### Answer:\n"
)

TRAIN_TEMPLATE     = _MC_BODY + "{answer}"
INFERENCE_TEMPLATE = _MC_BODY          # no answer appended


def format_as_multiple_choice(df: pd.DataFrame,
                               max_prompt_chars: int = 8000) -> list:
    """
    Convert each row into a multiple-choice SFT record.

    Output fields:
      example_id  — original sample ID
      text        — full prompt including correct answer letter (for training)
      input       — prompt without answer (for inference / label masking)
      output      — correct option letter, e.g. 'C'
      label       — integer index 0–4
    """
    records = []
    for _, row in df.iterrows():
        context = row['citing_prompt'][:max_prompt_chars]
        label   = int(row['label'])
        answer  = OPTION_LABELS[label]

        kwargs = dict(
            context   = context,
            holding_0 = row['holding_0'],
            holding_1 = row['holding_1'],
            holding_2 = row['holding_2'],
            holding_3 = row['holding_3'],
            holding_4 = row['holding_4'],
        )
        records.append({
            "example_id": row.get('example_id', ''),
            "text":       TRAIN_TEMPLATE.format(**kwargs, answer=answer),
            "input":      INFERENCE_TEMPLATE.format(**kwargs),
            "output":     answer,
            "label":      label,
        })
    return records


def format_as_classification(df: pd.DataFrame,
                              max_prompt_chars: int = 8000) -> list:
    """
    Convert each row into a flat classification record.
    Suitable for encoder-style models or custom training loops.

    Output fields:
      example_id, citing_prompt, holding_0~4, label, correct_holding
    """
    records = []
    for _, row in df.iterrows():
        record = {
            "example_id":    row.get('example_id', ''),
            "citing_prompt": row['citing_prompt'][:max_prompt_chars],
            "label":         int(row['label']),
            "correct_holding": row['correct_holding'],
        }
        for col in HOLDING_COLS:
            record[col] = row[col]
        records.append(record)
    return records


def save_jsonl(records: list, filepath) -> None:
    """Save a list of dicts to a JSONL file (one JSON object per line)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    size_mb = filepath.stat().st_size / 1024 / 1024
    print(f"  Saved {len(records):,} records → {filepath.name}  ({size_mb:.1f} MB)")
