"""
data_formatting.py
- Convert cleaned DataFrames into SFT (Supervised Fine-Tuning) format
- Provide prompt templates for training and inference
- Save formatted data as JSONL
"""

import json
import pandas as pd
from pathlib import Path


# ==================== Prompt Templates ====================
# Training template: includes both input and target output
TRAIN_TEMPLATE = (
    "Below is a US legislative bill. Write a concise summary.\n\n"
    "### Bill:\n{text}\n\n"
    "### Summary:\n{summary}"
)

# Inference template: input only, model generates the summary
INFERENCE_TEMPLATE = (
    "Below is a US legislative bill. Write a concise summary.\n\n"
    "### Bill:\n{text}\n\n"
    "### Summary:\n"
)


def format_for_sft(df: pd.DataFrame, max_text_chars: int = 12000) -> list:
    """
    Format each sample into SFT-ready structure.

    Each output record contains:
      - text:   Full prompt with bill + summary (for training loss computation)
      - input:  Prompt with bill only (for inference / label masking)
      - output: The reference summary (ground truth)

    Args:
        df: Cleaned DataFrame with 'text' and 'summary' columns
        max_text_chars: Truncate bill text to this many characters (rough limit;
                        precise truncation happens at tokenization stage)
    """
    formatted = []
    for _, row in df.iterrows():
        bill_text = row['text'][:max_text_chars]
        summary = row['summary']

        formatted.append({
            "text": TRAIN_TEMPLATE.format(text=bill_text, summary=summary),
            "input": INFERENCE_TEMPLATE.format(text=bill_text),
            "output": summary,
        })
    return formatted


def save_jsonl(records: list, filepath):
    """Save a list of dicts as a JSONL file (one JSON object per line)."""
    filepath = Path(filepath)
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"  Saved {len(records)} records -> {filepath}")
