"""
data_cleaning.py
- Load JSONL data files
- Clean and normalize text/summary fields
- Filter out anomalous samples (too short, too long, empty, duplicates)
"""

import re
import json
import pandas as pd
from pathlib import Path


def load_jsonl(filepath):
    """Load a JSONL file into a pandas DataFrame."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def clean_text(text: str) -> str:
    """
    Clean a single bill text string:
    - Normalize line endings
    - Collapse excessive blank lines (preserve paragraph structure)
    - Remove redundant inline whitespace
    - Strip control characters
    """
    if not isinstance(text, str) or len(text) == 0:
        return ""

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Replace inline whitespace sequences (tabs, spaces) with single space
    text = re.sub(r'[^\S\n]+', ' ', text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Remove control characters (keep newline \n and tab \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text.strip()


def clean_summary(summary: str) -> str:
    """
    Clean a summary string:
    - Summaries are typically single-paragraph, so collapse all whitespace
    """
    if not isinstance(summary, str) or len(summary) == 0:
        return ""

    summary = re.sub(r'\s+', ' ', summary)
    return summary.strip()


def filter_samples(df: pd.DataFrame,
                   min_text_len: int = 200,
                   max_text_len: int = 50000,
                   min_summary_len: int = 20,
                   max_summary_len: int = 2000) -> pd.DataFrame:
    """
    Filter out anomalous samples based on text/summary length.
    Also removes null values and duplicates.
    """
    original = len(df)

    # Drop rows with missing text or summary
    df = df.dropna(subset=['text', 'summary']).copy()

    # Drop empty strings
    df = df[df['text'].str.len() > 0]
    df = df[df['summary'].str.len() > 0]

    # Length-based filtering
    text_len = df['text'].str.len()
    summ_len = df['summary'].str.len()
    df = df[text_len.between(min_text_len, max_text_len)]
    df = df[summ_len.between(min_summary_len, max_summary_len)]

    # Remove duplicates by text content
    df = df.drop_duplicates(subset=['text'])

    removed = original - len(df)
    print(f"  Before: {original} | After: {len(df)} | Removed: {removed}")
    return df.reset_index(drop=True)


def preprocess_dataframe(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """
    Full preprocessing pipeline for a single DataFrame:
    1. Clean text and summary fields
    2. Filter anomalous samples
    3. Recompute length columns
    """
    print(f"\n--- Preprocessing: {name} ---")

    df = df.copy()

    # Apply text cleaning
    df['text'] = df['text'].apply(clean_text)
    df['summary'] = df['summary'].apply(clean_summary)
    if 'title' in df.columns:
        df['title'] = df['title'].apply(clean_summary)

    # Filter anomalous samples
    df = filter_samples(df)

    # Recompute length columns for downstream analysis
    df['text_len'] = df['text'].str.len()
    df['summary_len'] = df['summary'].str.len()

    return df
