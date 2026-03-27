"""
casehold_cleaning.py
Load, clean, and filter the CaseHOLD dataset.

CaseHOLD is a multiple-choice legal holding prediction task.
Each sample contains a citing context (with a <HOLDING> placeholder)
and five candidate holdings; the label indicates the correct one.

Fields per record:
  example_id     : unique sample identifier
  citing_prompt  : court opinion excerpt with <HOLDING> marker
  holding_0~4    : five candidate holding statements
  label          : correct option index (0–4)

Data source:
  Raw CSV files are downloaded directly from HuggingFace Hub (data/all/).
  This bypasses the deprecated dataset script loader present in datasets v4+.
"""

import re
import pandas as pd
from huggingface_hub import hf_hub_download


# ── Column names ──────────────────────────────────────────────────────────────
HOLDING_COLS = ['holding_0', 'holding_1', 'holding_2',
                'holding_3', 'holding_4']

# ── Filter thresholds ─────────────────────────────────────────────────────────
MIN_PROMPT_LEN  = 100
MAX_PROMPT_LEN  = 20000
MIN_HOLDING_LEN = 10
MAX_HOLDING_LEN = 2000


def clean_legal_text(text: str) -> str:
    """
    Clean a legal text string:
      - Normalize line endings
      - Collapse 3+ consecutive newlines to 2 (preserve paragraph breaks)
      - Remove redundant inline whitespace (keep newlines)
      - Strip per-line leading/trailing spaces
      - Remove non-printable control characters (keep \\n and \\t)
    """
    if not isinstance(text, str) or len(text) == 0:
        return ""

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.strip()


def clean_holding(text: str) -> str:
    """
    Clean a holding option string.
    Holdings are short single-paragraph statements — collapse all whitespace.
    """
    if not isinstance(text, str) or len(text) == 0:
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def validate_sample(row: dict) -> bool:
    """
    Return True if a sample passes all quality checks:
      - citing_prompt is within length bounds
      - all 5 holding options are present and within length bounds
      - label is a valid integer in [0, 4]
      - <HOLDING> token is present in citing_prompt
    """
    prompt = row.get('citing_prompt', '')

    if not (MIN_PROMPT_LEN <= len(prompt) <= MAX_PROMPT_LEN):
        return False

    if '<HOLDING>' not in prompt:
        return False

    for col in HOLDING_COLS:
        h = row.get(col, '')
        if not isinstance(h, str) or not (MIN_HOLDING_LEN <= len(h) <= MAX_HOLDING_LEN):
            return False

    try:
        if int(row.get('label', -1)) not in range(5):
            return False
    except (ValueError, TypeError):
        return False

    return True


def clean_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and filter a CaseHOLD DataFrame.

    Steps:
      1. Clean citing_prompt and all holding fields
      2. Coerce label to integer
      3. Remove samples that fail validate_sample()
      4. Add helper columns: prompt_len, correct_holding
    """
    df = df.copy()
    original_len = len(df)

    # Clean text fields
    df['citing_prompt'] = df['citing_prompt'].apply(clean_legal_text)
    for col in HOLDING_COLS:
        df[col] = df[col].apply(clean_holding)

    # Coerce label to numeric
    df['label'] = pd.to_numeric(df['label'], errors='coerce')

    # Filter
    mask = df.apply(validate_sample, axis=1)
    df = df[mask].reset_index(drop=True)

    # Helper columns
    df['prompt_len'] = df['citing_prompt'].str.len()
    df['correct_holding'] = df.apply(
        lambda r: r[f"holding_{int(r['label'])}"], axis=1
    )

    removed = original_len - len(df)
    print(f"  Before: {original_len:,} | After: {len(df):,} | "
          f"Removed: {removed:,} ({removed / original_len * 100:.1f}%)")
    return df


def load_and_clean_casehold() -> dict:
    """
    Download CaseHOLD CSV files from HuggingFace Hub and clean all three splits.

    Files downloaded:
      data/all/train.csv  (~85 MB)
      data/all/val.csv    (~10 MB)
      data/all/test.csv   (~11 MB)

    Returns dict with keys: 'train', 'validation', 'test'.
    """
    split_files = {
        'train':      'data/all/train.csv',
        'validation': 'data/all/val.csv',
        'test':       'data/all/test.csv',
    }

    splits = {}
    for split_name, repo_filename in split_files.items():
        print(f"\nDownloading {repo_filename} ...")
        local_path = hf_hub_download(
            repo_id   = "casehold/casehold",
            filename  = repo_filename,
            repo_type = "dataset",
        )
        print(f"  Cached at: {local_path}")

        df_raw = pd.read_csv(local_path)

        # The CSV was saved with a row-index column and numeric column headers.
        # According to the original casehold.py loader:
        #   row[0]  → example_id   (column 'Unnamed: 0')
        #   row[1]  → citing_prompt (column '0')
        #   row[2]  → holding_0    (column '1')
        #   row[3]  → holding_1    (column '2')
        #   row[4]  → holding_2    (column '3')
        #   row[5]  → holding_3    (column '4')
        #   row[6]  → holding_4    (column '5')
        #   row[12] → label        (column '11')
        col_map = {
            'Unnamed: 0': 'example_id',
            '0':          'citing_prompt',
            '1':          'holding_0',
            '2':          'holding_1',
            '3':          'holding_2',
            '4':          'holding_3',
            '5':          'holding_4',
            '11':         'label',
        }
        df_raw = df_raw.rename(columns=col_map)

        # Keep only the columns we need
        keep_cols = ['example_id', 'citing_prompt',
                     'holding_0', 'holding_1', 'holding_2',
                     'holding_3', 'holding_4', 'label']
        df_raw = df_raw[keep_cols]

        print(f"  Loaded {len(df_raw):,} rows  |  columns: {df_raw.columns.tolist()}")

        print(f"\n--- Cleaning split: {split_name} ---")
        splits[split_name] = clean_split(df_raw)

    return splits
