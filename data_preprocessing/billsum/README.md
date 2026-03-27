# BillSum Data Cleaning & Preprocessing

This repository contains the data preprocessing pipeline for the **BillSum** dataset, preparing it for LLM fine-tuning experiments (Full Fine-Tuning vs. Parameter-Efficient Fine-Tuning on legal summarization tasks).

## Dataset Overview

**BillSum** is a dataset of US Congressional and California legislative bills paired with human-written summaries. The raw data is sourced from [BillSum v4.1](https://github.com/FiscalNote/BillSum).

Each record contains:

| Field | Description |
|-------|-------------|
| `bill_id` | Unique bill identifier (e.g., `107_hr2256`) |
| `text` | Full bill text |
| `summary` | Human-written summary |
| `title` | Bill title |
| `text_len` | Character length of the bill text |
| `sum_len` | Character length of the summary |

## Preprocessing Pipeline

Run the full pipeline with:

```bash
cd "billsum data cleaning"
python src/run_preprocessing.py
```

### Step 1: Loading

Loads three raw JSONL files from `data/billsum_v4_1/`:
- `us_train_data_final_OFFICIAL.jsonl` — US training data
- `us_test_data_final_OFFICIAL.jsonl` — US test data
- `ca_test_data_final_OFFICIAL.jsonl` — California test data

### Step 2: Text Cleaning

Applied to both `text` and `summary` fields:
- Normalize line endings (`\r\n` → `\n`)
- Collapse excessive blank lines (3+ consecutive newlines → 2)
- Remove redundant inline whitespace
- Strip control characters (keep `\n` and `\t`)
- For summaries: collapse all whitespace into single spaces (single-paragraph format)

### Step 3: Sample Filtering

Remove anomalous samples based on the following criteria:
- **Text length**: must be between 200 and 50,000 characters
- **Summary length**: must be between 20 and 2,000 characters
- Drop rows with `null` or empty `text`/`summary`
- Remove duplicate bills (by `text` content)

### Step 4: Validation Split

5% of the cleaned training data is randomly sampled (seed=42) as a validation set.

### Step 5: SFT Formatting

Each sample is formatted using a prompt template suitable for Supervised Fine-Tuning (SFT):

```
Below is a US legislative bill. Write a concise summary.

### Bill:
{bill_text}

### Summary:
{summary}
```

Bill text is truncated to 12,000 characters at the character level. Precise truncation should be applied at the tokenization stage based on the target model's context window.

## Output

All outputs are saved to the `output/` directory.

### Cleaned Original Format

Retains the original field structure after cleaning and filtering. Useful for custom processing or analysis.

| File | Description | Records |
|------|-------------|---------|
| `train_cleaned.jsonl` | US training set | 15,988 |
| `val_cleaned.jsonl` | Validation set (5% split) | 842 |
| `test_us_cleaned.jsonl` | US test set | 2,905 |
| `test_ca_cleaned.jsonl` | CA test set | 655 |

### SFT-Formatted Data

Ready for LLM fine-tuning. Each line is a JSON object with three fields:

| Field | Description |
|-------|-------------|
| `text` | Full prompt including bill + summary (for training loss) |
| `input` | Prompt with bill only (for inference / label masking) |
| `output` | Reference summary (ground truth) |

| File | Description | Records | Size |
|------|-------------|---------|------|
| `train_sft.jsonl` | Training set | 15,988 | 276 MB |
| `val_sft.jsonl` | Validation set | 842 | 14.8 MB |
| `test_us_sft.jsonl` | US test set | 2,905 | 50 MB |
| `test_ca_sft.jsonl` | CA test set | 655 | 12.5 MB |

### Visualization

| File | Description |
|------|-------------|
| `length_distribution.png` | Histograms of text/summary character lengths across all three datasets |

## Data Statistics (After Cleaning)

### Sample Counts

| Dataset | Before Cleaning | After Cleaning | Removed |
|---------|----------------|----------------|---------|
| US Train | 18,949 | 16,830 → 15,988 (train) + 842 (val) | 2,119 (11.2%) |
| US Test | 3,269 | 2,905 | 364 (11.1%) |
| CA Test | 1,237 | 655 | 582 (47.1%) |

### Length Statistics

| Metric | US Train | US Test | CA Test |
|--------|----------|---------|---------|
| Avg text length | 8,138 chars | 8,106 chars | 9,117 chars |
| Avg summary length | 958 chars | 956 chars | 1,314 chars |
| Median compression ratio | 11.6% | 11.7% | 15.3% |

> **Note:** The CA test set has a higher removal rate (47%) because California bills tend to be longer and many exceed the filtering thresholds. CA data is kept separate and used exclusively for **cross-domain generalization evaluation** — it is never mixed into training.

## Usage

### For Fine-Tuning (e.g., with Hugging Face + PEFT)

```python
import json

# Load SFT-formatted training data
train_data = []
with open("output/train_sft.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        train_data.append(json.loads(line))

# Each item has: "text" (full prompt), "input" (bill only), "output" (summary)
sample = train_data[0]
print(sample["input"][:200])   # Bill prompt
print(sample["output"][:200])  # Reference summary
```

### For Custom Analysis

```python
import pandas as pd

# Load cleaned data in original format
train_df = pd.read_json("output/train_cleaned.jsonl", lines=True)
print(train_df.columns.tolist())
# ['bill_id', 'text', 'summary', 'title', 'text_len', 'sum_len', 'summary_len']
```

## Project Structure

```
billsum data cleaning/
├── data/
│   └── billsum_v4_1/
│       ├── us_train_data_final_OFFICIAL.jsonl   # Raw US training data
│       ├── us_test_data_final_OFFICIAL.jsonl    # Raw US test data
│       ├── ca_test_data_final_OFFICIAL.jsonl    # Raw CA test data
│       └── README.md                            # Original dataset README
├── src/
│   ├── data_check.py          # Quick data exploration script
│   ├── data_cleaning.py       # Text cleaning & sample filtering
│   ├── data_analysis.py       # Descriptive statistics & visualization
│   ├── data_formatting.py     # SFT prompt formatting & JSONL export
│   └── run_preprocessing.py   # Main pipeline entry point
├── output/                    # Generated by run_preprocessing.py
│   ├── train_sft.jsonl
│   ├── val_sft.jsonl
│   ├── test_us_sft.jsonl
│   ├── test_ca_sft.jsonl
│   ├── train_cleaned.jsonl
│   ├── val_cleaned.jsonl
│   ├── test_us_cleaned.jsonl
│   ├── test_ca_cleaned.jsonl
│   └── length_distribution.png
└── README.md                  # This file
```

## Dependencies

- Python 3.10+
- pandas
- matplotlib

```bash
pip install pandas matplotlib
```
