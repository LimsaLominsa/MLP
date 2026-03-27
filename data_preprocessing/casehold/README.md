# CaseHOLD Data Preprocessing

This repository contains the preprocessing pipeline for the **CaseHOLD** dataset,
preparing it for LLM fine-tuning experiments on legal holding prediction
(Full Fine-Tuning vs. Parameter-Efficient Fine-Tuning).

---

## Dataset Overview

**CaseHOLD** (**Case** law **HOLD**ing) is a multiple-choice legal NLP benchmark
derived from the Harvard Law School Case Law Access Project.

**Task**: Given a court opinion excerpt with a `<HOLDING>` placeholder, select
the correct legal holding from five candidates (A–E).

- **Type**: Multiple-choice classification
- **Domain**: US case law
- **Source**: [casehold/casehold](https://huggingface.co/datasets/casehold/casehold) on HuggingFace Hub
- **Paper**: Zheng et al., *When Does Pretraining Help?* (ACL 2021)

### Raw Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `example_id` | string | Unique sample identifier |
| `citing_prompt` | string | Court opinion excerpt with `<HOLDING>` marker |
| `holding_0` | string | Candidate holding option 0 (may be correct) |
| `holding_1` | string | Candidate holding option 1 |
| `holding_2` | string | Candidate holding option 2 |
| `holding_3` | string | Candidate holding option 3 |
| `holding_4` | string | Candidate holding option 4 |
| `label` | int | Index of the correct option (0–4) |

---

## Preprocessing Pipeline

### Run

```bash
cd "casehold preprocessing"
python src/run_casehold_preprocessing.py
```

### Step 1 — Download

Raw CSV files are downloaded directly from HuggingFace Hub via `hf_hub_download`,
bypassing the deprecated dataset-script loader removed in `datasets` v4+:

| Remote path | Split | Raw size |
|-------------|-------|----------|
| `data/all/train.csv` | train | ~85 MB |
| `data/all/val.csv` | validation | ~10 MB |
| `data/all/test.csv` | test | ~11 MB |

The CSV files use numeric column headers; the loader maps them to named fields:

```
CSV column  →  Field
──────────────────────────────────
Unnamed: 0  →  example_id
0           →  citing_prompt
1           →  holding_0
2           →  holding_1
3           →  holding_2
4           →  holding_3
5           →  holding_4
11          →  label
```

### Step 2 — Text Cleaning (`clean_legal_text`)

Applied to `citing_prompt`:
- Normalize line endings (`\r\n` → `\n`)
- Collapse 3+ consecutive blank lines → 2 (preserve paragraph structure)
- Remove redundant inline whitespace (preserve newlines)
- Strip per-line leading/trailing spaces
- Remove non-printable control characters (keep `\n` and `\t`)

Applied to `holding_0`–`holding_4` (`clean_holding`):
- Collapse all whitespace to single spaces (holdings are single-paragraph)

### Step 3 — Sample Filtering (`validate_sample`)

A sample is removed if **any** of the following conditions are met:

| Condition | Threshold |
|-----------|-----------|
| `citing_prompt` too short | < 100 chars |
| `citing_prompt` too long | > 20,000 chars |
| Any holding too short | < 10 chars |
| Any holding too long | > 2,000 chars |
| `<HOLDING>` token absent from prompt | — |
| `label` is not an integer in 0–4 | — |

### Step 4 — Helper Columns Added

| Column | Description |
|--------|-------------|
| `prompt_len` | Character length of `citing_prompt` after cleaning |
| `correct_holding` | Text of the holding at index `label` |

### Step 5 — Format and Save

Each cleaned split is saved in **two formats** plus a cleaned raw copy.

---

## Output Files

All outputs are written to `output/casehold/`.

### Sample Counts (after cleaning)

| Split | Records |
|-------|---------|
| train | 41,774 |
| validation | 5,223 |
| test | 5,221 |

---

### Format 1 — Multiple-Choice SFT (`*_mc.jsonl`)

Designed for instruction-tuned LLM fine-tuning and prompting.

**Fields per record:**

| Field | Type | Description |
|-------|------|-------------|
| `example_id` | string | Original sample ID |
| `text` | string | Full prompt **including** the correct answer letter (for training loss) |
| `input` | string | Prompt **without** answer (for inference / label masking) |
| `output` | string | Correct option letter: `"A"`, `"B"`, `"C"`, `"D"`, or `"E"` |
| `label` | int | Correct option index 0–4 |

**Prompt template (`input` field):**

```
You are a legal expert. The following excerpt is from a court opinion.
The token <HOLDING> marks a legal holding that has been removed.
Choose the correct holding from the five options below.

### Context:
{citing_prompt}

### Options:
A) {holding_0}
B) {holding_1}
C) {holding_2}
D) {holding_3}
E) {holding_4}

### Answer:
```

The `text` field appends the correct answer letter (e.g. `C`) after `### Answer:\n`.

**File sizes:**

| File | Records | Size |
|------|---------|------|
| `train_mc.jsonl` | 41,774 | 154 MB |
| `validation_mc.jsonl` | 5,223 | 19.3 MB |
| `test_mc.jsonl` | 5,221 | 19.3 MB |

---

### Format 2 — Classification Flat (`*_cls.jsonl`)

Designed for encoder-style models (e.g., BERT, LegalBERT) or custom training loops
that handle the five candidates separately.

**Fields per record:**

| Field | Type | Description |
|-------|------|-------------|
| `example_id` | string | Original sample ID |
| `citing_prompt` | string | Cleaned context (truncated to 8,000 chars) |
| `holding_0`–`holding_4` | string | Five candidate holdings |
| `label` | int | Correct option index 0–4 |
| `correct_holding` | string | Text of the correct holding |

**File sizes:**

| File | Records | Size |
|------|---------|------|
| `train_cls.jsonl` | 41,774 | 78 MB |
| `validation_cls.jsonl` | 5,223 | 9.75 MB |
| `test_cls.jsonl` | 5,221 | 9.75 MB |

---

### Cleaned Raw (`*_cleaned.jsonl`)

All original fields after cleaning, plus `prompt_len` and `correct_holding`.
Useful for custom analysis or alternative formatting.

| File | Records | Size |
|------|---------|------|
| `train_cleaned.jsonl` | 41,774 | 78 MB |
| `validation_cleaned.jsonl` | 5,223 | 9.75 MB |
| `test_cleaned.jsonl` | 5,221 | 9.75 MB |

---

### Visualization

| File | Description |
|------|-------------|
| `casehold_distribution.png` | Prompt length histograms + label distribution bar charts for all three splits |

---

## Usage

### Load SFT data for LLM fine-tuning

```python
import json

# Load as list of dicts
train = [json.loads(l) for l in open("output/casehold/train_mc.jsonl")]

sample = train[0]
print(sample["input"])    # Prompt shown to the model
print(sample["output"])   # Correct answer: "A" / "B" / "C" / "D" / "E"
print(sample["label"])    # Integer: 0 / 1 / 2 / 3 / 4
```

### Load classification data for encoder models

```python
import pandas as pd

train_df = pd.read_json("output/casehold/train_cls.jsonl", lines=True)
# Columns: example_id, citing_prompt, holding_0~4, label, correct_holding
print(train_df.head())
```

### Evaluation metric

CaseHOLD is a classification task evaluated by **Accuracy**:

```python
predictions = [...]  # list of predicted labels (int 0–4) or letters ("A"–"E")
references  = [...]  # list of ground-truth labels

accuracy = sum(p == r for p, r in zip(predictions, references)) / len(references)
print(f"Accuracy: {accuracy:.4f}")
```

For LLM outputs (letter format), convert before computing:

```python
LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
pred_idx = [LETTER_TO_IDX.get(p.strip(), -1) for p in predictions]
```

---

## Project Structure

```
casehold preprocessing/
├── src/
│   ├── casehold_cleaning.py          # Download, clean, filter
│   ├── casehold_analysis.py          # Statistics & visualization
│   ├── casehold_formatting.py        # SFT and classification formatting
│   └── run_casehold_preprocessing.py # Main pipeline entry point
└── output/
    └── casehold/
        ├── train_mc.jsonl            # SFT format — train        (154 MB)
        ├── validation_mc.jsonl       # SFT format — validation   (19 MB)
        ├── test_mc.jsonl             # SFT format — test         (19 MB)
        ├── train_cls.jsonl           # Classification — train    (78 MB)
        ├── validation_cls.jsonl      # Classification — validation (10 MB)
        ├── test_cls.jsonl            # Classification — test     (10 MB)
        ├── train_cleaned.jsonl       # Cleaned raw — train       (78 MB)
        ├── validation_cleaned.jsonl  # Cleaned raw — validation  (10 MB)
        ├── test_cleaned.jsonl        # Cleaned raw — test        (10 MB)
        └── casehold_distribution.png # Length & label plots
```

## Dependencies

```bash
pip install pandas huggingface-hub matplotlib
```

---

## Relation to BillSum

| Dimension | BillSum | CaseHOLD |
|-----------|---------|----------|
| Task type | Summarization (generative) | Holding prediction (classification) |
| Output | Free-form summary text | Option letter A–E |
| Evaluation | ROUGE-1/2/L + BERTScore | Accuracy |
| LLM format | `train_sft.jsonl` | `train_mc.jsonl` |
| Encoder format | — | `train_cls.jsonl` |
| Primary use | Full-FT vs LoRA — generation | Full-FT vs LoRA — classification |

Using both datasets together allows the research to evaluate the impact of
full fine-tuning vs. PEFT across two distinct legal NLP task types within a
single experimental framework.
