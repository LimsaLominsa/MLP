# legal-llm-finetuning

Research codebase for studying the effect of **Full Fine-Tuning vs. Parameter-Efficient Fine-Tuning (LoRA)** on small-scale LLMs for legal NLP tasks.

## Models

| Model | Size | Access |
|-------|------|--------|
| Qwen2.5-1.5B-Instruct | 1.5B | Public — no auth needed |
| Llama-3.2-1B-Instruct | 1B | **Gated** — requires HuggingFace account + license acceptance |

### Llama Access Setup (required before training)

Llama-3.2 is a gated model. Complete these steps **once** before using it:

1. Visit [https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and accept the license
2. Generate a HuggingFace token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (type: *Read*)
3. On the GCP server, authenticate:

```bash
pip install huggingface_hub
huggingface-cli login
# paste your token when prompted
```

Or set the environment variable directly (useful for scripts):

```bash
export HF_TOKEN="hf_your_token_here"   # add to ~/.bashrc to persist
```

After login, model weights download automatically the first time `train.py` runs.

---

## Tasks & Datasets

| Dataset | Task | Evaluation |
|---------|------|------------|
| BillSum | Legal summarization (generative) | ROUGE-1/2/L + BERTScore |
| CaseHOLD | Legal holding prediction (multiple-choice) | Accuracy |

### Token Length Statistics (verified locally)

| Dataset | Field | p95 tokens | max tokens | Config max_length |
|---------|-------|-----------|-----------|------------------|
| BillSum | input (bill text) | 2,941 | 4,652 | 16,384 (team config) |
| CaseHOLD | input (context + options) | 553 | 691 | **1,024** |
| CaseHOLD | output (answer letter) | 1 | 1 | 2,048 |

---

## GitHub & Data Policy

**Data files are NOT pushed to GitHub** — excluded via `.gitignore`.

Reasons:
- BillSum SFT files total ~430 MB; CaseHOLD MC files total ~193 MB
- GitHub enforces a 100 MB per-file hard limit and recommends repos stay under 1 GB
- Raw data is publicly available and can be regenerated from the preprocessing scripts

### What IS in the repo (push this)

```
configs/          YAML training configs
src/              All Python source code
scripts/          Server setup and launch scripts
README.md
.gitignore
```

### What is NOT in the repo (excluded)

```
data/             Preprocessed JSONL files (~600 MB total)
outputs/          Model checkpoints and predictions
*.jsonl
*.png
```

### Reproducing data on the GCP server

After cloning the repo on the server, regenerate the processed data:

```bash
# Install preprocessing deps
pip install pandas huggingface-hub matplotlib

# BillSum — requires raw files uploaded separately (or symlinked)
python src/data/billsum/run_preprocessing.py

# CaseHOLD — downloads raw CSVs automatically from HuggingFace Hub
python src/data/casehold/run_casehold_preprocessing.py
```

Alternatively, upload the processed `data/` directory directly via `gcloud storage cp` or `scp`.

---

## Project Structure

```
legal-llm-finetuning/
├── configs/
│   ├── lora_billsum_qwen.yaml       LoRA config — BillSum × Qwen2.5-1.5B
│   ├── lora_billsum_llama.yaml      LoRA config — BillSum × Llama-3.2-1B
│   ├── lora_casehold_qwen.yaml      LoRA config — CaseHOLD × Qwen2.5-1.5B
│   └── lora_casehold_llama.yaml     LoRA config — CaseHOLD × Llama-3.2-1B
├── src/
│   ├── data/
│   │   ├── billsum/                 BillSum preprocessing scripts
│   │   └── casehold/                CaseHOLD preprocessing scripts
│   ├── train/
│   │   ├── train.py                 LoRA training entry point
│   │   └── smoke_test.py            Local CPU validation + token length check
│   └── evaluate/
│       ├── eval_billsum.py          ROUGE + BERTScore evaluation
│       ├── eval_casehold.py         Accuracy evaluation
│       └── check_token_length.py    Token length analysis
├── scripts/
│   ├── setup_env.sh                 GCP server environment setup
│   └── run_experiment.sh            Single experiment launch
├── data/                            NOT in git — see Data Policy above
├── outputs/                         NOT in git — model checkpoints
├── .gitignore
└── README.md
```

---

## Local Validation (CPU only, no GPU needed)

Before deploying to GCP, verify your data and tokenizer setup locally:

```bash
python src/train/smoke_test.py
```

This checks:
- All data files exist and record counts are correct
- All required fields (`text`, `input`, `output`, `label`) are present
- Qwen tokenizer loads and tokenizes correctly
- Prints recommended `max_length` values for all configs

Expected output: `Smoke test PASSED`

---

## GCP Server Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/legal-llm-finetuning.git
cd legal-llm-finetuning

# 2. Set up Python environment
bash scripts/setup_env.sh
conda activate llm-ft

# 3. Authenticate HuggingFace (needed for Llama)
huggingface-cli login

# 4. Upload or regenerate data
# Option A: upload from local
# gcloud storage cp -r data/ gs://<your-bucket>/data/
# gcloud storage cp -r gs://<your-bucket>/data/ ./data/

# Option B: regenerate on server
python src/data/casehold/run_casehold_preprocessing.py   # downloads automatically
python src/data/billsum/run_preprocessing.py              # needs raw billsum files
```

---

## Step 1 — Train

```bash
# Single experiment via script
bash scripts/run_experiment.sh lora_billsum_qwen
bash scripts/run_experiment.sh lora_billsum_llama
bash scripts/run_experiment.sh lora_casehold_qwen
bash scripts/run_experiment.sh lora_casehold_llama

# Or directly
python src/train/train.py --config configs/lora_billsum_qwen.yaml
```

Checkpoints and the final LoRA adapter are saved under `outputs/<config_name>/`.

---

## Step 2 — Evaluate

**BillSum** (ROUGE + BERTScore):
```bash
python src/evaluate/eval_billsum.py \
    --predictions outputs/lora_billsum_qwen/predictions_test_us.jsonl \
    --output      outputs/lora_billsum_qwen/eval_test_us.json
```

**CaseHOLD** (Accuracy):
```bash
python src/evaluate/eval_casehold.py \
    --predictions outputs/lora_casehold_qwen/predictions_test.jsonl \
    --output      outputs/lora_casehold_qwen/eval_test.json
```

> An inference script for generating predictions will be added after GCP environment is confirmed.

---

## Experiment Design

| Experiment | Model | Dataset | Method |
|------------|-------|---------|--------|
| E1 | Qwen2.5-1.5B | BillSum | LoRA (r=16) |
| E2 | Llama-3.2-1B | BillSum | LoRA (r=16) |
| E3 | Qwen2.5-1.5B | CaseHOLD | LoRA (r=16) |
| E4 | Llama-3.2-1B | CaseHOLD | LoRA (r=16) |
| E5 | Qwen2.5-1.5B | BillSum | Full FT |
| E6 | Llama-3.2-1B | BillSum | Full FT |
| E7 | Qwen2.5-1.5B | CaseHOLD | Full FT |
| E8 | Llama-3.2-1B | CaseHOLD | Full FT |

Baseline: zero-shot inference (no fine-tuning).

---

## LoRA Configuration (fixed across all experiments)

| Param | Value |
|-------|-------|
| r | 16 |
| alpha | 32 |
| dropout | 0.05 |
| target_modules | q, k, v, o projections |
| learning_rate | 2e-4 |
| epochs | 1 |
| seed | 42 |
