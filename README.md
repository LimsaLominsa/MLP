# legal-llm-finetuning

Research codebase studying **LoRA vs. Full Fine-Tuning vs. QLoRA (4-bit)** on small-scale LLMs (~1B parameters) for legal NLP tasks.

**Status: All 8 experiments complete. Results committed to `results/`.**

---

## Models

| Model | Size | Access |
|-------|------|--------|
| Qwen2.5-1.5B-Instruct | 1.5B | Public — no auth needed |
| Llama-3.2-1B-Instruct | 1B | **Gated** — requires HuggingFace license acceptance |

### Llama Access Setup

```bash
huggingface-cli login   # paste token from huggingface.co/settings/tokens
# OR
export HF_TOKEN="hf_your_token_here"
```

---

## Tasks & Datasets

| Dataset | Task Type | Evaluation Metric |
|---------|-----------|-------------------|
| BillSum | Abstractive summarization | ROUGE-1/2/L, BERTScore-F1 |
| CaseHOLD | 5-way multiple-choice classification | Accuracy |

**BillSum test splits:** `test_us` (in-distribution US federal bills) and `test_ca` (out-of-distribution California bills).

### Token Length Statistics

| Dataset | p95 tokens | max tokens | `max_seq_length` used |
|---------|-----------|-----------|----------------------|
| BillSum (bill text) | 2,941 | 4,652 | 16,384 |
| CaseHOLD (context + options) | 553 | 691 | 1,024 |

---

## Experiment Design

8 experiments across 2 models × 2 datasets × 2 methods:

| ID | Model | Dataset | Method | Training Script | Config |
|----|-------|---------|--------|-----------------|--------|
| E1 | Qwen2.5-1.5B | BillSum | LoRA (r=16) | `train.py` | `lora_billsum_qwen.yaml` |
| E2 | Llama-3.2-1B | BillSum | LoRA (r=16) | `train.py` | `lora_billsum_llama.yaml` |
| E3 | Qwen2.5-1.5B | BillSum | Full FT | `train.py` | `full_billsum_qwen.yaml` |
| E4 | Llama-3.2-1B | BillSum | Full FT | `train.py` | `full_billsum_llama.yaml` |
| E5 | Qwen2.5-1.5B | CaseHOLD | LoRA (r=16) | `train.py` | `lora_casehold_qwen.yaml` |
| E6 | Llama-3.2-1B | CaseHOLD | LoRA (r=16) | `train.py` | `lora_casehold_llama.yaml` |
| E7 | Qwen2.5-1.5B | CaseHOLD | QLoRA 4-bit | `train_casehold_lora.py` | `qlora_casehold_qwen.yaml` |
| E8 | Llama-3.2-1B | CaseHOLD | QLoRA 4-bit | `train_casehold_lora.py` | `qlora_casehold_llama.yaml` |

**Server:** AutoDL H800 80GB · **Conda env:** `llm-ft` · **Seed:** 42 · **Epochs:** 1

### Hyperparameters

| Param | LoRA / Full FT | QLoRA |
|-------|---------------|-------|
| LoRA r / alpha | 16 / 32 | 16 / 32 |
| Target modules | q, k, v, o projections | q, k, v, o projections |
| learning_rate | 2e-4 | 2e-4 |
| Quantization | — | 4-bit NF4, double quant, bfloat16 compute |
| `load_best_model_at_end` | True | True |

---

## Results

### BillSum — Summarization (ROUGE-2, BERTScore-F1)

| Method | Model | test_us ROUGE-2 | test_ca ROUGE-2 | test_us BERTScore | test_ca BERTScore |
|--------|-------|:--------------:|:--------------:|:----------------:|:----------------:|
| LoRA | Qwen2.5-1.5B | 0.3725 | 0.1764 | 0.9024 | 0.8570 |
| LoRA | Llama-3.2-1B | 0.3741 | **0.2116** | 0.8988 | 0.8583 |
| Full FT | Qwen2.5-1.5B | 0.3688 | 0.1711 | 0.9018 | 0.8558 |
| Full FT | Llama-3.2-1B | **0.3839** | **0.2116** | 0.9003 | 0.8585 |

Full evaluation JSONs: `results/billsum/`

### CaseHOLD — Multiple-Choice Classification (Accuracy, n=5,221)

| Method | Model | Accuracy |
|--------|-------|:--------:|
| LoRA | Qwen2.5-1.5B | **0.8602** |
| LoRA | Llama-3.2-1B | **0.8617** |
| QLoRA 4-bit | Qwen2.5-1.5B | 0.7395 |
| QLoRA 4-bit | Llama-3.2-1B | 0.7179 |

Full evaluation JSONs (incl. per-class breakdown): `results/casehold/`

### Key Observations

1. **LoRA ≈ Full FT on BillSum** — ROUGE-2 difference < 0.02, suggesting LoRA is sufficient for summarization at this scale.
2. **Llama outperforms Qwen on domain generalization** — test_ca ROUGE-2: 0.2116 vs 0.1764 (LoRA). Both models never trained on CA bills.
3. **QLoRA 4-bit has ~12% accuracy drop vs LoRA** on CaseHOLD (0.74 vs 0.86). Root cause: 4-bit quantization degrades small (~1B) models significantly; additionally, the installed TRL version lacks `DataCollatorForCompletionOnlyLM`, so QLoRA trains with full-sequence loss rather than answer-only loss.
4. **No invalid predictions** in any CaseHOLD run — the instruction-tuned models reliably output A/B/C/D/E.

---

## Project Structure

```
legal-llm-finetuning/
├── configs/
│   ├── lora_billsum_qwen.yaml        LoRA — BillSum × Qwen2.5-1.5B
│   ├── lora_billsum_llama.yaml       LoRA — BillSum × Llama-3.2-1B
│   ├── full_billsum_qwen.yaml        Full FT — BillSum × Qwen2.5-1.5B
│   ├── full_billsum_llama.yaml       Full FT — BillSum × Llama-3.2-1B
│   ├── lora_casehold_qwen.yaml       LoRA — CaseHOLD × Qwen2.5-1.5B
│   ├── lora_casehold_llama.yaml      LoRA — CaseHOLD × Llama-3.2-1B
│   ├── qlora_casehold_qwen.yaml      QLoRA 4-bit — CaseHOLD × Qwen2.5-1.5B
│   ├── qlora_casehold_llama.yaml     QLoRA 4-bit — CaseHOLD × Llama-3.2-1B
│   ├── full_casehold_{qwen,llama}.yaml   (defined, not yet run)
│   ├── sweep_{billsum,casehold}_qwen.yaml  hyperparameter sweep configs
│   └── test_local_cpu.yaml           local smoke-test config
│
├── src/
│   ├── data/
│   │   ├── billsum/
│   │   │   ├── run_preprocessing.py      pipeline entry point
│   │   │   ├── data_cleaning.py
│   │   │   ├── data_formatting.py        produces SFT *_sft.jsonl
│   │   │   └── data_analysis.py
│   │   └── casehold/
│   │       ├── run_casehold_preprocessing.py
│   │       ├── casehold_cleaning.py
│   │       ├── casehold_formatting.py    produces *_mc.jsonl (multiple-choice)
│   │       └── casehold_analysis.py
│   ├── train/
│   │   ├── train.py                  LoRA + Full FT (BillSum & CaseHOLD)
│   │   ├── train_casehold_lora.py    QLoRA 4-bit (CaseHOLD only)
│   │   ├── train_sweep.py            hyperparameter sweep runner
│   │   ├── smoke_test.py             local CPU data validation
│   │   └── local_mini_test.py        local mini training test
│   └── evaluate/
│       ├── inference.py              generate predictions (all tasks)
│       ├── eval_billsum.py           ROUGE + BERTScore scorer
│       ├── eval_casehold.py          accuracy + per-class scorer
│       └── check_token_length.py     token length analysis
│
├── results/                          ✅ committed — all eval outputs
│   ├── billsum/
│   │   ├── lora_qwen_test_us.json    ROUGE + BERTScore for E1
│   │   ├── lora_qwen_test_ca.json
│   │   ├── lora_llama_test_us.json   ROUGE + BERTScore for E2
│   │   ├── lora_llama_test_ca.json
│   │   ├── full_qwen_test_us.json    ROUGE + BERTScore for E3
│   │   ├── full_qwen_test_ca.json
│   │   ├── full_llama_test_us.json   ROUGE + BERTScore for E4
│   │   └── full_llama_test_ca.json
│   └── casehold/
│       ├── lora_qwen_test.json       accuracy + per-class for E5
│       ├── lora_llama_test.json      accuracy + per-class for E6
│       ├── qlora_qwen_test.json      accuracy + per-class for E7
│       ├── qlora_llama_test.json     accuracy + per-class for E8
│       ├── qlora_casehold_qwen_run_summary.json   training metadata E7
│       └── qlora_casehold_llama_run_summary.json  training metadata E8
│
├── scripts/
│   ├── setup_env.sh                  AutoDL server env setup
│   └── run_experiment.sh             single experiment launcher
│
├── autodl_run.ipynb                  Jupyter notebook — full pipeline on AutoDL
├── data/                             NOT in git (~600 MB JSONL)
├── outputs/                          NOT in git (symlink → autodl-tmp, model weights)
├── logs/                             NOT in git (symlink → autodl-tmp)
├── .gitignore
└── README.md
```

---

## Data & Git Policy

| Path | In git | Notes |
|------|--------|-------|
| `configs/` | ✅ | All YAML configs |
| `src/` | ✅ | All source code |
| `results/` | ✅ | All eval JSON outputs |
| `autodl_run.ipynb` | ✅ | AutoDL execution notebook |
| `data/` | ❌ | ~600 MB JSONL, reproduce locally |
| `outputs/` | ❌ | Model weights (symlink on AutoDL → persistent disk) |
| `logs/` | ❌ | Training logs (symlink on AutoDL) |

---

## Reproducing the Pipeline

### 1. Environment Setup (AutoDL server)

```bash
git clone https://github.com/LimsaLominsa/MLP.git
cd MLP
bash scripts/setup_env.sh
conda activate llm-ft
huggingface-cli login   # required for Llama
```

### 2. Data Preprocessing

```bash
# CaseHOLD — auto-downloads from HuggingFace Hub
python src/data/casehold/run_casehold_preprocessing.py

# BillSum — requires raw billsum_v4_1/ files
python src/data/billsum/run_preprocessing.py
```

Output: `data/casehold/*_mc.jsonl`, `data/billsum/*_sft.jsonl`

### 3. Training

```bash
# LoRA or Full FT (BillSum and CaseHOLD LoRA)
python src/train/train.py --config configs/lora_billsum_qwen.yaml
python src/train/train.py --config configs/full_billsum_llama.yaml
python src/train/train.py --config configs/lora_casehold_qwen.yaml

# QLoRA 4-bit (CaseHOLD only)
python src/train/train_casehold_lora.py --config configs/qlora_casehold_qwen.yaml
```

Adapters saved to `outputs/<config_name>/final_adapter/`.

### 4. Inference

```bash
python src/evaluate/inference.py --config configs/lora_billsum_qwen.yaml --split test_us
python src/evaluate/inference.py --config configs/lora_casehold_qwen.yaml --split test
```

Predictions saved to `outputs/<config_name>/predictions_<split>.jsonl`.

### 5. Evaluation

```bash
# BillSum (ROUGE + BERTScore)
python src/evaluate/eval_billsum.py \
    --predictions outputs/lora_billsum_qwen/predictions_test_us.jsonl \
    --output      results/billsum/lora_qwen_test_us.json

# CaseHOLD (accuracy + per-class)
python src/evaluate/eval_casehold.py \
    --predictions outputs/lora_casehold_qwen/predictions_test.jsonl \
    --output      results/casehold/lora_qwen_test.json
```

### 6. Or: Run Everything via Notebook

The full pipeline (preprocessing → training → inference → evaluation) is available as a single Jupyter notebook for AutoDL:

```
autodl_run.ipynb
```

---

## Notes on QLoRA Implementation

`train_casehold_lora.py` uses TRL `SFTTrainer` + bitsandbytes 4-bit quantization. The installed TRL version on AutoDL does **not** export `DataCollatorForCompletionOnlyLM`, so the trainer falls back to full-sequence loss (all tokens supervised, not answer-only). This is the primary reason for the ~12% accuracy gap between QLoRA and LoRA. Future work could:

- Upgrade TRL to a version with `DataCollatorForCompletionOnlyLM`
- Implement manual label masking in the dataset preparation step
- Run Full FT on CaseHOLD (configs exist: `full_casehold_{qwen,llama}.yaml`) for a third comparison point
