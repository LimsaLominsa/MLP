# legal-llm-finetuning

Research codebase studying **LoRA vs. Full Fine-Tuning vs. QLoRA (4-bit)** on small-scale LLMs (~1B parameters) across legal and medical NLP tasks.

**Status:**
- **Phase 1–2 (Legal):** All 14 experiments complete. Results in `results/billsum/` and `results/casehold/`.
- **Phase 3a (Medical — NFCorpus Reranking):** All 8 experiments complete (incl. baseline). Results in `results/nfcorpus/`.
- **Phase 3b (Medical — PubMed Summarization):** In progress on AutoDL H800.

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

| Dataset | Domain | Task Type | Evaluation Metric | Train Size |
|---------|--------|-----------|-------------------|------------|
| BillSum | Legal | Abstractive summarization | ROUGE-1/2/L, BERTScore-F1 | 18,949 |
| CaseHOLD | Legal | 5-way multiple-choice classification | Accuracy | 45,000 |
| PubMed | Medical | Abstractive summarization | ROUGE-1/2/L, BERTScore-F1 | 20,000 |
| NFCorpus | Medical | Document reranking (top-5) | NDCG@5, MAP@5 | 2,590 |

**BillSum test splits:** `test_us` (in-distribution US federal bills) and `test_ca` (out-of-distribution California bills).

**NFCorpus:** Each sample contains a biomedical query and 5 candidate documents; the model must rank them by relevance. Small dataset (323 test samples).

**PubMed:** Scientific article summarization from PubMed abstracts (auto-downloaded from HuggingFace Hub).

### Token Length Statistics

| Dataset | p95 tokens | max tokens | `max_seq_length` used |
|---------|-----------|-----------|----------------------|
| BillSum (bill text) | 2,941 | 4,652 | 16,384 |
| CaseHOLD (context + options) | 553 | 691 | 1,024 |
| PubMed (article text) | ~2,000 | ~3,500 | 2,048 + 512 |
| NFCorpus (query + 5 docs) | ~800 | ~1,200 | 1,024 + 64 |

---

## Experiment Design

### Phase 1: LoRA vs Full FT vs QLoRA (8 experiments)

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

### Phase 2: Full FT CaseHOLD + Random Label Baseline (6 experiments)

| ID | Model | Dataset | Method | Training Script | Config |
|----|-------|---------|--------|-----------------|--------|
| E9  | Qwen2.5-1.5B | CaseHOLD | Full FT | `train.py` | `full_casehold_qwen.yaml` |
| E10 | Llama-3.2-1B | CaseHOLD | Full FT | `train.py` | `full_casehold_llama.yaml` |
| E11 | Qwen2.5-1.5B | BillSum  | LoRA (random labels) | `train.py` | `random_billsum_qwen.yaml` |
| E12 | Llama-3.2-1B | BillSum  | LoRA (random labels) | `train.py` | `random_billsum_llama.yaml` |
| E13 | Qwen2.5-1.5B | CaseHOLD | LoRA (random labels) | `train.py` | `random_casehold_qwen.yaml` |
| E14 | Llama-3.2-1B | CaseHOLD | LoRA (random labels) | `train.py` | `random_casehold_llama.yaml` |

### Phase 3a: Medical Domain — NFCorpus Reranking (8 experiments)

| ID | Model | Dataset | Method | Config | Status |
|----|-------|---------|--------|--------|--------|
| E21 | Qwen2.5-1.5B | NFCorpus | LoRA (r=16) | `lora_nfcorpus_qwen.yaml` | Done |
| E22 | Llama-3.2-1B | NFCorpus | LoRA (r=16) | `lora_nfcorpus_llama.yaml` | Done |
| E23 | Qwen2.5-1.5B | NFCorpus | Full FT | `full_nfcorpus_qwen.yaml` | Done |
| E24 | Llama-3.2-1B | NFCorpus | Full FT | `full_nfcorpus_llama.yaml` | Done |
| E25 | Qwen2.5-1.5B | NFCorpus | LoRA (random labels) | `random_nfcorpus_qwen.yaml` | Done |
| E26 | Llama-3.2-1B | NFCorpus | LoRA (random labels) | `random_nfcorpus_llama.yaml` | Done |
| B1  | Qwen2.5-1.5B | NFCorpus | Baseline (zero-shot) | `baseline_nfcorpus_qwen.yaml` | Done |
| B2  | Llama-3.2-1B | NFCorpus | Baseline (zero-shot) | `baseline_nfcorpus_llama.yaml` | Done |

### Phase 3b: Medical Domain — PubMed Summarization (8 experiments)

| ID | Model | Dataset | Method | Config | Status |
|----|-------|---------|--------|--------|--------|
| E15 | Qwen2.5-1.5B | PubMed | LoRA (r=16) | `lora_pubmed_qwen.yaml` | Running |
| E16 | Llama-3.2-1B | PubMed | LoRA (r=16) | `lora_pubmed_llama.yaml` | Running |
| E17 | Qwen2.5-1.5B | PubMed | Full FT | `full_pubmed_qwen.yaml` | Running |
| E18 | Llama-3.2-1B | PubMed | Full FT | `full_pubmed_llama.yaml` | Running |
| E19 | Qwen2.5-1.5B | PubMed | LoRA (random labels) | `random_pubmed_qwen.yaml` | Running |
| E20 | Llama-3.2-1B | PubMed | LoRA (random labels) | `random_pubmed_llama.yaml` | Running |
| B3  | Qwen2.5-1.5B | PubMed | Baseline (zero-shot) | `baseline_pubmed_qwen.yaml` | Running |
| B4  | Llama-3.2-1B | PubMed | Baseline (zero-shot) | `baseline_pubmed_llama.yaml` | Running |

**Random label experiments** shuffle the training set outputs (labels) while keeping inputs unchanged, creating a random input→output mapping. If LoRA models learn genuine task knowledge, their performance should far exceed random-label baselines.

**Baseline experiments** run inference directly with the pretrained model (no fine-tuning), establishing zero-shot performance for comparison.

**Server:** AutoDL H800/A800 80GB · **Seed:** 42

### Hyperparameters

#### Legal Tasks (Phase 1–2)

| Param | LoRA / Full FT | QLoRA |
|-------|---------------|-------|
| LoRA r / alpha | 16 / 32 | 16 / 32 |
| Target modules | q, k, v, o projections | q, k, v, o projections |
| learning_rate | 2e-4 (LoRA), 2e-5 (Full FT) | 2e-4 |
| Epochs | 1 | 1 |
| Effective batch size | 16 | 16 |
| Quantization | — | 4-bit NF4, double quant, bfloat16 compute |
| `load_best_model_at_end` | True | True |

#### PubMed Summarization (Phase 3b)

| Param | LoRA | Full FT |
|-------|------|---------|
| learning_rate | 2e-4 | 2e-5 |
| Epochs | 1 | 1 |
| Batch (per device × grad accum) | 4 × 4 = 16 | 4 × 4 = 16 |
| max_input / max_output | 2048 / 512 | 2048 / 512 |
| gradient_checkpointing | True | True |

#### NFCorpus Reranking (Phase 3a)

| Param | LoRA | Full FT |
|-------|------|---------|
| learning_rate | 2e-4 | 2e-5 |
| Epochs | 3 | 3 |
| Batch (per device × grad accum) | 8 × 2 = 16 | 8 × 2 = 16 |
| max_input / max_output | 1024 / 64 | 1024 / 64 |
| gradient_checkpointing | False (LoRA) / True (Full FT) | True |

---

## Results

### BillSum — Legal Summarization (ROUGE-2, BERTScore-F1)

| Method | Model | test_us ROUGE-2 | test_ca ROUGE-2 | test_us BERTScore | test_ca BERTScore |
|--------|-------|:--------------:|:--------------:|:----------------:|:----------------:|
| LoRA | Qwen2.5-1.5B | 0.3725 | 0.1764 | 0.9024 | 0.8570 |
| LoRA | Llama-3.2-1B | 0.3741 | **0.2116** | 0.8988 | 0.8583 |
| Full FT | Qwen2.5-1.5B | 0.3688 | 0.1711 | 0.9018 | 0.8558 |
| Full FT | Llama-3.2-1B | **0.3839** | **0.2116** | 0.9003 | 0.8585 |
| LoRA (random) | Qwen2.5-1.5B | 0.0161 | 0.0022 | 0.8262 | 0.8168 |
| LoRA (random) | Llama-3.2-1B | 0.0292 | 0.0144 | 0.8237 | 0.8079 |

Full evaluation JSONs: `results/billsum/`

### CaseHOLD — Legal Multiple-Choice Classification (Accuracy, n=5,221)

| Method | Model | Accuracy |
|--------|-------|:--------:|
| LoRA | Qwen2.5-1.5B | **0.8602** |
| LoRA | Llama-3.2-1B | **0.8617** |
| Full FT | Qwen2.5-1.5B | 0.8339 |
| Full FT | Llama-3.2-1B | **0.8632** |
| QLoRA 4-bit | Qwen2.5-1.5B | 0.7395 |
| QLoRA 4-bit | Llama-3.2-1B | 0.7179 |
| LoRA (random) | Qwen2.5-1.5B | 0.2300 |
| LoRA (random) | Llama-3.2-1B | 0.2099 |

Full evaluation JSONs (incl. per-class breakdown): `results/casehold/`

### NFCorpus — Medical Document Reranking (NDCG@5, MAP@5, n=323)

| Method | Model | NDCG@5 | MAP@5 |
|--------|-------|:------:|:-----:|
| LoRA | Qwen2.5-1.5B | **0.8943** | **0.8586** |
| LoRA | Llama-3.2-1B | 0.8913 | 0.8509 |
| Full FT | Qwen2.5-1.5B | 0.8859 | 0.8451 |
| Full FT | Llama-3.2-1B | 0.8820 | 0.8365 |
| LoRA (random) | Qwen2.5-1.5B | 0.7914 | 0.7012 |
| LoRA (random) | Llama-3.2-1B | 0.7873 | 0.6934 |
| Baseline (zero-shot) | Qwen2.5-1.5B | 0.8271 | 0.7384 |
| Baseline (zero-shot) | Llama-3.2-1B | 0.7879 | 0.6950 |

Full evaluation JSONs: `results/nfcorpus/`

### PubMed — Medical Summarization (ROUGE-2, BERTScore-F1)

> ⏳ **Experiments in progress on AutoDL H800.** Results will be added upon completion.

| Method | Model | ROUGE-2 | BERTScore-F1 |
|--------|-------|:-------:|:------------:|
| LoRA | Qwen2.5-1.5B | — | — |
| LoRA | Llama-3.2-1B | — | — |
| Full FT | Qwen2.5-1.5B | — | — |
| Full FT | Llama-3.2-1B | — | — |
| LoRA (random) | Qwen2.5-1.5B | — | — |
| LoRA (random) | Llama-3.2-1B | — | — |
| Baseline (zero-shot) | Qwen2.5-1.5B | — | — |
| Baseline (zero-shot) | Llama-3.2-1B | — | — |

Full evaluation JSONs: `results/pubmed/` (pending)

### Key Observations

#### Legal Domain (BillSum + CaseHOLD)

1. **LoRA ≈ Full FT on BillSum** — ROUGE-2 difference < 0.02, suggesting LoRA is sufficient for summarization at this scale.
2. **LoRA ≈ Full FT on CaseHOLD** — Llama Full FT (0.8632) slightly outperforms LoRA (0.8617); Qwen Full FT (0.8339) slightly underperforms LoRA (0.8602), possibly due to overfitting with full-parameter updates.
3. **Llama outperforms Qwen on domain generalization** — test_ca ROUGE-2: 0.2116 vs 0.1764 (LoRA). Both models never trained on CA bills.
4. **QLoRA 4-bit has ~12% accuracy drop vs LoRA** on CaseHOLD (0.74 vs 0.86). Root cause: 4-bit quantization degrades small (~1B) models significantly; additionally, the installed TRL version lacks `DataCollatorForCompletionOnlyLM`, so QLoRA trains with full-sequence loss rather than answer-only loss.
5. **Random label baselines confirm genuine learning** — On BillSum, random-label ROUGE-2 drops to near zero (0.016–0.029 vs 0.37+), a >90% relative decrease. On CaseHOLD, random-label accuracy (~0.21) is close to the random-guess baseline of 0.20 (5-way classification), while LoRA achieves 0.86.
6. **No invalid predictions** in any CaseHOLD run — the instruction-tuned models reliably output A/B/C/D/E.

#### Medical Domain (NFCorpus)

7. **LoRA > Full FT on small datasets** — LoRA (NDCG@5 = 0.894) outperforms Full FT (0.886) on NFCorpus. With only ~2,590 training samples, Full FT overfits while LoRA's parameter efficiency provides implicit regularization.
8. **High zero-shot baseline limits task discriminability** — The pretrained Qwen baseline already achieves NDCG@5 = 0.827 without any fine-tuning, indicating the model's pretrained knowledge can already perform basic query-document matching. The LoRA improvement over baseline is +0.067 (relative +8.1%).
9. **Random label ≈ baseline for Llama** — Llama random-label NDCG@5 (0.787) is nearly identical to its zero-shot baseline (0.788), confirming the random-label model learns nothing meaningful. Interestingly, Qwen random-label (0.791) is *lower* than its baseline (0.827), suggesting that training on corrupted labels can actively degrade pretrained capabilities.
10. **NFCorpus has limited benchmark utility for LLMs** — With small test size (n=323), high variance (std ≈ 0.14–0.16), and strong zero-shot performance, NFCorpus reranking provides limited discriminative power for comparing fine-tuning strategies on instruction-tuned LLMs.

---

## Project Structure

```
legal-llm-finetuning/
├── configs/
│   ├── lora_billsum_{qwen,llama}.yaml        LoRA — BillSum
│   ├── full_billsum_{qwen,llama}.yaml        Full FT — BillSum
│   ├── lora_casehold_{qwen,llama}.yaml       LoRA — CaseHOLD
│   ├── qlora_casehold_{qwen,llama}.yaml      QLoRA 4-bit — CaseHOLD
│   ├── full_casehold_{qwen,llama}.yaml       Full FT — CaseHOLD
│   ├── random_billsum_{qwen,llama}.yaml      Random Label — BillSum
│   ├── random_casehold_{qwen,llama}.yaml     Random Label — CaseHOLD
│   ├── lora_pubmed_{qwen,llama}.yaml         LoRA — PubMed
│   ├── full_pubmed_{qwen,llama}.yaml         Full FT — PubMed
│   ├── random_pubmed_{qwen,llama}.yaml       Random Label — PubMed
│   ├── lora_nfcorpus_{qwen,llama}.yaml       LoRA — NFCorpus
│   ├── full_nfcorpus_{qwen,llama}.yaml       Full FT — NFCorpus
│   ├── random_nfcorpus_{qwen,llama}.yaml     Random Label — NFCorpus
│   ├── baseline_nfcorpus_{qwen,llama}.yaml   Baseline (zero-shot) — NFCorpus
│   ├── sweep_{billsum,casehold}_qwen.yaml    Hyperparameter sweep configs
│   └── test_local_cpu.yaml                   Local smoke-test config
│
├── src/
│   ├── data/
│   │   ├── billsum/
│   │   │   ├── run_preprocessing.py          Pipeline entry point
│   │   │   ├── data_cleaning.py
│   │   │   ├── data_formatting.py            Produces SFT *_sft.jsonl
│   │   │   └── data_analysis.py
│   │   ├── casehold/
│   │   │   ├── run_casehold_preprocessing.py
│   │   │   ├── casehold_cleaning.py
│   │   │   ├── casehold_formatting.py        Produces *_mc.jsonl
│   │   │   └── casehold_analysis.py
│   │   ├── pubmed/
│   │   │   └── pubmed_formatting.py          Downloads & formats PubMed data
│   │   └── nfcorpus/
│   │       └── nfcorpus_formatting.py        Downloads & formats NFCorpus data
│   ├── train/
│   │   ├── train.py                  LoRA + Full FT (all tasks)
│   │   ├── train_casehold_lora.py    QLoRA 4-bit (CaseHOLD only)
│   │   ├── train_sweep.py            Hyperparameter sweep runner
│   │   ├── smoke_test.py             Local CPU data validation
│   │   └── local_mini_test.py        Local mini training test
│   └── evaluate/
│       ├── inference.py              Generate predictions (all tasks)
│       ├── eval_billsum.py           ROUGE + BERTScore scorer (BillSum & PubMed)
│       ├── eval_casehold.py          Accuracy + per-class scorer
│       ├── eval_rerank.py            NDCG@k + MAP@k scorer (NFCorpus)
│       └── check_token_length.py     Token length analysis
│
├── results/                          ✅ committed — all eval outputs
│   ├── billsum/                      12 JSON files (E1–E4, E11–E12)
│   ├── casehold/                     10 JSON files (E5–E10, E13–E14)
│   ├── nfcorpus/                     8 JSON files (E21–E26, B1–B2)
│   └── pubmed/                       (pending)
│
├── models/                           ✅ committed — LoRA adapters (~30-50 MB each)
│   ├── lora_billsum_{qwen,llama}/
│   ├── lora_casehold_{qwen,llama}/
│   ├── qlora_casehold_{qwen,llama}/
│   ├── random_billsum_{qwen,llama}/
│   ├── random_casehold_{qwen,llama}/
│   └── nfcorpus/
│       ├── lora_{qwen,llama}/
│       └── random_{qwen,llama}/
│
├── scripts/
│   ├── setup_env.sh                  AutoDL server env setup
│   └── run_experiment.sh             Single experiment launcher
│
├── autodl_run.ipynb                  Phase 1 pipeline (BillSum + CaseHOLD LoRA/Full FT)
├── autodl_run_phase2.ipynb           Phase 2 pipeline (Full FT CaseHOLD + Random Label)
├── autodl_run_phase3_pubmed.ipynb    Phase 3b pipeline (PubMed experiments)
├── autodl_run_phase3_nfcorpus.ipynb  Phase 3a pipeline (NFCorpus experiments)
├── data/                             NOT in git (symlink, ~600 MB JSONL)
├── outputs/                          NOT in git (symlink → autodl-tmp, full model weights)
├── logs/                             NOT in git (symlink → autodl-tmp)
├── .gitignore
└── README.md
```
│   └── run_experiment.sh             single experiment launcher
│
├── autodl_run.ipynb                  Jupyter notebook — Phase 1 pipeline on AutoDL
├── autodl_run_phase2.ipynb           Jupyter notebook — Phase 2 (Full FT CaseHOLD + Random Label)
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
| `configs/` | ✅ | All YAML configs (31 files) |
| `src/` | ✅ | All source code |
| `results/` | ✅ | All eval JSON outputs |
| `models/` | ✅ | LoRA adapters (~30-50 MB each) |
| `autodl_run*.ipynb` | ✅ | AutoDL execution notebooks (4 files) |
| `data/` | ❌ | ~600 MB JSONL; BillSum/CaseHOLD via scp, PubMed/NFCorpus auto-downloaded |
| `outputs/` | ❌ | Full FT model weights (~3 GB each, symlink on AutoDL) |
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

# PubMed — auto-downloads from HuggingFace Hub
python src/data/pubmed/pubmed_formatting.py

# NFCorpus — auto-downloads from HuggingFace Hub
python src/data/nfcorpus/nfcorpus_formatting.py
```

Output: `data/casehold/*_mc.jsonl`, `data/billsum/*_sft.jsonl`, `data/pubmed/*_sft.jsonl`, `data/nfcorpus/*_sft.jsonl`

#### Fix corrupt JSONL lines (run if training throws a JSON parse error)

```bash
python3 -c "
import json
good = []
bad = 0
with open('data/casehold/train_mc.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
            good.append(line)
        except json.JSONDecodeError as e:
            print(f'  skip Line {i}: {e}')
            bad += 1
with open('data/casehold/train_mc.jsonl', 'w') as f:
    f.writelines(good)
print(f'Finish. keep {len(good)} clauses, skip {bad} clauses.')
"
```

Replace `train_mc.jsonl` with `validation_mc.jsonl` or `test_mc.jsonl` to clean other splits. Same pattern applies to BillSum `*_sft.jsonl` files.

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
# BillSum / PubMed (ROUGE + BERTScore)
python src/evaluate/eval_billsum.py \
    --predictions outputs/lora_billsum_qwen/predictions_test_us.jsonl \
    --output      results/billsum/lora_qwen_test_us.json

# CaseHOLD (accuracy + per-class)
python src/evaluate/eval_casehold.py \
    --predictions outputs/lora_casehold_qwen/predictions_test.jsonl \
    --output      results/casehold/lora_qwen_test.json

# NFCorpus (NDCG@k + MAP@k)
python src/evaluate/eval_rerank.py \
    --predictions outputs/lora_nfcorpus_qwen/predictions_test.jsonl \
    --output      results/nfcorpus/lora_qwen_test.json
```

### 6. Or: Run Everything via Notebook

The full pipeline (preprocessing → training → inference → evaluation) is available as a single Jupyter notebook for AutoDL:

```
autodl_run.ipynb
```

---

## Notes on QLoRA Implementation

`train_casehold_lora.py` uses TRL `SFTTrainer` + bitsandbytes 4-bit quantization. The installed TRL version on AutoDL does **not** export `DataCollatorForCompletionOnlyLM`, so the trainer falls back to full-sequence loss (all tokens supervised, not answer-only). This is the primary reason for the ~12% accuracy gap between QLoRA and LoRA.

## Notes on Random Label Baseline

Random label experiments (E11–E14, E19–E20, E25–E26) use the same LoRA configuration as normal training, but the training set outputs are randomly shuffled (input text remains unchanged). This creates a sanity check:

- **BillSum:** Random-label ROUGE-2 collapses to 0.016–0.029 (vs 0.37+ with real labels), confirming LoRA learns genuine summarization ability.
- **CaseHOLD:** Random-label accuracy drops to ~0.21, close to 1/5 = 0.20 random-guess level (vs 0.86 with real labels). The slight difference from 0.20 is due to class distribution bias in the shuffled labels.
- **NFCorpus:** Random-label NDCG@5 (0.79) is comparable to zero-shot baseline (0.79–0.83), confirming the random-label model does not learn useful reranking signals. The relatively high absolute score is due to the pretrained model's inherent query-document matching ability (lexical/semantic overlap), not task-specific learning.

Random label data is generated on-the-fly in the respective `autodl_run_phase*.ipynb` notebooks (not committed to git).

## Notes on Baseline (Zero-Shot) Experiments

Baseline experiments run the pretrained model directly on test data without any fine-tuning. This establishes a lower bound and quantifies how much value fine-tuning adds:

- **NFCorpus:** Qwen zero-shot achieves NDCG@5 = 0.827, only 0.067 below LoRA (0.894). This high zero-shot floor suggests that instruction-tuned LLMs already possess strong query-document relevance judgment from pretraining.
- **PubMed:** Results pending.
