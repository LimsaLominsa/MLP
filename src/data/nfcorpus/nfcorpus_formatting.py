"""
nfcorpus_formatting.py
Download NFCorpus (BEIR) from HuggingFace and format into SFT JSONL
for passage reranking.

Dataset: BeIR/nfcorpus (medical information retrieval)
Task:    Given a query + 5 candidate passages, rank by relevance.

Each record contains:
  - text:      full prompt + ideal ranking (for training)
  - input:     prompt only (for inference)
  - output:    ideal ranking string, e.g. "2, 4, 1, 5, 3"
  - relevance: list of relevance scores for each candidate position
  - id:        query ID

Usage:
    python src/data/nfcorpus/nfcorpus_formatting.py [--output_dir data/nfcorpus]
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset


# ==================== Constants & Prompt Templates ====================

NUM_CANDIDATES = 5
MAX_PASSAGE_CHARS = 400  # ~100 tokens per passage, 5 × 100 = 500 tokens

TRAIN_TEMPLATE = (
    "Below are {n} candidate passages for a medical information query. "
    "Rank them from most relevant to least relevant.\n"
    "Output only the passage numbers separated by commas.\n\n"
    "### Query:\n{query}\n\n"
    "### Passages:\n{passages}\n\n"
    "### Ranking:\n{ranking}"
)

INFERENCE_TEMPLATE = (
    "Below are {n} candidate passages for a medical information query. "
    "Rank them from most relevant to least relevant.\n"
    "Output only the passage numbers separated by commas.\n\n"
    "### Query:\n{query}\n\n"
    "### Passages:\n{passages}\n\n"
    "### Ranking:\n"
)


# ==================== Data Loading ====================

def load_nfcorpus():
    """Load NFCorpus corpus, queries, and qrels from HuggingFace."""

    print("Downloading NFCorpus corpus...")
    corpus_ds = load_dataset("BeIR/nfcorpus", "corpus")
    corpus_key = list(corpus_ds.keys())[0]
    corpus = {}
    for row in corpus_ds[corpus_key]:
        doc_id = str(row["_id"])
        title = row.get("title", "")
        text = row.get("text", "")
        corpus[doc_id] = f"{title}. {text}" if title else text
    print(f"  Corpus: {len(corpus):,} documents")

    print("Downloading NFCorpus queries...")
    queries_ds = load_dataset("BeIR/nfcorpus", "queries")
    queries_key = list(queries_ds.keys())[0]
    queries = {}
    for row in queries_ds[queries_key]:
        queries[str(row["_id"])] = row["text"]
    print(f"  Queries: {len(queries):,} queries")

    print("Downloading NFCorpus qrels...")
    qrels_ds = load_dataset("BeIR/nfcorpus")
    qrels = {}
    for split_name in qrels_ds:
        qrels[split_name] = defaultdict(dict)
        for row in qrels_ds[split_name]:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            score = int(row["score"])
            qrels[split_name][qid][did] = score
        print(f"  QRels [{split_name}]: {len(qrels[split_name]):,} queries")

    return corpus, queries, qrels


# ==================== Record Building ====================

def build_reranking_records(queries, corpus, qrels_split,
                            all_doc_ids, num_candidates=5, seed=42):
    """Build reranking records for a given qrels split."""
    rng = random.Random(seed)
    records = []
    skipped = 0

    for qid, doc_scores in qrels_split.items():
        if qid not in queries:
            skipped += 1
            continue

        query_text = queries[qid]

        # Get relevant docs (score > 0) that exist in corpus
        relevant = {d: s for d, s in doc_scores.items()
                    if d in corpus and s > 0}
        if not relevant:
            skipped += 1
            continue

        # Sample relevant docs (at most half+1 of candidates)
        max_rel = min(len(relevant), (num_candidates + 1) // 2)
        rel_ids = rng.sample(sorted(relevant.keys()), max_rel)

        # Sample negatives from corpus (not positively labeled)
        num_neg = num_candidates - len(rel_ids)
        neg_pool = [d for d in all_doc_ids
                    if d not in doc_scores or doc_scores.get(d, 0) == 0]
        if len(neg_pool) < num_neg:
            skipped += 1
            continue
        neg_ids = rng.sample(neg_pool, num_neg)

        # Combine and shuffle
        candidates = [(d, relevant[d]) for d in rel_ids]
        candidates += [(d, 0) for d in neg_ids]
        rng.shuffle(candidates)

        # Format passages
        passages_lines = []
        relevance_scores = []
        for i, (did, score) in enumerate(candidates):
            passage = corpus[did][:MAX_PASSAGE_CHARS]
            passages_lines.append(f"{i+1}. {passage}")
            relevance_scores.append(score)

        passages_text = "\n".join(passages_lines)

        # Ideal ranking: sort by relevance descending, break ties by position
        ranked = sorted(range(num_candidates),
                        key=lambda x: (-relevance_scores[x], x))
        ranking_str = ", ".join(str(i + 1) for i in ranked)

        records.append({
            "text": TRAIN_TEMPLATE.format(
                n=num_candidates, query=query_text,
                passages=passages_text, ranking=ranking_str),
            "input": INFERENCE_TEMPLATE.format(
                n=num_candidates, query=query_text,
                passages=passages_text),
            "output": ranking_str,
            "relevance": relevance_scores,
            "id": qid,
        })

    if skipped:
        print(f"  Skipped {skipped} queries (missing data)")
    return records


# ==================== I/O ====================

def save_jsonl(records, filepath):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {filepath}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="NFCorpus reranking data formatting")
    parser.add_argument("--output_dir", default="data/nfcorpus")
    parser.add_argument("--num_candidates", type=int, default=NUM_CANDIDATES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)

    corpus, queries, qrels = load_nfcorpus()
    all_doc_ids = sorted(corpus.keys())

    # Map BEIR split names → output filenames
    split_map = {}
    for name in qrels:
        if "train" in name:
            split_map[name] = "train_sft.jsonl"
        elif "test" in name:
            split_map[name] = "test_sft.jsonl"
        elif "dev" in name or "validation" in name:
            split_map[name] = "val_sft.jsonl"

    has_val = any("val" in v for v in split_map.values())

    for split_name, filename in split_map.items():
        print(f"\nFormatting [{split_name}] → {filename}...")
        records = build_reranking_records(
            queries, corpus, qrels[split_name], all_doc_ids,
            num_candidates=args.num_candidates,
            seed=args.seed + abs(hash(split_name)) % 10000,
        )

        if not has_val and filename == "train_sft.jsonl":
            # No validation split available: carve 10% from train
            random.shuffle(records)
            val_size = max(1, len(records) // 10)
            val_records = records[:val_size]
            train_records = records[val_size:]
            save_jsonl(train_records, output_dir / "train_sft.jsonl")
            save_jsonl(val_records, output_dir / "val_sft.jsonl")
            has_val = True
        else:
            save_jsonl(records, output_dir / filename)

    # Summary
    print(f"\n{'='*50}")
    print(f"NFCorpus Reranking SFT data ready in {output_dir}/")


if __name__ == "__main__":
    main()
