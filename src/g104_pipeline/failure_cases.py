from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import load_config
from .io_utils import read_jsonl, write_jsonl


def _load_run_tables(run_dir: Path) -> pd.DataFrame:
    pred = pd.DataFrame(read_jsonl(run_dir / "predictions.jsonl"))
    faith = pd.DataFrame(read_jsonl(run_dir / "faithfulness_samples.jsonl"))

    if pred.empty:
        return pred

    faith = faith.rename(columns={"id": "sample_id"})
    merged = pred.merge(faith, left_on="id", right_on="sample_id", how="left")
    return merged


def build_failure_cases(config_path: str, max_cases: int = 12) -> str:
    cfg = load_config(config_path)
    output_root = Path(cfg["project"]["output_root"])
    seed = int(cfg.get("seeds", [cfg["seed"]])[0])

    pre_dir = output_root / "artifacts" / "pretrained" / str(seed)
    lora_dir = output_root / "artifacts" / "lora_sft" / str(seed)
    rnd_dir = output_root / "artifacts" / "random_label" / str(seed)

    pre = _load_run_tables(pre_dir).rename(columns={"prediction": "pred_pre", "is_correct": "ok_pre", "aopc": "aopc_pre", "confidence": "conf_pre"})
    lora = _load_run_tables(lora_dir).rename(columns={"prediction": "pred_lora", "is_correct": "ok_lora", "aopc": "aopc_lora", "confidence": "conf_lora"})
    rnd = _load_run_tables(rnd_dir).rename(columns={"prediction": "pred_rnd", "is_correct": "ok_rnd", "aopc": "aopc_rnd", "confidence": "conf_rnd"})

    if pre.empty or lora.empty:
        raise RuntimeError("Missing prediction artifacts for failure case extraction.")

    cols_keep = ["id", "label", "pred_pre", "ok_pre", "aopc_pre", "conf_pre"]
    merged = pre[cols_keep].merge(
        lora[["id", "pred_lora", "ok_lora", "aopc_lora", "conf_lora"]], on="id", how="inner"
    )

    if not rnd.empty:
        merged = merged.merge(rnd[["id", "pred_rnd", "ok_rnd", "aopc_rnd", "conf_rnd"]], on="id", how="left")

    cases: List[Dict] = []

    # Category A: lora improved accuracy but faithfulness did not improve.
    a = merged[(merged["ok_lora"] == True) & (merged["ok_pre"] == False)]
    a = a.sort_values(by=["conf_lora", "aopc_lora"], ascending=[False, True])
    for _, r in a.head(max_cases).iterrows():
        cases.append(
            {
                "id": r["id"],
                "category": "improved_accuracy_but_weak_faithfulness",
                "label": int(r["label"]),
                "pred_pre": int(r["pred_pre"]),
                "pred_lora": int(r["pred_lora"]),
                "aopc_pre": None if pd.isna(r.get("aopc_pre")) else float(r.get("aopc_pre")),
                "aopc_lora": None if pd.isna(r.get("aopc_lora")) else float(r.get("aopc_lora")),
                "confidence_pre": None if pd.isna(r.get("conf_pre")) else float(r.get("conf_pre")),
                "confidence_lora": None if pd.isna(r.get("conf_lora")) else float(r.get("conf_lora")),
            }
        )

    # Category B: lora regression cases.
    b = merged[(merged["ok_lora"] == False) & (merged["ok_pre"] == True)]
    b = b.sort_values(by=["conf_lora"], ascending=[False])
    for _, r in b.head(max_cases).iterrows():
        cases.append(
            {
                "id": r["id"],
                "category": "lora_regression",
                "label": int(r["label"]),
                "pred_pre": int(r["pred_pre"]),
                "pred_lora": int(r["pred_lora"]),
                "aopc_pre": None if pd.isna(r.get("aopc_pre")) else float(r.get("aopc_pre")),
                "aopc_lora": None if pd.isna(r.get("aopc_lora")) else float(r.get("aopc_lora")),
                "confidence_pre": None if pd.isna(r.get("conf_pre")) else float(r.get("conf_pre")),
                "confidence_lora": None if pd.isna(r.get("conf_lora")) else float(r.get("conf_lora")),
            }
        )

    # Category C: random-label accidentally high confidence (sanity control).
    if "conf_rnd" in merged.columns:
        c = merged.sort_values(by=["conf_rnd"], ascending=[False])
        for _, r in c.head(max_cases).iterrows():
            cases.append(
                {
                    "id": r["id"],
                    "category": "random_label_high_confidence_control",
                    "label": int(r["label"]),
                    "pred_rnd": int(r.get("pred_rnd", -1)),
                    "ok_rnd": None if pd.isna(r.get("ok_rnd")) else bool(r.get("ok_rnd")),
                    "confidence_rnd": None if pd.isna(r.get("conf_rnd")) else float(r.get("conf_rnd")),
                }
            )

    # Deduplicate and cap to 12 by default.
    seen = set()
    uniq: List[Dict] = []
    for c in cases:
        key = (c["id"], c["category"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
        if len(uniq) >= max_cases:
            break

    out_path = output_root / "reports" / "failure_cases.jsonl"
    write_jsonl(out_path, uniq)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 8-12 failure cases for qualitative analysis.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-cases", type=int, default=12)
    args = parser.parse_args()

    out = build_failure_cases(args.config, max_cases=args.max_cases)
    print(out)


if __name__ == "__main__":
    main()
