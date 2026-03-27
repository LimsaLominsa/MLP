from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import load_config
from .io_utils import write_json


def _pivot(df: pd.DataFrame, value_col: str, layer: int) -> pd.DataFrame:
    s = df[df["layer"] == layer]
    p = s.pivot_table(index="sample_id", columns="head", values=value_col, aggfunc="mean")
    p = p.sort_index().sort_index(axis=1).fillna(0.0)
    return p


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    denom = arr.sum(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return arr / denom


def _entropy_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, 1e-12, 1.0)
    return -(arr * np.log(arr)).sum(axis=1)


def _safe_js(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    js = 0.5 * (kl_pm + kl_qm)
    js = max(0.0, float(js))
    return float(np.sqrt(js))


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    xty = x.T @ y
    xxt = x.T @ x
    yyt = y.T @ y

    num = np.linalg.norm(xty, ord="fro") ** 2
    den = np.linalg.norm(xxt, ord="fro") * np.linalg.norm(yyt, ord="fro") + 1e-12
    return float(num / den)


def compute_rep_metrics_for_seed(config_path: str, seed: int) -> List[Dict]:
    cfg = load_config(config_path)
    output_root = Path(cfg["project"]["output_root"])
    layers = cfg.get("analysis", {}).get("cka_layers", [0, 4, 8, 12])

    base_path = output_root / "artifacts" / "pretrained" / str(seed) / "signals.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing pretrained signals: {base_path}")

    base = pd.read_parquet(base_path)

    results: List[Dict] = []

    for exp in ["lora_sft", "random_label"]:
        target_path = output_root / "artifacts" / exp / str(seed) / "signals.parquet"
        if not target_path.exists():
            continue

        tgt = pd.read_parquet(target_path)
        layer_rows: List[Dict] = []

        for layer in layers:
            b_att = _pivot(base, "attention", layer)
            t_att = _pivot(tgt, "attention", layer)
            common_idx = b_att.index.intersection(t_att.index)
            if len(common_idx) == 0:
                continue

            b_att = b_att.loc[common_idx]
            t_att = t_att.loc[common_idx]
            common_cols = b_att.columns.intersection(t_att.columns)
            b_arr = _normalize_rows(b_att[common_cols].to_numpy())
            t_arr = _normalize_rows(t_att[common_cols].to_numpy())

            js_vals = [_safe_js(b_arr[i], t_arr[i]) for i in range(b_arr.shape[0])]
            js_mean = float(np.mean(js_vals))

            ent_base = _entropy_rows(b_arr)
            ent_tgt = _entropy_rows(t_arr)
            ent_delta = float(np.mean(ent_tgt - ent_base))

            b_act = _pivot(base, "activation", layer)
            t_act = _pivot(tgt, "activation", layer)
            common_idx2 = b_act.index.intersection(t_act.index)
            common_cols2 = b_act.columns.intersection(t_act.columns)
            cka = _linear_cka(
                b_act.loc[common_idx2, common_cols2].to_numpy(),
                t_act.loc[common_idx2, common_cols2].to_numpy(),
            )

            layer_rows.append(
                {
                    "layer": int(layer),
                    "attention_js": js_mean,
                    "entropy_delta": ent_delta,
                    "cka_activation": cka,
                }
            )

        aggregate = {
            "experiment": exp,
            "seed": seed,
            "num_layers": len(layer_rows),
            "attention_js_mean": float(np.mean([x["attention_js"] for x in layer_rows])) if layer_rows else None,
            "entropy_delta_mean": float(np.mean([x["entropy_delta"] for x in layer_rows])) if layer_rows else None,
            "cka_activation_mean": float(np.mean([x["cka_activation"] for x in layer_rows])) if layer_rows else None,
            "layers": layer_rows,
        }

        out_path = output_root / "artifacts" / exp / str(seed) / "rep_metrics.json"
        write_json(out_path, aggregate)
        results.append(aggregate)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute representation shift metrics (JS/Entropy/CKA).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    rows = compute_rep_metrics_for_seed(args.config, args.seed)
    print(f"computed={len(rows)}")


if __name__ == "__main__":
    main()
