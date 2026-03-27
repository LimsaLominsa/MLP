"""Compute representation-shift metrics for medical tasks.

Reads signals.parquet from the BillSum-style layout:
  {output_root}/analysis/{split}/signals.parquet

Compares pretrained vs lora and pretrained vs random configs pairwise,
reusing the JS/entropy/CKA functions from rep_metrics.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .billsum_baseline import _load_config
from .io_utils import write_json
from .rep_metrics import _entropy_rows, _linear_cka, _normalize_rows, _safe_js


def compute_medical_rep_metrics(
    pretrained_config: str,
    target_config: str,
    label: str,
    split: str = "test",
    layers: list[int] | None = None,
) -> dict:
    """Compare signals from a pretrained config vs a target (lora/random) config."""
    cfg_base = _load_config(pretrained_config)
    cfg_tgt = _load_config(target_config)

    base_path = Path(cfg_base["project"]["output_root"]) / "analysis" / split / "signals.parquet"
    tgt_path = Path(cfg_tgt["project"]["output_root"]) / "analysis" / split / "signals.parquet"

    if not base_path.exists():
        raise FileNotFoundError(f"Missing pretrained signals: {base_path}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"Missing target signals: {tgt_path}")

    base = pd.read_parquet(base_path)
    tgt = pd.read_parquet(tgt_path)

    if layers is None:
        all_layers = sorted(base["layer"].unique())
        # Pick ~4 evenly spaced layers
        if len(all_layers) > 4:
            step = len(all_layers) // 4
            layers = [all_layers[i * step] for i in range(4)]
        else:
            layers = all_layers

    def _pivot(df, value_col, layer):
        s = df[df["layer"] == layer]
        p = s.pivot_table(index="sample_id", columns="head", values=value_col, aggfunc="mean")
        return p.sort_index().sort_index(axis=1).fillna(0.0)

    layer_rows = []
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

        layer_rows.append({
            "layer": int(layer),
            "attention_js": js_mean,
            "entropy_delta": ent_delta,
            "cka_activation": cka,
        })

    aggregate = {
        "label": label,
        "pretrained_config": pretrained_config,
        "target_config": target_config,
        "split": split,
        "num_layers": len(layer_rows),
        "attention_js_mean": float(np.mean([x["attention_js"] for x in layer_rows])) if layer_rows else None,
        "entropy_delta_mean": float(np.mean([x["entropy_delta"] for x in layer_rows])) if layer_rows else None,
        "cka_activation_mean": float(np.mean([x["cka_activation"] for x in layer_rows])) if layer_rows else None,
        "layers": layer_rows,
    }

    out_dir = Path(cfg_tgt["project"]["output_root"]) / "analysis" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rep_metrics.json"
    write_json(out_path, aggregate)
    print(f"Saved: {out_path}")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute medical task representation-shift metrics.")
    parser.add_argument("--pretrained-config", required=True, help="Config for pretrained baseline")
    parser.add_argument("--target-config", required=True, help="Config for lora or random-label condition")
    parser.add_argument("--label", required=True, help="Label for this comparison (e.g. pubmed_qwen_lora)")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    compute_medical_rep_metrics(
        pretrained_config=args.pretrained_config,
        target_config=args.target_config,
        label=args.label,
        split=args.split,
    )


if __name__ == "__main__":
    main()
