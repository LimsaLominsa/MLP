from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .billsum_baseline import _load_config
from .io_utils import write_json
from .rep_metrics import _entropy_rows, _linear_cka, _normalize_rows, _pivot, _safe_js


def compute_billsum_rep_metrics(
    baseline_config: str,
    target_config: str,
    split: str = "test_us",
) -> str:
    base_cfg = _load_config(baseline_config)
    tgt_cfg = _load_config(target_config)

    base_path = Path(base_cfg["project"]["output_root"]) / "analysis" / split / "signals.parquet"
    tgt_path = Path(tgt_cfg["project"]["output_root"]) / "analysis" / split / "signals.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing baseline signals: {base_path}")
    if not tgt_path.exists():
        raise FileNotFoundError(f"Missing target signals: {tgt_path}")

    base = pd.read_parquet(base_path)
    tgt = pd.read_parquet(tgt_path)
    layers = sorted(set(base["layer"].tolist()).intersection(set(tgt["layer"].tolist())))

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
        js_mean = float(sum(js_vals) / len(js_vals))

        ent_base = _entropy_rows(b_arr)
        ent_tgt = _entropy_rows(t_arr)
        ent_delta = float((ent_tgt - ent_base).mean())

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
        "baseline_model_name": base_cfg["model_name"],
        "target_model_name": tgt_cfg["model_name"],
        "target_adapter_dir": tgt_cfg.get("adapter_dir"),
        "split": split,
        "num_layers": len(layer_rows),
        "attention_js_mean": float(sum(x["attention_js"] for x in layer_rows) / len(layer_rows)) if layer_rows else None,
        "entropy_delta_mean": float(sum(x["entropy_delta"] for x in layer_rows) / len(layer_rows)) if layer_rows else None,
        "cka_activation_mean": float(sum(x["cka_activation"] for x in layer_rows) / len(layer_rows)) if layer_rows else None,
        "layers": layer_rows,
    }

    out_path = Path(tgt_cfg["project"]["output_root"]) / "analysis" / split / "rep_metrics.json"
    write_json(out_path, aggregate)
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BillSum representation-shift metrics.")
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--split", default="test_us", choices=["valid", "test_us", "test_ca"])
    args = parser.parse_args()

    out = compute_billsum_rep_metrics(
        baseline_config=args.baseline_config,
        target_config=args.target_config,
        split=args.split,
    )
    print(out)


if __name__ == "__main__":
    main()
