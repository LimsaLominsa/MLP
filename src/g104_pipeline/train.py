from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .config import load_config
from .io_utils import ensure_dir, read_jsonl, set_seed, write_json
from .hf_backend import train_hf_adapter
from .modeling import PairwiseChoiceModel


def train_one(config_path: str, experiment: str, seed: int) -> str:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    output_root = cfg["project"]["output_root"]

    train_rows = read_jsonl(data_cfg["train_file"])
    valid_rows = read_jsonl(data_cfg["valid_file"])

    out_dir = Path(output_root) / "artifacts" / experiment / str(seed)
    ensure_dir(out_dir)

    backend = cfg.get("backend", "mock")
    if backend == "hf_lora":
        resolved_backend = "hf_lora"
    else:
        resolved_backend = "mock"

    if resolved_backend == "hf_lora":
        hf_meta = train_hf_adapter(
            cfg=cfg,
            train_rows=train_rows,
            valid_rows=valid_rows,
            run_dir=out_dir,
            experiment=experiment,
            seed=seed,
        )
        write_json(out_dir / "hf_state.json", hf_meta)
    else:
        set_seed(seed)
        model = PairwiseChoiceModel(experiment=experiment, seed=seed)

        if experiment == "pretrained":
            # Intentionally no fitting: serves as frozen baseline.
            pass
        elif experiment == "lora_sft":
            model.fit(train_rows, randomize_labels=False)
        elif experiment == "random_label":
            model.fit(train_rows, randomize_labels=True)
        else:
            raise ValueError(f"Unsupported experiment type: {experiment}")

        model_path = out_dir / "model.pkl"
        model.save(str(model_path))

    train_meta: Dict[str, Any] = {
        "experiment": experiment,
        "seed": seed,
        "backend": resolved_backend,
        "model_name": cfg["model_name"],
        "lora_config": cfg["lora_config"],
        "train_args": cfg["train_args"],
        "train_size": len(train_rows),
        "valid_size": len(valid_rows),
    }
    write_json(out_dir / "train_meta.json", train_meta)
    return str(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one experiment run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True, choices=["pretrained", "lora_sft", "random_label"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    out_dir = train_one(args.config, args.experiment, args.seed)
    print(out_dir)


if __name__ == "__main__":
    main()
