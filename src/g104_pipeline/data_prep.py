from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .config import load_config
from .io_utils import ensure_dir, read_jsonl, set_seed, write_json, write_jsonl


def _extract_prompt(example: Dict[str, Any]) -> str:
    for k in ["citing_prompt", "prompt", "context", "question", "text"]:
        if k in example and example[k] is not None:
            return str(example[k])
    return ""


def _extract_options(example: Dict[str, Any]) -> List[str]:
    if "options" in example and isinstance(example["options"], list):
        return [str(x) for x in example["options"]]
    if "endings" in example and isinstance(example["endings"], list):
        return [str(x) for x in example["endings"]]
    if "choices" in example and isinstance(example["choices"], list):
        return [str(x) for x in example["choices"]]

    holding_keys = sorted([k for k in example.keys() if k.startswith("holding_")])
    if holding_keys:
        return [str(example[k]) for k in holding_keys]

    return []


def _extract_label(example: Dict[str, Any]) -> int:
    for k in ["label", "answer", "gold", "target"]:
        if k in example and example[k] is not None:
            try:
                return int(example[k])
            except Exception:
                pass
    return -1


def _to_unified(split: str, examples: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples):
        prompt = _extract_prompt(ex).strip()
        options = _extract_options(ex)
        label = _extract_label(ex)

        if not prompt or len(options) < 2:
            continue

        rec = {
            "id": f"{split}-{i}",
            "prompt": prompt,
            "options": options,
            "label": label,
            "split": split,
        }
        rows.append(rec)
    return rows


def _build_synthetic(output_cfg: Dict[str, Any], seed: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    set_seed(seed)

    legal_templates = [
        "The court considered whether evidence was admissible under the statute.",
        "The panel reviewed jurisdictional objections raised by the appellant.",
        "The defendant challenged the sufficiency of notice in the proceedings.",
        "The dispute centers on contract interpretation and enforceability.",
        "The judgment analyzes precedent for duty of care in negligence.",
    ]

    option_pool = [
        "The holding affirms admissibility based on statutory compliance.",
        "The holding reverses due to lack of jurisdiction.",
        "The holding vacates because notice was insufficient.",
        "The holding remands for contract damages calculation.",
        "The holding clarifies negligence duty under precedent.",
    ]

    def make_rec(split: str, i: int) -> Dict[str, Any]:
        label = random.randint(0, 4)
        prompt = legal_templates[label] + " " + f"Case note {split}-{i}."
        options = option_pool.copy()
        random.shuffle(options)
        # remap label after shuffle
        correct_text = option_pool[label]
        new_label = options.index(correct_text)
        return {
            "id": f"{split}-{i}",
            "prompt": prompt,
            "options": options,
            "label": new_label,
            "split": split,
        }

    train = [make_rec("train", i) for i in range(180)]
    valid = [make_rec("validation", i) for i in range(60)]
    test = [make_rec("test", i) for i in range(60)]
    return train, valid, test


def _load_casehold(dataset_name: str) -> Dict[str, Any]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "datasets package is required for CaseHOLD download. Install requirements or use synthetic mode."
        ) from e

    attempts = [
        (dataset_name, None),
        ("casehold/casehold", None),
        ("lexlms/lex_glue", "case_hold"),
        ("lex_glue", "case_hold"),
    ]

    errors: List[str] = []
    for name, subset in attempts:
        try:
            if subset is None:
                ds = load_dataset(name)
            else:
                ds = load_dataset(name, subset)
            return ds
        except Exception as e:  # pragma: no cover - environment dependent
            errors.append(f"{name}/{subset}: {e}")

    raise RuntimeError("Unable to download CaseHOLD dataset. Attempts:\n" + "\n".join(errors))


def _load_local_preprocessed_casehold(data_cfg: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], List[Dict]] | None:
    source_map = {
        "preprocessed_train_file": "train",
        "preprocessed_valid_file": "validation",
        "preprocessed_test_file": "test",
    }

    specified = {key: data_cfg.get(key) for key in source_map if data_cfg.get(key)}
    if not specified:
        return None

    missing = [key for key, path in specified.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "Configured local preprocessed CaseHOLD files are missing: "
            + ", ".join(f"{key}={data_cfg[key]}" for key in missing)
        )

    if len(specified) != len(source_map):
        missing_keys = [key for key in source_map if key not in specified]
        raise ValueError(
            "Either configure all local preprocessed CaseHOLD source files or none of them. "
            f"Missing keys: {missing_keys}"
        )

    train_rows = _to_unified("train", read_jsonl(data_cfg["preprocessed_train_file"]))
    valid_rows = _to_unified("validation", read_jsonl(data_cfg["preprocessed_valid_file"]))
    test_rows = _to_unified("test", read_jsonl(data_cfg["preprocessed_test_file"]))
    return train_rows, valid_rows, test_rows


def prepare_data(config_path: str) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    seed = int(cfg["seed"])

    ensure_dir(data_cfg["processed_dir"])
    ensure_dir(data_cfg["raw_dir"])

    dataset_name = str(data_cfg.get("dataset_name", "casehold/casehold"))
    local_preprocessed = _load_local_preprocessed_casehold(data_cfg)

    if local_preprocessed is not None:
        train_rows, valid_rows, test_rows = local_preprocessed
        dataset_name = "local_preprocessed_casehold"
    elif dataset_name == "synthetic":
        train_rows, valid_rows, test_rows = _build_synthetic(data_cfg, seed)
    else:
        ds = _load_casehold(dataset_name)

        train_rows = _to_unified("train", ds["train"])
        if "validation" in ds:
            valid_rows = _to_unified("validation", ds["validation"])
        else:
            valid_rows = train_rows[: max(1, min(512, len(train_rows) // 10))]

        if "test" in ds:
            raw_test = _to_unified("test", ds["test"])
            has_labels = any(r["label"] >= 0 for r in raw_test)
            test_rows = raw_test if has_labels else valid_rows.copy()
        else:
            test_rows = valid_rows.copy()

    write_jsonl(data_cfg["train_file"], train_rows)
    write_jsonl(data_cfg["valid_file"], valid_rows)
    write_jsonl(data_cfg["test_file"], test_rows)

    # fixed eval subset for all interpretability analysis
    subset_size = int(data_cfg.get("fixed_eval_subset_size", 256))
    source = test_rows if len(test_rows) > 0 else valid_rows
    set_seed(seed)
    chosen = source.copy()
    random.shuffle(chosen)
    fixed_eval = chosen[: min(subset_size, len(chosen))]
    write_jsonl(data_cfg["fixed_eval_subset_file"], fixed_eval)

    summary = {
        "train_size": len(train_rows),
        "valid_size": len(valid_rows),
        "test_size": len(test_rows),
        "fixed_eval_size": len(fixed_eval),
        "dataset_name": dataset_name,
    }
    write_json(Path(data_cfg["processed_dir"]) / "data_summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CaseHOLD dataset into unified JSONL schema.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    prepare_data(args.config)


if __name__ == "__main__":
    main()
