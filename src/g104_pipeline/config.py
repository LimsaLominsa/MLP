from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _resolve_path(config_dir: Path, value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return value

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_top = ["model_name", "lora_config", "seed", "train_args", "data", "project"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    config_dir = cfg_path.parent

    project_cfg = cfg.get("project", {})
    if "output_root" in project_cfg:
        project_cfg["output_root"] = _resolve_path(config_dir, project_cfg["output_root"])

    data_cfg = cfg.get("data", {})
    for key, value in list(data_cfg.items()):
        if key.endswith("_dir") or key.endswith("_file"):
            data_cfg[key] = _resolve_path(config_dir, value)

    return cfg
