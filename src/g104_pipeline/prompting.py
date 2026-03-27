from __future__ import annotations

from typing import Dict, List

OPTION_LABELS = ["A", "B", "C", "D", "E"]


def to_instruction_prompt(record: Dict) -> str:
    prompt = record["prompt"].strip()
    options: List[str] = record["options"]

    option_lines = []
    for idx, opt in enumerate(options):
        label = OPTION_LABELS[idx] if idx < len(OPTION_LABELS) else str(idx)
        option_lines.append(f"{label}) {opt.strip()}")

    return (
        "You are given a legal case prompt and candidate holdings.\n"
        "Choose the best option letter.\n\n"
        f"Prompt:\n{prompt}\n\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\n\nAnswer with the letter only."
    )
