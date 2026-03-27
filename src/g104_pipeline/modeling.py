from __future__ import annotations

import pickle
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .io_utils import set_seed

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class PairwiseChoiceModel:
    experiment: str
    seed: int

    def __post_init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.clf: LogisticRegression | None = None
        self.is_trained = False

    def _pair_text(self, prompt: str, option: str) -> str:
        return f"[PROMPT] {prompt} [OPTION] {option}"

    def fit(self, records: List[Dict], randomize_labels: bool = False) -> None:
        set_seed(self.seed)

        X: List[str] = []
        y: List[int] = []

        for rec in records:
            options = rec["options"]
            true_label = int(rec["label"])
            if randomize_labels:
                true_label = np.random.randint(0, len(options))

            for i, opt in enumerate(options):
                X.append(self._pair_text(rec["prompt"], opt))
                y.append(1 if i == true_label else 0)

        self.vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
        Xv = self.vectorizer.fit_transform(X)

        self.clf = LogisticRegression(max_iter=600, random_state=self.seed)
        self.clf.fit(Xv, y)
        self.is_trained = True

    def _heuristic_scores(self, record: Dict) -> np.ndarray:
        key = f"{self.seed}-{record.get('id', 'na')}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        seed_int = int(h[:8], 16)
        rng = np.random.default_rng(seed_int)
        arr = rng.random(len(record["options"]))
        return arr

    def predict_option_scores(self, record: Dict) -> np.ndarray:
        if self.experiment == "pretrained" or not self.is_trained:
            return self._heuristic_scores(record)

        assert self.vectorizer is not None and self.clf is not None
        pair_texts = [self._pair_text(record["prompt"], opt) for opt in record["options"]]
        X = self.vectorizer.transform(pair_texts)
        probs = self.clf.predict_proba(X)[:, 1]
        return probs

    def predict(self, record: Dict) -> Tuple[int, np.ndarray]:
        scores = self.predict_option_scores(record)
        pred = int(np.argmax(scores))
        return pred, scores

    def token_importance(self, record: Dict) -> Dict[str, float]:
        text = f"{record['prompt']} " + " ".join(record["options"])
        tokens = _tokenize(text)

        if not tokens:
            return {"<empty>": 1.0}

        # Lightweight attribution proxy: token frequency weighted by option confidence spread.
        _, scores = self.predict(record)
        confidence_spread = float(np.max(scores) - np.min(scores) + 1e-6)

        freq: Dict[str, float] = {}
        for t in tokens:
            freq[t] = freq.get(t, 0.0) + 1.0

        denom = sum(freq.values())
        return {k: (v / denom) * confidence_spread for k, v in freq.items()}

    def deletion_insertion_curve(self, record: Dict, topk_ratio: float, steps: int) -> Dict[str, List[float]]:
        original_pred, original_scores = self.predict(record)
        original_conf = float(original_scores[original_pred])

        prompt_tokens = _tokenize(record["prompt"])
        if not prompt_tokens:
            return {
                "deletion": [original_conf] * steps,
                "insertion": [original_conf] * steps,
                "aopc": 0.0,
            }

        imp = self.token_importance(record)
        ranked = sorted(prompt_tokens, key=lambda t: imp.get(t, 0.0), reverse=True)
        topk = max(1, int(len(ranked) * topk_ratio))
        selected = ranked[:topk]

        deletion_curve: List[float] = []
        insertion_curve: List[float] = []

        for s in range(1, steps + 1):
            frac = s / steps
            cut = max(1, int(len(selected) * frac))
            removed = set(selected[:cut])

            del_prompt = " ".join([t for t in prompt_tokens if t not in removed])
            ins_prompt = " ".join(selected[:cut])

            rec_del = dict(record)
            rec_ins = dict(record)
            rec_del["prompt"] = del_prompt
            rec_ins["prompt"] = ins_prompt

            _, del_scores = self.predict(rec_del)
            _, ins_scores = self.predict(rec_ins)

            deletion_curve.append(float(np.max(del_scores)))
            insertion_curve.append(float(np.max(ins_scores)))

        aopc = float(np.mean([original_conf - x for x in deletion_curve]))
        return {
            "deletion": deletion_curve,
            "insertion": insertion_curve,
            "aopc": aopc,
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PairwiseChoiceModel":
        with Path(path).open("rb") as f:
            return pickle.load(f)
