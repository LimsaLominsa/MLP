from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .io_utils import set_seed, softmax
from .prompting import OPTION_LABELS, to_instruction_prompt

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _require_hf():
    try:
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from torch.utils.data import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError as e:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "hf_lora backend requires torch/transformers/peft. Install requirements on cluster first."
        ) from e

    return {
        "torch": torch,
        "Dataset": Dataset,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "PeftModel": PeftModel,
    }


def _torch_dtype(torch_mod):
    if torch_mod.cuda.is_available():
        return torch_mod.bfloat16
    return torch_mod.float32


def _default_device(torch_mod):
    return "cuda" if torch_mod.cuda.is_available() else "cpu"


def _inference_model_kwargs(torch_mod) -> Dict[str, Any]:
    return {
        "torch_dtype": _torch_dtype(torch_mod),
        "trust_remote_code": True,
        # Interpretability stages need attentions/hidden states. Using eager avoids
        # the sdpa path that can drop attentions for Qwen-family models.
        "attn_implementation": "eager",
    }


def _answer_prompt(record: Dict[str, Any]) -> str:
    return to_instruction_prompt(record) + "\nAnswer:"


def _answer_text(label_idx: int) -> str:
    if label_idx < 0 or label_idx >= len(OPTION_LABELS):
        raise ValueError(f"Unsupported label index: {label_idx}")
    return f" {OPTION_LABELS[label_idx]}"


def _label_token_ids(tokenizer, n_options: int) -> List[int]:
    ids: List[int] = []
    for i in range(n_options):
        toks = tokenizer.encode(_answer_text(i), add_special_tokens=False)
        if len(toks) != 1:
            raise RuntimeError(
                f"Expected single-token answer label for option {i}, got token ids {toks}. "
                "Use a label representation that tokenizes to one token."
            )
        ids.append(int(toks[0]))
    return ids


def _tokenize_sft_instance(tokenizer, prompt: str, label_idx: int, max_len: int) -> Dict[str, List[int]]:
    ans_text = _answer_text(label_idx)

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ans_ids = tokenizer.encode(ans_text, add_special_tokens=False)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id
    if eos_id is None:
        eos_id = 0

    room = max_len - len(ans_ids) - 1
    if room < 8:
        room = max_len // 2
    if len(prompt_ids) > room:
        prompt_ids = prompt_ids[-room:]

    input_ids = prompt_ids + ans_ids + [eos_id]
    labels = ([-100] * len(prompt_ids)) + ans_ids + [eos_id]
    attn = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn,
    }


class _SFTDataset:
    def __init__(
        self,
        hf_mods,
        rows: List[Dict[str, Any]],
        tokenizer,
        max_len: int,
        randomize_labels: bool,
        seed: int,
    ) -> None:
        DatasetBase = hf_mods["Dataset"]

        class _Impl(DatasetBase):
            def __init__(self, items: List[Dict[str, List[int]]]) -> None:
                self.items = items

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, idx: int) -> Dict[str, List[int]]:
                return self.items[idx]

        rng = random.Random(seed)
        items: List[Dict[str, List[int]]] = []
        for r in rows:
            n_opts = len(r["options"])
            if n_opts < 2:
                continue
            lbl = int(r.get("label", -1))
            if lbl < 0:
                continue
            if randomize_labels:
                lbl = rng.randrange(0, n_opts)

            prompt = _answer_prompt(r)
            items.append(_tokenize_sft_instance(tokenizer, prompt, lbl, max_len=max_len))

        self.dataset = _Impl(items)


class _PadCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        import torch

        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        labels = []
        for ex in batch:
            pad_n = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.pad_id] * pad_n)
            attention_mask.append(ex["attention_mask"] + [0] * pad_n)
            labels.append(ex["labels"] + [-100] * pad_n)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train_hf_adapter(
    cfg: Dict[str, Any],
    train_rows: List[Dict[str, Any]],
    valid_rows: List[Dict[str, Any]],
    run_dir: Path,
    experiment: str,
    seed: int,
) -> Dict[str, Any]:
    hf = _require_hf()
    model_name = cfg["model_name"]
    max_len = int(cfg["train_args"].get("max_seq_length", 1024))
    set_seed(seed)

    if experiment == "pretrained":
        torch = hf["torch"]
        return {
            "hf_backend": True,
            "model_name": model_name,
            "experiment": experiment,
            "seed": seed,
            "adapter_dir": None,
            "max_seq_length": max_len,
            "device": _default_device(torch),
            "torch_dtype": str(_torch_dtype(torch)),
            "train_rows_used": 0,
            "valid_rows": len(valid_rows),
        }

    torch = hf["torch"]
    AutoTokenizer = hf["AutoTokenizer"]
    AutoModelForCausalLM = hf["AutoModelForCausalLM"]
    TrainingArguments = hf["TrainingArguments"]
    Trainer = hf["Trainer"]
    LoraConfig = hf["LoraConfig"]
    TaskType = hf["TaskType"]
    get_peft_model = hf["get_peft_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = _torch_dtype(torch)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    lcfg = cfg["lora_config"]
    peft_cfg = LoraConfig(
        r=int(lcfg.get("r", 16)),
        lora_alpha=int(lcfg.get("lora_alpha", 32)),
        lora_dropout=float(lcfg.get("lora_dropout", 0.05)),
        target_modules=list(lcfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    randomize = experiment == "random_label"
    train_ds = _SFTDataset(hf, train_rows, tokenizer, max_len=max_len, randomize_labels=randomize, seed=seed).dataset

    if valid_rows:
        eval_ds = _SFTDataset(hf, valid_rows, tokenizer, max_len=max_len, randomize_labels=False, seed=seed).dataset
    else:
        eval_ds = None

    targs = cfg["train_args"]
    local_out = run_dir / "trainer_ckpt"
    training_args = TrainingArguments(
        output_dir=str(local_out),
        num_train_epochs=float(targs.get("num_train_epochs", 1)),
        learning_rate=float(targs.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(targs.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(targs.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(targs.get("gradient_accumulation_steps", 8)),
        warmup_ratio=float(targs.get("warmup_ratio", 0.03)),
        logging_steps=int(targs.get("logging_steps", 20)),
        save_strategy=str(targs.get("save_strategy", "epoch")),
        evaluation_strategy=str(targs.get("eval_strategy", "no")),
        report_to=[],
        seed=seed,
        data_seed=seed,
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_PadCollator(tokenizer),
    )
    trainer.train()

    adapter_dir = run_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    train_rows_used = len(train_ds)

    meta = {
        "hf_backend": True,
        "model_name": model_name,
        "experiment": experiment,
        "seed": seed,
        "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
        "max_seq_length": max_len,
        "device": _default_device(torch),
        "torch_dtype": str(dtype),
        "train_rows_used": train_rows_used,
        "valid_rows": len(valid_rows),
    }
    return meta


@dataclass
class HFInferenceRunner:
    cfg: Dict[str, Any]
    run_dir: Path
    experiment: str

    def __post_init__(self) -> None:
        hf = _require_hf()
        self.hf = hf
        self.torch = hf["torch"]
        self.max_len = int(self.cfg["train_args"].get("max_seq_length", 1024))

        model_name = self.cfg["model_name"]
        AutoTokenizer = hf["AutoTokenizer"]
        AutoModelForCausalLM = hf["AutoModelForCausalLM"]
        PeftModel = hf["PeftModel"]

        adapter_dir = self.run_dir / "adapter"

        if self.experiment in {"lora_sft", "random_label"} and adapter_dir.exists():
            self.tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
            base = AutoModelForCausalLM.from_pretrained(
                model_name,
                **_inference_model_kwargs(self.torch),
            )
            self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **_inference_model_kwargs(self.torch),
            )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = _default_device(self.torch)
        self.model.to(self.device)
        self.model.eval()

    def _encode_prompt(self, prompt: str):
        tok = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
        )
        return {k: v.to(self.device) for k, v in tok.items()}

    def score_record(self, record: Dict[str, Any]) -> np.ndarray:
        prompt = _answer_prompt(record)
        inputs = self._encode_prompt(prompt)

        with self.torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits[0, -1, :]

        idx_ids = _label_token_ids(self.tokenizer, len(record["options"]))
        scores = [float(logits[i].item()) for i in idx_ids]
        return np.array(scores, dtype=float)

    def predict(self, record: Dict[str, Any]) -> Tuple[int, np.ndarray]:
        scores = self.score_record(record)
        pred = int(np.argmax(scores))
        probs = softmax(scores)
        return pred, probs

    def token_importance(self, record: Dict[str, Any]) -> Dict[str, float]:
        prompt = _answer_prompt(record)
        inputs = self._encode_prompt(prompt)

        with self.torch.no_grad():
            out = self.model(**inputs, output_attentions=True)

        att = out.attentions[-1][0].float()  # [heads, q, k]
        key_imp = att.mean(dim=0).mean(dim=0).detach().cpu().numpy()  # [k]

        ids = inputs["input_ids"][0].detach().cpu().tolist()
        toks = self.tokenizer.convert_ids_to_tokens(ids)

        d: Dict[str, float] = {}
        for t, v in zip(toks, key_imp):
            clean = t.replace("Ġ", "").replace("▁", "").strip().lower()
            if not clean:
                continue
            d[clean] = d.get(clean, 0.0) + float(max(v, 0.0))

        if not d:
            words = _WORD_RE.findall(prompt.lower())
            for w in words:
                d[w] = d.get(w, 0.0) + 1.0

        return d

    def deletion_insertion_curve(self, record: Dict[str, Any], topk_ratio: float, steps: int) -> Dict[str, Any]:
        base_pred, base_scores = self.predict(record)
        base_conf = float(base_scores[base_pred])

        prompt_words = _WORD_RE.findall(record["prompt"].lower())
        if not prompt_words:
            return {"deletion": [base_conf] * steps, "insertion": [base_conf] * steps, "aopc": 0.0}

        imp = self.token_importance(record)
        ranked = sorted(prompt_words, key=lambda w: imp.get(w, 0.0), reverse=True)

        topk = max(1, int(len(ranked) * topk_ratio))
        selected = ranked[:topk]

        deletion_curve: List[float] = []
        insertion_curve: List[float] = []

        for s in range(1, steps + 1):
            frac = s / steps
            cut = max(1, int(len(selected) * frac))
            removed = set(selected[:cut])

            del_prompt = " ".join([w for w in prompt_words if w not in removed])
            ins_prompt = " ".join(selected[:cut])

            rec_del = dict(record)
            rec_ins = dict(record)
            rec_del["prompt"] = del_prompt
            rec_ins["prompt"] = ins_prompt

            _, del_scores = self.predict(rec_del)
            _, ins_scores = self.predict(rec_ins)

            deletion_curve.append(float(np.max(del_scores)))
            insertion_curve.append(float(np.max(ins_scores)))

        aopc = float(np.mean([base_conf - x for x in deletion_curve]))
        return {
            "deletion": deletion_curve,
            "insertion": insertion_curve,
            "aopc": aopc,
        }

    def extract_signal_rows(self, record: Dict[str, Any], max_layer: int) -> List[Dict[str, Any]]:
        prompt = _answer_prompt(record)
        inputs = self._encode_prompt(prompt)

        with self.torch.no_grad():
            out = self.model(**inputs, output_attentions=True, output_hidden_states=True)

        attentions = out.attentions
        hidden_states = out.hidden_states

        n_layers = len(attentions)
        n_heads = attentions[0].shape[1]
        top_layer = min(max_layer, n_layers - 1)

        rows: List[Dict[str, Any]] = []
        for layer in range(top_layer + 1):
            att_layer = attentions[layer][0].float()  # [heads, q, k]
            hs = hidden_states[layer + 1][0].float()  # [seq, hidden]
            act_scalar = float(self.torch.norm(hs, dim=-1).mean().item())

            for head in range(n_heads):
                att_scalar = float(att_layer[head].mean().item())
                attr_scalar = float(abs(att_scalar) * act_scalar)
                rows.append(
                    {
                        "sample_id": record["id"],
                        "layer": layer,
                        "head": head,
                        "attention": att_scalar,
                        "activation": act_scalar,
                        "attribution": attr_scalar,
                    }
                )

        return rows


def evaluate_hf(
    cfg: Dict[str, Any],
    eval_rows: List[Dict[str, Any]],
    run_dir: Path,
    experiment: str,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    runner = HFInferenceRunner(cfg=cfg, run_dir=run_dir, experiment=experiment)

    correct = 0
    total_labeled = 0
    pred_rows: List[Dict[str, Any]] = []

    for rec in eval_rows:
        pred, probs = runner.predict(rec)
        label = int(rec.get("label", -1))

        if label >= 0:
            total_labeled += 1
            correct += int(pred == label)

        pred_rows.append(
            {
                "id": rec["id"],
                "split": rec.get("split", "test"),
                "label": label,
                "prediction": pred,
                "option_scores": [float(x) for x in probs.tolist()],
                "confidence": float(np.max(probs)),
                "is_correct": bool(pred == label) if label >= 0 else None,
            }
        )

    acc = float(correct / total_labeled) if total_labeled > 0 else None
    metrics = {
        "experiment": experiment,
        "seed": seed,
        "num_samples": len(eval_rows),
        "num_labeled": total_labeled,
        "accuracy": acc,
    }

    return metrics, pred_rows


def extract_hf_signals(
    cfg: Dict[str, Any],
    eval_rows: List[Dict[str, Any]],
    run_dir: Path,
    experiment: str,
    seed: int,
) -> List[Dict[str, Any]]:
    runner = HFInferenceRunner(cfg=cfg, run_dir=run_dir, experiment=experiment)
    max_layer = int(max(cfg.get("analysis", {}).get("cka_layers", [0, 4, 8, 12])))

    rows: List[Dict[str, Any]] = []
    for rec in eval_rows:
        part = runner.extract_signal_rows(rec, max_layer=max_layer)
        for r in part:
            r["experiment"] = experiment
            r["seed"] = seed
        rows.extend(part)

    return rows


def faithfulness_hf(
    cfg: Dict[str, Any],
    eval_rows: List[Dict[str, Any]],
    run_dir: Path,
    experiment: str,
    seed: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metric_cfg = cfg.get("metrics", {})
    topk_ratio = float(metric_cfg.get("faithfulness_topk_ratio", 0.2))
    deletion_steps = int(metric_cfg.get("deletion_steps", 10))

    runner = HFInferenceRunner(cfg=cfg, run_dir=run_dir, experiment=experiment)

    sample_rows: List[Dict[str, Any]] = []
    del_curves: List[List[float]] = []
    ins_curves: List[List[float]] = []
    aopcs: List[float] = []

    for rec in eval_rows:
        curve = runner.deletion_insertion_curve(rec, topk_ratio=topk_ratio, steps=deletion_steps)
        del_curves.append(curve["deletion"])
        ins_curves.append(curve["insertion"])
        aopcs.append(float(curve["aopc"]))

        sample_rows.append(
            {
                "id": rec["id"],
                "experiment": experiment,
                "seed": seed,
                "aopc": float(curve["aopc"]),
                "deletion_curve": curve["deletion"],
                "insertion_curve": curve["insertion"],
            }
        )

    mean_del = np.mean(np.array(del_curves), axis=0).tolist() if del_curves else []
    mean_ins = np.mean(np.array(ins_curves), axis=0).tolist() if ins_curves else []

    summary = {
        "experiment": experiment,
        "seed": seed,
        "num_samples": len(eval_rows),
        "mean_aopc": float(np.mean(aopcs)) if aopcs else None,
        "std_aopc": float(np.std(aopcs)) if aopcs else None,
        "mean_deletion_curve": [float(x) for x in mean_del],
        "mean_insertion_curve": [float(x) for x in mean_ins],
        "topk_ratio": topk_ratio,
        "steps": deletion_steps,
    }

    return summary, sample_rows
