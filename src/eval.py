"""Response-only evaluation: corpus-level NLL + perplexity (ADR 0004).

Computes the mean cross-entropy over *response* tokens only (prompt masked,
exactly as in training) on a held-out Dolly slice the model never trained on.
This is the apples-to-apples metric Run 002 lacked: two adapters (or the base
model) are comparable because the masking — and therefore the loss denominator
— is identical (see docs/math/03 §2).

Run via the Makefile:  `make eval ADAPTER=outputs/qwen-lora-adapter`
                       (`make eval ADAPTER=base` evaluates the un-adapted model)
"""

import math
import sys

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import build_example


def weighted_mean(values, weights):
    """Corpus mean: sum(vᵢ·wᵢ) / sum(wᵢ).

    Used to combine per-example mean-NLLs (each over wᵢ scored tokens) into one
    corpus-level NLL. Returns 0.0 when total weight is zero — never 0/0 NaN.
    """
    total = sum(weights)
    if total == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(cfg: dict, adapter_path: str | None) -> dict:
    torch.set_num_threads(cfg.get("num_threads", 12))

    model_id = cfg["model_id"]
    tok = AutoTokenizer.from_pretrained(adapter_path or model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    start = cfg.get("eval_start", cfg["n_train_examples"])
    n = cfg.get("n_eval_examples", 100)
    ds = load_dataset(cfg["dataset"], split=f"train[{start}:{start + n}]")

    losses, token_counts = [], []
    for ex in ds:
        e = build_example(
            tok, ex["instruction"], ex.get("context"), ex["response"],
            max_len=cfg["max_len"], mask_prompt=True,
        )
        if e is None:
            continue
        # Scored tokens = non-masked labels after the model's internal left-shift
        # (logits[:-1] vs labels[1:]); this is exactly HF's loss denominator.
        n_scored = sum(1 for label in e["labels"][1:] if label != -100)
        if n_scored == 0:
            continue
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([e["input_ids"]]),
                attention_mask=torch.tensor([e["attention_mask"]]),
                labels=torch.tensor([e["labels"]]),
            )
        losses.append(out.loss.item())
        token_counts.append(n_scored)

    nll = weighted_mean(losses, token_counts)
    return {
        "adapter": adapter_path or "base",
        "examples": len(losses),
        "response_tokens": sum(token_counts),
        "response_nll": nll,
        "perplexity": math.exp(nll) if nll else float("nan"),
    }


def main(config_path: str, adapter_path: str | None) -> None:
    cfg = load_config(config_path)
    if adapter_path in (None, "base"):
        adapter_path = None
    r = evaluate(cfg, adapter_path)
    print(
        f"[{r['adapter']}] response-NLL={r['response_nll']:.4f}  "
        f"perplexity={r['perplexity']:.2f}  "
        f"({r['examples']} examples, {r['response_tokens']} response tokens)"
    )


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/qwen_0.5b_lora.yaml"
    adapter = sys.argv[2] if len(sys.argv) > 2 else None
    main(config, adapter)
