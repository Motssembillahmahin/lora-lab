"""Merge a LoRA adapter into the base model, producing a standalone model.

Run via the Makefile:  `make merge`

This is the concrete form of  W' = W + (alpha / r) * B @ A  applied to every
adapted weight matrix. After merging there is zero added inference overhead and
no PEFT dependency at load time.
"""

import sys

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(adapter_path: str, out_path: str = "outputs/qwen-merged") -> None:
    # Resolve the base from the adapter itself (instruct or base track).
    base = PeftConfig.from_pretrained(adapter_path).base_model_name_or_path
    tok = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter_path)

    merged = model.merge_and_unload()  # folds B@A scaled by alpha/r into W
    merged.save_pretrained(out_path)
    tok.save_pretrained(out_path)
    print(f"Merged model saved to {out_path}")


if __name__ == "__main__":
    adapter = sys.argv[1] if len(sys.argv) > 1 else "outputs/qwen-lora-adapter"
    main(adapter)
