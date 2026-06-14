"""Chat with a fine-tuned LoRA adapter, CPU-only.

Run via the Makefile:  `make infer`  (or `make infer ADAPTER=outputs/...`)
"""

import sys

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Explain what a binary search is in two sentences.",
    "Write a haiku about databases.",
    "What is the difference between a process and a thread?",
]


def main(adapter_path: str) -> None:
    # The adapter records its own base model — load that, not a hardcoded id, so
    # this works for both the instruct and base tracks.
    base = PeftConfig.from_pretrained(adapter_path).base_model_name_or_path
    tok = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, adapter_path)  # attach LoRA adapter
    model.eval()

    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        print("=" * 70)
        print("PROMPT:", prompt)
        print(tok.decode(out[0], skip_special_tokens=True))
    print("=" * 70)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/qwen-lora-adapter")
