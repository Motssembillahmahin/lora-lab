"""Config-driven LoRA fine-tuning, CPU-only.

Run via the Makefile:  `make train`  (or `make train CONFIG=configs/other.yaml`)

This script is intentionally written to be *read*, not just run. Each section is
a step in the LoRA pipeline. See docs/math/ for the theory behind the knobs.
"""

import sys

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.data import build_example


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    # CPU-only: cap threads to the physical/logical cores we have.
    torch.set_num_threads(cfg.get("num_threads", 12))

    # --- 1. Tokenizer ---
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- 2. Base model (fp32 — CPU is happiest here) ---
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=torch.float32)
    model.config.use_cache = False  # needed with gradient checkpointing

    # --- 3. Attach LoRA adapters ---
    lc = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lc["r"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    # This line proves the whole point of LoRA — only a sliver is trainable.
    model.print_trainable_parameters()

    # --- 4. Dataset -> masked training examples ---
    # Loss is computed on the assistant's response only when mask_prompt is set:
    # prompt tokens get label -100. See docs/math/03-loss-masking.md + ADR 0003.
    ds = load_dataset(cfg["dataset"], split=f"train[:{cfg['n_train_examples']}]")
    mask_prompt = cfg.get("mask_prompt", True)

    def encode(ex: dict) -> dict:
        out = build_example(
            tok,
            ex["instruction"],
            ex.get("context"),
            ex["response"],
            max_len=cfg["max_len"],
            mask_prompt=mask_prompt,
        )
        # build_example returns None when the prompt alone fills max_len (no
        # response survives -> 0/0 NaN loss). Mark with empty ids, drop below.
        return out or {"input_ids": [], "attention_mask": [], "labels": []}

    n_before = len(ds)
    ds = ds.map(encode, remove_columns=ds.column_names)
    ds = ds.filter(lambda e: len(e["input_ids"]) > 0)
    n_dropped = n_before - len(ds)
    print(
        f"mask_prompt={mask_prompt}; kept {len(ds)}/{n_before} examples "
        f"({n_dropped} dropped: prompt >= max_len={cfg['max_len']})"
    )

    # Pads input_ids and labels (labels with -100), consuming the labels we built
    # rather than fabricating them from input_ids.
    collator = DataCollatorForSeq2Seq(tokenizer=tok, padding=True)

    # --- 5. Training args (CPU-friendly) ---
    tr = cfg["training"]
    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=tr["batch_size"],
        gradient_accumulation_steps=tr["grad_accum"],
        num_train_epochs=tr["epochs"],
        learning_rate=float(tr["lr"]),
        logging_steps=tr["logging_steps"],
        save_strategy="epoch",
        report_to="none",
        use_cpu=True,
        gradient_checkpointing=True,
        dataloader_num_workers=cfg.get("dataloader_workers", 2),
    )

    # --- 6. Train ---
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()

    # --- 7. Save the adapter (small — a few MB) ---
    model.save_pretrained(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print(f"Saved adapter to {cfg['output_dir']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/qwen_0.5b_lora.yaml")
