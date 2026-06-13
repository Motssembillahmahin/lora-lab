# LoRA Lab

A hands-on, math-first learning project for understanding **LoRA** and
**QLoRA** fine-tuning end to end — built to run on a CPU-only laptop.

This repo is deliberately structured like a real software project (uv-managed,
Makefile-driven, documented incrementally) while serving as a personal learning
lab. The aim is *understanding*, not a production model.

## Why this exists

Most LoRA tutorials hand you a script and stop. This repo goes the other way:
derive the math, understand every hyperparameter, version each experiment, and
write down what was learned as we go.

## Hardware target

CPU-only: AMD Ryzen 5 PRO 5650U (6c/12t), 14 GB RAM. No CUDA. We run plain LoRA
on small models (default: Qwen2.5-0.5B-Instruct). Real 4-bit QLoRA needs a GPU,
so we study its math here rather than train with it.

## Quickstart

This project uses [`uv`](https://docs.astral.sh/uv/) and a `Makefile`.

```bash
make help          # list every available command
make setup         # create venv + install deps via uv
make train         # run a LoRA fine-tune from configs/qwen_0.5b_lora.yaml
make infer         # chat with the fine-tuned adapter
make merge         # merge the adapter into the base model
```

## Layout

```
lora-lab/
├── CLAUDE.md            # context + working agreement for Claude Code
├── README.md
├── Makefile             # single entry point for all commands
├── pyproject.toml       # uv-managed project definition
├── configs/             # all hyperparameters live here (YAML)
├── src/                 # train / infer / merge
├── docs/                # ALL documentation lives here
│   ├── journal.md       # running narrative of the learning
│   ├── math/            # mathematical deep-dives
│   └── decisions/       # ADRs — one per meaningful choice
├── experiments/         # structured run log (tied to git SHAs)
├── .claude/agents/      # specialized subagents
├── data/                # datasets (gitignored)
└── outputs/             # adapters & checkpoints (gitignored)
```

## How to use it

Work in a loop: branch, tweak a config, `make train`, record the result in
`experiments/log.md` with the commit SHA, commit. The git history becomes the
record of what was learned. See `CLAUDE.md` for the full working agreement.
