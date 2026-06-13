# Learning Journal

Append-only narrative of the LoRA Lab. Newest entries at the bottom.

---

## Session 0 — scaffold

- Set up the repo: uv project, Makefile, src/ layout, configs, docs, agents.
- Decided constraints: CPU-only (Ryzen 5 5650U, 14 GB RAM), plain LoRA on
  Qwen2.5-0.5B-Instruct, no local QLoRA.
- Defined four subagents: math-tutor, experiment-runner, doc-writer,
  code-reviewer.
- Open learning threads, in rough order:
  1. Derive the LoRA low-rank update from scratch (docs/math/01).
  2. Understand rank vs. alpha (docs/math/02).
  3. Fix loss masking — currently training on prompt tokens (docs/math/03 + code).
  4. Study QLoRA's NF4 / double quantization mathematically (docs/math/04).
  5. First training run + first experiment varying r.
- Next: open the repo in Claude Code, let it init git + uv, then pull the math
  thread first.
