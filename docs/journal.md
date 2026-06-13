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

---

## Session 1 — git/uv init, LoRA derivation, first run

- Claude Code did the first-tasks list: `git init` (main) + scaffold commit,
  `make setup` (torch 2.12.0+cpu, transformers 5.12, peft/datasets/accelerate),
  walked the repo + Makefile, then I chose **math first, then a run**.
- Wrote `docs/math/01-lora-derivation.md` (via the math-tutor agent): full
  derivation of $W' = W_0 + \frac{\alpha}{r}BA$ — shapes, low-rank hypothesis,
  $B(Ax)$ forward + multiplication-order FLOPs, A-random/B-zero init (and the
  both-zero saddle), three α framings incl. rsLoRA, gradient flow + Adam memory,
  free merge. Three open empirical questions at the end.
- Ran `make train` on the baseline (branch `exp/01-baseline-r8`). The payoff:
  `trainable params: 1,081,344 (0.2184%)` matched a by-hand **GQA** decomposition
  to the exact parameter — q/o are 896², but k/v project to only 128 (2 KV heads
  × 64). The square-matrix estimate (~1.38M) overcounts by ~300k. Folded the
  verified numbers back into the doc (§1, §6) and added open-question #4: a single
  global $r$ is much looser capacity on the narrow k/v than on q/o.
- Numbers: loss 2.54 → ~1.91 (noisy, batch=1×8, 300 ex), 25m39s wall,
  **4.3 GB peak RAM**, 4.2 MB adapter. CPU-only LoRA on 0.5B is comfortable here.
- Things I learned / want to chase:
  - The optimizer-memory argument is the real reason this fits, not just param
    count — Adam $m,v$ only for the 1.08M trainable params (~8.6 MB).
  - **Loss masking gap** (src/train.py:75): we train on prompt+response, so the
    loss isn't a clean instruction signal. That's the next fix (docs/math/03).
- Next thread options: fix loss masking, or run the rank sweep for math/02.
