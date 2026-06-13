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

---

## Session 2 — loss masking (spec → TDD → A/B)

- Pulled the loss-masking thread properly: brainstormed approaches, chose
  **Approach A** (manual prefix masking, no new deps), wrote the spec
  (`docs/math/03-loss-masking.md` + ADR 0003) and got it reviewed *before*
  touching code.
- Key math from 03: causal-LM loss with HF's internal label shift; `-100`
  changes the loss *denominator* (so masked/unmasked loss are NOT comparable);
  full-seq loss is a convex combo `|P|/(|P|+|R|)·L_P + |R|/(|P|+|R|)·L_R`, so
  training on the prompt spends a `|P|/(|P|+|R|)` slice of every gradient learning
  to generate the instruction (~80% for a 400/100 split).
- Built it **test-first**. The failing test caught a real surprise: transformers
  5.12 `apply_chat_template(tokenize=True)` returns a `BatchEncoding`, not a token
  list — needs `return_dict=True` + `["input_ids"]`. 7 tests pin the invariants
  (prompt masked, response unmasked, input_ids untouched, prompt-ids an exact
  prefix of full-ids, all-prompt example filtered). `src/data.py` holds the pure
  `build_example`; `train.py` swapped to `DataCollatorForSeq2Seq`.
- Run 002 (masked): dropped 3/300 examples (prompt ≥ max_len — the NaN guard),
  18m15s, 3.84 GB, train_loss 2.06 (not comparable to Run 001 — different
  denominator).
- **Honest result: inconclusive.** Compared *generations* (not loss) for masked
  vs unmasked: very similar. Masked obeyed "two sentences"; unmasked's haiku was
  a touch better. The confound: the base is already *Instruct*-tuned, so 300 ex /
  1 epoch of LoRA barely shifts behavior and the masking effect washes out. The
  fix is the right objective; this experiment just can't isolate its benefit.
- Lesson: eyeballing 3 greedy generations is a weak metric. Next time measure
  response-only eval loss on a held-out set (apples-to-apples), and/or pick a
  setup where the base is actually weak so there's signal to move.
- Next: a held-out eval harness, or the rank sweep (math/02).

---

## Session 3 — held-out eval harness

- Built the metric Run 002 was missing: `src/eval.py` + `make eval`, a
  response-only token-weighted NLL on a disjoint Dolly slice (`train[300:400]`).
  TDD'd the pure `weighted_mean` first (corpus mean, not mean-of-means; zero-token
  guard returns 0.0 not NaN). ADR 0004 records choosing the custom loop over
  `Trainer.evaluate()` (whose batch=1 eval_loss is mean-of-means).
- Three-way result (same 98 examples / 7821 response tokens, so comparable):
  base ppl 8.53 → Run 001 (unmasked) 7.55 → Run 002 (masked) **7.45**.
- **The metric earned its keep.** Eyeballing 3 generations (Session 2) said
  "no difference"; the token-weighted NLL shows masked beats unmasked by ΔNLL
  ≈ 0.013 (~1.3% ppl) — small but in the predicted direction, and both adapters
  clearly beat the base.
- Stayed honest: the gap is small, single-seed, and Run 002 trained on 297 vs
  300 examples — so I logged it as "directionally right, not yet attributable,"
  with the seed-controlled repeat as the way to actually prove it.
- Lesson reinforced: pick the metric before trusting the conclusion. The math
  (03 §2) said train loss wasn't comparable; eval NLL on a fixed denominator is.
- Next: seed-controlled masking repeat, or the rank sweep (math/02) — now there's
  a real number to plot against r.
