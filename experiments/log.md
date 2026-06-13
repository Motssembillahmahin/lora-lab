# Experiment Log

One entry per training run. Each entry is tied to a git commit SHA so any result
can be reproduced by checking out that commit's config + code.

Template:

```
## Run NNN — <short name>
- commit: <git sha>
- hypothesis: <what this run tests, one variable changed>
- config: <the diff from the previous run>
- final loss: <eyeballed trend>
- wall time: <minutes>
- trainable params: <count / total>
- observation: <what happened, surprises>
- next: <what this suggests trying>
```

---

## Run 001 — baseline r=8, attention-only
- commit: f3cd7a5 (branch `exp/01-baseline-r8`)
- hypothesis: sanity-check the pipeline end-to-end and ground the §1/§6 math —
  does the printed trainable-param count match the GQA decomposition, and does
  the run fit in 14 GB RAM?
- config: `configs/qwen_0.5b_lora.yaml` unchanged (baseline). r=8, alpha=16
  (α/r=2), dropout=0.05, target=[q,k,v,o], 300 Dolly examples, max_len=512,
  batch=1 × grad_accum=8 → 38 optim steps, 1 epoch, lr=2e-4 (linear decay).
- final loss: train_loss 2.076 avg; per-log trend 2.54 → 2.11 → 2.18 → 1.96 →
  2.04 → 1.95 → 1.91. Downward but noisy — expected at batch=1×8 on 300 examples.
- wall time: 25m39s total (train_runtime 1319s ≈ 22 min + ~3.5 min model/data
  download). ~34 s/step on 12 CPU threads.
- trainable params: 1,081,344 / 495,114,112 = 0.2184%. Matches hand GQA count
  exactly (q/o 896², k/v 128×896 → 45,056/layer × 24). Adapter on disk = 4.2 MB.
- peak RAM: 4.3 GB resident — well under 14 GB, no swap.
- observation: pipeline works end-to-end; the measured param count validated the
  derivation to the parameter (see docs/math/01 §6). NOTE: loss is computed over
  the *whole* sequence incl. prompt tokens (src/train.py:75) — so this loss is
  not a clean instruction-following signal yet.
- next: (a) fix loss masking before trusting loss numbers; (b) rank sweep
  r∈{2,4,8,16,32} for docs/math/02 open-question 1; (c) try per-module
  rank_pattern (open-question 4) since r=8 is loose on the narrow k/v.
