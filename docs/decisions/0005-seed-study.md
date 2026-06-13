# ADR 0005 — Paired seed study for the masking effect

- Status: Accepted
- Date: 2026-06-13
- Relates to: ADR 0003 (masking), ADR 0004 (eval). Closes the open question from
  Run 002's eval (`experiments/log.md`).

## Context

The eval (ADR 0004) showed masked beating unmasked by ΔNLL ≈ 0.013 (~1.3% ppl) —
the right direction, but from a **single** training run each, and Run 002 also
trained on 297 vs Run 001's 300 examples. So the gap could be seed noise or the
example-count confound, not masking. We want to know which.

## Decision

Run a **paired seed study** (`src/study.py`, `make study`):

1. **Seed plumbing.** Add `seed` to the cfg; `train()` calls
   `transformers.set_seed(seed)` (LoRA `A` init, dropout, shuffle).
2. **Paired design.** For each seed, train masked *and* unmasked with the *same*
   seed, so shared init/shuffle cancel and the per-seed delta isolates masking.
3. **Same examples.** The all-prompt filter (`src/data.py`) now fires regardless
   of `mask_prompt`, so both arms train on the identical surviving examples —
   removes the 297-vs-300 confound.
4. **Readout.** `summarize()` reports masked vs unmasked mean±std, each paired
   per-seed delta, and whether the delta sign is **consistent** across seeds.
5. **Budget.** 3 seeds at `n_train=150` (half size, ~1 h on this CPU), chosen
   over full size (~2 h) / more seeds for tractability.

## Alternatives considered

- **Unpaired (independent seeds per arm).** Simpler bookkeeping but lower
  statistical power — paired cancels the shared-init variance, which dominates
  at small N. Rejected.
- **More seeds / full size.** Better statistics, but 2 h+ on CPU. Rejected for
  now; the infra scales (`make study SEEDS=… N=…`) if we want more later.

## Consequences

- This is an **effect-direction + spread** readout, not a formal significance
  test: N=3 is too small for a credible p-value. We report the paired mean delta,
  its std, and sign-consistency, and interpret honestly (a consistent small
  positive delta is suggestive, not proof).
- **Half-size runs reduce signal** per run; the absolute NLLs won't match the
  full-size Run 001/002 numbers and shouldn't be compared to them — only masked
  vs unmasked *within* this study are comparable.
- Adapters land under `outputs/study/` (gitignored).
