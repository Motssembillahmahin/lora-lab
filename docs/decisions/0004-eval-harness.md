# ADR 0004 — Response-only eval harness (token-weighted NLL)

- Status: Accepted
- Date: 2026-06-13
- Relates to: ADR 0003 (masking), Run 002 (`experiments/log.md`) which exposed
  the need. Math: [`docs/math/03-loss-masking.md`](../math/03-loss-masking.md) §2.

## Context

Run 002 compared masked vs unmasked adapters by eyeballing three greedy
generations — a weak, non-reproducible metric. We need a single honest number
to rank adapters: the mean cross-entropy over **response tokens only**, on a
held-out slice the model never trained on. Because masking is applied
identically at eval time, the loss denominator is the same for every adapter, so
the numbers are directly comparable (unlike train loss across masked/unmasked
runs — math/03 §2).

## Decision

Add `src/eval.py` + `make eval ADAPTER=… CONFIG=…` (`ADAPTER=base` evaluates the
un-adapted model as a reference line). It:

1. Loads a **disjoint** Dolly slice via `eval_start`/`n_eval_examples`
   (defaults `train[300:400]`, after the `train[:300]` training slice).
2. Masks each example with the same `build_example` used in training
   (`mask_prompt=True` always at eval).
3. Computes a **corpus-level token-weighted NLL**:
   `Σ(per-example NLL × scored-tokens) / Σ(scored-tokens)`, where scored tokens
   are the non-`-100` labels *after* the model's internal left-shift (HF's own
   loss denominator). Reports NLL and perplexity `exp(NLL)`.

Chosen over `Trainer.evaluate()` (Approach A), whose `batch_size=1` `eval_loss`
is a mean-of-per-example-means rather than a token-weighted corpus mean —
slightly wrong and less transparent. The custom loop is a few lines more but is
exact and demonstrates the denominator idea from math/03 §2.

## Consequences

- **Comparable across adapters and the base.** Same eval set, same masking, same
  denominator → NLL/perplexity rank adapters honestly. This is the metric future
  experiments (rank sweep, etc.) report.
- **`weighted_mean` is unit-tested** (`tests/test_eval.py`), including the
  zero-token guard (returns 0.0, never 0/0 NaN). The model loop is integration,
  verified by `make eval`.
- **Eval is forward-only** (no grad), so it is much cheaper than a training run
  and can be run after every experiment.
- **Confound caveat stands.** A better metric does not remove Run 002's confound
  (an already-instruct-tuned base + tiny data); it just measures the gap
  precisely instead of by eye.
