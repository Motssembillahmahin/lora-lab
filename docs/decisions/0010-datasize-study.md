# ADR 0010 — Data-size study (base model), with a disjoint-eval guard

- Status: Accepted
- Date: 2026-06-15
- Relates to: Runs 007–009 (all traced to the small-data/noise-limited regime).
  Uses the eval harness (ADR 0004) and seeding (ADR 0005).

## Context

Every base-model surprise — masking's win not transferring (008), the refuted
mechanism (009), the 17× seed variance — points to n=150 being too little data:
response learning saturates at a ~0.10 NLL drop and small effects drown in noise.
The question: does more data de-saturate response learning, and does masking's
effect re-emerge, stay dead, or grow?

## Decision

Sweep `n_train ∈ {150, 300, 600, 1200}` on the base model, training a **masked**
and an **unmasked** adapter at each (paired, single seed, 1 epoch), and eval
response-NLL. `src/datasize.py` / `make datasize`. Plot response-NLL vs n (masked
vs unmasked lines + the un-adapted floor).

**Correctness guard (the reason this needed care):** prior runs evaluated on
`train[300:400]`. Training on `train[:n]` with n>300 would *swallow* that eval
slice — a data leak. `make_datasize_configs` pins `eval_start = max(n_values)`
(=1200), so eval is `train[1200:1300]`, disjoint from every arm's `train[:n]`.
This invariant is unit-tested (`test_eval_slice_disjoint_from_every_arm`).

## Alternatives considered

- **Keep eval at train[300:400]** — rejected: leaks for n>300. Non-negotiable.
- **Vary epochs instead of n** — epochs add gradient steps on the *same* data
  (risks memorization); unique-example count is the cleaner "data size" axis for a
  de-saturation question. Kept epochs=1.
- **Multiple seeds per point** — would multiply an already ~5h study; single seed
  is acceptable because (a) variance should *shrink* with n (one of the things
  we're testing) and (b) the trend across 8× data is the signal, not per-point CIs.

## Consequences

- ~5h wall-clock (8 train+eval runs; n=1200 alone is ~80 min on CPU).
- **Absolute NLLs are NOT comparable to Runs 002–009** — different eval slice
  (`train[1200:1300]` vs `train[300:400]`). Only *within-study* comparisons
  (across n, masked vs unmasked) are valid. Logged explicitly.
- Single seed: the small-n masking delta stays noisy (Run 008 showed ±0.0035 at
  n=150); the point is whether it firms up and trends at larger n.
