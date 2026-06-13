# ADR 0006 — Rank sweep methodology (α = 2r), plotted

- Status: Accepted
- Date: 2026-06-13
- Relates to: `docs/math/02-rank-and-alpha.md` (the "why"), ADR 0004 (the eval
  metric), ADR 0005 (seeding). Answers `math/01` open-question 1.

## Context

`math/01` open-question 1 asks whether `r=8` actually captures the update or
whether eval loss plateaus earlier. Now that `make eval` gives a comparable
held-out response perplexity, we can sweep `r` and read the curve.

## Decision

Add `src/sweep.py` + `make sweep`. For each `r ∈ {2,4,8,16,32}`:

1. **Hold α = 2r** (so α/r = 2 fixed). This isolates rank *capacity*; holding α
   fixed would also shrink α/r as r grows, confounding capacity with an
   effective-LR change (math/02 §3).
2. Train a **masked** adapter (the established default, ADR 0003/0005), **single
   seed**, `n_train=150` — justified by Run 003's tiny seed variance (±0.0002),
   so one seed per r shows the trend at ~1/3 the cost of 3 seeds.
3. Eval each with the response-only harness (ADR 0004) → `(r, perplexity)`.
4. Emit a summary table and a committed matplotlib figure
   `docs/math/assets/02-rank-sweep.png` (log2 x-axis), embedded in math/02.

`make_sweep_configs` (pure, deep-copies the base cfg) is unit-tested for the
α=2r invariant, distinct output dirs, and non-mutation of the caller's config.
The train/eval/plot loop is integration, verified by the run.

## Alternatives considered

- **Fixed α while sweeping r** — rejected as the primary sweep (confounds
  capacity with effective-LR), though math/02 §3 explains it as a teaching foil.
- **2D (r × α) grid** — most illustrative of α/r but 3–4× the runs (hours on
  CPU). Deferred; the harness takes `RANKS=`/`SEED=` so it can be extended.
- **matplotlib dependency** — added to the `dev` extra. Accepted: a real
  committed figure beats an ASCII plot for a learning artifact; small cost.

## Consequences

- Produces a real perplexity-vs-r curve to read for the plateau/still-dropping/
  U-shape signatures (math/02). Single seed + small data are honest caveats.
- **α/r vs α/√r confound stands** (math/02 §3): a plateau under α=2r could be
  partly the rsLoRA 1/√r under-scaling, not purely low intrinsic rank. The
  `use_rslora=True` control is noted as future work, not run here.
- New dev dep (matplotlib) → `make setup` now also installs the plotting stack.
