# ADR 0007 — Per-module rank allocation at a fixed budget

- Status: Accepted
- Date: 2026-06-14
- Answers: `docs/math/01-lora-derivation.md` open-question 4. Builds on the rank
  sweep (ADR 0006, Run 004/005) and the eval harness (ADR 0004).

## Context

Qwen2.5-0.5B uses GQA: `q_proj`/`o_proj` are 896×896 (wide), `k_proj`/`v_proj`
project to only 128 (narrow). A *uniform* `r=8` is therefore a much looser
constraint on `k/v` (rank 8 of 128 = 6.25% of full rank) than on `q/o`
(8 of 896 = 0.89%). So `q/o` is the rank-starved part. Open-question 4: at a
fixed parameter budget, does spending rank where it's scarce (`q/o`) beat
spreading it uniformly — or the reverse?

## Decision

A **budget-matched** three-arm experiment (`src/allocation.py`, `make alloc`),
all three at **exactly 1,081,344** trainable params (per-layer LoRA cost: `q/o`
≈ 1792·r each, `k/v` ≈ 1024·r each; each arm sums to 45,056/layer × 24):

| arm | q/o rank | k/v rank |
|-----|---------:|---------:|
| uniform | 8 | 8 |
| wide-heavy | 12 | 1 |
| narrow-heavy | 4 | 15 |

Same budget → lower held-out perplexity = better *allocation*, not more capacity.
Implemented via PEFT `rank_pattern` + `alpha_pattern` (now wired into
`train.py`), with `alpha_pattern = 2·rank` per module so every module keeps
α/r = 2 (vanilla, the established default). Masked, single seed, n=150 — same
regime as Runs 003–005. The equal param count is verified at run time by
`print_trainable_parameters()`.

## Alternatives considered

- **Non-budget-matched** (just add rank to `q/o`, leave `k/v` at 8) — rejected:
  conflates *where* the rank goes with *how much* total, which the rank sweep
  (Run 004/005) already studied. The fixed budget is the whole point.
- **rsLoRA scaling** — kept vanilla α/r; Run 005 showed it's the right choice at
  the r=8-scale optimum and avoids re-opening the scaling confound here.
- **Per-module sweep (2D)** — far more runs; this 3-arm cut is the cheap, legible
  first probe.

## Consequences

- Isolates allocation cleanly (identical budget). Reading: if wide-heavy wins,
  rank is better spent where the matrix is wide/starved; if uniform wins,
  even spread is fine; if narrow-heavy wins, the narrow `k/v` were the bottleneck.
- **rank-1 on `k/v`** (wide-heavy) is deliberately extreme — it tests whether
  `k/v` adaptation can be nearly dropped in favor of `q/o`.
- Same honest caveats as the sweeps: single seed (seed variance ±0.0002, Run
  003), tiny data, instruct base → expect small absolute gaps.
