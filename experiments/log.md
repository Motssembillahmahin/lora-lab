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
- commit: d5d8126 (branch `exp/01-baseline-r8`)
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

## Run 002 — prompt loss masking on (Approach A)
- commit: 52e741e (branch `feat/loss-masking`)
- hypothesis: masking prompt tokens (loss on response only) improves
  instruction-following vs the unmasked Run 001. Judge by generation, not loss
  (masked/unmasked loss aren't comparable — different denominators, math/03 §2).
- config: only `mask_prompt: true` changed (was effectively false in Run 001).
  Everything else identical: r=8, α=16, [q,k,v,o], 300 Dolly, max_len=512,
  batch=1×8, 1 epoch, lr=2e-4. One-variable experiment.
- final loss: train_loss 2.06. **NOT comparable to Run 001's 2.076** — averaged
  over response tokens only. Recorded for completeness, not for comparison.
- wall time: 18m15s (train_runtime 1081s; no model download this time).
- peak RAM: 3.84 GB.
- trainable params: 1,081,344 (unchanged — LoRA config didn't move). The masking
  filter dropped **3/300** examples whose prompt alone ≥ max_len=512 (the NaN
  guard, math/03 §5); trained on 297.
- observation: **inconclusive — no decisive generation difference.** On the 3
  infer prompts: masked obeyed "in two sentences" (2 vs 1); unmasked's haiku was
  marginally better-formed; process-vs-thread answer was identical. Most likely
  confound: the base is Qwen2.5-0.5B-*Instruct* (already instruction-tuned), and
  300 ex / 1 epoch of r=8 LoRA barely moves it, washing out the masking effect.
  The fix is correct (right objective) but this setup can't isolate its benefit.
- next: to actually measure masking's effect — (a) hold out a small eval set and
  compute *response-only* eval loss for both adapters (apples-to-apples, since
  eval masking is identical), rather than eyeballing 3 generations; (b) increase
  signal: more examples / more epochs, or a task the instruct base is weak at;
  (c) the |R|-weighted argument (math/03 §3) predicts a bigger effect when
  prompts dwarf responses — worth constructing such a slice deliberately.

## Eval — held-out response NLL (ADR 0004)
- harness: `make eval`, `src/eval.py`. Same held-out slice `train[300:400]`
  (disjoint from training), same masking → comparable denominators. 98/100
  examples scored (2 dropped: prompt ≥ max_len), 7821 response tokens each.
- numbers:

  | adapter            | response-NLL | perplexity | vs base |
  |--------------------|-------------:|-----------:|--------:|
  | base (no adapter)  | 2.1440       | 8.53       | —       |
  | Run 001 (unmasked) | 2.0219       | 7.55       | −11.5%  |
  | Run 002 (masked)   | **2.0088**   | **7.45**   | −12.7%  |

- read: both adapters clearly beat the base (LoRA works). Masked edges out
  unmasked by ΔNLL ≈ 0.013 (~1.3% perplexity) — small, but in the direction
  math/03 predicts. The generation eyeball (Run 002) couldn't see this; the
  token-weighted metric can.
- honest caveats: (1) no seed-variance estimate — a single training run each, so
  0.013 could be partly noise; (2) Run 002 trained on 297 vs Run 001's 300
  examples, a second uncontrolled variable; (3) the effect is small, consistent
  with the instruct-base confound. To *attribute* the gain to masking: repeat
  with fixed seeds across N runs, or a same-297-examples unmasked control.
- next: seed-controlled repeat, or move to the rank sweep (math/02) now that
  there's a real metric to plot against r.

## Run 003 — paired seed study: masking IS real (ADR 0005)
- harness: `make study SEEDS=0,1,2 N=150`, `src/study.py`. For each seed, trained
  a masked and an unmasked adapter with the *same* seed on the *same* 147 examples
  (all-prompt filter now fires for both arms), then evaluated both on the held-out
  slice. Paired design → per-seed delta isolates masking.
- wall time: ~1 h for all 6 train+eval pairs (n_train=150, ~19 steps/run).
- numbers (held-out response-NLL):

  | seed | masked | unmasked | Δ (unmasked − masked) |
  |-----:|-------:|---------:|----------------------:|
  | 0    | 2.0201 | 2.0387   | +0.0187 |
  | 1    | 2.0202 | 2.0390   | +0.0188 |
  | 2    | 2.0198 | 2.0398   | +0.0200 |
  | **mean** | **2.0200 ± 0.0002** | **2.0392 ± 0.0005** | **+0.0191 ± 0.0006** |

  Masked ppl 7.54 vs unmasked 7.68 (~1.8% lower).
- read: **masking's benefit is real, not seed noise.** The effect (+0.0191) is
  ~30× the std of the paired delta (±0.0006), and the sign is consistent across
  all 3 seeds. Removing the 297-vs-300 confound (same 147 examples both arms) the
  gap held — actually slightly larger than Run 002's single-seed 0.013. Seed
  variance is tiny here (±0.0002), i.e. fp32 CPU LoRA on fixed data is very stable.
- honest caveats: (1) N=3 → this is effect-direction + spread, not a formal
  p-value; the case rests on consistent sign + signal ≫ spread, which is strong
  but not a significance test. (2) Absolute NLLs are at n_train=150 and are NOT
  comparable to Run 001/002 (n_train=300) — only masked-vs-unmasked *within* this
  study is. (3) Effect is small in absolute terms (~1.8% ppl), as expected for an
  already-instruct-tuned base + tiny data; the *mechanism* (math/03) predicts a
  bigger gap when prompts dwarf responses.
- verdict: the masking question is closed — keep `mask_prompt: true`.
- next: rank sweep (math/02), now with `make study`/`make eval` as the metric.

## Run 004 — rank sweep r∈{2,4,8,16,32}, α=2r (ADR 0006)
- harness: `make sweep RANKS=2,4,8,16,32 SEED=0 N=150`, `src/sweep.py`. Masked,
  single seed, α=2r (α/r=2 held fixed → isolates capacity, math/02 §3). Each r
  trained + evaluated on the held-out slice. Plot: `docs/math/assets/02-rank-sweep.png`.
- wall time: ~1 h for 5 train+eval runs.
- numbers (held-out response perplexity):

  | r  | α  | response-NLL | perplexity | Δppl vs r/2 |
  |---:|---:|-------------:|-----------:|------------:|
  | 2  | 4  | 2.0413       | 7.70       | —           |
  | 4  | 8  | 2.0279       | 7.60       | −0.10       |
  | 8  | 16 | 2.0201       | 7.54       | −0.06       |
  | 16 | 32 | 2.0164       | 7.51       | −0.03       |
  | 32 | 64 | 2.0164       | 7.51       |  0.00       |

- read: **monotonic decrease that plateaus by r=16** (r=16 and r=32 identical to
  4 decimals). This is reading (a) from math/02 §2 — **low intrinsic rank**. Most
  of the gain is captured by r=8; r=16 adds a sliver (ΔNLL 0.0037, ~0.4% ppl);
  r=32 adds nothing. No U-shape, so no overfitting at high rank on this slice.
  Answers 01 open-question 1: r=8 is a sound default with headroom; the task's
  update genuinely lives in a thin subspace, as the LoRA hypothesis predicts.
- honest caveats: (1) single seed (justified — Run 003 variance ±0.0002; the
  0.0037 r=8→16 gain is ~18× that, so real, but small); (2) n_train=150, tiny
  data — absolute ppl not comparable to Run 001/002, only within-sweep across r;
  (3) **α/r vs α/√r confound (math/02 §3)**: the plateau under α=2r could be
  partly the rsLoRA 1/√r under-scaling rather than purely low intrinsic rank — a
  `use_rslora=True` sweep is the control to disentangle; (4) global r under GQA
  (01 OQ4) — per-module rank_pattern is the next probe.
- verdict: keep r=8 as default; r=16 if squeezing the last ~0.4% ppl is worth 2×
  the adapter/optimizer cost. Diminishing returns are clear.
- next: rsLoRA control sweep (disentangle the §3 confound), or per-module
  rank_pattern (01 OQ4), or move to a non-instruct base for a stronger signal.

## Run 005 — rsLoRA control sweep (α/√r): the plateau WAS an artifact
- harness: `make sweep RANKS=2,4,8,16,32 SEED=0 N=150 RSLORA=1`. Identical to
  Run 004 except scaling = α/√r (`use_rslora=True`), the control pre-registered in
  math/02 §3 / ADR 0006. Plot: `docs/math/assets/02-rank-sweep-rslora.png`;
  overlay vs Run 004: `docs/math/assets/02-rank-sweep-comparison.png`.
- numbers (held-out response perplexity), vs Run 004:

  | r  | vanilla α/r (Run 004) | rsLoRA α/√r (Run 005) |
  |---:|----------------------:|----------------------:|
  | 2  | 7.70                  | 7.65 |
  | 4  | 7.60                  | 7.54 |
  | 8  | 7.54                  | **7.51** |
  | 16 | 7.51                  | 7.58 |
  | 32 | 7.51                  | 7.95 |

- read: **the §3 confound was real — Run 004's plateau was partly a scaling
  artifact.** Under variance-correct α/√r the curve is a **U-shape** (reading (c)):
  it bottoms at r=8 (7.51) then *rises* sharply (r=32 → 7.95). Mechanism: α/√r >
  α/r for r>1 and increasingly so with r, so high-r rsLoRA adapters get larger
  effective updates; combined with more capacity on ~150 examples they **overfit**
  (held-out ppl worsens). Vanilla α/r under-scaled those same high-r adapters by
  ~1/√r, which *flattened* the rise into a benign-looking plateau. rsLoRA is also
  slightly better at low r (r≤8) where the bigger effective update helps without
  overfitting. Crossover ≈ r=8.
- synthesis: **r=8 is the optimum under both scalings** — robustly the right
  practical choice (best single number anywhere: rsLoRA r=8 = 7.51). But the
  earlier "r=8 has comfortable headroom" (Run 004 reading (a)) was too rosy: the
  headroom was the under-scaling hiding overfitting. Correct picture on this data:
  r=8 is the knee, and going higher is plateau-at-best (α/r) or harmful (α/√r).
- honest caveats: single seed (but the r=32 jump 7.51→7.95 is ~300× seed std, so
  unambiguous); tiny data is *why* the U appears — on a larger corpus the high-r
  rise would likely flatten (overfitting is the data regime, not a LoRA flaw);
  α/√r vs α/r is now disentangled, confound closed.
- verdict: keep r=8 + vanilla α/r as the default. (rsLoRA r=8 edges it by 0.03 ppl
  but buys nothing above r=8 and hurts there.) The §3 question is answered.
- next: per-module rank_pattern (01 OQ4), or a non-instruct base / more data so the
  effects aren't all in the ~0.1–0.4 ppl range.

## Run 006 — per-module rank allocation @ fixed budget (ADR 0007, 01 OQ4)
- harness: `make alloc SEED=0 N=150`, `src/allocation.py`. Three arms, **all
  exactly 1,081,344 trainable params** (verified live by print_trainable_parameters
  ×3), α/r=2 per module via rank_pattern+alpha_pattern, vanilla, masked, seed 0,
  n=150. Same budget → lower ppl = better *allocation*. Plot:
  `docs/math/assets/rank-allocation.png`.
- numbers (held-out response perplexity):

  | arm          | q/o | k/v | response-NLL | ppl  |
  |--------------|----:|----:|-------------:|-----:|
  | narrow-heavy |   4 |  15 | **2.0186**   | 7.53 |
  | uniform      |   8 |   8 | 2.0201       | 7.54 |
  | wide-heavy   |  12 |   1 | 2.0217       | 7.55 |

- read: **the GQA "q/o is rank-starved, feed it" hypothesis is NOT supported —
  mildly contradicted.** narrow-heavy (rank into k/v) is best; wide-heavy (rank-1
  k/v) is worst; uniform sits between. The total spread is **~0.02 ppl** — tiny,
  though the ordering is above seed noise (full range ≈0.0031 NLL ≈ 15× the
  ±0.0002 seed std from Run 003). Honest takeaway: **allocation barely matters
  here; the only clear signal is don't starve a module to rank 1.**
- why this reconciles with Runs 004/005: the task's intrinsic rank is low (≤~8),
  so q/o gains nothing from 12 (we already saw r>8 plateaus/overfits), while
  crushing k/v to rank 1 drops it *below* its small intrinsic need → the wide-heavy
  penalty. The "fraction-of-full-rank" intuition (k/v 8/128 vs q/o 8/896) was the
  wrong lens: what matters is the intrinsic rank of each module's *update*, which
  is low everywhere — not how much of the full matrix rank you're using.
- honest caveats: single seed (ordering likely real but effect tiny — a few seeds
  would firm up the ranking given how small 0.02 ppl is); tiny data; instruct base.
  rank-1 k/v was a deliberately extreme probe and behaved as the one clear loser.
- verdict: **keep uniform r=8** — simplest and statistically indistinguishable
  from the best. Per-module reallocation isn't worth it on this model/task.
- next: a non-instruct base and/or 10× data — every effect across Runs 002–006
  lives in a ~0.02–0.4 ppl band because the instruct base + tiny data ceiling
  everything. A weaker base would make these experiments decisive.

## Run 007 — new baseline on the NON-instruct base (ADR 0008)
- config: `configs/qwen_0.5b_base_lora.yaml` (model_id `Qwen/Qwen2.5-0.5B`, the
  base LM). Same recipe as before: masked, r=8, α=16, q/k/v/o, 300 Dolly, 1 epoch.
  Base tokenizer ships ChatML, so the pipeline ran unchanged.
- train: train_loss 2.286, 22m02s, 3.99 GB peak, adapter 4.2 MB, 1,081,344
  trainable params, dropped 3/300 (prompt ≥ max_len).
- eval (held-out train[300:400], 98 ex / 7861 response tokens, same-tokenizer so
  base-vs-adapter is apples-to-apples):

  | track            | floor (no adapter) | + adapter        | gain (ΔNLL / Δppl) |
  |------------------|-------------------:|-----------------:|-------------------:|
  | instruct (Run 002) | 2.1440 / 8.53    | 2.0088 / 7.45    | 0.135 / −12.7%     |
  | **base (Run 007)** | **2.3291 / 10.27** | **2.1708 / 8.77** | **0.158 / −14.6%** |

- read: pivot worked **in direction, modestly in magnitude.** The base has a
  higher floor (10.27 vs 8.53 — genuinely worse at ChatML responses, as expected)
  and LoRA improves it more (ΔNLL 0.158 vs 0.135). But it's not dramatically
  worse than instruct — Qwen2.5-0.5B base clearly absorbed instruction-like data
  in pretraining. So the base isn't a blank slate.
- why it still matters: the *total* fine-tuning effect here (14.6% ppl) is ~8×
  the masking effect and ~6× the rank-sweep range we were splitting hairs over on
  the instruct base. That headroom is the point — re-running masking / rank /
  allocation on THIS base should make those differential effects decisive instead
  of ~0.02–0.4 ppl noise-adjacent. (Not yet measured — that's the next step.)
- honest caveats: single seed; train_loss not comparable across tracks (different
  base); the base→adapter gain is real and clean (same tokenizer/eval slice), but
  whether *downstream* effects amplify is a hypothesis to test, not yet shown.
- next: re-run the masking A/B (Run 003) and/or rank sweep (Run 004/005) on the
  base config and see if the effects grow. Optionally bump data (n_train↑) since
  the base benefits more from examples.

## Run 008 — masking A/B on the NON-instruct base: the win did NOT transfer
- harness: `make study CONFIG=configs/qwen_0.5b_base_lora.yaml SEEDS=0,1,2 N=150`.
  Identical to Run 003 (instruct) except the base model. Paired, masked vs unmasked
  per seed, held-out response-NLL. Plot: `docs/math/assets/masking-effect-by-base.png`.
- numbers:

  | track | masked NLL | unmasked NLL | paired Δ (unmasked−masked) | sign |
  |-------|-----------:|-------------:|---------------------------:|:----:|
  | instruct (Run 003) | 2.0200 ± 0.0002 | 2.0392 ± 0.0005 | **+0.0191 ± 0.0006** | consistent |
  | base (Run 008)     | 2.2260 ± 0.0035 | 2.2281 ± 0.0013 | **+0.0020 ± 0.0023** | **flips (−,+,+)** |

- read: **prediction refuted.** I expected masking's benefit to *grow* on the base
  (more headroom). Instead it *collapsed*: +0.0191 (consistent, 32× its std) on
  instruct → +0.0020 (≈ its own std 0.0023, sign flips across seeds) on the base.
  On the base, masked vs unmasked is statistically indistinguishable.
- mechanism (hypothesis, ties to math/03 §4): a base model must *learn the ChatML
  structure / instruction-following format*, and the prompt tokens carry that
  signal. Masking discards it. The already-formatted instruct model didn't need it
  (so masking was pure gain there); the base is nearer the "training on the prompt
  helps" regime, so masking gives back roughly what it saves. Masking's value is
  **regime-dependent**, not universal — exactly the §4 caveat, now demonstrated.
- second finding: **seed variance jumped ~17×** (instruct masked ±0.0002 →
  base masked ±0.0035). Base-model LoRA is much less stable run-to-run — a
  single-seed study would have been misleading here; the paired multi-seed design
  caught it.
- honest note: I recommended this pivot *and* predicted masking would amplify; the
  pivot was worth it (clean, counterintuitive finding + a testable mechanism) but
  the specific prediction was wrong. Logged as-is.
- verdict: keep `mask_prompt: true` as default (still the right objective in
  theory, neutral-to-slightly-positive and harmless on the base) — but drop the
  claim that masking is a reliable win everywhere. It's a clear win on an
  already-instruction-tuned base; ~neutral on a base LM.
- next: (a) test the mechanism — does unmasked train_loss on PROMPT tokens drop a
  lot for the base (it's learning the format) but barely for instruct? (b) rank
  sweep on the base; (c) more data (n_train↑) to cut the now-large seed variance.

## Run 009 — mechanism probe: §3 confirmed, my §4 story refuted (ADR 0009)
- harness: `make mechanism SEED=0 N=150`, `src/mechanism.py`. Trained one UNMASKED
  adapter per track, measured prompt-NLL and response-NLL drop (floor − tuned) on
  the held-out slice. Plot: `docs/math/assets/mechanism-prompt-vs-response.png`.
- numbers (NLL drop from unmasked training):

  | track | prompt: floor→tuned (drop) | response: floor→tuned (drop) |
  |-------|---------------------------:|-----------------------------:|
  | instruct | 3.3975 → 2.1331 (**+1.26**) | 2.1440 → 2.0387 (+0.11) |
  | base     | 3.8042 → 2.9871 (**+0.82**) | 2.3291 → 2.2291 (+0.10) |

- **confirmed (math/03 §3), vividly:** for BOTH models, unmasked training puts
  ~90% of its NLL improvement into PROMPT tokens (+1.26 / +0.82) vs ~+0.10 on
  response. Unmasked training overwhelmingly learns to predict tokens we never
  generate — the convex-combination waste argument, measured directly.
- **refuted (my Run 008 explanation):** I predicted the *base* would drop
  prompt-NLL MORE (learning the ChatML format it lacks). The reverse — *instruct*
  drops it more (1.26 vs 0.82). So "the base needs prompt signal to learn the
  format" does NOT explain why masking failed to transfer.
- revised explanation: response-NLL drop under *unmasked* is ~identical (0.11
  instruct, 0.10 base). The masking benefit = the EXTRA response gain from
  concentrating gradient on response. Instruct converts it (unmasked 0.105 →
  masked 0.124, +0.019); the base barely does (0.100 → 0.103, +0.003). The base's
  response learning is near-saturated / noise-limited on 150 examples (cf. Run
  008's 17× seed variance), so masking's small benefit drowns in noise rather than
  being mechanistically cancelled by format-learning.
- honest note: this is my SECOND wrong prediction in a row on the base (Run 008
  amplification, Run 009 mechanism). Both corrected by measurement. The probe
  still earned its keep: it nailed the §3 macro-claim and killed a plausible-but-
  wrong story I'd otherwise have believed.
- caveats: single seed, n=150; prompt-NLL is dominated by the (hard, ~arbitrary)
  instruction TEXT, not just format scaffolding — so "prompt-NLL drop" partly
  reflects fitting the Dolly instruction distribution, not only ChatML structure.
- next: (a) more data (n_train↑) to lift the base out of the noise-limited regime
  and re-test masking; (b) rank sweep on the base; (c) per-segment split of prompt
  into scaffolding vs instruction-text to separate format-learning from memorizing.

## Run 010 — data-size study (ADR 0010): an incident, then a minimal result
### Incident (logged honestly)
- First launch was wrong on two counts: (1) **wrong config** — `make datasize`
  inherited the Makefile's global `CONFIG ?= …instruct`, so it ran the INSTRUCT
  model, not the base (my invocation error, missing `CONFIG=`). (2) **machine
  resource exhaustion** — load avg ~12 on 6 cores + 2.1 GiB swap (memory pressure
  → disk swapping) drove per-step time from ~33 s to ~2089 s (~60×). Killed after
  ~20 h on the n=600 arm (5th of 8). The full n∈{...,1200} plan is infeasible on
  this CPU box in a degraded state.
- Fixes: `make datasize` now defaults `CONFIG := configs/qwen_0.5b_base_lora.yaml`
  (the study is base-specific); base config `num_threads: 12 → 6` (1 per physical
  core — 12 oversubscribed under other load). After killing, load fell to 0.52 and
  the relaunch ran at 27 s/it (normal).

### Minimal result — n=600, base, masked vs unmasked (single seed)
- eval on the disjoint slice `train[600:700]` (eval_start pinned to max(n)=600).
  Floor (base, no adapter): 2.1465. (Lower than Run 007's 2.33 only because this
  is a different, easier eval slice — absolute NLLs are NOT comparable across
  slices, only within a study.)

  | n (base) | masked NLL | unmasked NLL | Δ (unmasked−masked) | eval slice |
  |---------:|-----------:|-------------:|--------------------:|------------|
  | 150 (Run 008) | 2.2260 | 2.2281 | +0.0020 (sign flipped) | train[300:400] |
  | 600 (Run 010) | **1.9587** | **1.9662** | **+0.0075** | train[600:700] |

- read: at n=600 masking clearly helps the base (masked better by 0.0075, ~3.7× the
  n=150 delta and ~2× the n=150 seed std). Response-NLL dropped 0.19 from floor vs
  ~0.10 at n=150 → response learning **de-saturated** with 4× data. This reconciles
  Run 009: masking isn't *dead* on the base, it was *noise-limited at n=150* and
  re-emerges with more data — the direction I'd originally (over-confidently)
  predicted, now with actual support.
- honest caveats (this is suggestive, not conclusive): (1) single seed per arm;
  (2) the n=150 and n=600 deltas are on **different eval slices**, so the growth
  conflates data size with slice difficulty — a clean version needs n=150 and
  n=600 on the SAME slice, multiple seeds; (3) only one n beyond 150, so no real
  curve. The full multi-n study remains the right experiment when the machine is
  fresh (reboot/cool) and run with num_threads=6, n capped where it doesn't swap.
- next: clean re-test — masked vs unmasked at n∈{150,600} on ONE fixed slice, 2-3
  seeds, base, num_threads=6; or rank sweep on the base.
