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
