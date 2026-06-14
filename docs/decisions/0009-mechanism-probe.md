# ADR 0009 — Prompt-NLL mechanism probe

- Status: Accepted
- Date: 2026-06-15
- Explains: Run 008 (masking's win didn't transfer to the base). Math:
  `docs/math/03-loss-masking.md` §4. Builds on the eval harness (ADR 0004).

## Context

Run 008 showed prompt masking is a consistent win on the instruct base
(+0.0191 NLL) but ~neutral on the base LM (+0.0020, sign flips). The proposed
explanation (math/03 §4): a base model uses prompt tokens to *learn the ChatML
format / instruction structure*, so unmasked training spends useful signal on the
prompt — signal that masking throws away. The instruct model already knows the
format, so it loses nothing by masking. We want to test this directly rather than
leave it as a plausible story.

## Decision

Measure how much **unmasked** training lowers the model's loss on **prompt
tokens** vs **response tokens**, for each track:

1. Add a prompt-loss eval mode: `invert_label_mask` flips the response-masked
   labels (from `build_example`) into prompt-masked labels, and `evaluate(...,
   target="prompt")` scores loss on the prompt tokens. (`make eval TARGET=prompt`.)
2. `src/mechanism.py` / `make mechanism`: train one **unmasked** adapter per track
   (base + instruct, seed 0, n=150), then report prompt-NLL and response-NLL
   drops (un-adapted floor − adapter) for each.

Prediction: the base's prompt-NLL drop ≫ the instruct's. If so, unmasked training
really is teaching the base the format (the signal masking discards), which is why
masking helps the base less.

## Alternatives considered

- **Inspect generations qualitatively** — softer, not quantitative; the prompt-NLL
  delta is a direct measurement of "did it learn to predict the format."
- **Reuse the existing study adapters** — the instruct n=150 adapters were
  overwritten by Run 008; retraining both tracks fresh at matched (seed, n) keeps
  the comparison clean and self-contained.

## Consequences

- `evaluate()` return keys generalized: `response_nll`/`response_tokens` →
  `nll`/`scored_tokens`, plus a `target` field (callers study/sweep/allocation
  updated to read `nll`). `invert_label_mask` is unit-tested (incl. involution).
- A confirmed mechanism turns Run 008 from a surprising negative into an
  understood, predictable phenomenon — and sharpens the rule: mask when the base
  already knows the format, consider not masking when it must learn it.
- Single seed, n=150, tiny data — same caveats as the other runs.
