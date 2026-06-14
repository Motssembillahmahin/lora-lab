# ADR 0008 — Pivot to the non-instruct base model

- Status: Accepted
- Date: 2026-06-14
- Relates to: Runs 002–006 (`experiments/log.md`), which all hit the same ceiling.

## Context

Every result since masking (Runs 002–006) landed in a **~0.02–0.4 ppl band**:
masking +1.8%, rank sweep ~2.5%, allocation ~0.3%. The cause is the base: we
fine-tuned `Qwen2.5-0.5B-**Instruct**`, which is *already* instruction-tuned, so
300 examples of LoRA barely move it and every effect is hair-splitting near the
model's ceiling. The methodology is solid (eval harness, seeding,
pre-registration, budget-matching); the signal is the problem.

## Decision

Add a **non-instruct base** track: `Qwen/Qwen2.5-0.5B` (the base LM, not
`-Instruct`), via a new config `configs/qwen_0.5b_base_lora.yaml`. The base model
has not seen instruction tuning, so fine-tuning should move it substantially —
larger, cleaner effects for the masking / rank / allocation questions.

Verified before committing: the base tokenizer **ships the same ChatML
`chat_template`** (`apply_chat_template` renders identically, `add_generation_prompt`
works, prompt is an exact token prefix), so `build_example` and the whole
masking/eval pipeline work **unchanged** — no prompt-format change needed.

Made the tooling base-agnostic so both tracks work:
- `src/eval.py` now loads `cfg["model_id"]` instead of a hardcoded constant.
- `src/infer.py` / `src/merge.py` resolve the base from the adapter's own
  `PeftConfig.base_model_name_or_path`, so they always match the adapter.

The instruct config (`qwen_0.5b_lora.yaml`) stays intact — Runs 001–006 remain
reproducible at their SHAs.

## Alternatives considered

- **Change the default config's `model_id` in place** — rejected: would desync the
  default config from the historical instruct runs and muddle reproducibility. A
  separate config is cleaner and keeps both tracks.
- **Switch prompt format to Alpaca-style** (`### Instruction/### Response`) — not
  needed: the base ships ChatML, so reusing it keeps masked/eval comparable across
  tracks. Could revisit if ChatML on a base model underperforms.
- **A larger base / more data** — orthogonal and heavier; the base-model swap is
  the cheapest change with the biggest expected signal gain.

## Consequences

- Expect **much larger effect sizes**: the raw base should be poor at producing
  ChatML-formatted responses (high perplexity), and the adapter should improve it
  far more than the ~1 ppl seen on the instruct base. Future masking/rank
  experiments should become decisive rather than hair-splitting.
- One-time ~1 GB download of the base weights.
- `CLAUDE.md` still names the instruct base as the convention; left as-is for now
  (both tracks exist). Worth revisiting if the base track becomes primary.
