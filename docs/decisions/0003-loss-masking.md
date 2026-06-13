# ADR 0003 — Mask prompt tokens in the training loss

- Status: Accepted
- Date: 2026-06-13
- Supersedes / relates to: builds on Run 001 (`experiments/log.md`), which
  exposed the problem. Math: [`docs/math/03-loss-masking.md`](../math/03-loss-masking.md).

## Context

`src/train.py` tokenizes the full chat string (prompt + response) and uses
`DataCollatorForLanguageModeling(mlm=False)`, which copies `input_ids → labels`
wholesale. The causal-LM loss is therefore averaged over **every** token,
including the user's instruction and context. The model is trained to generate
the prompt as well as the response.

Run 001's reported loss (`train_loss ≈ 2.076`) is thus not a clean
instruction-following signal: for Dolly examples with long `context`, the prompt
can be the large majority of the sequence, so most of the gradient is spent
learning the *input* distribution rather than the input→output mapping (full
derivation in the math doc). We want to fix this before running any experiment
whose conclusion depends on the loss.

## Decision

Mask prompt tokens so the loss is computed over response tokens only. Concretely
(**Approach A — manual prefix masking**):

1. Tokenize the prompt with `apply_chat_template(messages[:-1], add_generation_prompt=True)`
   to get `prompt_len` (the assistant header is part of the masked prefix).
2. Tokenize the full conversation, truncate to `max_len`.
3. `labels = full_ids.copy()`; set `labels[:prompt_len] = -100` (PyTorch's
   `ignore_index`); keep response tokens (and their `<|im_end|>`) as targets.
4. Swap the collator to `DataCollatorForSeq2Seq`, which pads `labels` with `-100`
   and consumes the labels we built instead of fabricating them.
5. Add a `mask_prompt: true` flag to `configs/qwen_0.5b_lora.yaml` so the previous
   whole-sequence behavior stays reachable for a clean A/B.
6. Filter out examples whose prompt alone fills `max_len` (no unmasked label →
   loss over zero tokens → NaN). Log how many were dropped.

## Alternatives considered

- **B — patch the chat template + `return_assistant_tokens_mask`.** Qwen2.5's
  template has no `{% generation %}` block (verified: native assistant mask
  returns all zeros), so this requires forking and maintaining the Jinja
  template. More general (multi-turn), but more magic and unnecessary for the
  single-turn Dolly data. Revisit if/when we do multi-turn.
- **C — `trl`'s `DataCollatorForCompletionOnlyLM`.** Least code, but adds the
  `trl` dependency and hides the masking mechanism behind a response-template
  match. Rejected: against the minimal-deps constraint and the "understand it"
  ethos of this repo.

A was chosen for transparency (every masked token is inspectable and
unit-tested), zero new dependencies, and sufficiency for single-turn data.

## Consequences

- **Loss scale changes.** Masked loss averages over fewer tokens (different
  denominator), so it is *not numerically comparable* to Run 001. The honest
  comparison between masked and unmasked is **generation quality** on the
  `make infer` prompts, plus a same-config A/B via `mask_prompt`.
- **Single-turn only.** The prefix approach assumes one user turn + one assistant
  turn. Multi-turn would need Approach B.
- **New test surface.** First use of `make test`: a `tests/` unit test pins the
  masking invariants (prompt masked, response unmasked, prefix assumption holds,
  all-prompt example filtered).
- **Enables Run 002.** A meaningful experiment (masked vs unmasked) now becomes
  possible; previously the loss was measuring the wrong objective.
