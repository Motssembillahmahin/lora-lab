"""Dataset preparation with prompt loss masking (ADR 0003).

The math is in docs/math/03-loss-masking.md. The one idea: build `labels`
aligned to `input_ids`, then blank the prompt prefix with -100 so the loss
(and gradient) is computed over the assistant's response tokens only.

`build_example` is kept pure and tokenizer-only so the masking invariants can
be unit-tested without torch/peft (see tests/test_masking.py).
"""

IGNORE_INDEX = -100  # PyTorch CrossEntropyLoss default; HF models honor it.


def build_example(tok, instruction, context, response, max_len, mask_prompt=True):
    """Tokenize one instruction/response pair into a masked training example.

    Returns a dict with `input_ids`, `attention_mask`, `labels` (all aligned,
    same length), or `None` if the prompt alone fills `max_len` so no response
    token survives truncation (would make the loss a 0/0 NaN — math/03 §5).
    """
    user = instruction if not context else f"{instruction}\n\n{context}"
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
    ]

    # Full conversation, truncated from the right to the context window.
    # transformers returns a BatchEncoding for tokenize=True; pull the ids list.
    full_ids = tok.apply_chat_template(messages, tokenize=True, return_dict=True)["input_ids"]
    full_ids = full_ids[:max_len]

    # Prompt incl. the assistant header (add_generation_prompt=True): this is
    # exactly the prefix the model sees before it should start generating.
    # Computed regardless of mask_prompt so the all-prompt filter fires the same
    # way for masked and unmasked runs (clean paired control — ADR 0005).
    prompt_ids = tok.apply_chat_template(
        messages[:-1], tokenize=True, return_dict=True, add_generation_prompt=True
    )["input_ids"]
    prompt_len = len(prompt_ids)
    if prompt_len >= len(full_ids):
        return None  # no response tokens survive truncation -> drop this example

    labels = list(full_ids)
    if mask_prompt:
        labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }
