"""Unit tests pinning the prompt loss-masking invariants (ADR 0003).

These assert the math from docs/math/03-loss-masking.md holds in code:
- prompt tokens get label -100, response tokens stay as targets,
- input_ids are never altered by masking,
- the prompt-ids are an exact prefix of the full-ids (the BPE-seam risk),
- an all-prompt example (no response survives max_len) is filtered out.

Uses the real Qwen tokenizer (cached locally from the first training run);
no network needed.
"""

import pytest

from src.data import build_example

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    t = AutoTokenizer.from_pretrained(MODEL_ID)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


def _ids(tok, messages, **kw):
    # transformers 5.x returns a BatchEncoding for tokenize=True; pull the list.
    return tok.apply_chat_template(messages, tokenize=True, return_dict=True, **kw)["input_ids"]


def _prompt_len(tok, instruction, context):
    user = instruction if not context else f"{instruction}\n\n{context}"
    msgs = [{"role": "user", "content": user}]
    return len(_ids(tok, msgs, add_generation_prompt=True))


def test_labels_align_with_input_ids(tok):
    ex = build_example(tok, "What is a B-tree?", None, "A balanced search tree.", max_len=512)
    assert len(ex["labels"]) == len(ex["input_ids"])
    assert len(ex["attention_mask"]) == len(ex["input_ids"])


def test_prompt_tokens_are_masked(tok):
    instruction, response = "Explain a hash map.", "Keys map to buckets via a hash."
    ex = build_example(tok, instruction, None, response, max_len=512)
    plen = _prompt_len(tok, instruction, None)
    assert ex["labels"][:plen] == [-100] * plen


def test_response_tokens_are_unmasked(tok):
    instruction, response = "Explain a hash map.", "Keys map to buckets via a hash."
    ex = build_example(tok, instruction, None, response, max_len=512)
    plen = _prompt_len(tok, instruction, None)
    # response label positions equal the corresponding input_ids (real targets)
    assert ex["labels"][plen:] == ex["input_ids"][plen:]
    assert any(label != -100 for label in ex["labels"])


def test_prompt_ids_are_exact_prefix_of_full_ids(tok):
    # The Approach-A correctness assumption; if this fails, the BPE seam shifted
    # and the length-based split is off by a token (see math/03 §5).
    cases = [
        ("Summarize this.", "Birds migrate south in winter to find food."),
        ("Write a function.", "def f(x):\n    return x + 1"),
        ("Continue:", "...and then the system crashed."),  # response starts mid-thought
    ]
    for instruction, response in cases:
        ex = build_example(tok, instruction, None, response, max_len=512)
        plen = _prompt_len(tok, instruction, None)
        full = _ids(
            tok,
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
        )
        assert ex["input_ids"][:plen] == full[:plen]
        assert full[:plen] == _ids(
            tok, [{"role": "user", "content": instruction}], add_generation_prompt=True
        )


def test_input_ids_unchanged_by_masking(tok):
    instruction, response = "Explain a hash map.", "Keys map to buckets via a hash."
    masked = build_example(tok, instruction, None, response, max_len=512, mask_prompt=True)
    unmasked = build_example(tok, instruction, None, response, max_len=512, mask_prompt=False)
    assert masked["input_ids"] == unmasked["input_ids"]


def test_mask_prompt_false_keeps_every_label(tok):
    ex = build_example(tok, "hi", None, "hello", max_len=512, mask_prompt=False)
    assert ex["labels"] == ex["input_ids"]
    assert -100 not in ex["labels"]


def test_all_prompt_example_is_filtered(tok):
    # max_len so small the prompt alone fills it -> no response survives -> None.
    ex = build_example(
        tok,
        "This is a fairly long instruction that on its own exceeds the cap.",
        None,
        "irrelevant response",
        max_len=8,
    )
    assert ex is None
