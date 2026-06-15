"""Unit tests for the data-size study configs (ADR 0010).

The load-bearing invariant: the held-out eval slice must stay DISJOINT from
train[:n] for every arm, or larger-n training swallows the eval set and the
numbers are a data leak. These tests pin that guard.
"""

from src.datasize import make_datasize_configs

BASE = {
    "model_id": "Qwen/Qwen2.5-0.5B",
    "dataset": "databricks/databricks-dolly-15k",
    "max_len": 512,
    "n_train_examples": 999,  # sentinel; should be overridden per arm, base untouched
    "eval_start": 300,
    "n_eval_examples": 100,
    "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj"]},
    "training": {"batch_size": 1, "grad_accum": 8, "epochs": 1, "lr": 2e-4, "logging_steps": 5},
}


def test_eval_slice_disjoint_from_every_arm():
    n_values = [150, 300, 600, 1200]
    cfgs = make_datasize_configs(BASE, n_values, seed=0)
    for cfg in cfgs:
        # eval starts at/after the largest training set -> no overlap with train[:n]
        assert cfg["eval_start"] >= max(n_values)
        assert cfg["eval_start"] >= cfg["n_train_examples"]


def test_masked_and_unmasked_per_n():
    cfgs = make_datasize_configs(BASE, [150, 300], seed=0)
    assert len(cfgs) == 4  # 2 n × {masked, unmasked}
    by_n = {}
    for c in cfgs:
        by_n.setdefault(c["n_train_examples"], []).append(c["mask_prompt"])
    assert by_n[150] == [True, False]
    assert by_n[300] == [True, False]


def test_distinct_output_dirs():
    cfgs = make_datasize_configs(BASE, [150, 300, 600], seed=0)
    dirs = [c["output_dir"] for c in cfgs]
    assert len(set(dirs)) == len(dirs)


def test_explicit_eval_start_is_respected():
    cfgs = make_datasize_configs(BASE, [150, 300], seed=0, eval_start=5000)
    assert all(c["eval_start"] == 5000 for c in cfgs)


def test_seed_set_and_base_not_mutated():
    cfgs = make_datasize_configs(BASE, [150], seed=7)
    assert all(c["seed"] == 7 for c in cfgs)
    assert BASE["n_train_examples"] == 999
    assert "output_dir" not in BASE
