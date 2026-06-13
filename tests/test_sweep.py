"""Unit tests for the rank-sweep config construction (ADR 0006).

The sweep holds alpha = 2r (alpha/r = 2 fixed) so the curve isolates rank
*capacity*, not effective-LR shrinkage (see docs/math/02 §3). These tests pin
that invariant and that each rank gets its own output dir.
"""

from src.sweep import make_sweep_configs

BASE = {
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "dataset": "databricks/databricks-dolly-15k",
    "max_len": 512,
    "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "target_modules": ["q_proj"]},
    "training": {"batch_size": 1, "grad_accum": 8, "epochs": 1, "lr": 2e-4, "logging_steps": 5},
}


def test_alpha_is_two_r_for_every_rank():
    cfgs = make_sweep_configs(BASE, [2, 4, 8, 16, 32], seed=0, n_train=150)
    for cfg in cfgs:
        assert cfg["lora"]["alpha"] == 2 * cfg["lora"]["r"]


def test_each_rank_is_set_and_in_order():
    rs = [2, 4, 8, 16, 32]
    cfgs = make_sweep_configs(BASE, rs, seed=0, n_train=150)
    assert [c["lora"]["r"] for c in cfgs] == rs


def test_output_dirs_are_distinct_per_rank():
    cfgs = make_sweep_configs(BASE, [2, 4, 8], seed=0, n_train=150)
    dirs = [c["output_dir"] for c in cfgs]
    assert len(set(dirs)) == len(dirs)


def test_sweep_sets_seed_size_and_masking():
    cfgs = make_sweep_configs(BASE, [8], seed=7, n_train=150)
    cfg = cfgs[0]
    assert cfg["seed"] == 7
    assert cfg["n_train_examples"] == 150
    assert cfg["mask_prompt"] is True


def test_base_config_not_mutated():
    # deepcopy expected: sweeping must not clobber the caller's base lora.r/alpha.
    make_sweep_configs(BASE, [2, 32], seed=0, n_train=150)
    assert BASE["lora"]["r"] == 8
    assert BASE["lora"]["alpha"] == 16
