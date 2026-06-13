"""Unit tests for the per-module rank allocation configs (ADR 0007).

Budget-matched: all three arms target the same trainable-param count, varying
only WHERE the rank goes (wide q/o vs narrow k/v). Tests pin the rank_pattern /
alpha_pattern construction; the param-count match itself is verified at run time
by print_trainable_parameters().
"""

from src.allocation import make_allocation_configs

BASE = {
    "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    "dataset": "databricks/databricks-dolly-15k",
    "max_len": 512,
    "lora": {
        "r": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    },
    "training": {"batch_size": 1, "grad_accum": 8, "epochs": 1, "lr": 2e-4, "logging_steps": 5},
}


def test_three_named_arms():
    cfgs = make_allocation_configs(BASE, seed=0, n_train=150)
    assert set(cfgs) == {"uniform", "wide-heavy", "narrow-heavy"}


def test_alpha_pattern_is_two_times_rank_per_module():
    cfgs = make_allocation_configs(BASE, seed=0, n_train=150)
    for cfg in cfgs.values():
        rp = cfg["lora"]["rank_pattern"]
        ap = cfg["lora"]["alpha_pattern"]
        assert rp and ap
        assert all(ap[m] == 2 * rp[m] for m in rp)


def test_wide_heavy_favors_qo_narrow_favors_kv():
    cfgs = make_allocation_configs(BASE, seed=0, n_train=150)
    wide = cfgs["wide-heavy"]["lora"]["rank_pattern"]
    narrow = cfgs["narrow-heavy"]["lora"]["rank_pattern"]
    assert wide["q_proj"] > wide["k_proj"]
    assert narrow["k_proj"] > narrow["q_proj"]


def test_common_fields_set_and_distinct_dirs():
    cfgs = make_allocation_configs(BASE, seed=3, n_train=150)
    for cfg in cfgs.values():
        assert cfg["seed"] == 3
        assert cfg["n_train_examples"] == 150
        assert cfg["mask_prompt"] is True
    dirs = [c["output_dir"] for c in cfgs.values()]
    assert len(set(dirs)) == len(dirs)


def test_base_config_not_mutated():
    make_allocation_configs(BASE, seed=0, n_train=150)
    assert "rank_pattern" not in BASE["lora"]
    assert BASE["lora"]["r"] == 8
