"""Per-module rank allocation experiment (ADR 0007, answers 01 open-question 4).

A uniform r is a looser constraint on the narrow GQA k/v projections (output dim
128) than on the wide q/o (896). This asks: at a *fixed* trainable-param budget,
does spending rank on the wide matrices (where it's scarce) beat spreading it
evenly — or the opposite? All three arms hit the same param count (1,081,344);
only the allocation differs, so lower perplexity = better allocation.

Per-layer LoRA cost: q/o ≈ 1792·r each, k/v ≈ 1024·r each. The three arms below
each sum to 45,056/layer (× 24 layers = 1,081,344). Verified at run time by
print_trainable_parameters().

Run via the Makefile:  `make alloc`
"""

import copy
import sys

from src.eval import evaluate
from src.train import load_config, train

# module -> rank, per arm. alpha is set to 2·rank per module (α/r = 2, vanilla).
ALLOCATIONS = {
    "uniform": {"q_proj": 8, "o_proj": 8, "k_proj": 8, "v_proj": 8},
    "wide-heavy": {"q_proj": 12, "o_proj": 12, "k_proj": 1, "v_proj": 1},
    "narrow-heavy": {"q_proj": 4, "o_proj": 4, "k_proj": 15, "v_proj": 15},
}


def make_allocation_configs(base_cfg, seed, n_train, out_root="outputs/alloc"):
    """One cfg per allocation arm. Pure; deep-copies base_cfg. Sets rank_pattern and
    alpha_pattern (=2·rank) so every module keeps α/r = 2."""
    cfgs = {}
    for name, ranks in ALLOCATIONS.items():
        cfg = copy.deepcopy(base_cfg)
        cfg["lora"]["rank_pattern"] = dict(ranks)
        cfg["lora"]["alpha_pattern"] = {m: 2 * r for m, r in ranks.items()}
        cfg["mask_prompt"] = True
        cfg["seed"] = seed
        cfg["n_train_examples"] = n_train
        cfg["output_dir"] = f"{out_root}/{name}"
        cfgs[name] = cfg
    return cfgs


def run_allocation(base_cfg, seed, n_train):
    rows = []
    for name, cfg in make_allocation_configs(base_cfg, seed, n_train).items():
        rp = cfg["lora"]["rank_pattern"]
        print(f"\n===== training {name}  (q/o={rp['q_proj']}, k/v={rp['k_proj']}) =====")
        adapter = train(cfg)
        res = evaluate(cfg, adapter)
        res.update({"arm": name, "rank_pattern": rp})
        rows.append(res)
        print(f"[{name}] response-NLL={res['response_nll']:.4f} ppl={res['perplexity']:.2f}")
    return rows


def summarize(rows):
    print("\n============== rank allocation summary ==============")
    print(f"{'arm':>13} {'q/o':>4} {'k/v':>4} {'response-NLL':>13} {'perplexity':>11}")
    for row in rows:
        rp = row["rank_pattern"]
        print(
            f"{row['arm']:>13} {rp['q_proj']:>4} {rp['k_proj']:>4} "
            f"{row['response_nll']:>13.4f} {row['perplexity']:>11.2f}"
        )
    best = min(rows, key=lambda row: row["perplexity"])
    print(f"best (same param budget): {best['arm']} ppl={best['perplexity']:.2f}")
    print("=====================================================")


def main(config_path, seed, n_train):
    base = load_config(config_path)
    rows = run_allocation(base, seed, n_train)
    summarize(rows)


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/qwen_0.5b_lora.yaml"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    main(config, seed, n)
