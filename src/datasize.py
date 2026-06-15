"""Data-size study (ADR 0010): how do response-NLL and the masking effect change
as n_train grows on the base model?

Runs 007–009 all traced to the small-data / noise-limited regime (n=150). This
sweeps n_train, training a masked AND an unmasked adapter at each (paired, single
seed), to see whether response learning de-saturates and whether masking's effect
re-emerges, stays dead, or grows.

CRITICAL: the eval slice must be disjoint from train[:n] for EVERY arm, else
larger-n training swallows the eval set (data leak). make_datasize_configs pins
eval_start to max(n_values) so eval = train[max_n : max_n+n_eval] never overlaps
any arm's training data.

Run via the Makefile:  `make datasize`
"""

import copy
import sys

from src.eval import evaluate
from src.train import load_config, train

PLOT_PATH = "docs/math/assets/datasize-response-nll.png"


def make_datasize_configs(base_cfg, n_values, seed, eval_start=None, out_root="outputs/datasize"):
    """One cfg per (n_train, mask_prompt) arm. Pure; deep-copies base_cfg.

    eval_start defaults to max(n_values) so the eval slice is disjoint from every
    arm's train[:n] — the data-leak guard this whole study depends on.
    """
    if eval_start is None:
        eval_start = max(n_values)
    cfgs = []
    for n in n_values:
        for mask in (True, False):
            cfg = copy.deepcopy(base_cfg)
            cfg["n_train_examples"] = n
            cfg["mask_prompt"] = mask
            cfg["seed"] = seed
            cfg["eval_start"] = eval_start
            cfg["output_dir"] = f"{out_root}/n{n}-{'masked' if mask else 'unmasked'}"
            cfgs.append(cfg)
    return cfgs


def run_datasize(base_cfg, n_values, seed):
    cfgs = make_datasize_configs(base_cfg, n_values, seed)
    # floor: un-adapted base on the same disjoint eval slice (one reference point).
    floor = evaluate(cfgs[0], None)["nll"]
    print(f"[floor] base, no adapter: response-NLL={floor:.4f}")
    rows = []
    for cfg in cfgs:
        n, mask = cfg["n_train_examples"], cfg["mask_prompt"]
        tag = f"n{n}-{'masked' if mask else 'unmasked'}"
        print(f"\n===== training {tag} (eval_start={cfg['eval_start']}) =====")
        adapter = train(cfg)
        res = evaluate(cfg, adapter)
        res.update({"n": n, "mask_prompt": mask})
        rows.append(res)
        print(f"[{tag}] response-NLL={res['nll']:.4f} ppl={res['perplexity']:.2f}")
    return floor, rows


def plot_results(floor, rows, path=PLOT_PATH):
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    ns = sorted({r["n"] for r in rows})
    masked = [next(r["nll"] for r in rows if r["n"] == n and r["mask_prompt"]) for n in ns]
    unmasked = [next(r["nll"] for r in rows if r["n"] == n and not r["mask_prompt"]) for n in ns]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ns, masked, marker="o", color="#2f855a", label="masked")
    ax.plot(ns, unmasked, marker="s", color="#c05621", label="unmasked")
    ax.axhline(floor, color="gray", ls=":", label=f"base floor ({floor:.2f})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("n_train (examples, 1 epoch)")
    ax.set_ylabel("held-out response NLL")
    ax.set_title("Base model: response NLL vs data size (masked vs unmasked)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plot to {path}")


def summarize(floor, rows):
    ns = sorted({r["n"] for r in rows})
    print(f"\n=============== data-size study (base) — floor {floor:.4f} ===============")
    print(f"{'n':>6} {'masked NLL':>11} {'unmasked NLL':>13} {'Δ (unmasked-masked)':>20}")
    for n in ns:
        m = next(r["nll"] for r in rows if r["n"] == n and r["mask_prompt"])
        u = next(r["nll"] for r in rows if r["n"] == n and not r["mask_prompt"])
        print(f"{n:>6} {m:>11.4f} {u:>13.4f} {u - m:>+20.4f}")
    print("============================================================================")


def main(config_path, n_values, seed):
    base = load_config(config_path)
    floor, rows = run_datasize(base, n_values, seed)
    summarize(floor, rows)
    plot_results(floor, rows)


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/qwen_0.5b_base_lora.yaml"
    ns = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [150, 300, 600, 1200]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(config, ns, seed)
