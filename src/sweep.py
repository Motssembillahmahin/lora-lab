"""Rank sweep: train at several r (alpha=2r), eval each, plot perplexity vs r.

Holds alpha/r = 2 fixed so the curve isolates rank *capacity* rather than
effective-LR shrinkage (docs/math/02 §3). Single seed, masked, on a small slice
(the seed study found seed variance ≈ ±0.0002, so one seed shows the trend).

Run via the Makefile:  `make sweep`   (override RANKS=2,4,8,16,32 SEED=0 N=150)
"""

import copy
import sys

from src.eval import evaluate
from src.train import load_config, train

PLOT_PATH = "docs/math/assets/02-rank-sweep.png"


def make_sweep_configs(base_cfg, r_values, seed, n_train, use_rslora=False, out_root="outputs/sweep"):
    """One cfg per rank, with alpha=2r and a distinct output dir. Pure; deep-copies
    base_cfg so the caller's nested lora dict is never mutated.

    use_rslora=True switches the scaling from alpha/r to alpha/sqrt(r) (the control
    that disentangles a real plateau from the 1/sqrt(r) under-scaling — ADR 0006 §3).
    """
    tag = "rslora" if use_rslora else "vanilla"
    cfgs = []
    for r in r_values:
        cfg = copy.deepcopy(base_cfg)
        cfg["lora"]["r"] = r
        cfg["lora"]["alpha"] = 2 * r
        cfg["use_rslora"] = use_rslora
        cfg["mask_prompt"] = True
        cfg["seed"] = seed
        cfg["n_train_examples"] = n_train
        cfg["output_dir"] = f"{out_root}/{tag}-r{r}"
        cfgs.append(cfg)
    return cfgs


def run_sweep(base_cfg, r_values, seed, n_train, use_rslora=False):
    rows = []
    for cfg in make_sweep_configs(base_cfg, r_values, seed, n_train, use_rslora=use_rslora):
        r, alpha = cfg["lora"]["r"], cfg["lora"]["alpha"]
        scale = "α/√r" if use_rslora else "α/r"
        print(f"\n===== training r={r} (alpha={alpha}, scaling={scale}) =====")
        adapter = train(cfg)
        res = evaluate(cfg, adapter)
        res.update({"r": r, "alpha": alpha})
        rows.append(res)
        print(f"[r={r}] response-NLL={res['response_nll']:.4f} ppl={res['perplexity']:.2f}")
    return rows


def plot_results(rows, path=PLOT_PATH):
    import os

    import matplotlib

    matplotlib.use("Agg")  # headless / CPU box
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    rs = [row["r"] for row in rows]
    ppls = [row["perplexity"] for row in rows]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rs, ppls, marker="o", color="#2b6cb0")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rs)
    ax.set_xticklabels([str(r) for r in rs])
    ax.set_xlabel("LoRA rank r   (α = 2r, so α/r = 2 fixed)")
    ax.set_ylabel("held-out response perplexity")
    ax.set_title("Rank sweep — eval perplexity vs r")
    ax.grid(True, alpha=0.3)
    for r, p in zip(rs, ppls):
        ax.annotate(f"{p:.2f}", (r, p), textcoords="offset points", xytext=(0, 8), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"Saved plot to {path}")
    return path


def summarize(rows):
    print("\n================ rank sweep summary ================")
    print(f"{'r':>4} {'alpha':>6} {'response-NLL':>13} {'perplexity':>11}")
    for row in rows:
        print(f"{row['r']:>4} {row['alpha']:>6} {row['response_nll']:>13.4f} {row['perplexity']:>11.2f}")
    best = min(rows, key=lambda row: row["perplexity"])
    print(f"best perplexity: r={best['r']} ppl={best['perplexity']:.2f}")
    print("====================================================")


def main(config_path, r_values, seed, n_train, use_rslora=False):
    base = load_config(config_path)
    rows = run_sweep(base, r_values, seed, n_train, use_rslora=use_rslora)
    summarize(rows)
    path = PLOT_PATH.replace(".png", "-rslora.png") if use_rslora else PLOT_PATH
    plot_results(rows, path)


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/qwen_0.5b_lora.yaml"
    ranks = [int(r) for r in sys.argv[2].split(",")] if len(sys.argv) > 2 else [2, 4, 8, 16, 32]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 150
    rslora = str(sys.argv[5]).lower() in ("1", "true", "yes") if len(sys.argv) > 5 else False
    main(config, ranks, seed, n, use_rslora=rslora)
