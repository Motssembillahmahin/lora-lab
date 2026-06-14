"""Paired seed study: is prompt masking's edge real or noise? (ADR 0005)

For each seed, trains a masked and an unmasked adapter with that *same* seed
(shared LoRA init / shuffle, so the only difference is masking) on the *same*
examples (the all-prompt filter now fires for both — src/data.py), then evals
both with the response-only harness. Reports per-seed deltas + mean±std.

Run via the Makefile:  `make study`
"""

import statistics
import sys

from src.eval import evaluate
from src.train import load_config, train


def mean_std(values):
    """Mean and population standard deviation. Empty -> (0.0, 0.0)."""
    if not values:
        return (0.0, 0.0)
    mean = sum(values) / len(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return (mean, sd)


def run_study(base_cfg, seeds, n_train, out_root="outputs/study"):
    """Train+eval masked and unmasked adapters across seeds; return result rows."""
    rows = []
    for seed in seeds:
        for mask in (True, False):
            cfg = dict(base_cfg)
            cfg["seed"] = seed
            cfg["mask_prompt"] = mask
            cfg["n_train_examples"] = n_train
            tag = f"{'masked' if mask else 'unmasked'}-s{seed}"
            cfg["output_dir"] = f"{out_root}/{tag}"
            print(f"\n===== training {tag} =====")
            adapter = train(cfg)
            r = evaluate(cfg, adapter)
            r.update({"seed": seed, "mask_prompt": mask, "tag": tag})
            rows.append(r)
            print(f"[{tag}] response-NLL={r['nll']:.4f} ppl={r['perplexity']:.2f}")
    return rows


def summarize(rows):
    """Print masked vs unmasked means and the paired per-seed deltas."""
    masked = {r["seed"]: r["nll"] for r in rows if r["mask_prompt"]}
    unmasked = {r["seed"]: r["nll"] for r in rows if not r["mask_prompt"]}
    seeds = sorted(set(masked) & set(unmasked))

    m_mean, m_sd = mean_std([masked[s] for s in seeds])
    u_mean, u_sd = mean_std([unmasked[s] for s in seeds])
    # delta > 0 means masking is better (lower NLL).
    deltas = [unmasked[s] - masked[s] for s in seeds]
    d_mean, d_sd = mean_std(deltas)

    print("\n================ seed study summary ================")
    print(f"seeds: {seeds}")
    print(f"masked   response-NLL: {m_mean:.4f} ± {m_sd:.4f}")
    print(f"unmasked response-NLL: {u_mean:.4f} ± {u_sd:.4f}")
    for s in seeds:
        print(f"  seed {s}: delta (unmasked-masked) = {unmasked[s] - masked[s]:+.4f}")
    print(f"paired delta (unmasked - masked): {d_mean:+.4f} ± {d_sd:.4f}")
    same_sign = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
    verdict = "consistent" if same_sign else "INCONSISTENT (sign flips across seeds)"
    print(f"sign of delta across seeds: {verdict}")
    print("====================================================")
    return {"masked": (m_mean, m_sd), "unmasked": (u_mean, u_sd), "delta": (d_mean, d_sd)}


def main(config_path, seeds, n_train):
    base = load_config(config_path)
    rows = run_study(base, seeds, n_train)
    summarize(rows)


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/qwen_0.5b_lora.yaml"
    seed_list = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    main(config, seed_list, n)
