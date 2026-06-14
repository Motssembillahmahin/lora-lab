"""Mechanism probe for Run 008 (ADR 0009).

Run 008 found prompt masking helps the instruct base but ~not the base LM.
Hypothesis (math/03 §4): a base model uses the prompt tokens to *learn the ChatML
format*, so unmasked training spends real signal on the prompt — which masking
discards. If true, unmasked training should drop the model's loss on PROMPT
tokens a lot for the base, but barely for the already-formatted instruct model.

This trains an UNMASKED adapter on each track, then measures how much it lowers
prompt-NLL vs response-NLL (vs the un-adapted floor) on the held-out slice.

Run via the Makefile:  `make mechanism`
"""

import sys

from src.eval import evaluate
from src.train import load_config, train

TRACKS = {
    "instruct": "configs/qwen_0.5b_lora.yaml",
    "base": "configs/qwen_0.5b_base_lora.yaml",
}


def run_mechanism(seed=0, n_train=150):
    rows = []
    for track, cfg_path in TRACKS.items():
        cfg = load_config(cfg_path)
        cfg["mask_prompt"] = False  # unmasked: prompt tokens ARE in the training loss
        cfg["seed"] = seed
        cfg["n_train_examples"] = n_train
        cfg["output_dir"] = f"outputs/mech/{track}-unmasked"
        print(f"\n===== training {track} unmasked (seed={seed}, n={n_train}) =====")
        adapter = train(cfg)
        for target in ("prompt", "response"):
            floor = evaluate(cfg, None, target=target)["nll"]
            tuned = evaluate(cfg, adapter, target=target)["nll"]
            rows.append(
                {"track": track, "target": target, "floor": floor, "tuned": tuned,
                 "delta": floor - tuned}
            )
            print(f"[{track}/{target}] floor={floor:.4f} tuned={tuned:.4f} drop={floor - tuned:+.4f}")
    return rows


def summarize(rows):
    def drop(track, target):
        return next(r["delta"] for r in rows if r["track"] == track and r["target"] == target)

    print("\n=========== mechanism: NLL drop from UNMASKED training ===========")
    print(f"{'track':>9} {'target':>9} {'floor':>8} {'tuned':>8} {'drop (floor-tuned)':>19}")
    for r in rows:
        print(
            f"{r['track']:>9} {r['target']:>9} {r['floor']:>8.4f} {r['tuned']:>8.4f} "
            f"{r['delta']:>+19.4f}"
        )
    print("\nprompt-NLL drop (does unmasked training teach the format?):")
    print(f"  base    : {drop('base', 'prompt'):+.4f}")
    print(f"  instruct: {drop('instruct', 'prompt'):+.4f}")
    print("mechanism holds if the base's prompt-NLL drop >> the instruct's.")
    print("==================================================================")


def main(seed, n_train):
    summarize(run_mechanism(seed, n_train))


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    main(seed, n)
