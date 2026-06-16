"""NF4 codec demo: compare reconstruction error on synthetic Gaussian and real Qwen weights.

Run via:  make quant-demo
Or:       uv run python -m src.quant_demo [n] [block_size]
"""

import sys

import torch

from src.quant import (
    absmax_dequantize,
    absmax_quantize,
    double_quant_bits,
    nf4_dequantize,
    nf4_quantize,
    reconstruction_benchmark,
)


def _nf4_roundtrip_mse(x: torch.Tensor, block_size: int) -> float:
    import math

    n = x.shape[0]
    n_blocks = math.ceil(n / block_size)
    padded_len = n_blocks * block_size
    padded = torch.zeros(padded_len)
    padded[:n] = x
    blocks = padded.reshape(n_blocks, block_size)
    scales = blocks.abs().max(dim=1).values
    safe_scales = scales.clone()
    safe_scales[safe_scales == 0.0] = 1.0
    x_norm = blocks / safe_scales.unsqueeze(1)
    indices = nf4_quantize(x_norm.flatten())
    x_hat_norm = nf4_dequantize(indices).reshape(n_blocks, block_size)
    x_hat = (x_hat_norm * safe_scales.unsqueeze(1)).flatten()[:n]
    return ((x - x_hat) ** 2).mean().item()


def _int4_roundtrip_mse(x: torch.Tensor, block_size: int) -> float:
    q, scales = absmax_quantize(x, block_size)
    x_hat = absmax_dequantize(q, scales, block_size)
    return ((x - x_hat) ** 2).mean().item()


def run_synthetic(n: int = 4096, block_size: int = 64, seed: int = 42):
    print(f"\n=== Synthetic N(0,1) weights  (n={n}, block_size={block_size}) ===")
    result = reconstruction_benchmark(n=n, block_size=block_size, seed=seed)
    nf4_mse = result["nf4_mse"]
    int4_mse = result["int4_mse"]
    print(f"  fp32 MSE  : {result['fp32_mse']:.6f}  (lossless reference)")
    print(f"  NF4  MSE  : {nf4_mse:.6f}")
    print(f"  INT4 MSE  : {int4_mse:.6f}")
    ratio = int4_mse / nf4_mse if nf4_mse > 0 else float("inf")
    print(f"  INT4/NF4  : {ratio:.2f}×  (NF4 wins on Gaussian; this should be > 1)")


def run_real_weights(block_size: int = 64):
    print("\n=== Real Qwen2.5-0.5B  q_proj weights ===")
    print("  Loading model (CPU, no GPU)…")
    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32
        )
    except Exception as e:
        print(f"  Could not load Qwen2.5-0.5B: {e}")
        print("  Skipping real-weight benchmark.")
        return

    # First transformer layer q_proj
    w = model.model.layers[0].self_attn.q_proj.weight.detach().float()
    print(f"  q_proj shape: {tuple(w.shape)}  ({w.numel():,} params)")
    flat = w.flatten()

    nf4_mse = _nf4_roundtrip_mse(flat, block_size)
    int4_mse = _int4_roundtrip_mse(flat, block_size)
    fp32_mse = 0.0

    print(f"  fp32 MSE  : {fp32_mse:.6f}  (lossless reference)")
    print(f"  NF4  MSE  : {nf4_mse:.6f}")
    print(f"  INT4 MSE  : {int4_mse:.6f}")
    ratio = int4_mse / nf4_mse if nf4_mse > 0 else float("inf")
    print(f"  INT4/NF4  : {ratio:.2f}×")

    mean = flat.mean().item()
    std = flat.std().item()
    print(f"  weight stats: mean={mean:.4f}, std={std:.4f}  (pretrained ≈ N(0,σ))")

    del model


def run_bit_accounting():
    print("\n=== Double-quant bit accounting (n=65536 params) ===")
    print(f"  {'Config':35s} {'base':>6} {'scale':>8} {'super':>8} {'total':>8}")

    # Pre-double-quant baseline: fp16 first-level scales (no second quantization)
    n = 65536
    import math
    for bs, label in [(64, "fp16 scales, B=64 (no dq)"), (32, "fp16 scales, B=32 (no dq)")]:
        n_blocks = math.ceil(n / bs)
        scale_bits = (n_blocks * 16) / n  # fp16 = 16 bits
        total = 4.0 + scale_bits
        print(f"  {label:35s} {4.0:>6.3f} {scale_bits:>8.4f} {'---':>8} {total:>8.4f}")

    print(f"  {'---':35s}")

    # With double-quant: INT8 first-level scales + fp32 super-scales
    configs = [
        ("INT8 scales, B=64, B₂=256 (paper)", 65536, 64, 256),
        ("INT8 scales, B=32, B₂=256", 65536, 32, 256),
        ("INT8 scales, B=64, B₂=128", 65536, 64, 128),
    ]
    for label, n, bs, sbs in configs:
        from src.quant import double_quant_bits

        r = double_quant_bits(n, bs, sbs)
        print(
            f"  {label:35s} {r['base_bits']:>6.3f} {r['scale_bits']:>8.4f} "
            f"{r['superscale_bits']:>8.4f} {r['total']:>8.4f}"
        )
    print("  (scale = INT8 scales; super = fp32 super-scales)")


def main(n: int = 4096, block_size: int = 64):
    run_synthetic(n=n, block_size=block_size)
    run_real_weights(block_size=block_size)
    run_bit_accounting()
    print()


if __name__ == "__main__":
    _n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    _b = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    main(_n, _b)
