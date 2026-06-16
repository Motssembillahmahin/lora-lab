"""NF4 codec, blockwise absmax quantization, and double-quant bit accounting.

Pure numpy/torch demo — no bitsandbytes, no CUDA. Demonstrates the quantization
math behind QLoRA (Dettmers et al., 2023) on CPU.

Real QLoRA requires bitsandbytes + CUDA kernels for dequant-on-the-fly during
the forward pass. This module implements the codec math so the numbers are
auditable without GPU hardware.
"""

import math

import torch

# NF4 codebook — 16 quantile levels of N(0,1) normalized to [-1, 1].
# Equal-mass bins: levels = Φ⁻¹(k/16), k = 1/16..15/16, plus explicit 0.0 and
# ±1.0 endpoints, normalized so max(|level|) = 1.
# Source: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," 2023.
_NF4 = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]

_NF4_TENSOR = torch.tensor(_NF4, dtype=torch.float32)


def nf4_levels() -> list[float]:
    """Return the 16 NF4 codebook levels."""
    return list(_NF4)


def nf4_quantize(x: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor quantize a float tensor to NF4 indices in [0, 15].

    x should be pre-normalized to [-1, 1] (apply absmax scaling first).
    """
    flat = x.flatten().float()
    dists = (flat.unsqueeze(1) - _NF4_TENSOR.unsqueeze(0)).abs()  # (n, 16)
    return dists.argmin(dim=1).to(torch.long)


def nf4_dequantize(indices: torch.Tensor) -> torch.Tensor:
    """Map NF4 indices back to float codebook values."""
    return _NF4_TENSOR[indices.long()]


def absmax_quantize(x: torch.Tensor, block_size: int = 64):
    """Blockwise absmax quantization to symmetric INT4 (stored as int32, range [-7, 7]).

    Each block of block_size elements is independently scaled by max(|block|) / 7,
    capping the range at ±7 quantization steps.

    Returns:
        quantized: int32 tensor of length len(x), values in [-7, 7]
        scales:    float32 tensor of length ceil(len(x) / block_size)
    """
    flat = x.flatten().float()
    n = flat.shape[0]
    n_blocks = math.ceil(n / block_size)

    padded = torch.zeros(n_blocks * block_size, dtype=torch.float32)
    padded[:n] = flat
    blocks = padded.reshape(n_blocks, block_size)

    max_abs = blocks.abs().max(dim=1).values  # (n_blocks,)
    scales = max_abs / 7.0

    safe_scales = scales.clone()
    safe_scales[safe_scales == 0.0] = 1.0  # avoid div-by-zero on all-zero blocks

    normalized = blocks / safe_scales.unsqueeze(1)
    quantized = normalized.round().clamp(-7, 7).to(torch.int32)

    return quantized.flatten()[:n], scales


def absmax_dequantize(q: torch.Tensor, scales: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Reconstruct float values from blockwise absmax INT4 quantized data."""
    n = q.shape[0]
    n_blocks = math.ceil(n / block_size)

    padded = torch.zeros(n_blocks * block_size, dtype=torch.int32)
    padded[:n] = q
    blocks = padded.reshape(n_blocks, block_size).float()

    dequantized = blocks * scales.unsqueeze(1)
    return dequantized.flatten()[:n]


def double_quant_bits(
    n_params: int, block_size: int = 64, scale_block_size: int = 256
) -> dict:
    """Bits-per-parameter accounting for NF4 + double quantization.

    Level 1 scales: INT8, one per block_size weights.
    Level 2 super-scales: fp32, one per scale_block_size level-1 scales.

    Returns dict: base_bits, scale_bits, superscale_bits, total.
    """
    n_blocks = math.ceil(n_params / block_size)
    n_superblocks = math.ceil(n_blocks / scale_block_size)

    base_bits = 4.0
    scale_bits = (n_blocks * 8) / n_params          # INT8 per block
    superscale_bits = (n_superblocks * 32) / n_params  # fp32 per super-block

    return {
        "base_bits": base_bits,
        "scale_bits": scale_bits,
        "superscale_bits": superscale_bits,
        "total": base_bits + scale_bits + superscale_bits,
    }


def reconstruction_benchmark(n: int = 4096, block_size: int = 64, seed: int = 42) -> dict:
    """MSE comparison: NF4 vs uniform INT4 vs fp32 on N(0,1) input.

    NF4 pipeline: absmax-normalize each block to [-1,1] → NF4 codec → rescale.
    INT4 pipeline: absmax_quantize / absmax_dequantize (uniform grid ±7 steps).
    fp32: lossless reference (MSE = 0.0).

    Returns dict: nf4_mse, int4_mse, fp32_mse.
    """
    torch.manual_seed(seed)
    x = torch.randn(n)

    # --- NF4 ---
    n_blocks = math.ceil(n / block_size)
    padded_len = n_blocks * block_size
    padded = torch.zeros(padded_len)
    padded[:n] = x
    blocks = padded.reshape(n_blocks, block_size)

    block_scales = blocks.abs().max(dim=1).values  # max|x| per block, shape (n_blocks,)
    safe_scales = block_scales.clone()
    safe_scales[safe_scales == 0.0] = 1.0

    x_norm_blocks = blocks / safe_scales.unsqueeze(1)  # normalize each block to [-1, 1]
    indices = nf4_quantize(x_norm_blocks.flatten())
    x_nf4_norm = nf4_dequantize(indices).reshape(n_blocks, block_size)
    x_nf4 = (x_nf4_norm * safe_scales.unsqueeze(1)).flatten()[:n]
    nf4_mse = ((x - x_nf4) ** 2).mean().item()

    # --- Uniform INT4 ---
    q_int4, scales_int4 = absmax_quantize(x, block_size)
    x_int4 = absmax_dequantize(q_int4, scales_int4, block_size)
    int4_mse = ((x - x_int4) ** 2).mean().item()

    return {"nf4_mse": nf4_mse, "int4_mse": int4_mse, "fp32_mse": 0.0}
