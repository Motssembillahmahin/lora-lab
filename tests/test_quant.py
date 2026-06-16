"""
TDD tests for src/quant.py — NF4 codec, blockwise absmax, double-quant
bit accounting, and reconstruction benchmark.

These tests are intentionally RED until src/quant.py is implemented.
No bitsandbytes, no CUDA — pure numpy/torch demo.
"""

import math

import numpy as np
import pytest
import torch

from src.quant import (
    absmax_dequantize,
    absmax_quantize,
    double_quant_bits,
    nf4_dequantize,
    nf4_levels,
    nf4_quantize,
    reconstruction_benchmark,
)


# ---------------------------------------------------------------------------
# 1. NF4 levels
# ---------------------------------------------------------------------------


def test_nf4_levels_count():
    """nf4_levels() returns exactly 16 values."""
    levels = nf4_levels()
    assert len(levels) == 16


def test_nf4_levels_sorted():
    """The 16 levels are sorted in ascending order."""
    levels = nf4_levels()
    for i in range(len(levels) - 1):
        assert levels[i] <= levels[i + 1], (
            f"levels not sorted at index {i}: {levels[i]} > {levels[i+1]}"
        )


def test_nf4_levels_near_symmetric():
    """NF4 endpoints are exactly ±1.0; both negative and positive values exist.

    NF4 is NOT perfectly symmetric (the negative and positive halves are derived
    from the normal quantile function independently, with asymmetric zero handling).
    The true invariant: the codebook spans both sides and is normalized to ±1.
    """
    levels = nf4_levels()
    assert levels[0] == pytest.approx(-1.0, abs=1e-6), "min level should be -1.0"
    assert levels[-1] == pytest.approx(1.0, abs=1e-6), "max level should be 1.0"
    assert any(v < 0 for v in levels), "should have negative levels"
    assert any(v > 0 for v in levels), "should have positive levels"


def test_nf4_levels_normalized():
    """max(abs(levels)) == 1.0 (levels are normalized to [-1, 1])."""
    levels = nf4_levels()
    assert max(abs(v) for v in levels) == pytest.approx(1.0, abs=1e-6)


def test_nf4_levels_contains_zero():
    """0.0 is among the 16 NF4 levels."""
    levels = nf4_levels()
    assert any(v == pytest.approx(0.0, abs=1e-6) for v in levels), (
        f"0.0 not found in levels: {levels}"
    )


# ---------------------------------------------------------------------------
# 2. NF4 quantize / dequantize
# ---------------------------------------------------------------------------


def test_nf4_quantize_returns_indices():
    """nf4_quantize(tensor) returns integer indices in [0, 15]."""
    torch.manual_seed(42)
    x = torch.randn(64)
    indices = nf4_quantize(x)
    assert indices.dtype in (torch.int32, torch.int64, torch.uint8, torch.long), (
        f"expected integer dtype, got {indices.dtype}"
    )
    assert indices.min().item() >= 0
    assert indices.max().item() <= 15


def test_nf4_quantize_roundtrip_exact_levels():
    """Quantizing the NF4 levels themselves and dequantizing recovers them exactly."""
    levels = nf4_levels()
    x = torch.tensor(levels, dtype=torch.float32)
    indices = nf4_quantize(x)
    recovered = nf4_dequantize(indices)
    assert recovered.shape == x.shape
    for i, (orig, rec) in enumerate(zip(x.tolist(), recovered.tolist())):
        assert orig == pytest.approx(rec, abs=1e-6), (
            f"roundtrip failed at index {i}: {orig} -> {rec}"
        )


def test_nf4_dequantize_shape_preserved():
    """Output shape of nf4_dequantize matches the input index tensor shape."""
    torch.manual_seed(42)
    x = torch.randn(3, 8)
    indices = nf4_quantize(x.flatten())
    recovered = nf4_dequantize(indices)
    assert recovered.shape == x.flatten().shape


def test_nf4_quantize_gaussian_error_small():
    """On a 1000-sample N(0,1) tensor, MSE after roundtrip < 0.05.

    NF4 is designed for Gaussian weights; this is a loose sanity bound.
    The tensor is normalised to [-1, 1] before quantization (matching the
    intended use: absmax normalise -> NF4 encode -> store).
    """
    torch.manual_seed(42)
    x = torch.randn(1000)
    # Normalise to [-1, 1] as NF4 expects
    scale = x.abs().max()
    x_norm = x / scale
    indices = nf4_quantize(x_norm)
    x_hat = nf4_dequantize(indices) * scale
    mse = ((x - x_hat) ** 2).mean().item()
    assert mse < 0.05, f"NF4 roundtrip MSE too high: {mse:.4f}"


# ---------------------------------------------------------------------------
# 3. Blockwise absmax
# ---------------------------------------------------------------------------


def test_absmax_scale_per_block():
    """absmax_quantize(x, block_size=8) produces one scale per block."""
    torch.manual_seed(42)
    x = torch.randn(40)
    _q, scales = absmax_quantize(x, block_size=8)
    expected_n_blocks = math.ceil(len(x) / 8)
    assert len(scales) == expected_n_blocks, (
        f"expected {expected_n_blocks} scales, got {len(scales)}"
    )


def test_absmax_roundtrip_recovers_sign():
    """After absmax quantize+dequantize, sign is preserved for non-tiny elements.

    Small values can round to the zero quantization step, producing x_hat=0 (sign=0)
    even when x is nonzero. We only check sign agreement for elements where BOTH
    x and x_hat are clearly nonzero.
    """
    torch.manual_seed(42)
    x = torch.randn(64)
    q, scales = absmax_quantize(x, block_size=8)
    x_hat = absmax_dequantize(q, scales, block_size=8)
    both_nonzero = (x.abs() > 1e-6) & (x_hat.abs() > 1e-6)
    assert (torch.sign(x[both_nonzero]) == torch.sign(x_hat[both_nonzero])).all(), (
        "sign mismatch after absmax roundtrip"
    )


def test_absmax_scale_is_max_abs_per_block():
    """Each scale == max(|x_block|) / 7 for symmetric 4-bit quantization."""
    torch.manual_seed(42)
    x = torch.randn(32)
    block_size = 8
    _q, scales = absmax_quantize(x, block_size=block_size)
    n_blocks = math.ceil(len(x) / block_size)
    for b in range(n_blocks):
        block = x[b * block_size : (b + 1) * block_size]
        expected_scale = block.abs().max().item() / 7.0
        assert scales[b].item() == pytest.approx(expected_scale, rel=1e-5), (
            f"block {b}: scale {scales[b].item()} != expected {expected_scale}"
        )


def test_absmax_roundtrip_error_bounded():
    """Max absolute error after absmax roundtrip <= (max_abs / 7) / 2.

    This is one half of a quantization step — the theoretical worst case
    for nearest-neighbour quantization.
    """
    torch.manual_seed(42)
    x = torch.randn(64)
    block_size = 8
    q, scales = absmax_quantize(x, block_size=block_size)
    x_hat = absmax_dequantize(q, scales, block_size=block_size)
    max_abs = x.abs().max().item()
    quant_step = max_abs / 7.0
    max_err = (x - x_hat).abs().max().item()
    assert max_err <= quant_step / 2 + 1e-6, (
        f"max error {max_err:.4f} exceeds bound {quant_step/2:.4f}"
    )


# ---------------------------------------------------------------------------
# 4. Double-quant bit accounting
# ---------------------------------------------------------------------------


def test_double_quant_bits_per_param():
    """double_quant_bits returns total bits/param close to ~4.127 for standard config.

    Config: n_params=640, block_size=64, scale_block_size=256.
    Derivation sketch:
      - 640 params -> 10 blocks of 64 -> 10 fp32 scales (32 bits each)
      - Double-quant: 10 scales -> ceil(10/256)*1 super-scale (fp32) + 10 int8 values
      - base bits/param = 4.0
      - scale overhead ≈ 10*8 / 640 = 0.125 bits/param
      - super-scale overhead ≈ 32 / 640 ≈ 0.05 bits/param
      Total ≈ 4.127 (may vary slightly with exact formula; test within 0.05)
    """
    result = double_quant_bits(n_params=640, block_size=64, scale_block_size=256)
    total = result["total"]
    assert total == pytest.approx(4.127, abs=0.05), (
        f"expected ~4.127 bits/param, got {total:.4f}"
    )


def test_double_quant_breakdown():
    """double_quant_bits returns a dict with keys: base_bits, scale_bits,
    superscale_bits, total."""
    result = double_quant_bits(n_params=640, block_size=64, scale_block_size=256)
    assert isinstance(result, dict)
    for key in ("base_bits", "scale_bits", "superscale_bits", "total"):
        assert key in result, f"missing key '{key}' in result dict"
    assert result["base_bits"] == pytest.approx(4.0, abs=1e-9)


def test_double_quant_scale_overhead_decreases():
    """Total bits with double-quant < total bits without (naive 4 + fp32-scale overhead).

    Without double-quant: 4 bits/param + 32 bits per block / block_size
      = 4 + 32/64 = 4.5 bits/param.
    With double-quant, the scale is itself stored in 8 bits, lowering overhead.
    """
    result = double_quant_bits(n_params=640, block_size=64, scale_block_size=256)
    naive_bits = 4.0 + 32.0 / 64  # 4.5 bits/param (fp32 scale, not double-quantised)
    assert result["total"] < naive_bits, (
        f"double-quant total ({result['total']:.4f}) should be < naive "
        f"({naive_bits:.4f})"
    )


# ---------------------------------------------------------------------------
# 5. Reconstruction benchmark (smoke tests)
# ---------------------------------------------------------------------------


def test_benchmark_runs_and_returns_dict():
    """reconstruction_benchmark(n=256, block_size=64) returns a dict with
    keys 'nf4_mse', 'int4_mse', 'fp32_mse'."""
    result = reconstruction_benchmark(n=256, block_size=64)
    assert isinstance(result, dict)
    for key in ("nf4_mse", "int4_mse", "fp32_mse"):
        assert key in result, f"missing key '{key}' in benchmark result"


def test_benchmark_fp32_mse_is_zero():
    """fp32 roundtrip is lossless: fp32_mse == 0.0."""
    result = reconstruction_benchmark(n=256, block_size=64)
    assert result["fp32_mse"] == pytest.approx(0.0, abs=1e-12), (
        f"fp32_mse should be 0.0, got {result['fp32_mse']}"
    )


def test_benchmark_int4_worse_than_nf4_on_gaussian():
    """For N(0,1) input, nf4_mse < int4_mse (NF4 optimized for Gaussian).

    This tests a structural property of NF4, not an exact value.
    Uses a large-enough sample (n=4096) so the ordering is stable.
    """
    result = reconstruction_benchmark(n=4096, block_size=64)
    assert result["nf4_mse"] < result["int4_mse"], (
        f"expected NF4 MSE ({result['nf4_mse']:.6f}) < "
        f"INT4 MSE ({result['int4_mse']:.6f}) on Gaussian input"
    )
