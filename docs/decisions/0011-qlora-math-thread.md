# ADR 0011 — QLoRA math thread as CPU-runnable codec + math doc (no training)

- Status: Accepted
- Date: 2026-06-16
- Relates to: ADR 0002 (CPU-only constraint), docs/math/04-qlora-quantization.md,
  src/quant.py

## Context

The original Session 0 plan had four learning threads: math derivation, training
pipeline, rank/alpha experiments, and QLoRA quantization. The first three were
completed across Sessions 1–11. The QLoRA thread was deferred until now.

The hard constraint (no GPU, no bitsandbytes) means actual 4-bit training is
impossible on this machine. The savings QLoRA delivers — storing weights in NF4
and dequantizing on-the-fly inside the matmul kernel — depend on CUDA fused
kernels that bitsandbytes provides. On CPU the only honest option is to
dequantize the full weight matrix before the matmul, which costs *more* RAM than
keeping fp32 around (temporary dequant buffer + original int4), defeating the
point.

But the mathematical content — NF4 codec, blockwise absmax, double quantization,
the memory equation — is fully learnable and demonstrable without a GPU. The
codec is pure arithmetic on float tensors; you can implement it, test it, and
measure its reconstruction quality on real model weights locally.

## Decision

Implement the QLoRA thread as three artifacts:

1. `docs/math/04-qlora-quantization.md` — full mathematical treatment:
   uniform quantization baseline → NF4 derivation (equal-mass quantile bins for
   Gaussian weights, minimising expected squared error in the Lloyd-Max sense) →
   double-quant bit accounting (4.127 bits/param total) → QLoRA assembly
   (dequant-on-the-fly forward pass, gradient flow through the frozen quantized
   weight, memory equation: 84 → 14.6 → 4.1 GB for a 7B model at fp16/NF4/NF4+dq)
   → paged optimizers → explicit CPU caveat. 703 lines.

2. `src/quant.py` — TDD'd NF4 codec: `nf4_levels()`, `nf4_quantize()`,
   `nf4_dequantize()`, `absmax_quantize()`, `absmax_dequantize()`,
   `double_quant_bits()`, `reconstruction_benchmark()`. 19 tests written first
   (RED = ImportError), then implemented to GREEN.

3. `src/quant_demo.py` — demo script (invoked via `make quant-demo`): runs the
   benchmark on synthetic Gaussian data AND a real Qwen2.5-0.5B q_proj weight,
   prints NF4 vs INT4 reconstruction MSE, plus a double-quant bit accounting
   table.

The CPU caveat is made explicit in the doc and the demo output: bitsandbytes
CUDA kernels provide the fused dequant-on-the-fly matmul; on CPU you would have
to dequantize the whole weight matrix before the matmul, defeating the memory
saving. We study and verify the codec math; we do not claim to run QLoRA training
locally.

## Alternatives considered

- **Skip QLoRA entirely** (it's CPU-incompatible for training) — rejected: the
  information-theoretic argument for NF4 (equal-mass bins on a Gaussian source)
  and the memory equation are core learning goals, not just implementation
  details. You can understand why QLoRA works without being able to run it.

- **Use bitsandbytes on CPU** (it has partial CPU support for quantize/dequantize)
  — rejected: adds a GPU-library dependency to a deliberately CPU-clean project,
  and the math is cleaner and more instructive written in plain torch. The codec
  is ~50 lines; the library would obscure more than it reveals.

## Consequences

- 19 new tests (53 total in suite); all green, no regressions to existing tests.
- `make quant-demo` gives a live comparison of NF4 vs INT4 reconstruction error
  on real Qwen2.5-0.5B weights, reproducible on CPU in seconds.
- The CPU limit is clearly documented at the math level and in the demo output —
  no future confusion about whether QLoRA training is in scope for this machine.
- docs/math/04 completes the four-thread plan from Session 0.
