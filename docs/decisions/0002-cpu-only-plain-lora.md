# 0002 — CPU-only, plain LoRA (no local QLoRA)

## Context

The target machine is an AMD Ryzen 5 PRO 5650U laptop: 6c/12t, 14 GB RAM, no
usable CUDA/ROCm. QLoRA's defining feature — 4-bit NF4 quantization via
bitsandbytes — depends on CUDA kernels and does not run meaningfully on CPU.

## Decision

- Run **plain LoRA** locally on a small model (Qwen2.5-0.5B-Instruct) in fp32.
- Treat **QLoRA as a study topic**, not a local training method. We will explain
  its math (NF4, double quantization, paged optimizers) in docs/math/04 and may
  demonstrate the quantization ideas with small numpy/torch experiments, but we
  will not pretend to run 4-bit fine-tuning on this CPU.
- If real QLoRA practice is wanted later, the path is a free Colab T4 GPU,
  documented as an option rather than a local default.

## Consequences

- Local runs stay honest about hardware limits.
- The learning still covers QLoRA conceptually and mechanically.
- Model size and dataset slices stay small so runs finish in minutes.
