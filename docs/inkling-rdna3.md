# Inkling 2 Mini on RDNA3: notes for this branch

This branch carries experimental patches on top of `b12600` that make Inkling 2 Mini
(`inkling2_loop`) faster and able to run longer contexts on an AMD RX 7900 XTX through
HIP/ROCm. It is not an upstream PR. It is shared so people can see the numbers, try
it, and reuse whatever is useful.

Tested only on: RX 7900 XTX (gfx1100, 24 GB), ROCm 7.2.1, Ubuntu 24.04 container,
`ggml-org/Inkling-2-Mini-GGUF` IQ4_NL with `mmproj-F16`. **Not tested on NVIDIA** or on
any other AMD card. The HIP code lives in `ggml-cuda`, so it also compiles for CUDA.

## Results

| | b12600 | this branch |
|---|---|---|
| `llama-bench` tg128 | 31.9 t/s | 64.0 t/s |
| tg128 at 16k depth | 30.9 | 63.5 |
| pp4096 | 854 | 1802 |
| pp4096 at 16k depth | 492 | 1467 |
| Largest context that fit with the vision projector | 131072 (q8_0 K/V) | 393216 (q4_0 K/V) |
| wikitext-2 PPL (ctx 2048, q8_0 K/V) | 5.9465 | 5.9472 |

At 393216 context with q4_0 K/V, a 382k-token prompt with five 4K images at the end
retrieved 10 of 10 facts spread through the text, at a 23.2 GB VRAM peak. Server flags
used: `-fa on -ub 256 --cache-type-k q4_0 --cache-type-v q4_0 --image-max-tokens 1024`.

## What changed

1. `model : inkling2_loop re-share the loop slots after the scale pass`: a backport of
   upstream PR #32704 (Daniel Han). It is not exercised by IQ4_NL, so it is unverified
   here.
2. Decode speed:
   - Fused HIP kernels for Inkling's small ops (L2 clip, SITU, loop controller,
     depth-attention mix, short conv, concat+state store, concat+rmsnorm, MoE router
     norm). They cut decode from ~7,650 to ~3,000 kernels per token.
   - A new `GGML_OP_INK_REL_BIAS`.
   - Graph memoization in `inkling2.cpp`.
   - A retuned RDNA3 mmvq table for Q8_0/IQ4_NL single-column matvecs.
   - A q8_1 input cache.
3. Long context:
   - Flash attention computes Inkling's relative-position bias in-kernel from per-cell
     positions, instead of a dense per-head F16 mask.
   - Long prompt batches run attention per K/V head group.
   - Fixes from a code review.
4. Tests: `test-backend-ops` cases for every new kernel and fused path (CPU vs GPU),
   plus `test-ink-rel-bias`, which compares the bias bit for bit against the original
   unfused chain up to positions past 1M. 10,000+ cases pass on gfx1100.

## Scope: what can affect other models or cards

- Inkling only: the model code, `GGML_OP_INK_REL_BIAS`, and the flash-attention bias
  path, which is only taken when an op carries the bias.
- RDNA3 only: the mmvq table change, limited to Q8_0/IQ4_NL weights and single-token
  matvecs.
- **Every model, every CUDA/HIP card:** the q8_1 input cache on single-token matvecs.
  Disable it with `GGML_CUDA_NO_Q8_CACHE=1`.
- **Any model whose graph matches the patterns:** the fusion matchers. They are shaped
  for Inkling but run on every graph. Disable them with `GGML_INK_NO_FUSE=1`.
