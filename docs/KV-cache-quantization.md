# KV Cache Quantization with TurboQuant

TurboQuant adds five runtime-only KV cache quantization types that compress
the K/V cache far beyond the standard `q8_0` while keeping decode quality via a
Walsh-Hadamard rotation (WHT) that Gaussianizes the cache vectors before
quantization:

| Type               | Enum                       | Size            | Compression vs f16 |
|--------------------|----------------------------|-----------------|--------------------|
| `turbo2`           | `GGML_TYPE_TURBO2_0` (43)  | 2.125 bits/value | 7.5x              |
| `turbo3`           | `GGML_TYPE_TURBO3_0` (44)  | 3.125 bits/value | 5.1x              |
| `turbo4`           | `GGML_TYPE_TURBO4_0` (47)  | 4.125 bits/value | 3.9x              |
| `turbo5`           | `GGML_TYPE_TURBO5_0` (51)  | 5.125 bits/value | 3.1x              |
| `turbo6`           | `GGML_TYPE_TURBO6_0` (52)  | 6.125 bits/value | 2.6x              |

These are KV-cache-only types: they are never stored in model files. The
corresponding model-weight quantization types are `TQ3_1S` (45) and `TQ4_1S`
(46) - 3/4-bit WHT-rotated Lloyd-Max quantization, block size 32, exposed in
`llama-quantize` as `TQ3_1S` / `TQ4_1S`.

## Usage

```bash
llama-cli -m model.gguf -c 8192 -ngl 99 \
    --cache-type-k q8_0 --cache-type-v turbo3
```

The flag parser accepts any combination of `f32`, `f16`, `bf16`, `q8_0`,
`q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1` and `turbo2` to `turbo6` for K and V.
Which combinations have backend kernels is narrower and backend-dependent;
`turbo5` and `turbo6` support a smaller set of pairs, listed below. A
quantized V cache with a `q8_0` K cache is the common configuration.

On Metal, flash attention accepts exactly three mixed pairs: turbo K + turbo V,
`q8_0` K + turbo V, and turbo K + `q8_0` V, where turbo is `turbo2/3/4`. Any
other mixed pair aborts in `ggml_metal_op_flash_attn_ext`.

Turbo KV types require flash attention. If a turbo cache type is requested
with flash attention disabled, it is enabled automatically (a warning is
printed) before the quantized-V check runs, so a turbo V cache never reaches
that error. A quantized V cache of another type with flash attention explicitly
disabled is an error, matching upstream behavior.

The same flags work in `llama-server`, `llama-bench`, and `llama-perplexity`.

## turbo5 and turbo6

`turbo5` and `turbo6` sit between `q8_0` and `turbo4`. They use the same WHT
rotation as the other turbo types plus a Lloyd-Max codebook for the rotated
N(0, 1/128) distribution and a per-block corrected L2 norm:

| Type     | Block             | Layout                                                        | Centroids |
|----------|-------------------|---------------------------------------------------------------|-----------|
| `turbo5` | 128 values / 82 B | `f16` norm + 64 B of 4-bit magnitudes + 16 B of sign bits      | 32 (antisymmetric) |
| `turbo6` | 128 values / 98 B | `f16` norm + 64 B of 4-bit low bits + 32 B of 2-bit high bits | 64        |

The `turbo5` codebook is antisymmetric, so it stores a magnitude index and a
sign bit per value instead of a 5-bit code, which keeps the decode to a single
byte-permute lookup in the CUDA kernels.

Like `turbo2`/`turbo3`/`turbo4`, a symmetric `turbo5` or `turbo6` cache has its
K rewritten to `q8_0` by the auto-asymmetric rule on models with GQA ratio 6 or
more. On Qwen2.5-0.5B (GQA 7:1) `turbo6` K gives about 22 PPL and `turbo5` K
about 32 against 8.0 for `f16`, while `turbo6` V is within noise of `f16`.

Kernels exist on the CPU and CUDA backends (HIP builds the same CUDA kernels).
Metal, Vulkan and SYCL have no `SET_ROWS` kernel for these types, so on those
backends context creation fails with an error unless the KV cache is kept in
host memory (`-nkvo`). `GET_ROWS` is not implemented for any turbo type. The
blocks hold 128 values; head dims that are not a multiple of 128 are zero-padded
up to one, as for the other turbo types.

CUDA flash attention (head dims 128 and 256 after padding) covers these K/V pairs
in a default build:

- `turbo5` or `turbo6` K with V of the same type, `turbo3`, `turbo4`, `q8_0` or `f16`
- `turbo6` K with `turbo5` V
- `q8_0` K with `turbo5` or `turbo6` V (the auto-asymmetric result)

`-DGGML_CUDA_FA_ALL_QUANTS=ON` adds the remaining pairs with `f16`, `q8_0`,
`turbo2`, `turbo3` and `turbo4`. The fused GQA-packed MMA kernel, which is the fast
path for long contexts, additionally covers `turbo5`/`turbo5`, `turbo6`/`turbo6`,
`turbo5`/`turbo3`, `turbo6`/`turbo3` (head dims 128 and 256) and `turbo5`/`turbo4`,
`turbo6`/`turbo4`, `turbo6`/`turbo5` (head dim 256).

## Model-specific quality

Models with attention sinks can be unusually sensitive to K-cache quantization. GPT-OSS is a known case: even `q8_0` K changes the output distribution substantially, and lower-bit K types degrade it further despite normal codec and kernel accuracy. Use `f16` K for GPT-OSS and other sink-heavy models. Validate a quantized V cache separately against an `f16` K/V baseline before deploying it.

Short output samples are not a sufficient quality check for this class of model because the text can remain fluent while token probabilities move significantly. Use `llama-perplexity --kl-divergence` or an equivalent logit comparison when selecting cache types.

## Rotation

K and V vectors are rotated by a fixed 128x128 orthonormal Walsh-Hadamard
matrix before quantization and inverse-rotated after dequantization. Head
dimensions that are not multiples of 128 are zero-padded to the next multiple
of 128 for turbo types. MLA models have no separate V cache (V is a view of
K), so V rotation and padding are skipped for them, and the K and V cache types
must be identical.

## Environment knobs

This table is the source of truth for these variables. If it and the code ever
disagree, the code wins and both need fixing in the same PR.

| Variable                        | Default | Effect                                                          |
|---------------------------------|---------|-----------------------------------------------------------------|
| `TURBO_LAYER_ADAPTIVE`          | auto    | Layer-adaptive KV precision. Unset: Boundary V (mode 7) auto-enables when V is `turbo2` and the model has 8+ layers; `0` opts out |
| `TURBO_AUTO_ASYMMETRIC`         | `1`     | Auto-select asymmetric K/V types for large-GQA models (`0` disables) |
| `TURBO_SPARSE_V`                | `1`     | Metal: sparse-V dequant skip for `turbo3` V (`0` disables)     |
| `GGML_TQ_NATIVE`                | unset   | CUDA: `1` keeps `TQ4_1S` weights native and uses the fused TQ kernels instead of converting them to `q8_0` at load; decode about 30% faster and weight VRAM about 1.7x smaller, prefill about 2x slower |
| `GGML_TQ_MMQ`                   | unset   | CUDA: `1` opts into the native int8 MMQ path for TQ3_1S/TQ4_1S prefill instead of the dequant-to-f16 + cuBLAS fallback. AMD only: the TQ MMQ config exists for RDNA4 (TQ3_1S) and for CDNA/RDNA2/RDNA3/RDNA4 (TQ4_1S). Off by default. TQ4_1S also needs `GGML_TQ_NATIVE=1`, because otherwise the weights are converted to q8_0 at load and this path is not reached; TQ3_1S is never converted, so it needs only this knob |
| `GGML_CUDA_FUSE_CHAIN`          | unset   | CUDA: `0` disables elementwise chain fusion, so ADD/MUL/DIV/SCALE/CLAMP/SQR and the SILU/SIGMOID/SOFTPLUS/GELU/RELU/NEG unaries run as separate kernels |
| `GGML_CUDA_Q8CACHE`             | unset   | CUDA: `0` disables the per-graph shared `q8_1` activation cache in `mmvq`, so a repeated activation is quantized again per use |
| `LLAMA_ATTN_ROT_K_OVERRIDE` / `LLAMA_ATTN_ROT_V_OVERRIDE` | off | Enable upstream #21038 attention rotation for K / V; needs a quantized type and a head dim that is a multiple of 64 |
| `LLAMA_ATTN_ROT_K_NROT`         | `64`    | Rotation tile size for the optional upstream K path (`0` = upstream's largest power-of-two divisor of the head dim) |
| `LLAMA_ATTN_ROT_DISABLE`        | `0`     | Hard lock-out: force rotation off on both sides (`1` disables) |

`TURBO_LAYER_ADAPTIVE` is not simply `0`. With the variable unset, Boundary V
(mode 7) turns on automatically when the V cache type is `turbo2` and the model
has 8 or more layers: the first two and last two V layers become `q8_0` and
every other V layer becomes `turbo2`. Setting the variable explicitly selects
the mode; modes 5, 6 and 7 assign the V type per layer regardless of `-ctv`,
so mode 7 on a `-ctv turbo3` or `-ctv turbo4` run downgrades the non-edge V
layers to `turbo2`. The full mode legend is in `src/llama-kv-cache.cpp`
(`kv_adaptive_mode`).

Upstream attention rotation is off by default: TurboQuant manages rotation
itself (the WHT applied at cache write is equivalent and interacts with the
cache types). `LLAMA_ATTN_ROT_*` only affects the optional upstream rotation
path for models that benefit from it.

## Model-weight quantization (TQ3_1S / TQ4_1S)

```bash
llama-quantize model-f16.gguf model-tq4.gguf TQ4_1S
```

`TQ3_1S` and `TQ4_1S` are first-class weight types with CPU, CUDA/HIP (warp
cooperative mmvq), Metal, and Vulkan kernels. ggml-sycl has no TQ kernels, and
its generic dequantize path leaves TQ weights in the WHT-rotated domain, so TQ
models are unsupported on SYCL. For TQ MoE weights, the small-batch
`MUL_MAT_ID` path routes experts on device and stays CUDA-graph capturable;
only the large-batch fallback dequantizes to f16 and synchronizes the stream.

## Context checkpoint offload and the disk prompt cache

The server adds a disk tier for two kinds of KV data so long contexts keep
their rollback and reuse state without pinning it in RAM:

- **Context checkpoints** (`--ctx-checkpoints N`, default 32) snapshot the
  evicted KV window of the context so a task restart can roll back mid-prompt
  (for example after a BPE-boundary retokenization of the prompt tail) instead
  of re-prefilling. The snapshot payload is spilled to a per-slot `.ccn` file
  in the disk cache directory; only the lightweight `n_tokens`/`pos_min`/`
  pos_max` metadata stays in RAM. Checkpoints are created only for completion
  tasks on models that need them (SWA, or contexts without partial sequence
  removal), and are erased when invalidated.
- **Prompt cache states** (`--cache-ram N` + `--cache-disk PATH`) persist the
  prompt KV state so a fresh process can restore the prefix. The in-RAM pool
  is capped by `--cache-ram` (MiB); there is a separate disk pool capped by
  `--cache-disk-max` (MiB, entries survive eviction from the RAM pool). A state
  larger than the RAM cap is *written through* to a `.dpc` file instead of
  held in RAM: the write goes to the background disk thread and the RAM is
  freed immediately, so a 500 MiB state at a 64 MiB RAM cap costs no decode
  stalls.

All disk I/O for `.dpc` and `.ccn` payloads runs on a single background writer
thread. The server enqueues jobs (write-through states, checkpoint spills, and
the `.ckpt` sidecar written by the SLOT_SAVE endpoint) and returns to decoding
immediately; every file is written to a `.tmp` path and atomically renamed, so
a crash never leaves a torn `.dpc`/`.ccn`/`.ckpt`. Before exit the server
drains the writer queue (joins the thread), then flushes the RAM-tier entries
to disk, so a clean shutdown preserves everything. On startup the disk index is
rebuilt by scanning the `.dpc` files and matching prompts by token prefix.

The `.ccn`/`.ckpt`/`.dpc` payloads are stored raw (uncompressed). The turbo
cache types are already nearly incompressible: a zstd level-3 run over
turbo4-at-64k production states measured a median ratio of 1.065x at ~150 MiB/s
(versus ~780 MiB/s for the raw write), so compression was explicitly rejected
(see the measurement log). Adding a compressor here would burn more CPU than
it would save disk space.

### Chunked state store (`cache.dcp`/`cache.dci`)

Large states write through to one append-only chunk store per cache directory
instead of one `.dpc` file per state. The state blob is sliced into fixed-cell
sheets; identical sheets dedup to one stored chunk, so two states sharing a
token prefix share chunks and the disk usage grows only by the new suffix.

- `cache.dcp` - the record log (chunk payloads + state references + tombstones).
- `cache.dci` - index snapshot, always rebuildable from a full `cache.dcp` scan.
- `cache.lock` - advisory single-writer lock (flock).
- `<fp>.dts` - token-only sidecar per state, lets a fresh boot find a state to
  restore before reading any chunk bytes.

On a config change the fingerprint mismatch wipes the whole directory
(`.dpc`/`.dts`/`.ccn`/`.dcp`/`.dci`), matching the legacy pointer-cache
behavior. The disk pool budget counts `cache.dcp + cache.dci` as payload
bytes: eviction drops state references and dead (refcount-zero) chunks are
reclaimed by an on-demand compact (dead/live ratio > 0.35). The store is
enabled by default; `TURBO_DISK_CHUNK_STORE=0` disables it and falls back to
the plain per-state `.dpc` writer.

**Rollback to the legacy `.dpc` layout** is a one-pass operation. Set
`TURBO_DISK_CHUNK_STORE=0` and start the server once: the boot-time index load
prunes `cache.dcp`, `cache.dci`, `cache.lock`, and any `.dts` sidecars from
every cache directory, leaving only the .dpc/.ccn tree that older builds read
unchanged. The store never modifies or removes `.dpc`/`.ccn` files, so a mixed
tree stays readable on either code path.

## Fit strategies

`--fit-strategy` selects the fit engine that sizes the model placement and the
KV cache together (`docs/fit-strategies.md`). `maxctx` and `rpc-aware` take a
context floor from `--fit-strategy-min-ctx`. When the cache types are set (for
example `--cache-type-k q8_0 --cache-type-v turbo3`), the fit measurement
accounts for the quantized KV working set, so the fitted `n_ctx` respects the
selected cache types.