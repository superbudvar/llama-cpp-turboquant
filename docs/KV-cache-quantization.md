# KV Cache Quantization with TurboQuant

TurboQuant adds three runtime-only KV cache quantization types that compress
the K/V cache far beyond the standard `q8_0` while keeping decode quality via a
Walsh-Hadamard rotation (WHT) that Gaussianizes the cache vectors before
quantization:

| Type               | Enum                       | Size            | Compression vs f16 |
|--------------------|----------------------------|-----------------|--------------------|
| `turbo2`           | `GGML_TYPE_TURBO2_0` (43)  | 2 bits/value    | 6.4x               |
| `turbo3`           | `GGML_TYPE_TURBO3_0` (44)  | 3.25 bits/value | 4.9x               |
| `turbo4`           | `GGML_TYPE_TURBO4_0` (47)  | 4.25 bits/value | 3.8x               |

These are KV-cache-only types: they are never stored in model files. The
corresponding model-weight quantization types are `TQ3_1S` (45) and `TQ4_1S`
(46) - 3/4-bit WHT-rotated Lloyd-Max quantization, block size 32, exposed in
`llama-quantize` as `TQ3_1S` / `TQ4_1S`.

## Usage

```bash
llama-cli -m model.gguf -c 8192 -ngl 99 \
    --cache-type-k q8_0 --cache-type-v turbo3
```

Any combination of `f16`, `q8_0`, `turbo2`, `turbo3`, `turbo4` for K and V is
supported; mixing quantized V with unquantized K is the common configuration.

Turbo KV types require flash attention. If a turbo cache type is requested
with flash attention disabled, it is enabled automatically (a warning is
printed). A quantized V cache with flash attention explicitly disabled is an
error, matching upstream behavior for all quantized V types.

The same flags work in `llama-server`, `llama-bench`, and `llama-perplexity`.

## Rotation

K and V vectors are rotated by a fixed 128x128 orthonormal Walsh-Hadamard
matrix before quantization and inverse-rotated after dequantization. Head
dimensions that are not multiples of 128 are zero-padded to the next multiple
of 128 for turbo types. MLA models have no separate V cache (V is a view of
K), so V rotation and padding are skipped for them.

## Environment knobs

| Variable                        | Default | Effect                                                          |
|---------------------------------|---------|-----------------------------------------------------------------|
| `TURBO_LAYER_ADAPTIVE`          | `0`     | Layer-adaptive KV precision; `7` = Boundary V (first/last layers in `q8_0`, middle in turbo) |
| `TURBO_AUTO_ASYMMETRIC`         | `1`     | Auto-select asymmetric K/V types for large-GQA models (`0` disables) |
| `TURBO_SPARSE_V`                | `1`     | Sparse-V dequant skip in flash attention (`0` disables)        |
| `LLAMA_ATTN_ROT_K_OVERRIDE`     | off     | Enable upstream #21038 attention rotation for K                |
| `LLAMA_ATTN_ROT_V_OVERRIDE`     | off     | Enable upstream #21038 attention rotation for V                |
| `LLAMA_ATTN_ROT_DISABLE`        | `0`     | Hard lock-out: force rotation off on both sides (`1` disables) |

Upstream attention rotation is off by default: TurboQuant manages rotation
itself (the WHT applied at cache write is equivalent and interacts with the
cache types). `LLAMA_ATTN_ROT_*` only affects the optional upstream rotation
path for models that benefit from it.

## Model-weight quantization (TQ3_1S / TQ4_1S)

```bash
llama-quantize model-f16.gguf model-tq4.gguf TQ4_1S
```

`TQ3_1S` and `TQ4_1S` are first-class weight types with CUDA/HIP (warp
cooperative mmvq), Metal, and Vulkan kernels. MoE models disable CUDA graphs
for TQ `MUL_MAT_ID` automatically.

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
