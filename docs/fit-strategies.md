# Fit strategies

Selectable fit engine. `--fit-strategy NAME` picks which engine drives the
memory fit (the same placement machinery behind `--fit`), `--fit-strategy-min-ctx N`
sets the context floor for the strategies that need one.

## Flags

| Flag | Type | Meaning |
| --- | --- | --- |
| `--fit-strategy` | `upstream` \| `rpc-aware` \| `maxctx` | Fit engine to use. Default `upstream` (stock byte-identical fit). |
| `--fit-strategy-min-ctx` | int | Context floor the strategy must reach. Required for `rpc-aware` and `maxctx`. |
| `--fit-max-attempts` | int | Maximum measured probes without improving a safe `rpc-aware` plan. Default `32`; `0` disables the limit. |
| `--fit-probe-ceiling` | bool | Reuse a cached ceiling if present. Active allocation-to-OOM probes are disabled for safety. |

Rules:

- Unknown strategy name throws (case-insensitive).
- `--fit-strategy rpc-aware|maxctx` without `--fit-strategy-min-ctx` throws.
- Non-`upstream` strategy combined with `-fit off` throws.
- `--fit-strategy upstream` requires nothing and behaves exactly like stock `--fit`.

The flags are accepted by the fit-capable tools: `llama-cli`, `llama-server`,
`llama-perplexity`, `llama-imatrix`, `llama-tts`, `llama-completion`,
`llama-results`, `llama-cvector-generator`, `llama-mtmd-cli`, `llama-mtmd-debug`,
and `llama-fit-params`. `llama-tokenize`, `llama-batched-bench`, and
`llama-export-lora` accept and ignore them.

## Strategies

### upstream

The stock `common_fit_params` solver, byte-for-byte identical to the base tree.
This is the default and the byte-parity guarantee:

```sh
git rev-parse --verify bd1bf025f^{commit} >/dev/null && \
  test "$(git diff bd1bf025f -- common/fit.cpp | wc -l)" -eq 0
```

### maxctx

Finds the largest `n_ctx` the model placement can hold. Uses the same
two-anchor interpolation as stock fit, then confirms the interpolated context
with a measurement. The resulting `n_ctx` is equivalent to what stock fit
reports, not a strict binary-search maximum.

Errors (it does not silently ship an unfitted placement) when even the floor
`--fit-strategy-min-ctx` does not fit.

### rpc-aware

Places decode-hot dense tensors on the local non-RPC device and routes the
routed expert tensors to the device with the most free memory after the dense
weights are placed, selected by interconnect quality. Rules:

- The MoE pool is never forced onto `--main-gpu`.
- If the user `--main-gpu` is an RPC device, the strategy overrides it to the
  first non-RPC device and warns.
- A pool grant of 0 disables the MoE cache explicitly (`mode=off`); a grant of 0
  with `budget_mib=0` would otherwise mean "unlimited" in the cache layer.
- Explicit `--moe-cache N` wins over the strategy budget. Explicit `--moe-cache off`
  is never overridden.
- Candidate search uses a no-allocation tensor-placement pass to count exact
  per-device model bytes, while context and compute buffers are measured. It
  rejects candidates that miss a device margin before allocating model weights.
- The first candidate that reaches the absolute context cap with all margins
  satisfied is accepted immediately. If no candidate reaches the cap, the solver
  compares candidates and keeps the largest margin-valid context.
- Once a margin-valid plan exists, `--fit-max-attempts` bounds further no-improvement
  probes. At the limit the best safe plan is loaded, and its search cursor is cached
  so a later load can continue. The cached plan is remeasured against current device
  memory before it is reused; a changed baseline restarts the search. If no safe plan
  is found, fitting does not fall back to an unreserved plan.
- `--fit-target` values follow model-device order. Logs name each device and its
  effective margin; do not assume the first value is the local GPU when RPC is
  enumerated first.

The server margin accounting (`fit_params_target`) is re-attributed to the
strategy's effective main GPU before the candidate loop, so reserved margins
land on the device the placement actually uses.

## RPC fallback

If `--rpc` hosts are configured but no RPC device registers (server down, cold
start), the engine warns loudly and falls back to a local-only fit. It never
silently fits as if RPC did not matter. A device that reports `free=0/total=0`
during measurement (RPC device that died mid-fit) is excluded from the
candidate set and logged.

The interconnect link matrix (latency + bandwidth) is probed directly; a failed
probe uses conservative defaults and warns, never aborts.

## Pool budget policy

`common_fit_strategy_pool_budget` computes the bounded MoE cache budget:

- explicit `--moe-cache N` wins;
- otherwise `min(clamp(remaining/2, max(2*working_set, 256 MiB), 4 GiB), remaining)` (see `common/fit-strategy.cpp:1807-1824`);
- when `remaining < floor` the grant is 0 and the pool is disabled with a warning
  (a 0 budget with cache `mode=on` means unlimited, so disabling is explicit).

The budget is returned in bytes and converted to MiB before it reaches the
cache layer (`config.budget_mb` is a MiB cap). When the experts land on the
CPU pool, the grant is sized from the dense device's spare VRAM instead of
being disabled.

Sizing uses device total minus measured placement usage, not the probe's free
value: the probe snapshot can predate our own allocations. A fixed reserve
(1536 MiB plus the margin) keeps the driver away from queue-eviction territory
on near-full devices.

## Expert splits

Partial expert placements exist but are opt-in via `TQ_FIT_EXPERT_SPLIT=1`:
static residency packs expert layers onto the dense device up to its spare
VRAM, then falls back to a partial RPC offload. Measured on the dev box
(Qwen3.6-35B-A3B over 1 Gbps RPC, 16 GiB local GPU), both lose to the CPU pool
plus a large MoE cache - static residency hit 9.3 t/s vs 19.1 t/s for the
cache, and near-full VRAM wedged amdgpu queues during load. The adaptive cache
keeps the hottest experts; a fixed layer split cannot.

## Activation profile

`common_activation_profile` records per-tensor decode counters (token count,
routed expert set, KV rows touched). `common_activation_profile_sink` appends one
JSONL row per profile to a path. v1 ships the schema and the sink only; the
decode-path hooks that feed the profile are a follow-up.

## Manual plan caveat

A caller-provided `common_fit_plan` is validated by measurement, never trusted
blindly. A plan with out-of-range device ids or margins the devices cannot hold
is rejected loudly with an error status.

## Byte-parity guarantee

`common/fit.cpp` is pinned to the base tree anchor `bd1bf025f`. All strategy
work lives in `common/fit-strategy.{h,cpp}`. The parity gate above is part of
the test suite (`test-fit-strategy`).
