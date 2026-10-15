# Instructions for the common/ fit-strategy engine (TurboQuant fork)

Fit placement on multi-GPU / RPC split models.

### Which file is the operative engine

- `common/fit-strategy.cpp` is the real solver for `--fit-strategy rpc-aware` / `maxctx`. The log line `common_fit_strategy_apply_impl: successfully fit params to free device memory` confirms it ran.
- `common/fit.cpp` is the stock/upstream path: back-to-front fill `for (id = nd - 1; id >= 0; id--)` where the LAST device fills first. It is a fallback only; do not debug rpc-aware placement there.

### Byte-parity guarantee (replaces the original ac-fs.1)

The engine code lives in `common/fit-strategy.{h,cpp}`. `common/fit.cpp` is not modified from the no-strategy invocation. The original byte-identical constraint has been replaced by inspection-grade acceptance: "the custom logic is provably OUT of the shared code path". See `.agents/plans/2026-10-03-fit-strategy-cleanup.md` for disposition (ac-fs.1 superseded).

### rpc_frac scan direction decides which GPU gets packed

- `fit_solve_rpc_aware` compares candidates by achieved context. If a margin-valid candidate reaches the absolute cap (trained context or explicit `--ctx-size`), later candidates cannot improve the primary objective and the solver accepts it immediately. Otherwise it keeps searching and returns the best context across splits.
- Scanning top-down (largest rpc_frac first) packs the RPC device to the brim while the local device stays half-idle (observed: RPC 0 MiB free, ROCm 4 GiB free). Scanning bottom-up keeps the smallest RPC share that still reaches the same max context and packs the local device first (observed: ROCm 611 MiB free, RPC 4 GiB free). See the loop at fit-strategy.cpp:~764.
- Context is the primary objective: a split that reaches a higher context always beats a smaller rpc_frac at lower context. Tie-break (scan direction) only applies among splits at the same max context.
- Candidate order starts local-only to preserve decode locality, then considers local+RPC. Candidate measurements use no-allocation tensor placement plus context/compute measurement; model weight buffers are not allocated during the search. The final real load is the physical check.
- CPU spill runs only after every reservation-shed level fails to reach the minimum; its base is the RPC-split candidate kept inside the loop (`have_rpc_spill`), so the spill stays rocm+rpc+cpu. A local-only spill base only remains when no RPC device is usable.
- CPU-spill rpc-split re-brackets rpc_frac per n_gpu_layers (`fit_spill_split_at_ngl`): the old spill reused the full-offload ceiling (the max RPC fraction), which overflows nothing once layers move to CPU and fills RPC first while the local device stays half-idle. A spill must re-scan the split at each layer count and pack the local device first. The coarse 0.05 scan catches windows wider than the step; a bisection handles narrower windows straddled by the scan. See fit-strategy.cpp:~587.

### Companion reservations shed before CPU spill

- The server registers soft VRAM reservations (`common_fit_reserve`) for its companions before fitting the main model: mmproj (priority 0) and the draft model (priority 1, only for a separate draft; MTP stays a hard fit margin). Each is a `{device, bytes, priority, tag}` entry in `fit_params_reserves`; priorities are lower-releases-first.
- `fit_solve_rpc_aware` (and `maxctx`) re-run the whole candidate suite per shed level: no shed, then cumulative release by ascending priority. The first level with a full-offload candidate reaching min-ctx wins. CPU spill runs only at the last level (all reservations shed), because moving a companion off VRAM is cheaper than spilling model layers.
- Positions: mmproj sheds before spec before model layers; if shedding mmproj alone suffices, the spec (draft) stays on VRAM.
- The winning plan's `shed_reserves` (tags) is saved to the fit cache, published back into `params.fit_params_shed_tags`, and used by the server to load the tagged companions on CPU (draft forced to CPU via `devices.clear()/n_gpu_layers=0`, mmproj via `mparams.use_gpu=false`).
- The cache key includes the reserve set; a cache hit is validated against the margins matching its own `shed_reserves`.

### Fit probe output is not the real allocation

- During the fit probe phase every device prints `model buffer size = 0.00 MiB` and `offloaded N/N layers`, and the breakdown `unaccounted` column can go negative (reported free includes other processes, e.g. the desktop session on ROCm). These lines are measurement artifacts.
- The probe loads with `no_alloc = true` and `load_mode = mmap` (forced at `common/fit-strategy.cpp:409` and `:1814`), not `load_mode = none`. MMAP is required: the probe reports exact per-tensor byte counts from the tensor table, and `load_mode` is part of the fit measurement cache key (`fit-strategy.cpp:178`), so changing it would invalidate every cached placement. Because `no_alloc` never reads tensor data, `llama_model_loader::init_mappings` is called without prefetch when `no_alloc` is set (`src/llama-model.cpp:1683`), which skips `MAP_POPULATE`. With prefetch, every probe read the entire GGUF from disk (~11 GiB, ~70 s per probe on a 12 GB model).
- The real placement only appears in the final `MODEL READY - device memory breakdown` line.
- `n_ctx_seq < n_ctx_train` warnings during probing are expected; the final context is printed right after MODEL READY.
- `--fit-target` values are indexed by the model device order used by `fit_analyze`, not by the intuitive local-then-RPC order. Check the new per-device margin log before interpreting a comma-separated target list.
- `--fit-probe-ceiling` no longer allocates near the OOM threshold. Active cold and warm OOM probes are disabled; use the reported free memory and explicit margins.

### Config semantics

- `--fit-strategy-min-ctx` is a floor, not a cap. With `--ctx-size` unset the solver grows context up to `n_ctx_train`.
- The rpc_frac feasibility interval: low rpc_frac overflows the local device, high rpc_frac overflows RPC (each device has a 2 GiB `context_reserve`; RPC also subtracts a 2 GiB `rpc_overhead`). The interval is bracketed at the minimum context, then context grows per candidate.
