# Instructions for the common/ fit-strategy engine (TurboQuant fork)

Fit placement on multi-GPU / RPC split models.

### Which file is the operative engine

- `common/fit-strategy.cpp` is the real solver for `--fit-strategy rpc-aware` / `maxctx`. The log line `common_fit_strategy_apply_impl: successfully fit params to free device memory` confirms it ran.
- `common/fit.cpp` is the stock/upstream path: back-to-front fill `for (id = nd - 1; id >= 0; id--)` where the LAST device fills first. It is a fallback only; do not debug rpc-aware placement there.

### Byte-parity guarantee (replaces the original ac-fs.1)

The engine code lives in `common/fit-strategy.{h,cpp}`. `common/fit.cpp` is not modified from the no-strategy invocation. The original byte-identical constraint has been replaced by inspection-grade acceptance: "the custom logic is provably OUT of the shared code path". See `.agents/plans/2026-10-03-fit-strategy-cleanup.md` for disposition (ac-fs.1 superseded).

### rpc_frac scan direction decides which GPU gets packed

- `fit_solve_rpc_aware` scans feasible rpc_frac values and keeps the split reaching the largest context (`fit_grow_context`). Selection is strict `ctx > best_ctx`, so the FIRST scanned split that reaches the max context wins the tie.
- Scanning top-down (largest rpc_frac first) packs the RPC device to the brim while the local device stays half-idle (observed: RPC 0 MiB free, ROCm 4 GiB free). Scanning bottom-up keeps the smallest RPC share that still reaches the same max context and packs the local device first (observed: ROCm 611 MiB free, RPC 4 GiB free). See the loop at fit-strategy.cpp:~764.
- Context is the primary objective: a split that reaches a higher context always beats a smaller rpc_frac at lower context. Tie-break (scan direction) only applies among splits at the same max context.
- Selection tiers (dense): (1) local-only offload reaching min-ctx wins immediately on the main GPU; (2) the local+RPC split keeps growing to its max context and wins when it reaches min-ctx - CPU spill is never considered in case 2; (3) only when tier 2 fails does the CPU-spill phase run, and its base is the RPC-split candidate kept inside the loop (`have_rpc_spill`), so the spill stays rocm+rpc+cpu. A local-only spill base only remains when no RPC device is usable.
- CPU-spill rpc-split re-brackets rpc_frac per n_gpu_layers (`fit_spill_split_at_ngl`): the old spill reused the full-offload ceiling (the max RPC fraction), which overflows nothing once layers move to CPU and fills RPC first while the local device stays half-idle. A spill must re-scan the split at each layer count and pack the local device first. The coarse 0.05 scan catches windows wider than the step; a bisection handles narrower windows straddled by the scan. See fit-strategy.cpp:~587.

### Companion reservations shed before CPU spill

- The server registers soft VRAM reservations (`common_fit_reserve`) for its companions before fitting the main model: mmproj (priority 0) and the draft model (priority 1, only for a separate draft; MTP stays a hard fit margin). Each is a `{device, bytes, priority, tag}` entry in `fit_params_reserves`; priorities are lower-releases-first.
- `fit_solve_rpc_aware` (and `maxctx`) re-run the whole candidate suite per shed level: no shed, then cumulative release by ascending priority. The first level with a full-offload candidate reaching min-ctx wins. CPU spill runs only at the last level (all reservations shed), because moving a companion off VRAM is cheaper than spilling model layers.
- Positions: mmproj sheds before spec before model layers; if shedding mmproj alone suffices, the spec (draft) stays on VRAM.
- The winning plan's `shed_reserves` (tags) is saved to the fit cache, published back into `params.fit_params_shed_tags`, and used by the server to load the tagged companions on CPU (draft forced to CPU via `devices.clear()/n_gpu_layers=0`, mmproj via `mparams.use_gpu=false`).
- The cache key includes the reserve set; a cache hit is validated against the margins matching its own `shed_reserves`.

### Fit probe output is not the real allocation

- During the fit probe phase (`load_mode = none`) every device prints `model buffer size = 0.00 MiB` and `offloaded N/N layers`, and the breakdown `unaccounted` column can go negative (reported free includes other processes, e.g. the desktop session on ROCm). These lines are measurement artifacts.
- The real placement only appears in the final `MODEL READY - device memory breakdown` line.
- `n_ctx_seq < n_ctx_train` warnings during probing are expected; the final context is printed right after MODEL READY.

### Config semantics

- `--fit-strategy-min-ctx` is a floor, not a cap. With `--ctx-size` unset the solver grows context up to `n_ctx_train`.
- The rpc_frac feasibility interval: low rpc_frac overflows the local device, high rpc_frac overflows RPC (each device has a 2 GiB `context_reserve`; RPC also subtracts a 2 GiB `rpc_overhead`). The interval is bracketed at the minimum context, then context grows per candidate.