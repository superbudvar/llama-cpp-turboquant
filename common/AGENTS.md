# Instructions for the common/ fit-strategy engine (TurboQuant fork)

Fit placement on multi-GPU / RPC split models.

### Which file is the operative engine

- `common/fit-strategy.cpp` is the real solver for `--fit-strategy rpc-aware` / `maxctx`. The log line `common_fit_strategy_apply_impl: successfully fit params to free device memory` confirms it ran.
- `common/fit.cpp` is the stock/upstream path: back-to-front fill `for (id = nd - 1; id >= 0; id--)` where the LAST device fills first. It is a fallback only; do not debug rpc-aware placement there.

### rpc_frac scan direction decides which GPU gets packed

- `fit_solve_rpc_aware` scans feasible rpc_frac values and keeps the split reaching the largest context (`fit_grow_context`). Selection is strict `ctx > best_ctx`, so the FIRST scanned split that reaches the max context wins the tie.
- Scanning top-down (largest rpc_frac first) packs the RPC device to the brim while the local device stays half-idle (observed: RPC 0 MiB free, ROCm 4 GiB free). Scanning bottom-up keeps the smallest RPC share that still reaches the same max context and packs the local device first (observed: ROCm 611 MiB free, RPC 4 GiB free). See the loop at fit-strategy.cpp:~764.
- Context is the primary objective: a split that reaches a higher context always beats a smaller rpc_frac at lower context. Tie-break (scan direction) only applies among splits at the same max context.

### Fit probe output is not the real allocation

- During the fit probe phase (`load_mode = none`) every device prints `model buffer size = 0.00 MiB` and `offloaded N/N layers`, and the breakdown `unaccounted` column can go negative (reported free includes other processes, e.g. the desktop session on ROCm). These lines are measurement artifacts.
- The real placement only appears in the final `MODEL READY - device memory breakdown` line.
- `n_ctx_seq < n_ctx_train` warnings during probing are expected; the final context is printed right after MODEL READY.

### Config semantics

- `--fit-strategy-min-ctx` is a floor, not a cap. With `--ctx-size` unset the solver grows context up to `n_ctx_train`.
- The rpc_frac feasibility interval: low rpc_frac overflows the local device, high rpc_frac overflows RPC (each device has a 2 GiB `context_reserve`; RPC also subtracts a 2 GiB `rpc_overhead`). The interval is bracketed at the minimum context, then context grows per candidate.