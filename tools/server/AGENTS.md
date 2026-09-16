# Instructions for tools/server/ (TurboQuant fork)

### Reading the router log

- Child instance output is forwarded to the router log with a `[NNNNN]` bracket that is the child's **port** (`LOG("[%5d] %s", port, buffer)`, server-models.cpp:991), never a pid. Router's own lines carry no bracket. To map a log segment to a process, match `--port NNNNN` in the ps args.
- The prefix clock is monotonic per-process uptime in `M.s.ms.us` (common/log.cpp:101), not wall clock; it restarts per spawned instance, so two `[port]` segments cannot be compared by timestamp alone.
- A change to the router ini triggers `load_models: (reload) unloading model ... (source updated or removed)` followed by a fresh spawn on a new port. A backend crash instead shows `http client error: Failed to read connection` or `instance ... exited with status 1`, then `ensure_model: model ... is not loaded, loading...` and an on-demand respawn (the router sets `--sleep-idle-seconds` and lazily wakes dead/evicted backends). Spawn args are identical in both cases - read the unload/exit lines, not the args, to tell config reload from crash-respawn.

### Backend crash identification

- `ggml_abort` prints its backtrace tool output to the same stream: `[New LWP ...]`, `[Inferior 1 (process N) detached]`. These gdb-style lines are the abort handler, not an interactive debugger. The `(process N)` is the backend pid - cross-reference it with `--port` to find the crashed instance.
- With `--rpc`, any remote-node failure aborts the whole backend from `RPC_STATUS_ASSERT` (`ggml-rpc.cpp:509`, killed in `ggml_backend_rpc_buffer_set_tensor`). `send failed (bytes_sent=..., size_to_send=...)` shows how far a large tensor transfer got; the remote node's TCP port can still be open afterwards, so an open port does NOT mean healthy RPC.

### Prompt-cache tier semantics (`--cache-ram`, `--cache-disk`)

- The disk write-through branch in `alloc()` is gated on `limit_size > 0` (server-task.cpp:1694). `--cache-ram 0` therefore does NOT mean "disable RAM, use disk only": with `limit_size == 0` no state ever write-throughs at save time, and `can_promote` (load_from_disk_tier) is `limit_size == 0 || ...`, so every disk entry is promoted into RAM - the RAM tier grows unbounded. A nonzero limit smaller than every state (e.g. 3072 MiB with 3.7-6.1 GiB states) is the pure-disk case: every alloc write-throughs and every restore logs `RAM budget exhausted`.
- Cache health greps: `restored entry ... directly from disk` = disk hit, `wrote through state ... .dpc` = disk write, `promoted entry ... into RAM` = disk->RAM restore, `pruned .dpc to X ... (budget Y)` = disk prune, `update: prompt cache: N prompts, X MiB (limit Y MiB, Z tokens)` = RAM residency. Disk entries persist across restarts keyed by a token hash in the `meta` fingerprint file; a config change invalidates the whole directory (fingerprint mismatch clears `.dpc`/`.ccn`).