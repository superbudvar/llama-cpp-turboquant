# KV disk prompt-cache refactor plan

Status: in progress. DCP/DTS integrity hardening, storage-source extraction, and
upstream state/save safeguards are implemented locally; focused tests pass.
Hardware validation of block-streamed state round-trip and wider integration
checks remain outstanding.

## Progress

- DCP generation-header rewrite errors now fail commit. A wrapped `pwrite`
  failure test verifies that the generation stays uncommitted and the batch is
  absent after reopen.
- DCI now has an independent version and CRC32C. Older snapshots are rejected
  and rebuilt from DCP; counts, offsets, and chunk references are bounded.
- DTS publication reports errors. A failed publication tombstones the committed
  state instead of leaving an unreachable store entry. The writer distinguishes
  `not_stored`, `stored_in_ram`, and `queued_to_disk`; queue admission does not
  cause a live slot to be cleared.
- The low-level chunk store, disk-tier implementation, and `.ccn` spill helpers
  now live in separate source files. Disk state and worker ownership are grouped
  in `server_prompt_cache_disk`; RAM prompt selection remains in
  `server_prompt_cache`.
- Upstream `09dd093b3` state serialization, slot-preservation, and empty-checkpoint
  safeguards were applied without merging upstream.
- `test-chunk-store` and `test-prompt-cache-pool` pass. `test-save-load-state`
  builds, but its run requires a model file that is not available in this
  workspace. CUDA block-streamed runtime validation remains outstanding.

The goal is to isolate the fork's persistent prompt-cache and per-slot
checkpoint-spill code from the large server implementation files, then integrate
upstream's block-KV state save/load and slot-preservation changes. Preserve
current DPC/DTS formats, prompt matching, pruning, and cache restore behavior.

The original comparison was local `HEAD` `efbf3311111caf2252bf667a75229f3a4f7bf296`
against `origin/feature/turboquant-kv-cache`
`6e09de1b44ae328f641d1367bb4c875a8ffab43f`. The fit-engine changes have since
been committed as `a3cc8a032`; the cache work in this plan is the current
uncommitted change set.

## Separate the two disk-backed features

1. **Persistent prompt-cache disk tier**: `.dpc` states, `.dts` indexes,
   `cache.dcp`/`cache.dci` chunk store, async write-through, shared-pool pruning,
   and restore. This is the `--cache-disk` path.
2. **Per-slot checkpoint spill**: `.ccn` payload spill/rehydrate/compaction
   while a live slot keeps checkpoint metadata in RAM. This is scratch state
   for the active server context, not a DPC/DTS entry.

Do not confuse either with the fit-result cache or the RAM prompt-cache tier.

## Initial code shape

- Before this work, `server_prompt_cache` owned RAM entries and most disk-tier state in
  `tools/server/server-task.h:932-1119`: disk indexes, write-through scratch,
  chunk-store handle, async jobs, locks, and writer/vacuum threads.
- `tools/server/server-task.cpp` contains RAM allocation/matching and the disk
  fallback (`1663-1874`), queue admission (`4209-4245`), the async state worker
  (`4689-4762`), and disk restore (`load_from_disk_tier`).
- The low-level chunk store and DPC/DTS types are also exposed from
  `server-task.h:659-930`; its implementation and codecs occupy a large part of
  `server-task.cpp`.
- `tools/server/server-context.cpp` manages the per-model directory/fingerprint
  and load/save lifecycle (`1249-1297`) and owns the `.ccn` spill call site
  (`3183-3276`).
- `tests/test-chunk-store.cpp` directly exercised the chunk-store scenarios;
  `tests/CMakeLists.txt:177-179` links it through `server-context`.

The incoming branch's current `server-task.{cpp,h}` does not contain the local
DPC/DTS disk tier. Its relevant new commit is `09dd093b3`, which changes
`src/llama-kv-cache.cpp` and `tools/server/server-context.cpp`, not the disk
store implementation.

## Upstream behavior to preserve in our cache

1. **Block-streamed state save/load** (`09dd093b3`): upstream removes the
   refusal in `llama_kv_cache::state_write` and `state_read_sinfo` because the
   block-streamed KV buffer is full-sized pinned host memory. The restore write
   resets the resident GPU mirror and bumps its graph generation. Port this
   low-level change and test the assumptions; do not remove the guard without
   retaining the full-buffer, synchronization, and mirror-reset invariants.
2. **Do not wipe live prompt state on an immediate save failure**: upstream
   records both `saved` and `loaded`, and only clears the slot when save
   succeeded and no better prompt restored (`server-context.cpp:2002-2019`).
3. **Drop empty checkpoints**: upstream discards a checkpoint whose KV payload
   is empty before keeping its position/token metadata (`server-context.cpp:2815-2822`).
   In our branch, this check must run before `.ccn` spill; non-empty checkpoints
   still spill as they do today.
4. **Keep one LCP selector**: upstream's simplified RAM scan uses the same
   eligibility/tie-break rule as the local `select_cache_candidate`. Preserve
   the local shared selector and disk fallback rather than adding a second
   matching policy.

### Async save result needs an explicit meaning

Local `server_slot::prompt_save()` returns `commit_disk_state()`'s boolean.
For RAM entries that means the entry is stored in RAM. For a write-through
entry, `commit_disk_state()` moves a serialized copy into the async queue and
returns true immediately (`server-task.cpp:4209-4245`); the worker may later fail
and drop it (`4689-4762`). Thus `true` currently means “stored in RAM or queued,”
not “durably committed.”

The extracted API should return an explicit outcome such as
`not_stored`, `stored_in_ram`, or `queued_to_disk`. The server should preserve
the upstream immediate-failure guard. It must not report a queued state as
durable/indexed until the worker commits it. A worker failure should remain
visible and result in a cache miss, not a corrupt restore. Keep disk I/O off the
decode thread; decide at implementation time whether queued states defer slot
clearing or whether a completion signal is required for idle-slot clearing.

## Recommended module boundary

The preferred end state is a disk-tier owner with a small server-facing API:

- `tools/server/server-prompt-cache-store.{h,cpp}` owns the low-level append-only
  chunk store, record codec, CRC/recovery, lineage/refcount, and compact logic.
- `server_prompt_cache_disk` owns disk state and worker threads. Its API is
  declared in `server-task.h`; DPC/DTS codecs/index, async writer/vacuum,
  write-through, disk lookup/restore, and shared-pool pruning are implemented in
  `server-prompt-cache-disk.cpp`.
- `tools/server/server-task.{h,cpp}` retains RAM prompt entries, shared LCP
  candidate selection, slot state APIs, and delegates disk work to the disk
  owner. Remove disk queues, indexes, and store internals from the RAM cache
  class.
- `server-checkpoint-spill.cpp` owns `.ccn` payload store/load/compact helpers;
  declarations remain in `server-task.h`. `server-context.cpp` retains
  checkpoint policy and calls the helpers.

Do not start with only the chunk-store class and call the cache extracted: that
is a useful first vertical slice (the existing chunk-store tests provide a
boundary), but it leaves DPC/DTS indexes, async writes, disk restore, and
pruning entangled in `server_prompt_cache`. Probe both the low-level-only cut
and the full disk-tier owner in disposable isolation, then keep the smallest
cohesive boundary that materially removes those internals from `server-task`.

## Merge plan

The previous conflict is narrow and separate from the main disk tier. At
`create_checkpoint`, local code spills a non-empty payload to `.ccn`
(`server-context.cpp:3254-3275`); upstream drops empty checkpoint payloads
(`09dd093b3`, `server-context.cpp:2801-2828`). Compose both: drop an empty
checkpoint first, otherwise spill it. Extracting `.ccn` I/O shrinks the local
hunk, but the call site still needs this ordering.

When retrying the upstream merge in an isolated worktree:

1. Preserve the local disk modules and benchmark evidence; do not take the
   upstream `server-task` tree wholesale, which has no DPC/DTS tier.
2. Port the `llama-kv-cache.cpp` save/load invariant and server slot/checkpoint
   safeguards into the local architecture.
3. Resolve the checkpoint hunk as described above and keep upstream state
   changes independent from disk-tier ownership.

## Implementation status and remaining checks

1. **Implemented locally:** storage modules, DCP/DCI/DTS integrity handling,
   explicit queued-write outcomes, and upstream state/save safeguards. The focused
   chunk-store and prompt-cache-pool suites pass, and `llama-server` builds.
2. **Still required:** run a model-backed block-streamed save/restore test on
   CUDA; the local save/load test could not run without a model file.
3. **Next integration check:** assess the upstream merge in an isolated
   worktree, preserve benchmark evidence, and resolve the `.ccn` checkpoint hunk.
   Do not merge as part of this change unless separately authorized.

Acceptance is unchanged RAM prompt matching, disk restore/write-through
semantics, and on-disk format compatibility; correct state save/load under block
KV streaming; safe save-failure and empty-checkpoint handling; and an upstream
merge that retains the local DPC/DTS functionality and benchmark records. The
actual merge and full streamed-model validation are not part of this planning
step.
