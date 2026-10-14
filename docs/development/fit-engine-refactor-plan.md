# Fit engine and cache refactor plan

Status: in progress. The measurement seam, cache consolidation, and MTP search
session are implemented in the working tree; server preflight extraction and
upstream merge remain outstanding.

This plan separates the reusable memory probe and cache responsibilities from
the stock fit algorithm and the RPC-aware fit policy. The goal is to make the
fork's additions easier to preserve when merging upstream, without changing
placement behavior or weakening memory validation.

The baseline is `efbf3311111caf2252bf667a75229f3a4f7bf296`. The current working
tree is uncommitted and includes bounded-search, probe extraction, and cache
changes beyond that commit.

## Current structure

- `common/fit.cpp` owns `common_fit_params`, the stock back-to-front device
  placement algorithm, and fork-added measurement/accounting code. That file
  currently contains no-allocation tensor placement measurement, parent/child
  companion measurement, per-layer device byte capture, MoE-cache headroom, and
  cache feedback integration. These responsibilities do not all belong to the
  stock placement algorithm.
- `common/fit-strategy.cpp` owns the RPC-aware and max-context policies. It
  analyzes devices/tensors, builds candidate placements, validates measured
  memory, handles reservations and uses `common_fit_params` as its conservative
  fallback.
- `common/fit-cache.cpp` owns the disk fit-plan cache. `common/common.cpp`
  dispatches to stock or custom fitting, and `tools/server/server-context.cpp`
  measures/reserves MTP and mmproj companions.
- `common/fit.cpp` stock placement fills devices back-to-front. The RPC-aware
  solver instead compares local/RPC candidate splits to maximize context. Keep
  those objectives separate; share measurement contracts, not the stock
  placement loop.

Relevant source owners: `common/fit.cpp` (`common_get_device_memory_data_impl`,
`common_params_fit_impl`), `common/fit.h` (fit and memory APIs),
`common/fit-strategy.cpp` (`fit_analyze`, `fit_solve_rpc_aware`,
`common_fit_strategy_apply_impl`), `common/fit-cache.cpp`, `common/common.cpp`
(`common_init_result::update_fit_delta`), and `tools/server/server-context.cpp`
(target-context MTP preflight).

## Recommended module boundary

After a disposable seam probe, extract the shared measurement contract into
`common/fit-probe.{h,cpp}`:

- measured per-device model, context, and compute bytes;
- no-allocation placement probes;
- parent/child measurement used by external draft/MTP preflight;
- any layer-placement byte data needed by the stock fitter;
- device inventory and validation results required by both fit policies.

Keep `common_fit_params` and its stock placement policy in `common/fit.cpp`.
Keep RPC-aware candidate policy in `common/fit-strategy.{h,cpp}`. Keep cache
serialization and feedback under `common/fit-cache.{h,cpp}`. Keep the
server-facing MTP/mmproj orchestration in a small server helper only if the
probe shows that its caller signature stays simple; `server-context.cpp` should
then retain a small orchestration call rather than the measurement/convergence
implementation.

Reuse these stock-fit properties where the RPC-aware policy needs them:

1. Use measured per-device bytes as the acceptance oracle; do not accept a
   placement from a model-only estimate.
2. Preserve explicit user-placement guards, known loader layer-slot semantics,
   and conservative handling of devices with unknown memory.
3. Reuse the existing MoE-cache eligibility/budget calculations rather than
   duplicating a second cache-budget formula.
4. Retain `common_fit_params` as the last-resort stock fitter, with all hard
   companion reserves folded into its margins when RPC-aware placement cannot
   find a safe plan.

Do not transplant the stock back-to-front allocation loop into the RPC-aware
solver. Its placement objective and ordering are different.

## Cache responsibilities and optimization

There are two related but distinct cache concepts here: the **fit-result cache**
under `~/.cache/llama-cpp/fit-cache` (or `$LLAMA_FIT_CACHE`) and the server's
prompt-cache tiers (`--cache-ram` / `--cache-disk`). This plan concerns only the
fit-result cache.

### Separate reusable result from search progress

Keep the validated fit result and the optimization cursor as separate sections
of one cache record:

- **Fit result:** placement, context, shed reservations, projections, and memory
  feedback. Reuse it only after a fresh memory measurement validates it.
- **Search progress:** solver-policy revision, candidate/fraction cursor,
  incumbent score, and device baseline. Resume only when the fit identity,
  solver-policy revision, and baseline checks still match.

Store both in the same record and update them atomically. Do not put the cursor
in a second file: the incumbent and cursor must not become inconsistent after a
crash or feedback update. In code, prefer a nested `common_fit_search_progress`
member over adding more unrelated cursor fields directly to `common_fit_plan`.

Keep `--fit-max-attempts` out of the fit identity key. It changes how much
search work this load performs, not whether a validated placement fits. Record
a search-policy revision separately so a change to candidate ordering or cursor
meaning resets progress without discarding useful memory feedback.

The existing fit key intentionally identifies the model and device capacities,
fit targets/reserves, strategy/context, KV/FA settings, speculative draft, and
mmproj, while live memory load is checked separately. Preserve that safety
boundary. In particular, do not reuse an incumbent from a different MTP reserve
without measuring it against the current margins.

### Converge feedback updates through one serializer

The current cache has several manually maintained format paths:

- `common_fit_cache_save` serializes a plan;
- `common_fit_cache_refresh_deltas` patches a byte offset in the body;
- `common_fit_cache_save_with_model_size_error` separately deserializes and
  serializes the plan after feedback.

The search cursor adds more fields that each path must preserve. Replace these
with one checked decoder, one encoder, and one atomic read-modify-write update
that applies delta EMA, model-size error, projections, ceilings, and search
progress together. Keep the exact cache path/key passed through the feedback
flow; never search for an arbitrary model entry to update.

Retain the current correction rule: use the EMA when present, otherwise the raw
model-size error, never both. Keep fit-plan reuse gated by live validation. The
existing tests already check that fit feedback updates preserve the cursor; add
coverage for all supported on-disk versions and malformed/truncated progress
sections.

### Version compatibility decision

The committed cache format at the inspected HEAD is v6; the current writer is
v9. The loader accepts v6-v9 and defaults absent search fields. Cache tests
construct v6, v7, and v8 fixtures, load them, and verify that a subsequent save
rewrites them as v9. Older incomplete cursors restart at the candidate boundary
when their fractional cursor or solver-policy revision is absent. The nested
search-progress record carries a policy revision; changing candidate/cursor
semantics invalidates progress but retains a freshly validated incumbent. Do
not broaden compatibility to unverified historical formats just to avoid a
version bump.

### MTP convergence and progress identity

MTP preflight iterates the target fit while its measured reserve changes. The
exact reserve and target values participate in the cache key, so intermediate
iterations may produce different entries. Use one process-local search session
for the entire MTP convergence loop so the 32-probe no-improvement budget is
shared across iterations. Persist only the converged incumbent/cursor under the
final stable fit identity. On a later load, revalidate before resuming. If the
reserve changes materially, keep the validated fit result only if it still fits
and reset the search cursor.

Do not weaken the exact fit key to force reuse across different companion
reserves. If repeated real loads show cache fragmentation from harmless reserve
variation, first measure its frequency and impact; then consider a separate
progress fingerprint with a reserve tolerance while still revalidating the
incumbent against the current exact margins.

## Merge boundary and separate conflict

The fit module extraction should make future fitter edits additive and easier to
port, but it does not resolve the conflict found in the prior merge attempt. The
observed conflict is in prompt-checkpoint handling: this branch spills checkpoint
payloads to disk, while `origin/feature/turboquant-kv-cache` drops empty
checkpoints that cannot serialize KV state. Resolve that behavior separately
when retrying the merge. Preserve the benchmark evidence deleted by the incoming
branch unless the caller explicitly approves its removal.

## Staged implementation and acceptance

1. **Seam probe - complete:** a disposable worktree confirmed that the shared
   measurement implementation can move without changing its consumers' call
   signatures. The implementation uses a private internal declaration and a
   public `fit-probe.h` measurement contract.
2. **Extract measurement - complete in working tree:** `fit-probe.cpp` now owns
   normal, no-allocation, and parent/child measurement implementations;
   `common_fit_params` remains in `fit.cpp` and the RPC-aware policy remains in
   `fit-strategy.cpp`.
3. **Consolidate cache I/O - complete in working tree:** feedback now uses one
   load/update/save operation, and v6-v8 migration fixtures pass. Search
   progress is a nested `common_fit_search_progress` section with a policy
   revision. The MTP convergence loop shares a process-local search session
   across iterations and defers intermediate cache writes until the final
   converged target load.
4. **Server preflight extraction - deferred:** it is not required for the fit
   probe/cache boundary, and the existing loader depends on local model/draft
   parameters and callbacks. Revisit only if an isolated helper prototype yields
   a small stable interface. Keep this distinct from the checkpoint merge
   conflict.
5. **Upstream merge - not done:** preserve fit modules and benchmark artifacts,
   resolve the checkpoint conflict separately, then run fit/cache tests and
   build the affected server targets.

Acceptance is behavior-preserving stock fitting, unchanged safe RPC-aware
placement and cache feedback, bounded search that resumes only from a compatible
cursor, and a merge result that retains the fork's fit engine and benchmark
evidence. The cache and fit-strategy suites pass with the small CTest fixture.
The caller selected the existing behavior for soft context rejects: treat the
failed context as a non-fit point and continue the bounded search. The test now
asserts that behavior. A full Qwen3.8 external-MTP server load remains
unverified.
