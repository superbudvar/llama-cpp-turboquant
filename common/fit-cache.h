#pragma once

#include "fit-strategy.h"

#include <cstdint>
#include <string>

// Disk cache for successful fit plans (fork feature).
//
// A cache hit substitutes the cached placement for the candidate probe suite
// on repeated loads of the same model with the same configuration. The cached
// plan is not trusted blindly: the caller re-validates it with a single
// measurement probe and falls back to a fresh solve when memory no longer
// fits. Cache location: ~/.cache/llama-cpp/fit-cache (or $LLAMA_FIT_CACHE).
// Per-model entries are <sha256-of-key>.bin, written atomically (tmp+rename).
//
// Cache file format (little-endian):
//   header (32 bytes):
//     u32 magic      = 0x46495443 ("FITC")
//     u32 version    = 1
//     u64 key_hash   = first 8 bytes of the key SHA-256
//     i64 timestamp  = unix seconds of the last successful fit
//     u32 body_size
//   body (per plan, S3 format):
//     i32 n_roles
//     i32 role_devices[n_roles]
//     i32 n_gpu_layers
//     i32 tensor_split_len
//     f32 tensor_split[tensor_split_len]
//     i32 n_ctx
//     u64 pool_grant_mib
//
// The serialized plan carries no tensor_buft_overrides: the solvers never
// populate them (only the synthetic manual-plan test path does), so cached
// plans always have an empty override list.

// SHA-256 of the identity key for path_model + configuration + device set.
// The key hashes model identity (path, mtime, size), device identity (name,
// type, total bytes, rpc flag), fit configuration (target, strategy,
// min-ctx) and RPC endpoints. Speculative draft and mmproj companions are
// hashed too: they load after the fit and consume leftover VRAM, so they must
// not share a cached plan with a run that lacks them. Device memory CAPACITY
// is hashed; device memory LOAD is validated separately on every cache hit.
uint64_t common_fit_cache_key_hash(
        const common_params & params,
        const std::string & path_model,
        const common_fit_analysis & analysis);

// Load a cached plan. Returns false on miss: no entry, stale (>24h), key
// mismatch, corrupted file, or --fit-cache-clear requested (also removes the
// entry). A miss never throws.
bool common_fit_cache_load(
        const common_params & params,
        const std::string & path_model,
        const common_fit_analysis & analysis,
        common_fit_plan & plan_out);

// Save a plan under the current key. Best-effort, returns false on failure.
// Called on the fit success path; re-saving on a cache hit refreshes the
// freshness timestamp (keeps hot models from going stale).
bool common_fit_cache_save(
        const common_params & params,
        const std::string & path_model,
        const common_fit_analysis & analysis,
        const common_fit_plan & plan,
        uint64_t override_key = 0);

// Refresh just the delta_free_ema field in the exact matching cache entry.
// Called after MODEL READY: update_fit_delta() has computed the new EMA but
// the cache was already saved (before projected_free existed). This patches
// the delta into the existing cache so the NEXT load applies it.
// Never scans or patches another configuration's cache entry. Returns false if
// no matching entry exists or refresh fails (non-fatal).
bool common_fit_cache_refresh_deltas(
        const common_params & params,
        const std::string & path_model,
        const common_fit_analysis & analysis,
        const std::vector<int64_t> & delta_free_ema,
        std::string * cache_path_out = nullptr);

// Save/update the exact cache entry with projected_free and discrepancy data
// after MODEL READY. The ceiling field is retained for cache compatibility.
bool common_fit_cache_save_with_model_size_error(
        const common_params & params,
        const std::string & path_model,
        const common_fit_plan & plan,
        const std::vector<int64_t> & projected_free,
        const std::vector<int64_t> & model_size_error,
        const std::vector<int64_t> & ceiling_free_warm,
        const std::string * cache_path = nullptr);
