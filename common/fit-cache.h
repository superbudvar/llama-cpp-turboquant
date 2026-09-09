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
// min-ctx) and RPC endpoints. Device memory CAPACITY is hashed; device memory
// LOAD is validated separately on every cache hit.
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
        const common_fit_plan & plan);