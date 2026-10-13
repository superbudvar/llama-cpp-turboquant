#pragma once

#include "common.h"
#include "fit.h"

#include <functional>
#include <string>
#include <vector>

// Selectable fit-strategy engine (fork feature).
// Custom strategies live here, OUT of common/fit.cpp (byte-parity anchor).
//
// MEASUREMENT-IS-THE-ORACLE: every allocation-affecting param is serialized
// into candidate mparams/cparams and measured with common_get_device_memory_data
// (fit.h:96), not modeled. The probe seam below lets synthetic tests override
// the measurement itself.

enum common_fit_role {
    COMMON_FIT_ROLE_ATTN_QKV,
    COMMON_FIT_ROLE_ATTN_O,
    COMMON_FIT_ROLE_DENSE_FFN,
    COMMON_FIT_ROLE_EXPERT_UP,
    COMMON_FIT_ROLE_EXPERT_GATE,
    COMMON_FIT_ROLE_EXPERT_DOWN,
    COMMON_FIT_ROLE_NORM,
    COMMON_FIT_ROLE_EMBED,
};

struct common_fit_tensor_meta {
    common_fit_role role;
    std::string tensor_id; // stable GGUF tensor name
    size_t bytes;          // relative ordering only; sizing comes from measurement
};

struct common_fit_device {
    int id;
    ggml_backend_dev_t dev;
    bool is_rpc;
    int64_t free_bytes;
    int64_t total_bytes;
    size_t link_bw_mbps;    // 0 = unknown -> conservative default
    size_t link_latency_us; // 0 = unknown
};

struct common_fit_moe_meta {
    int32_t n_expert;
    int32_t n_expert_used;
    size_t expert_bytes;
    size_t working_set_bytes;
};

struct common_fit_analysis {
    std::vector<common_fit_device> devices;
    std::vector<common_fit_tensor_meta> tensors;
    common_fit_moe_meta moe;
    bool rpc_available;   // an RPC device was enumerated and responsive
    bool rpc_hosts_set;   // params.rpc_hosts was non-empty at parse (cold-start signal)
    std::string model_id;
    uint32_t loader_ngl = 0; // n_layer_all: unconditional loader slot count (nextn included)
    uint32_t n_ctx_train = 0;
};

struct common_fit_plan {
    // explicit per-role device assignment (device ids from common_fit_analysis.devices)
    std::vector<int> role_devices; // indexed by common_fit_role
    int32_t n_gpu_layers = -1;
    std::vector<float> tensor_split;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
    int32_t n_ctx = 0;
    size_t pool_grant_mib = 0; // 0 = no pool
    // tags of soft reservations this plan released ("mmproj", "spec"). the
    // caller moves those companions off VRAM before loading them.
    std::vector<std::string> shed_reserves;
    // per-device EMA of (actual_free - projected_free) after each successful
    // MODEL READY. negative = fit overestimates free space, so margin should be
    // increased. indexed by device id from common_fit_analysis.devices.
    // version 2+: stored in cache; version 1 has no delta (initialized to 0).
    std::vector<int64_t> delta_free_ema;
    // projected free bytes per device at fit time (cold_free - used from probe).
    // used to compute actual delta after MODEL READY: delta = actual_free - projected_free.
    // indexed by device id. populated by fit_solve_rpc_aware before returning.
    std::vector<int64_t> projected_free;
    // true when delta_free_ema was applied to margins before computing projected_free.
    // lets subsequent runs know this projected_free came from delta-adjusted margins.
    bool projected_with_delta = false;
    // per-device maximum observed free bytes after a successful MODEL READY. set by
    // fit_probe_ceiling() or recorded after a successful load. subsequent fits clamp
    // the backend-reported free to min(free, ceiling) to avoid overestimating when
    // free fluctuates due to driver overhead or fragmented pools.
    // indexed by device id. version 4+: stored in cache; earlier versions have empty.
    std::vector<int64_t> ceiling_free;
    // warm counterpart of ceiling_free: measured AFTER model weights are loaded but
    // BEFORE KV cache/context init. captures the driver overhead at the moment the
    // context will be created, preventing the cold-only ceiling from masking the
    // ~800 MiB ROCm driver allocation growth during init. version 6+ or same as
    // ceiling_free when not separately captured.
    std::vector<int64_t> ceiling_free_warm;
    // per-device model_size_error: actual_model_used - projected_model_from_probe.
    // this is the dominant component of delta_free_ema but converges slowly via EMA.
    // storing it directly avoids the multi-load convergence lag. version 5+: cached.
    std::vector<int64_t> model_size_error;
};

// Measure seam: mirrors the common_get_device_memory_data signature (fit.h:96).
// The public apply() uses the real function; synthetic tests override it.
using common_fit_measure_fn = std::function<common_device_memory_data_vec(
    const char * path_model,
    const llama_model_params * mparams,
    const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
    uint32_t & hp_ngl,
    uint32_t & hp_n_ctx_train,
    uint32_t & hp_n_expert,
    ggml_log_level log_level)>;

struct common_fit_probe {
    common_fit_measure_fn measure;
    common_fit_measure_fn measure_no_alloc;
};

// Full entry point: analyze -> solve -> serialize -> validate.
// plan != nullptr = manual placement: skip solve, validate the plan, fail loudly on conflict.
// plan_out != nullptr = return the computed plan (including projected_free and delta_free_ema).
// Returns the same status contract as common_fit_params (fit.cpp:1422-1427).
common_params_fit_status common_fit_strategy_apply(
        common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_plan * plan = nullptr,
        common_fit_plan * plan_out = nullptr);

// Internal overload with an injectable measure seam (used by tests).
common_params_fit_status common_fit_strategy_apply_impl(
        common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_plan * plan,
        const common_fit_probe & probe,
        common_fit_plan * plan_out = nullptr);

// Bounded MoE pool budget. user_budget_mib != 0 wins.
// Else: min(clamp(remaining/2, max(2*working_set, 256 MiB), 4 GiB), remaining).
// When remaining < floor: grant = 0 (caller must WRN).
size_t common_fit_strategy_pool_budget(size_t working_set_bytes, size_t remaining_vram, size_t user_budget_mib);

// Compatibility entry point. Active warm OOM allocation probes are disabled
// for safety; this returns an empty ceiling vector.
std::vector<int64_t> common_fit_probe_ceiling_warm(llama_model * model, const common_params & params);

// Activation profile schema (Issue 3, schema + sink only - no decode-path hooks in v1).
struct common_activation_profile {
    int version = 1;
    std::string model_id;
    struct counters_t {
        std::string tensor_id;
        common_fit_role role;
        int64_t n_tokens;
        int64_t expert_set;
        int64_t kv_touched;
    };
    std::vector<counters_t> counters;
};

// JSONL sink: write one profile row. Returns true on success.
bool common_activation_profile_sink(const common_activation_profile & profile, const std::string & path);

// Device ID query functions for proper backend device mapping
// Get the actual GGML backend device index for a logical device index
LLAMA_API int llama_model_get_backend_device_id(const struct llama_model * model, int device_index);

// Get a device handle by backend device index
LLAMA_API ggml_backend_dev_t llama_model_get_device_by_backend_id(const struct llama_model * model, int backend_device_id);

// RPC device-id validation result. duplicate id is soft (warn, allow);
// non-local id and out-of-range id (id < 0 or id == INT_MAX) are hard
// (caller must abort). see fit_validate_rpc_device_ids for the contract.
enum fit_rpc_device_id_error {
    FIT_RPC_DEVICE_ID_OK = 0,
    FIT_RPC_DEVICE_ID_DUPLICATE,    // log WRN, caller may continue
    FIT_RPC_DEVICE_ID_NON_LOCAL,    // log ERR, caller must abort
    FIT_RPC_DEVICE_ID_INVALID,      // log ERR, caller must abort
};

// Validate a user-provided list of RPC device IDs against the local id space.
// First failure wins; per-id checks run in (a) o[k] order:
//   id < 0 or id == INT_MAX -> INVALID
//   id >= local_device_count -> NON_LOCAL
//   id already seen -> DUPLICATE (warn, allow)
// Empty list is OK. The function reports only; callers must abort on
// NON_LOCAL or INVALID.
fit_rpc_device_id_error fit_validate_rpc_device_ids(
    const std::vector<int> & device_ids,
    int local_device_count);
