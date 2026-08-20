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
};

struct common_fit_plan {
    // explicit per-role device assignment (device ids from common_fit_analysis.devices)
    std::vector<int> role_devices; // indexed by common_fit_role
    int32_t n_gpu_layers = -1;
    std::vector<float> tensor_split;
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;
    int32_t n_ctx = 0;
    size_t pool_grant_mib = 0; // 0 = no pool
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
};

// Full entry point: analyze -> solve -> serialize -> validate.
// plan != nullptr = manual placement: skip solve, validate the plan, fail loudly on conflict.
// Returns the same status contract as common_fit_params (fit.cpp:1422-1427).
common_params_fit_status common_fit_strategy_apply(
        common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_plan * plan = nullptr);

// Internal overload with an injectable measure seam (used by tests).
common_params_fit_status common_fit_strategy_apply_impl(
        common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_plan * plan,
        const common_fit_probe & probe);

// Bounded MoE pool budget. user_budget_mib != 0 wins.
// Else: min(clamp(remaining/2, max(2*working_set, 256 MiB), 4 GiB), remaining).
// When remaining < floor: grant = 0 (caller must WRN).
size_t common_fit_strategy_pool_budget(size_t working_set_bytes, size_t remaining_vram, size_t user_budget_mib);

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