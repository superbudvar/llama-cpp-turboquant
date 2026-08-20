#include "fit-strategy.h"

#include "../src/llama-ext.h" // llama_model_get_moe_tensor_info (internal)

#include <algorithm>
#include <cstring>
#include <stdexcept>

common_fit_strategy common_fit_strategy_from_string(const std::string & s) {
    if (s == "upstream")    return COMMON_FIT_STRATEGY_UPSTREAM;
    if (s == "rpc-aware")   return COMMON_FIT_STRATEGY_RPC_AWARE;
    if (s == "maxctx")      return COMMON_FIT_STRATEGY_MAXCTX;
    throw std::invalid_argument("unknown fit strategy: " + s);
}

size_t common_fit_strategy_pool_budget(size_t working_set_bytes, size_t remaining_vram, size_t user_budget_mib) {
    if (user_budget_mib != 0) {
        return user_budget_mib;
    }
    const size_t floor = std::max(2 * working_set_bytes, size_t(256) << 20);
    if (remaining_vram < floor) {
        return 0; // caller must WRN: grant 0, never over-commit
    }
    const size_t half  = remaining_vram / 2;
    const size_t cap   = size_t(4) << 30;
    return std::min(std::min(std::max(half, floor), cap), remaining_vram);
}

bool common_activation_profile_sink(const common_activation_profile & profile, const std::string & path) {
    // Schema + sink only (Issue 3). Consumers are follow-up work.
    return path.size() > 0 && profile.version > 0;
}

common_params_fit_status common_fit_strategy_apply_impl(
        common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_plan * plan,
        const common_fit_probe & probe) {
    // Contract surface (Slice 0). analyze/solve/serialize/validate land in Issue 2.
    (void) params;
    (void) mparams;
    (void) cparams;
    (void) plan;
    (void) probe;
    return COMMON_FIT_STRATEGY_UPSTREAM == params.fit_strategy
        ? COMMON_PARAMS_FIT_STATUS_SUCCESS
        : COMMON_PARAMS_FIT_STATUS_ERROR;
}

common_params_fit_status common_fit_strategy_apply(
        common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_plan * plan) {
    common_fit_probe probe;
    probe.measure = [](const char * path_model,
                       const llama_model_params * mparams,
                       const llama_context_params * cparams,
                       std::vector<ggml_backend_dev_t> & devs,
                       uint32_t & hp_ngl,
                       uint32_t & hp_n_ctx_train,
                       uint32_t & hp_n_expert,
                       ggml_log_level log_level) {
        return common_get_device_memory_data(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);
    };
    return common_fit_strategy_apply_impl(params, mparams, cparams, plan, probe);
}