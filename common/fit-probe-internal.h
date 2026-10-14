#pragma once

#include "fit-probe.h"
#include "../src/llama-ext.h"

#include <cstdint>
#include <string>
#include <vector>

struct per_device_layer_costs {
    size_t n_layer_all = 0;
    size_t n_device = 0;
    std::vector<int64_t> bytes;
    std::vector<int64_t> model_bytes;
    std::vector<int64_t> context_compute;
    std::vector<int32_t> layer_dev;
};

std::vector<llama_device_memory_data> common_get_device_memory_data_impl(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level,
        std::vector<llama_moe_tensor_info> * moe_tensors = nullptr,
        llama_context * ctx_parent = nullptr,
        uint32_t * n_layer_all = nullptr,
        bool no_alloc = false,
        per_device_layer_costs * per_layer_costs = nullptr);
