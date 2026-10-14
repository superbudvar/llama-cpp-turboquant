#pragma once

#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <vector>

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t model;
    size_t context;
    size_t compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model and context, then return the measured per-device footprint.
common_device_memory_data_vec common_get_device_memory_data(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level);

// Measure the requested tensor placement without allocating model weight
// buffers. Context and compute allocations are still measured.
common_device_memory_data_vec common_get_device_memory_data_no_alloc(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level);

// Load a no-allocation parent context, then measure a child model/context on it.
common_device_memory_data_vec common_get_device_memory_data_with_parent(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        const char * path_parent,
        const llama_model_params * mparams_parent,
        const llama_context_params * cparams_parent,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level,
        bool no_alloc_child = false);
