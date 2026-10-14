#include "fit-probe-internal.h"

#include "common.h"
#include "fit.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct common_fit_logger_guard {
    ggml_log_callback original_callback;
    void * original_user_data;
    ggml_log_level min_level;

    explicit common_fit_logger_guard(ggml_log_level min_level) : min_level(min_level) {
        llama_log_get(&original_callback, &original_user_data);
        llama_log_set(callback, this);
    }

    ~common_fit_logger_guard() {
        llama_log_set(original_callback, original_user_data);
    }

    static void callback(ggml_log_level level, const char * text, void * user_data) {
        const common_fit_logger_guard * guard = (const common_fit_logger_guard *) user_data;
        const ggml_log_level level_eff = level >= guard->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        guard->original_callback(level_eff, text, guard->original_user_data);
    }
};


std::vector<llama_device_memory_data> common_get_device_memory_data_impl(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams_in,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level,
        std::vector<llama_moe_tensor_info> * moe_tensors,
        llama_context * ctx_parent,
        uint32_t * n_layer_all,
        bool no_alloc,
        per_device_layer_costs * per_layer_costs) {
    common_fit_logger_guard logger_guard(log_level);

    // stack-local cparams: the linear-interpolation branch below mutates
    // cparams->n_ctx to probe the model at min_ctx and at an interpolated
    // value. the caller's cparams would leak that mutation back, leaving
    // params.n_ctx = 4096 for every subsequent probe and the post-fit real
    // load. take a copy here and use it everywhere; the caller's cparams
    // is never written through.
    llama_context_params cparams_local = *cparams_in;
    llama_context_params * cparams = &cparams_local;

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = no_alloc;
    mparams_copy.load_mode = LLAMA_LOAD_MODE_MMAP;

    llama_model_ptr model(llama_model_load_from_file(path_model, mparams_copy));
    if (model == nullptr) {
        throw std::runtime_error("failed to load model");
    }

    llama_context_params cparams_copy = *cparams;
    if (ctx_parent != nullptr) {
        cparams_copy.ctx_other = ctx_parent;
    }

    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams_copy));
    if (ctx == nullptr) {
        throw std::runtime_error("failed to create llama_context from model");
    }

    const size_t nd = llama_model_n_devices(model.get());
    std::vector<llama_device_memory_data> ret(nd + 1);

    if (per_layer_costs) {
        per_layer_costs->n_layer_all = llama_model_n_layer(model.get()) + llama_model_n_layer_nextn(model.get());
        per_layer_costs->n_device = nd;
        per_layer_costs->bytes.assign(per_layer_costs->n_layer_all * per_layer_costs->n_device, 0);
        per_layer_costs->model_bytes.assign(nd + 1, 0);
        per_layer_costs->context_compute.assign(nd, 0);
        per_layer_costs->layer_dev.assign(per_layer_costs->n_layer_all, -1);
        const size_t count = llama_model_get_tensor_devices(model.get(), nullptr, 0);
        std::vector<llama_tensor_device_info> tensor_info(count);
        llama_model_get_tensor_devices(model.get(), tensor_info.data(), tensor_info.size());
        for (const auto & ti : tensor_info) {
            int layer = -1;
            if (strstr(ti.name, "blk.") == ti.name) {
                const char * dot1 = ti.name + 4;
                layer = atoi(dot1);
            }
            size_t dev_id = nd; // default: host (CPU)
            if (ti.dev) {
                for (size_t i = 0; i < nd; i++) {
                    if (ti.dev == llama_model_get_device(model.get(), i)) {
                        dev_id = i;
                        break;
                    }
                }
            }
            per_layer_costs->model_bytes[dev_id] += (int64_t) ti.bytes;
            if (layer >= 0 && (size_t)layer < per_layer_costs->n_layer_all && dev_id < nd) {
                per_layer_costs->bytes[layer * nd + dev_id] += ti.bytes;
                per_layer_costs->layer_dev[layer] = (int32_t) dev_id;
            }
        }
    }

    llama_memory_breakdown memory_breakdown = llama_get_memory_breakdown(ctx.get());

    for (const auto & [buft, mb] : memory_breakdown) {
        if (ggml_backend_buft_is_host(buft)) {
            ret.back().mb.model   += mb.model;
            ret.back().mb.context += mb.context;
            ret.back().mb.compute += mb.compute;
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < nd; i++) {
            if (dev == llama_model_get_device(model.get(), i)) {
                ret[i].mb.model   += mb.model;
                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                break;
            }
        }
    }

    if (per_layer_costs) {
        for (size_t i = 0; i < ret.size() && i < per_layer_costs->model_bytes.size(); i++) {
            ret[i].mb.model = (size_t) per_layer_costs->model_bytes[i];
        }
    }

    {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (cpu_dev == nullptr) {
            throw std::runtime_error("no CPU backend found");
        }
        size_t free;
        size_t total;
        ggml_backend_dev_memory(cpu_dev, &free, &total);
        ret.back().free  = free;
        ret.back().total = total;
    }
    for (size_t i = 0; i < nd; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model.get(), i);

        size_t free;
        size_t total;
        ggml_backend_dev_memory(dev, &free, &total);

        // Some non-GPU accelerator backends, such as BLAS, report 0/0 and rely on
        // the host-memory fallback. For GPU-like backends, keep 0/0 so --fit does
        // not assign anything to a device with an unknown memory budget.
        if (free == 0 && total == 0) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                LOG_WRN("%s: device %s did not report memory; --fit will not use it\n",
                        __func__, ggml_backend_dev_name(dev));
            } else {
                free  = ret.back().free;
                total = ret.back().total;
            }
        }
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs.clear();
    for (int i = 0; i < llama_model_n_devices(model.get()); i++) {
        devs.push_back(llama_model_get_device(model.get(), i));
    }

    if (n_layer_all) {
        // the loader always reserves n_layer_all + 1 slots regardless of
        // load_mtp: a skipped in-file MTP block still occupies its layer slot
        *n_layer_all = llama_model_n_layer(model.get()) + llama_model_n_layer_nextn(model.get());
    }
    hp_ngl         = llama_model_n_layer(model.get());
    if (mparams->load_mtp) {
        hp_ngl    += llama_model_n_layer_nextn(model.get());
    }
    hp_n_ctx_train = llama_model_n_ctx_train(model.get());
    hp_n_expert    = llama_model_n_expert(model.get());

    if (moe_tensors) {
        const size_t count = llama_model_get_moe_tensor_info(model.get(), nullptr, 0);
        moe_tensors->resize(count);
        const size_t written = llama_model_get_moe_tensor_info(model.get(), moe_tensors->data(), moe_tensors->size());
        GGML_ASSERT(written == count);
    }

    common_memory_breakdown_print(ctx.get());

    return ret;
}

common_device_memory_data_vec common_get_device_memory_data(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    std::vector<llama_device_memory_data> impl = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level, nullptr, nullptr, nullptr, false);

    common_device_memory_data_vec ret(impl.size());
    for (size_t i = 0; i < impl.size(); i++) {
        ret[i].total   = impl[i].total;
        ret[i].free    = impl[i].free;
        ret[i].model   = impl[i].mb.model;
        ret[i].context = impl[i].mb.context;
        ret[i].compute = impl[i].mb.compute;
    }
    return ret;
}

common_device_memory_data_vec common_get_device_memory_data_no_alloc(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level) {
    per_device_layer_costs costs;
    std::vector<llama_device_memory_data> impl = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert,
            log_level, nullptr, nullptr, nullptr, true, &costs);

    common_device_memory_data_vec ret(impl.size());
    for (size_t i = 0; i < impl.size(); i++) {
        ret[i].total   = impl[i].total;
        ret[i].free    = impl[i].free;
        ret[i].model   = impl[i].mb.model;
        ret[i].context = impl[i].mb.context;
        ret[i].compute = impl[i].mb.compute;
    }
    return ret;
}

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
        bool no_alloc_child) {
    common_fit_logger_guard logger_guard(log_level);

    llama_model_params mparams_parent_copy = *mparams_parent;
    mparams_parent_copy.no_alloc  = true;
    mparams_parent_copy.load_mode = LLAMA_LOAD_MODE_MMAP;

    llama_model_ptr model_parent(llama_model_load_from_file(path_parent, mparams_parent_copy));
    if (model_parent == nullptr) {
        throw std::runtime_error("failed to load parent model");
    }

    llama_context_ptr ctx_parent(llama_init_from_model(model_parent.get(), *cparams_parent));
    if (ctx_parent == nullptr) {
        throw std::runtime_error("failed to create parent llama_context");
    }

    per_device_layer_costs child_costs;
    std::vector<llama_device_memory_data> impl = common_get_device_memory_data_impl(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert,
            log_level, nullptr, ctx_parent.get(), nullptr, no_alloc_child,
            no_alloc_child ? &child_costs : nullptr);

    common_device_memory_data_vec ret(impl.size());
    for (size_t i = 0; i < impl.size(); i++) {
        ret[i].total   = impl[i].total;
        ret[i].free    = impl[i].free;
        ret[i].model   = impl[i].mb.model;
        ret[i].context = impl[i].mb.context;
        ret[i].compute = impl[i].mb.compute;
    }
    return ret;
}
