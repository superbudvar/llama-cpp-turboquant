#include "fit-strategy.h"

#include "../src/llama-ext.h" // llama_model_get_moe_tensor_info, llama_model_get_tensor_devices (internal)
#include "common.h"

#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// local copy of the guard exception from fit.cpp. fit.cpp's copy is private to
// that translation unit and the probe seam never throws it, so the strategy
// engine keeps its own for guard throws + the outer catch (same-TU catch works).
class common_params_fit_exception : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

static const char * fit_role_name(common_fit_role role) {
    switch (role) {
        case COMMON_FIT_ROLE_ATTN_QKV:    return "attn_qkv";
        case COMMON_FIT_ROLE_ATTN_O:      return "attn_o";
        case COMMON_FIT_ROLE_DENSE_FFN:   return "dense_ffn";
        case COMMON_FIT_ROLE_EXPERT_UP:   return "expert_up";
        case COMMON_FIT_ROLE_EXPERT_GATE: return "expert_gate";
        case COMMON_FIT_ROLE_EXPERT_DOWN: return "expert_down";
        case COMMON_FIT_ROLE_NORM:        return "norm";
        case COMMON_FIT_ROLE_EMBED:       return "embed";
    }
    return "unknown";
}

// classify a GGUF tensor name into a fit role. expert tensors are matched
// first so dense checks never catch the _exps variants.
static common_fit_role fit_classify_role(const std::string & name) {
    const bool is_expert = name.find("_exps.weight") != std::string::npos ||
                           name.find("_chexps.weight") != std::string::npos;
    if (is_expert) {
        if (name.find(".ffn_gate_up") != std::string::npos || name.find(".ffn_up") != std::string::npos) {
            return COMMON_FIT_ROLE_EXPERT_UP;
        }
        if (name.find(".ffn_gate") != std::string::npos) {
            return COMMON_FIT_ROLE_EXPERT_GATE;
        }
        if (name.find(".ffn_down") != std::string::npos) {
            return COMMON_FIT_ROLE_EXPERT_DOWN;
        }
        return COMMON_FIT_ROLE_EXPERT_UP;
    }
    if (name.find(".attn_qkv") != std::string::npos) {
        return COMMON_FIT_ROLE_ATTN_QKV;
    }
    if (name.find(".attn_q") != std::string::npos || name.find(".attn_k") != std::string::npos ||
        name.find(".attn_v") != std::string::npos) {
        return COMMON_FIT_ROLE_ATTN_QKV;
    }
    if (name.find(".attn_o") != std::string::npos || name.find(".attn_output") != std::string::npos) {
        return COMMON_FIT_ROLE_ATTN_O;
    }
    if (name.find(".ffn_gate") != std::string::npos || name.find(".ffn_up") != std::string::npos ||
        name.find(".ffn_down") != std::string::npos || name.find(".ffn_gate_up") != std::string::npos) {
        return COMMON_FIT_ROLE_DENSE_FFN;
    }
    if (name.find("token_embd") != std::string::npos || name.compare(0, 6, "output") == 0) {
        return COMMON_FIT_ROLE_EMBED;
    }
    if (name.find("norm") != std::string::npos) {
        return COMMON_FIT_ROLE_NORM;
    }
    return COMMON_FIT_ROLE_NORM;
}

static bool fit_is_rpc_device(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return false;
    }
    const char * reg_name = ggml_backend_reg_name(reg);
    return reg_name != nullptr && strstr(reg_name, "RPC") != nullptr;
}

// the device list index of the best non-RPC device by free memory, or -1
static int fit_best_local(const common_fit_analysis & analysis, int exclude = -1) {
    int best = -1;
    for (const auto & d : analysis.devices) {
        if (d.is_rpc || d.id == exclude || d.free_bytes <= 0) {
            continue;
        }
        if (best < 0 || d.free_bytes > analysis.devices[best].free_bytes) {
            best = d.id;
        }
    }
    return best;
}

static int fit_best_rpc(const common_fit_analysis & analysis) {
    int best = -1;
    for (const auto & d : analysis.devices) {
        if (!d.is_rpc || d.free_bytes <= 0) {
            continue;
        }
        if (best < 0 || d.free_bytes > analysis.devices[best].free_bytes) {
            best = d.id;
        }
    }
    return best;
}

// first non-RPC device, mirroring server_effective_main_gpu
static int fit_effective_main_gpu(const common_fit_analysis & analysis, int user_main_gpu) {
    if (user_main_gpu >= 0 && user_main_gpu < (int) analysis.devices.size() &&
        !analysis.devices[user_main_gpu].is_rpc) {
        return user_main_gpu;
    }
    for (const auto & d : analysis.devices) {
        if (!d.is_rpc) {
            return d.id;
        }
    }
    return analysis.devices.empty() ? -1 : analysis.devices[0].id;
}

// rpc-aware: shift the margin of an RPC main_gpu onto the first non-RPC device.
// margins are indexed by model device order (same space as analysis.devices).
static void fit_reattribute_margins(
        std::vector<size_t> & margins, const common_fit_analysis & analysis, int user_main_gpu) {
    if (user_main_gpu < 0 || user_main_gpu >= (int) analysis.devices.size()) {
        return;
    }
    if (!analysis.devices[user_main_gpu].is_rpc) {
        return;
    }
    for (const auto & d : analysis.devices) {
        if (!d.is_rpc && d.id < (int) margins.size() && user_main_gpu < (int) margins.size()) {
            std::swap(margins[user_main_gpu], margins[d.id]);
            return;
        }
    }
}

struct fit_measure_result {
    common_device_memory_data_vec memory;
    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0;
    uint32_t hp_nct = 0;
    uint32_t hp_nex = 0;
};

// measure a concrete plan through the probe seam. returns false when the
// context cannot be created at this placement (soft reject); hard errors and
// model-not-found rethrow.
static bool fit_measure_plan(
        const common_fit_probe & probe,
        const common_fit_plan & plan,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        const std::string & path_model,
        fit_measure_result & out) {
    if (!probe.measure) {
        throw std::runtime_error("fit probe has no measure function");
    }
    std::vector<float> split_storage;
    std::vector<llama_model_tensor_buft_override> override_storage;
    llama_model_params cp = *mparams;
    llama_context_params cc = *cparams;
    split_storage = plan.tensor_split;
    override_storage = plan.tensor_buft_overrides;
    cp.n_gpu_layers = plan.n_gpu_layers;
    cp.use_extra_bufts = false;
    cp.tensor_split = split_storage.empty() ? nullptr : split_storage.data();
    cp.tensor_buft_overrides = override_storage.empty() ? nullptr : override_storage.data();
    if (plan.n_ctx > 0) {
        cc.n_ctx = plan.n_ctx;
    }
    try {
        out.memory = probe.measure(path_model.c_str(), &cp, &cc,
                out.devs, out.hp_ngl, out.hp_nct, out.hp_nex, GGML_LOG_LEVEL_INFO);
    } catch (const std::runtime_error & e) {
        if (std::string(e.what()).find("failed to create llama_context") != std::string::npos) {
            return false;
        }
        throw;
    }
    return true;
}

static bool fit_same_devices(const std::vector<ggml_backend_dev_t> & devs, const common_fit_analysis & analysis) {
    if (devs.size() != analysis.devices.size()) {
        return false;
    }
    for (size_t i = 0; i < devs.size(); i++) {
        if (devs[i] != analysis.devices[i].dev) {
            return false;
        }
    }
    return true;
}

// mirror fit.cpp:894-900 per-device margin check
static int64_t fit_device_used(const common_device_memory_data & dmd) {
    return (int64_t) dmd.model + (int64_t) dmd.context + (int64_t) dmd.compute;
}

static bool fit_fits_margins(
        const common_device_memory_data_vec & memory,
        const std::vector<int64_t> & margins,
        size_t nd) {
    if (memory.size() != nd + 1 || margins.size() < nd) {
        return false;
    }
    for (size_t id = 0; id < nd; id++) {
        if (fit_device_used(memory[id]) > INT64_MAX ||
            memory[id].free - fit_device_used(memory[id]) < margins[id]) {
            return false;
        }
    }
    return true;
}

// remaining VRAM on a device after the projected model/context usage, the
// margin, and a reserve. the reserve falls back to 10% of free when the raw
// unaccounted value is negative (i.e. the probe free already covers the model).
static int64_t fit_device_remaining(const common_device_memory_data & dmd, int64_t margin) {
    const int64_t unaccounted_raw = dmd.total - dmd.free - fit_device_used(dmd);
    const int64_t unaccounted = unaccounted_raw < 0 ? (int64_t) (dmd.free * 10 / 100) : unaccounted_raw;
    return dmd.free - margin - unaccounted;
}

// load model metadata without allocating buffers, mirror fit.cpp:302-330
static common_fit_analysis fit_analyze(
        const common_params & params,
        llama_model_params * mparams,
        const std::string & path_model,
        uint32_t & hp_ngl) {
    const llama_model_params default_mparams = llama_model_default_params();

    // honor-or-abort guard, mirror fit.cpp:702-728
    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        throw common_params_fit_exception("n_gpu_layers already set by user to " +
                std::to_string(mparams->n_gpu_layers) + ", abort");
    }
    if (mparams->split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        throw common_params_fit_exception("fit-strategy not implemented for LLAMA_SPLIT_MODE_TENSOR, abort");
    }
    if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
        throw common_params_fit_exception("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
    }
    if (mparams->tensor_split) {
        for (size_t id = 0; id < llama_max_devices(); id++) {
            if (mparams->tensor_split[id] != 0.0f) {
                throw common_params_fit_exception("model_params::tensor_split already set by user, abort");
            }
        }
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        throw common_params_fit_exception("model_params::tensor_buft_overrides already set by user, abort");
    }

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc  = true;
    mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;

    llama_model_ptr model(llama_model_load_from_file(path_model.c_str(), mparams_copy));
    if (model == nullptr) {
        throw std::runtime_error("failed to load model");
    }

    common_fit_analysis analysis;
    analysis.rpc_hosts_set = !params.rpc_hosts.empty();

    const int32_t nd = llama_model_n_devices(model.get());
    for (int32_t i = 0; i < nd; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model.get(), i);

        const bool is_rpc = fit_is_rpc_device(dev);
        if (is_rpc) {
            analysis.rpc_available = true;
        }

        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);

        // GPU devices reporting 0/0 are dead or excluded, mirror fit.cpp:379-391
        if (free == 0 && total == 0) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                LOG_WRN("%s: device %s did not report memory; excluding it from the fit\n",
                        __func__, ggml_backend_dev_name(dev));
            }
        }

        common_fit_device d;
        d.id = (int) i;
        d.dev = dev;
        d.is_rpc = is_rpc;
        d.free_bytes = (int64_t) free;
        d.total_bytes = (int64_t) total;
        d.link_bw_mbps = 0;    // unknown -> conservative default
        d.link_latency_us = 0; // unknown
        analysis.devices.push_back(d);
    }

    if (analysis.rpc_hosts_set && !analysis.rpc_available) {
        LOG_WRN("%s: rpc_hosts configured but no RPC device responded; fitting local devices only\n", __func__);
    }

    // per-tensor roles for relative ordering. the no_alloc load assigns dummy
    // buffers, so the read-back helper returns the intended device placement.
    const size_t n_tensors = llama_model_get_tensor_devices(model.get(), nullptr, 0);
    std::vector<llama_tensor_device_info> tensor_infos(n_tensors);
    if (n_tensors > 0) {
        llama_model_get_tensor_devices(model.get(), tensor_infos.data(), n_tensors);
    }
    analysis.tensors.reserve(n_tensors);
    for (const auto & info : tensor_infos) {
        common_fit_tensor_meta meta;
        meta.role = fit_classify_role(info.name);
        meta.tensor_id = info.name;
        meta.bytes = info.bytes;
        analysis.tensors.push_back(meta);
    }

    // MoE metadata: working set = per-expert bytes of one layer times n_expert_used
    analysis.moe.n_expert = llama_model_n_expert(model.get());
    analysis.moe.n_expert_used = llama_model_n_expert_used(model.get());
    const size_t moe_count = llama_model_get_moe_tensor_info(model.get(), nullptr, 0);
    std::vector<llama_moe_tensor_info> moe_tensors(moe_count);
    if (moe_count > 0) {
        llama_model_get_moe_tensor_info(model.get(), moe_tensors.data(), moe_count);
    }
    size_t layer_expert_bytes = 0;
    for (const auto & mt : moe_tensors) {
        if (mt.layer == 0) {
            layer_expert_bytes += mt.expert_size;
        }
    }
    analysis.moe.expert_bytes = layer_expert_bytes;
    analysis.moe.working_set_bytes = layer_expert_bytes * analysis.moe.n_expert_used;

    analysis.model_id = path_model;

    hp_ngl = llama_model_n_layer(model.get());
    if (mparams->load_mtp) {
        hp_ngl += llama_model_n_layer_nextn(model.get());
    }

    return analysis;
}

struct fit_candidate_spec {
    std::string label;
    int dense_main;              // primary device for dense roles
    std::vector<int> split_devices; // devices receiving dense tensors via tensor_split
    int pool;                    // expert pool device id, -1 = CPU
};

// build a plan from a candidate spec. n_ctx <= 0 means "leave cparams untouched".
static common_fit_plan fit_plan_from_spec(
        const common_fit_analysis & analysis,
        const fit_candidate_spec & spec,
        uint32_t hp_ngl,
        int32_t n_ctx) {
    common_fit_plan plan;
    plan.role_devices.assign(8, -1);
    const int dense_main = spec.dense_main;

    for (size_t r = 0; r < plan.role_devices.size(); r++) {
        if (r == COMMON_FIT_ROLE_EXPERT_UP || r == COMMON_FIT_ROLE_EXPERT_GATE ||
            r == COMMON_FIT_ROLE_EXPERT_DOWN) {
            plan.role_devices[r] = spec.pool;
        } else {
            plan.role_devices[r] = dense_main;
        }
    }

    plan.n_gpu_layers = (int32_t) (hp_ngl + 1);
    plan.n_ctx = n_ctx;

    plan.tensor_split.assign(llama_max_devices(), 0.0f);
    const float frac = 1.0f / spec.split_devices.size();
    for (int id : spec.split_devices) {
        plan.tensor_split[id] = frac;
    }

    if (analysis.moe.n_expert > 0) {
        ggml_backend_buffer_type_t pool_buft = spec.pool >= 0
            ? ggml_backend_dev_buffer_type(analysis.devices[spec.pool].dev)
            : ggml_backend_cpu_buffer_type();
        plan.tensor_buft_overrides = {
            { common_moe_cache_tensor_override_pattern(), pool_buft },
            { nullptr, nullptr },
        };
    }

    return plan;
}

// solve for rpc-aware: dense decode-hot on local non-RPC devices, experts on a
// pool that is never the main_gpu device. first fit wins; returns false when no
// candidate fits (caller falls back to the stock solver).
static bool fit_solve_rpc_aware(
        const common_fit_analysis & analysis,
        const std::vector<int64_t> & margins,
        const common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_probe & probe,
        const std::string & path_model,
        uint32_t hp_ngl,
        common_fit_plan & out_plan,
        fit_measure_result & out_measure) {
    const int effective_main = fit_effective_main_gpu(analysis, params.main_gpu);
    if (params.main_gpu >= 0 && params.main_gpu < (int) analysis.devices.size() &&
        analysis.devices[params.main_gpu].is_rpc) {
        LOG_WRN("%s: main_gpu %d is an RPC device; using local device %d instead\n",
                __func__, params.main_gpu, effective_main);
    }

    const int local_best = fit_best_local(analysis);
    if (local_best < 0) {
        throw std::runtime_error("rpc-aware strategy found no local (non-RPC) device to place the model on");
    }

    // dense placements: single local, split across local, split local+rpc
    std::vector<std::vector<int>> dense_splits;
    dense_splits.push_back({ local_best });
    std::vector<int> all_local;
    for (const auto & d : analysis.devices) {
        if (!d.is_rpc && d.free_bytes > 0) {
            all_local.push_back(d.id);
        }
    }
    if (all_local.size() > 1) {
        dense_splits.push_back(all_local);
    }
    const int rpc_best = fit_best_rpc(analysis);
    if (rpc_best >= 0 && !all_local.empty()) {
        std::vector<int> mix = all_local;
        mix.push_back(rpc_best);
        dense_splits.push_back(mix);
    }

    // pool options: best local excluding main_gpu, best RPC, CPU
    std::vector<int> pool_opts;
    if (analysis.moe.n_expert > 0) {
        const int local_pool = fit_best_local(analysis, effective_main);
        if (local_pool >= 0) {
            pool_opts.push_back(local_pool);
        }
        if (rpc_best >= 0) {
            pool_opts.push_back(rpc_best);
        }
        pool_opts.push_back(-1); // CPU
    }

    std::vector<fit_candidate_spec> candidates;
    for (const auto & ds : dense_splits) {
        const int dense_main = ds.front();
        if (analysis.moe.n_expert > 0) {
            for (int pool : pool_opts) {
                std::string label = "dense=" + std::string(ggml_backend_dev_name(analysis.devices[dense_main].dev));
                label += ",pool=" + std::string(pool >= 0 ? ggml_backend_dev_name(analysis.devices[pool].dev) : "cpu");
                candidates.push_back({ label, dense_main, ds, pool });
            }
        } else {
            candidates.push_back({ std::string("dense=") + ggml_backend_dev_name(analysis.devices[dense_main].dev),
                    dense_main, ds, dense_main });
        }
    }

    const int32_t n_ctx = cparams->n_ctx > 0 ? (int32_t) cparams->n_ctx : 0;
    for (const auto & cand : candidates) {
        const common_fit_plan plan = fit_plan_from_spec(analysis, cand, hp_ngl, n_ctx);
        fit_measure_result m;
        if (!fit_measure_plan(probe, plan, mparams, cparams, path_model, m)) {
            LOG_INF("%s: candidate %s cannot create a context, skipping\n", __func__, cand.label.c_str());
            continue;
        }
        if (!fit_same_devices(m.devs, analysis)) {
            LOG_WRN("%s: candidate %s changed the device set, skipping\n", __func__, cand.label.c_str());
            continue;
        }
        if (!fit_fits_margins(m.memory, margins, analysis.devices.size())) {
            LOG_INF("%s: candidate %s does not fit within margins, skipping\n", __func__, cand.label.c_str());
            continue;
        }
        out_plan = plan;
        out_measure = m;
        LOG_INF("%s: selected candidate %s\n", __func__, cand.label.c_str());
        return true;
    }

    return false;
}

// solve for maxctx: two-anchor interpolation per placement, mirror fit.cpp:667-669.
// the floor is fit_strategy_min_ctx; an unreachable floor is a hard error.
static common_fit_plan fit_solve_maxctx(
        const common_fit_analysis & analysis,
        const std::vector<int64_t> & margins,
        const common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_probe & probe,
        const std::string & path_model,
        uint32_t hp_ngl,
        fit_measure_result & out_measure) {
    if (params.fit_strategy_min_ctx <= 0) {
        throw common_params_fit_exception("maxctx strategy requires --fit-strategy-min-ctx");
    }
    const int32_t target_ctx = params.fit_strategy_min_ctx;
    const int32_t upper_ctx = params.n_ctx > 0 ? params.n_ctx : 0; // 0 = full train context

    const int local_best = fit_best_local(analysis);
    if (local_best < 0) {
        throw std::runtime_error("maxctx strategy found no usable local device");
    }

    // placements: full GPU offload first, experts to CPU second
    std::vector<fit_candidate_spec> placements;
    placements.push_back({ "full-gpu", local_best, { local_best }, local_best });
    if (analysis.moe.n_expert > 0) {
        placements.push_back({ "experts-cpu", local_best, { local_best }, -1 });
    }

    const size_t nd = analysis.devices.size();
    for (const auto & placement : placements) {
        // anchor 1: upper context
        const common_fit_plan plan_upper = fit_plan_from_spec(analysis, placement, hp_ngl, upper_ctx);
        fit_measure_result m_upper;
        if (!fit_measure_plan(probe, plan_upper, mparams, cparams, path_model, m_upper) ||
            !fit_same_devices(m_upper.devs, analysis)) {
            LOG_INF("%s: placement %s cannot create a context at the upper context, skipping\n",
                    __func__, placement.label.c_str());
            continue;
        }
        if (fit_fits_margins(m_upper.memory, margins, nd)) {
            LOG_INF("%s: placement %s fits at the upper context\n", __func__, placement.label.c_str());
            out_measure = m_upper;
            return plan_upper;
        }

        // anchor 2: target (floor) context
        const common_fit_plan plan_target = fit_plan_from_spec(analysis, placement, hp_ngl, target_ctx);
        fit_measure_result m_target;
        if (!fit_measure_plan(probe, plan_target, mparams, cparams, path_model, m_target) ||
            !fit_same_devices(m_target.devs, analysis)) {
            LOG_INF("%s: placement %s cannot create a context at the floor context, skipping\n",
                    __func__, placement.label.c_str());
            continue;
        }
        if (!fit_fits_margins(m_target.memory, margins, nd)) {
            LOG_INF("%s: placement %s cannot meet margins even at the floor context, skipping\n",
                    __func__, placement.label.c_str());
            continue;
        }

        // interpolate between the two anchors
        int64_t sum_upper = 0;
        int64_t sum_target = 0;
        int64_t budget = 0;
        for (size_t id = 0; id < nd; id++) {
            sum_upper += fit_device_used(m_upper.memory[id]);
            sum_target += fit_device_used(m_target.memory[id]);
            budget += m_upper.memory[id].free - margins[id];
        }
        const int32_t hp_nct = (int32_t) m_upper.hp_nct;
        int32_t n_ctx = target_ctx;
        const int64_t ctx_span = sum_upper - sum_target;
        if (ctx_span > 0 && budget <= sum_upper) {
            const int32_t upper = upper_ctx > 0 ? upper_ctx : hp_nct;
            n_ctx = target_ctx + (int32_t) ((int64_t) (upper - target_ctx) *
                    (budget - sum_target) / ctx_span);
            n_ctx = std::max(n_ctx - n_ctx % 256, target_ctx); // round down for CUDA
        }

        // final confirmation measure
        const common_fit_plan plan_int = fit_plan_from_spec(analysis, placement, hp_ngl, n_ctx);
        fit_measure_result m_int;
        if (!fit_measure_plan(probe, plan_int, mparams, cparams, path_model, m_int) ||
            !fit_same_devices(m_int.devs, analysis) ||
            !fit_fits_margins(m_int.memory, margins, nd)) {
            // interpolation overshot; the floor is known to fit
            LOG_INF("%s: placement %s did not confirm at %" PRIi32 ", using the floor context\n",
                    __func__, placement.label.c_str(), n_ctx);
            out_measure = m_target;
            return plan_target;
        }

        LOG_INF("%s: placement %s selected context %" PRIi32 "\n", __func__, placement.label.c_str(), n_ctx);
        out_measure = m_int;
        return plan_int;
    }

    throw std::runtime_error("maxctx strategy could not fit the model at the minimum context of " +
            std::to_string(target_ctx) + " tokens; try a smaller --fit-strategy-min-ctx or free device memory");
}

static void fit_serialize_moe_cache(
        common_params & params,
        llama_context_params * cparams,
        const common_fit_analysis & analysis,
        const common_fit_plan & plan,
        const fit_measure_result & measure,
        const std::vector<int64_t> & margins) {
    common_moe_cache_params & mc = params.moe_cache;

    // no experts -> nothing to pool
    if (analysis.moe.n_expert <= 0) {
        return;
    }

    // explicit OFF is never overridden
    if (mc.mode_explicit && mc.mode == COMMON_MOE_CACHE_MODE_OFF) {
        mc.fit_selected = false;
        return;
    }

    const int pool_id = plan.role_devices[COMMON_FIT_ROLE_EXPERT_UP];
    int64_t remaining = 0;
    if (pool_id >= 0 && pool_id < (int) measure.memory.size() && pool_id < (int) margins.size()) {
        remaining = fit_device_remaining(measure.memory[pool_id], margins[pool_id]);
        if (remaining < 0) {
            remaining = 0;
        }
    }

    const size_t user_budget_mib = (mc.mode_explicit && mc.mode != COMMON_MOE_CACHE_MODE_OFF)
        ? mc.budget_mib : 0;
    const size_t grant = common_fit_strategy_pool_budget(
            analysis.moe.working_set_bytes, (size_t) remaining, user_budget_mib);

    if (grant == 0) {
        // budget_mib == 0 means unlimited, so OFF must be explicit
        mc.mode = COMMON_MOE_CACHE_MODE_OFF;
        mc.budget_mib = 0;
        mc.fit_selected = false;
        LOG_WRN("%s: no VRAM remaining for an MoE pool on device %d; pool disabled\n", __func__, pool_id);
    } else {
        if (mc.mode_explicit && mc.mode == COMMON_MOE_CACHE_MODE_SOFT) {
            mc.mode = COMMON_MOE_CACHE_MODE_SOFT;
        } else {
            mc.mode = COMMON_MOE_CACHE_MODE_ON;
        }
        mc.budget_mib = grant;
        mc.fit_selected = true;
        LOG_INF("%s: MoE pool budget %zu MiB on device %s\n", __func__, grant,
                pool_id >= 0 ? ggml_backend_dev_name(analysis.devices[pool_id].dev) : "cpu");
    }

    // remap the context params to the resolved mode, mirroring common_context_params_to_llama
    switch (mc.mode) {
        case COMMON_MOE_CACHE_MODE_OFF:
            cparams->moe_cache_mode = LLAMA_MOE_CACHE_MODE_OFF;
            break;
        case COMMON_MOE_CACHE_MODE_AUTO:
            cparams->moe_cache_mode = LLAMA_MOE_CACHE_MODE_AUTO;
            break;
        case COMMON_MOE_CACHE_MODE_ON:
        case COMMON_MOE_CACHE_MODE_SOFT:
            cparams->moe_cache_mode = LLAMA_MOE_CACHE_MODE_ON;
            break;
    }
    cparams->moe_cache_budget_mib = mc.budget_mib;
}

static void fit_serialize(
        const common_fit_plan & plan,
        common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_analysis & analysis,
        const fit_measure_result & measure,
        const std::vector<int64_t> & margins) {
    // tensor_split into caller buffers (the model params point into these)
    std::fill(std::begin(params.tensor_split), std::end(params.tensor_split), 0.0f);
    const size_t n_split = std::min(plan.tensor_split.size(), (size_t) llama_max_devices());
    for (size_t i = 0; i < n_split; i++) {
        params.tensor_split[i] = plan.tensor_split[i];
    }
    mparams->tensor_split = params.tensor_split;

    // tensor_buft_overrides: patterns are static string literals, safe to copy
    params.tensor_buft_overrides = plan.tensor_buft_overrides;
    mparams->tensor_buft_overrides = params.tensor_buft_overrides.empty()
        ? nullptr : params.tensor_buft_overrides.data();

    mparams->n_gpu_layers = plan.n_gpu_layers;
    mparams->use_extra_bufts = false;

    if (plan.n_ctx > 0) {
        cparams->n_ctx = plan.n_ctx;
    }

    fit_serialize_moe_cache(params, cparams, analysis, plan, measure, margins);
}

static void fit_validate(
        const common_fit_plan & plan,
        const common_fit_analysis & analysis,
        const common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_probe & probe,
        const std::string & path_model) {
    // one final measure with the serialized placement
    fit_measure_result final_measure;
    if (!fit_measure_plan(probe, plan, mparams, cparams, path_model, final_measure)) {
        throw std::runtime_error("final placement cannot create a context");
    }
    if (!fit_same_devices(final_measure.devs, analysis)) {
        throw std::runtime_error("final placement changed the device set");
    }
    for (size_t id = 0; id < analysis.devices.size(); id++) {
        if (final_measure.memory[id].free - fit_device_used(final_measure.memory[id]) <
                (int64_t) params.fit_params_target[id]) {
            throw std::runtime_error("final placement exceeds the per-device margin on device " +
                    std::string(ggml_backend_dev_name(analysis.devices[id].dev)));
        }
    }

    // placement read-back: load with the serialized placement and assert roles
    llama_model_params rparams = *mparams;
    rparams.no_alloc  = true;
    rparams.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model_ptr model(llama_model_load_from_file(path_model.c_str(), rparams));
    if (model == nullptr) {
        throw std::runtime_error("failed to load model for placement verification");
    }

    // expected device set per role. non-expert roles may land on any split
    // device or on a host (CPU) buffer (input layer, token_embd).
    std::vector<ggml_backend_dev_t> allowed_non_expert;
    allowed_non_expert.push_back(nullptr);
    for (size_t i = 0; i < plan.tensor_split.size() && i < analysis.devices.size(); i++) {
        if (plan.tensor_split[i] > 0.0f) {
            allowed_non_expert.push_back(analysis.devices[i].dev);
        }
    }
    for (int id : plan.role_devices) {
        if (id >= 0 && id < (int) analysis.devices.size()) {
            allowed_non_expert.push_back(analysis.devices[id].dev);
        }
    }
    const int pool = plan.role_devices[COMMON_FIT_ROLE_EXPERT_UP];
    ggml_backend_dev_t pool_dev = nullptr;
    if (pool >= 0 && pool < (int) analysis.devices.size()) {
        pool_dev = analysis.devices[pool].dev;
    }

    const size_t n_tensors = llama_model_get_tensor_devices(model.get(), nullptr, 0);
    std::vector<llama_tensor_device_info> tensor_infos(n_tensors);
    if (n_tensors > 0) {
        llama_model_get_tensor_devices(model.get(), tensor_infos.data(), n_tensors);
    }
    for (const auto & info : tensor_infos) {
        const common_fit_role role = fit_classify_role(info.name);
        const bool is_expert = role == COMMON_FIT_ROLE_EXPERT_UP ||
                               role == COMMON_FIT_ROLE_EXPERT_GATE ||
                               role == COMMON_FIT_ROLE_EXPERT_DOWN;
        // a CPU pool may land on the CPU buffer or on a device host buffer
        // (the CPU override path selects from buft_list_cpu, which can be host)
        const bool ok = is_expert
            ? (pool < 0 ? info.host : info.dev == pool_dev)
            : std::find(allowed_non_expert.begin(), allowed_non_expert.end(), info.dev) != allowed_non_expert.end();
        if (!ok) {
            throw std::runtime_error(std::string("tensor ") + info.name + " (role " + fit_role_name(role) +
                    ") landed on an unexpected device; got " +
                    (info.dev ? ggml_backend_dev_name(info.dev) : "cpu") +
                    " expected " + (pool_dev ? ggml_backend_dev_name(pool_dev) : "cpu") +
                    " (is_expert " + (is_expert ? "1" : "0") + ", pool " + std::to_string(pool) + ")");
        }
    }
}

common_params_fit_status common_fit_strategy_apply_impl(
        common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_plan * plan,
        const common_fit_probe & probe) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
    try {
        if (params.fit_strategy == COMMON_FIT_STRATEGY_UPSTREAM) {
            // never reached through the dispatch, but honor the contract
            return COMMON_PARAMS_FIT_STATUS_SUCCESS;
        }
        if (params.fit_strategy_min_ctx <= 0 && params.n_ctx <= 0) {
            throw common_params_fit_exception("fit strategy requires -c or --fit-strategy-min-ctx");
        }

        uint32_t hp_ngl = 0;
        const common_fit_analysis analysis = fit_analyze(params, mparams, params.model.path, hp_ngl);

        // margins are indexed by model device order; re-attribute an RPC main_gpu
        if (params.fit_strategy == COMMON_FIT_STRATEGY_RPC_AWARE) {
            fit_reattribute_margins(params.fit_params_target, analysis, params.main_gpu);
        }
        std::vector<int64_t> margins;
        margins.reserve(analysis.devices.size());
        for (size_t id = 0; id < analysis.devices.size(); id++) {
            margins.push_back(id < params.fit_params_target.size()
                ? (int64_t) params.fit_params_target[id] : 0);
        }

        common_fit_plan final_plan;
        fit_measure_result win_measure;
        bool have_plan = false;

        if (plan == nullptr) {
            if (params.fit_strategy == COMMON_FIT_STRATEGY_RPC_AWARE) {
                have_plan = fit_solve_rpc_aware(
                        analysis, margins, params, mparams, cparams, probe, params.model.path,
                        hp_ngl, final_plan, win_measure);
                if (!have_plan) {
                    // exhaust: fall back to the stock solver with the re-attributed margins
                    LOG_WRN("%s: no rpc-aware candidate fit; falling back to the upstream fit\n", __func__);
                    return common_fit_params(params.model.path.c_str(), mparams, cparams,
                            params.tensor_split, params.tensor_buft_overrides.data(), &params.moe_cache,
                            params.fit_params_target.data(),
                            params.fit_strategy_min_ctx > 0 ? params.fit_strategy_min_ctx : params.fit_params_min_ctx,
                            GGML_LOG_LEVEL_INFO);
                }
            } else {
                final_plan = fit_solve_maxctx(
                        analysis, margins, params, mparams, cparams, probe, params.model.path,
                        hp_ngl, win_measure);
                have_plan = true;
            }
        } else {
            // manual plan: measure it once, a conflict is a hard error
            final_plan = *plan;
            for (int id : final_plan.role_devices) {
                if (id >= (int) analysis.devices.size()) {
                    throw std::runtime_error("manual plan references unknown device id " + std::to_string(id));
                }
            }
            if (!fit_measure_plan(probe, final_plan, mparams, cparams, params.model.path, win_measure)) {
                throw std::runtime_error("manual plan cannot create a context");
            }
            if (!fit_same_devices(win_measure.devs, analysis)) {
                throw std::runtime_error("manual plan changed the device set");
            }
            if (!fit_fits_margins(win_measure.memory, margins, analysis.devices.size())) {
                throw std::runtime_error("manual plan does not fit within the per-device margins");
            }
            have_plan = true;
        }

        if (!have_plan) {
            throw std::runtime_error("fit strategy produced no plan");
        }

        fit_serialize(final_plan, params, mparams, cparams, analysis, win_measure, margins);
        fit_validate(final_plan, analysis, params, mparams, cparams, probe, params.model.path);

        LOG_TRC("%s: successfully fit params to free device memory\n", __func__);
    } catch (const common_params_fit_exception & e) {
        LOG_WRN("%s: failed to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_FAILURE;
    } catch (const std::runtime_error & e) {
        LOG_ERR("%s: encountered an error while trying to fit params to free device memory: %s\n", __func__, e.what());
        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }
    const int64_t t1_us = llama_time_us();
    LOG_TRC("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return status;
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

size_t common_fit_strategy_pool_budget(size_t working_set_bytes, size_t remaining_vram, size_t user_budget_mib) {
    // explicit user budget wins
    if (user_budget_mib != 0) {
        return user_budget_mib * 1024 * 1024;
    }

    const size_t floor = std::max<size_t>(2 * working_set_bytes, 256ull * 1024 * 1024);
    const size_t cap   = 4ull * 1024 * 1024 * 1024;

    // nothing left for the pool
    if (remaining_vram < floor) {
        return 0;
    }

    const size_t half = remaining_vram / 2;
    const size_t clamped = std::min(std::max(half, floor), cap);
    return std::min(clamped, remaining_vram);
}

common_fit_strategy common_fit_strategy_from_string(const std::string & s) {
    if (s == "upstream")  { return COMMON_FIT_STRATEGY_UPSTREAM;  }
    if (s == "rpc-aware") { return COMMON_FIT_STRATEGY_RPC_AWARE; }
    if (s == "maxctx")    { return COMMON_FIT_STRATEGY_MAXCTX;    }
    return COMMON_FIT_STRATEGY_UPSTREAM;
}