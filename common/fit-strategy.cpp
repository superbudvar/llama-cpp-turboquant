#include "fit-strategy.h"
#include "fit-cache.h"

#include "../src/llama-ext.h" // llama_model_get_moe_tensor_info, llama_model_get_tensor_devices (internal)
#include "common.h"


#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <fstream>
#include <map>
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

// parse the layer index from a "blk.<N>." tensor name; -1 if the tensor does
// not belong to a block layer (embeddings, output, rope, etc.)
static int fit_tensor_layer(const std::string & name) {
    const std::string prefix = "blk.";
    if (name.compare(0, prefix.size(), prefix) != 0) {
        return -1;
    }
    size_t i = prefix.size();
    if (i >= name.size() || name[i] < '0' || name[i] > '9') {
        return -1;
    }
    int il = 0;
    for (; i < name.size() && name[i] >= '0' && name[i] <= '9'; i++) {
        il = il * 10 + (name[i] - '0');
    }
    return (i < name.size() && name[i] == '.') ? il : -1;
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
            LOG_DBG("%s: fit_measure_plan FAIL (no context) n_ctx=%u split0=%.3f\n",
                    __func__, cc.n_ctx, cp.tensor_split ? cp.tensor_split[0] : -1.0f);
            return false;
        }
        throw;
    }
    LOG_DBG("%s: fit_measure_plan OK n_ctx=%u split0=%.3f ndevs=%zu\n",
            __func__, cc.n_ctx, cp.tensor_split ? cp.tensor_split[0] : -1.0f,
            out.devs.size());
    return true;
}

static bool fit_same_devices(const std::vector<ggml_backend_dev_t> & devs, const common_fit_analysis & analysis) {
    if (devs.size() != analysis.devices.size()) {
        LOG_DBG("%s: fit_same_devices FAIL size devs=%zu analysis=%zu\n",
                __func__, devs.size(), analysis.devices.size());
        return false;
    }
    for (size_t i = 0; i < devs.size(); i++) {
        const char * ndev = devs[i] ? ggml_backend_dev_name(devs[i]) : "(null)";
        const char * nad  = analysis.devices[i].dev ? ggml_backend_dev_name(analysis.devices[i].dev) : "(null)";
        const bool same = devs[i] == analysis.devices[i].dev;
        LOG_DBG("%s: fit_same_devices[%zu] probe=%s (%p) analysis=%s (%p) match=%d\n",
                __func__, i, ndev, (const void *) devs[i], nad, (const void *) analysis.devices[i].dev,
                same ? 1 : 0);
        if (!same) {
            LOG_DBG("%s: fit_same_devices FAIL at index %zu (name probe=%s analysis=%s)\n",
                    __func__, i, ndev, nad);
            return false;
        }
    }
    LOG_DBG("%s: fit_same_devices PASS (%zu devices)\n", __func__, devs.size());
    return true;
}

// mirror fit.cpp:894-900 per-device margin check
static int64_t fit_device_used(const common_device_memory_data & dmd) {
    return (int64_t) dmd.model + (int64_t) dmd.context + (int64_t) dmd.compute;
}

static bool fit_fits_margins(
        const common_device_memory_data_vec & memory,
        const std::vector<int64_t> & margins,
        const common_fit_analysis & analysis,
        size_t * fail_dev = nullptr) {
    const size_t nd = analysis.devices.size();
    if (fail_dev) { *fail_dev = (size_t) -1; }
    LOG_DBG("%s: memory.size()=%zu nd+1=%zu margins.size()=%zu nd=%zu\n",
            __func__, memory.size(), nd + 1, margins.size(), nd);
    if (memory.size() != nd + 1 || margins.size() < nd) {
        LOG_DBG("%s: fit_fits_margins FAIL size guard\n", __func__);
        return false;
    }
    for (size_t id = 0; id < nd; id++) {
        const int64_t used = (int64_t) fit_device_used(memory[id]);
        // Use the cold-captured free VRAM (analysis.devices[id].free_bytes), not
        // the live free re-queried inside each measurement probe. Every candidate
        // measurement allocates a context on the GPU and ROCm's buffer pool
        // retains those allocations, so the live free shrinks across the loop.
        // The last candidate's check would then see a falsely-small headroom and
        // collapse to the minimum context, while the larger contexts the earlier
        // passes proved are discarded. The real model load reuses the same pool
        // buffers, so the cold free is the correct basis for the fit. The model,
        // context and compute sizes come from the GGUF/tensor metadata via
        // llama_get_memory_breakdown, so `used` is exact.
        const int64_t free_cold = analysis.devices[id].free_bytes;
        const int64_t rem = free_cold - used;
        LOG_DBG("%s: dev[%zu] free_cold=%lld used=%lld (m%d c%d k%d) rem=%lld margin=%lld -> %s\n",
                __func__, id,
                (long long) free_cold, (long long) used,
                (int) memory[id].model, (int) memory[id].context, (int) memory[id].compute,
                (long long) rem, (long long) margins[id],
                (used > INT64_MAX || rem < margins[id]) ? "REJECT" : "ok");
        if (used > INT64_MAX ||
            rem < margins[id]) {
            LOG_DBG("%s: fit_fits_margins FAIL dev[%zu]\n", __func__, id);
            if (fail_dev) { *fail_dev = id; }
            return false;
        }
    }
    LOG_DBG("%s: fit_fits_margins PASS\n", __func__);
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

    // solver math keys on the loader span: the loader always reserves
    // n_layer_all + 1 slots regardless of load_mtp, and a skipped in-file MTP
    // block still occupies its layer slot. count nextn unconditionally here.
    // memory-side accounting (the probe) derives its own gated count.
    const uint32_t loader_ngl = llama_model_n_layer(model.get()) + llama_model_n_layer_nextn(model.get());
    analysis.loader_ngl = loader_ngl;
    hp_ngl = loader_ngl;

    return analysis;
}

struct fit_candidate_spec {
    std::string label;
    int dense_main;              // primary device for dense roles
    std::vector<int> split_devices; // devices receiving dense tensors via tensor_split
    int pool;                    // expert pool device id, -1 = CPU
};

// build a plan from a candidate spec. n_ctx <= 0 means "leave cparams untouched".
// ngl_override >= 0 forces a partial offload (CPU spill); -1 means full offload.
static common_fit_plan fit_plan_from_spec(
        const common_fit_analysis & analysis,
        const fit_candidate_spec & spec,
        uint32_t loader_ngl,
        int32_t n_ctx,
        float rpc_frac_override = -1.0f,
        int32_t ngl_override = -1) {
    common_fit_plan plan;
    plan.role_devices.assign(8, -1);
    const int dense_main = spec.dense_main;

    // device ID mapping guard: dense_main and pool must be valid logical indices
    if (dense_main < 0 || dense_main >= (int) analysis.devices.size()) {
        throw std::runtime_error("fit_plan_from_spec: dense_main " + std::to_string(dense_main) +
                " is out of range for " + std::to_string(analysis.devices.size()) + " devices");
    }
    if (spec.pool >= (int) analysis.devices.size()) {
        throw std::runtime_error("fit_plan_from_spec: pool " + std::to_string(spec.pool) +
                " is out of range for " + std::to_string(analysis.devices.size()) + " devices");
    }

    for (size_t r = 0; r < plan.role_devices.size(); r++) {
        if (r == COMMON_FIT_ROLE_EXPERT_UP || r == COMMON_FIT_ROLE_EXPERT_GATE ||
            r == COMMON_FIT_ROLE_EXPERT_DOWN) {
            // Expert-weight override anchor. The MoE cache is a scheduler-level
            // session that spans every backend in the split (see
            // ggml_backend_sched_set_moe_cache), so it is NOT confined to one
            // device. role_devices[EXPERT_UP] is only used to derive the MoE
            // cache budget and as a backwards-compatible allowed device in
            // validation; the per-device split is handled by tensor_split.
            plan.role_devices[r] = spec.pool;
        } else {
            plan.role_devices[r] = dense_main;
        }
    }

    plan.n_gpu_layers = (ngl_override >= 0) ? ngl_override : (int32_t) (loader_ngl + 1);
    plan.n_ctx = n_ctx;

    // tensor_split: distribute the FULL model across split devices, proportional
    // to available VRAM. This is "best endeavours" weight splitting: dense prefers
    // local (local has more VRAM, so it gets the larger share), experts overflow
    // to RPC. KV cache follows the weights because the whole layer lands on one
    // device.
    plan.tensor_split.assign(llama_max_devices(), 0.0f);
    if (spec.split_devices.size() == 1) {
        plan.tensor_split[spec.split_devices[0]] = 1.0f;
    } else {
        const int local_device = spec.split_devices.front();
        // derive the RPC member of the weight split from split_devices, not from
        // spec.pool. pool only selects the MoE expert cache device; the tensor
        // split must follow the devices actually named in split_devices. using
        // pool here (which can be CPU or the local device) wrongly zeroes the
        // RPC share and can dump the whole model onto one GPU.
        int rpc_device = -1;
        for (int d : spec.split_devices) {
            if (d != local_device && analysis.devices[d].is_rpc) {
                rpc_device = d;
                break;
            }
        }

        const int64_t context_reserve = 2LL * 1024 * 1024 * 1024; // 2 GiB per device for KV/compute
        // The RPC backend stages weight copies on the local GPU and keeps its
        // own command/staging buffers, so it cannot use the full reported VRAM.
        // Leave headroom so the offload does not OOM the RPC device mid-load.
        const int64_t rpc_overhead = 2LL * 1024 * 1024 * 1024;
        const int64_t local_avail = analysis.devices[local_device].total_bytes - context_reserve;
        const int64_t rpc_avail   = rpc_device >= 0 ? analysis.devices[rpc_device].total_bytes - context_reserve - rpc_overhead : 0;

        size_t total_model = 0;
        for (const auto & t : analysis.tensors) {
            total_model += t.bytes;
        }

        // RPC gets as much of the full model as it can hold; local gets the rest.
        float rpc_frac = 0.0f;
        if (rpc_device >= 0 && total_model > 0) {
            rpc_frac = std::min(1.0f, (float)rpc_avail / (float)total_model);
        }
        if (rpc_frac_override >= 0.0f) {
            rpc_frac = rpc_frac_override;
        }
        plan.tensor_split[local_device] = 1.0f - rpc_frac;
        if (rpc_device >= 0) {
            plan.tensor_split[rpc_device] = rpc_frac;
        }
        LOG_DBG("%s: VRAM-prop tensor_split: total_model=%zu MiB, local_avail=%zu MiB, rpc_avail=%zu MiB, rpc_frac=%.4f\n",
                __func__, total_model / 1024 / 1024, local_avail / 1024 / 1024,
                rpc_avail / 1024 / 1024, rpc_frac);
    }

    if (analysis.moe.n_expert > 0) {
        // The MoE cache is a scheduler-level session that already spans every
        // split backend (see ggml_backend_sched_set_moe_cache), so it follows
        // the expert placement automatically. The fit only derives its total
        // budget from the free VRAM across all split devices; it does not pin
        // the cache to a single device.
    }

    return plan;
}

// grow the context for a fixed candidate placement, from the minimum context
// (fit_strategy_min_ctx) up to the upper bound (params.n_ctx, or the full train
// context when 0), honoring per-device margins. rpc_frac_override selects the
// weight split for an rpc-split placement; pass -1.0f for a single-device
// placement. returns the chosen n_ctx (>= 0) or -1 when even the minimum context
// does not fit. sets out_plan/out_measure to the confirming placement.
static int32_t fit_grow_context(
        const common_fit_analysis & analysis,
        const std::vector<int64_t> & margins,
        const common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_probe & decode_probe,
        const fit_candidate_spec & spec,
        float rpc_frac_override,
        uint32_t loader_ngl,
        const std::string & path_model,
        common_fit_plan & out_plan,
        fit_measure_result & out_measure,
        int32_t ngl_override = -1) {
    const size_t nd = analysis.devices.size();
    // an explicit --ctx-size is a hard requirement: fit model + exactly that
    // context and never grow or shrink it. the fit then only sheds model layers
    // (n_gpu_layers) to CPU/RPC if the full offload does not fit. when only
    // --fit-strategy-min-ctx is set we maximize context from the minimum upward
    // (the hi bound below is then the train context).
    const int32_t target_ctx = (params.n_ctx > 0)
        ? (int32_t) params.n_ctx
        : (int32_t) params.fit_strategy_min_ctx;

    const auto fits = [&](const common_fit_plan & p, fit_measure_result & m) -> bool {
        const bool ok_m = fit_measure_plan(decode_probe, p, mparams, cparams, path_model, m);
        if (!ok_m) { LOG_DBG("%s: grow fits FAIL measure n_ctx=%u\n", __func__, p.n_ctx); return false; }
        if (!fit_same_devices(m.devs, analysis)) { LOG_DBG("%s: grow fits FAIL devs n_ctx=%u\n", __func__, p.n_ctx); return false; }
        const bool ok_fm = fit_fits_margins(m.memory, margins, analysis);
        LOG_DBG("%s: grow fits n_ctx=%u measure=%d margins=%d -> %s\n", __func__, p.n_ctx, ok_m ? 1 : 0, ok_fm ? 1 : 0, ok_fm ? "ok" : "FAIL");
        return ok_fm;
    };

    // the floor (minimum context) must fit, otherwise this placement is infeasible
    common_fit_plan plan_lo = fit_plan_from_spec(analysis, spec, loader_ngl, target_ctx, rpc_frac_override, ngl_override);
    fit_measure_result m_lo;
    if (!fits(plan_lo, m_lo)) {
        return -1;
    }

    // upper bound: capped by --ctx-size, else the model's full train context
    int32_t hi = params.n_ctx > 0 ? (int32_t) params.n_ctx
        : (int32_t) (m_lo.hp_nct > 0 ? m_lo.hp_nct : (target_ctx * 4));
    if (hi <= target_ctx) {
        out_plan = plan_lo;
        out_measure = m_lo;
        return target_ctx;
    }

    // if the upper bound already fits there is nothing to search
    common_fit_plan plan_hi = fit_plan_from_spec(analysis, spec, loader_ngl, hi, rpc_frac_override, ngl_override);
    fit_measure_result m_hi;
    if (fits(plan_hi, m_hi)) {
        out_plan = plan_hi;
        out_measure = m_hi;
        return hi;
    }

    // binary search the largest context that still fits within margins
    int32_t lo = target_ctx;
    int32_t best_ctx = target_ctx;
    common_fit_plan best_plan = plan_lo;
    fit_measure_result best_m = m_lo;
    while (hi - lo > 256) {
        int32_t mid = lo + (hi - lo) / 2;
        mid -= mid % 256;
        if (mid <= lo) mid = lo + 256;
        common_fit_plan pm = fit_plan_from_spec(analysis, spec, loader_ngl, mid, rpc_frac_override, ngl_override);
        fit_measure_result mm;
        if (fits(pm, mm)) {
            lo = mid;
            best_ctx = mid;
            best_plan = pm;
            best_m = mm;
        } else {
            hi = mid;
        }
    }
    out_plan = best_plan;
    out_measure = best_m;
    return best_ctx;
}

// for an rpc-split CPU spill at a fixed n_gpu_layers, find the rpc_frac split and
// grow the context from it. the full-offload scan pinned rpc_frac to the RPC
// ceiling; once layers move to CPU neither device overflows at that ceiling, so
// the RPC-maximizing split fills RPC first and strands the local device. this
// re-brackets the feasible interval at this layer count (coarse 0.05 scan, then
// bisection when the window is narrower than the step) and keeps the split that
// reaches the largest context, using the smallest rpc_frac (least RPC offload)
// on ties. returns the chosen n_ctx (>= 0) or -1 when no split reaches the
// minimum context; sets out_frac to the chosen rpc_frac.
static int32_t fit_spill_split_at_ngl(
        const common_fit_analysis & analysis,
        const std::vector<int64_t> & margins,
        const common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_probe & probe,
        const fit_candidate_spec & spec,
        uint32_t loader_ngl,
        const std::string & path_model,
        int32_t ngl,
        float scan_hi,
        common_fit_plan & out_plan,
        fit_measure_result & out_measure,
        float & out_frac) {
    out_frac = -1.0f;
    const int32_t min_ctx = (int32_t) (params.n_ctx > 0 ? params.n_ctx : params.fit_strategy_min_ctx);
    const float step = 0.05f;
    const size_t nd = analysis.devices.size();

    struct row_t {
        float f;
        bool ok;
        size_t fail_dev;
    };
    std::vector<row_t> rows;
    std::vector<float> feas;

    // coarse scan top-down, mirroring the full-offload bracket. rpc_frac
    // feasibility is an interval: low f overflows the local device, high f
    // overflows RPC.
    for (float f = scan_hi; f >= -step * 0.5f; f -= step) {
        if (f < 0.0f) { f = 0.0f; }
        common_fit_plan plan = fit_plan_from_spec(analysis, spec, loader_ngl, min_ctx, f, ngl);
        fit_measure_result m;
        const bool ok_measure = fit_measure_plan(probe, plan, mparams, cparams, path_model, m);
        const bool ok_devs    = ok_measure && fit_same_devices(m.devs, analysis);
        size_t fail = (size_t) -1;
        const bool ok_margin  = ok_devs && fit_fits_margins(m.memory, margins, analysis, &fail);
        LOG_DBG("%s: ngl=%d rpc_frac=%.4f measure=%d devs=%d margin=%d -> %s\n",
                __func__, ngl, f, ok_measure ? 1 : 0, ok_devs ? 1 : 0, ok_margin ? 1 : 0,
                ok_margin ? "FEASIBLE" : "rejected");
        rows.push_back({ f, ok_margin, fail });
        if (ok_margin) {
            feas.push_back(f);
        }
        if (f <= 0.0f) { break; }
    }

    // window narrower than one step: the scan straddles it (high rows overflow
    // RPC, low rows overflow the local device). bisect into the window; when no
    // feasible split exists the search converges on the balancing point and
    // returns no feasible result, which is the correct verdict for that ngl.
    if (feas.empty()) {
        float f_lo = 0.0f;
        float f_hi = scan_hi;
        bool have_lo = false;
        bool have_hi = false;
        for (const auto & r : rows) {
            if (!r.ok && r.fail_dev < nd && analysis.devices[r.fail_dev].is_rpc) {
                f_hi = std::min(f_hi, r.f); // feasible window lies below RPC overflow
                have_hi = true;
            } else if (!r.ok && r.fail_dev < nd) {
                f_lo = std::max(f_lo, r.f); // ... and above local-device overflow
                have_lo = true;
            }
        }
        if (have_lo && have_hi) {
            for (int it = 0; it < 13 && f_hi - f_lo > 1e-4f; it++) {
                const float fm = (f_lo + f_hi) * 0.5f;
                common_fit_plan plan = fit_plan_from_spec(analysis, spec, loader_ngl, min_ctx, fm, ngl);
                fit_measure_result m;
                const bool ok_measure = fit_measure_plan(probe, plan, mparams, cparams, path_model, m);
                const bool ok_devs    = ok_measure && fit_same_devices(m.devs, analysis);
                size_t fail = (size_t) -1;
                const bool ok_margin  = ok_devs && fit_fits_margins(m.memory, margins, analysis, &fail);
                LOG_DBG("%s: ngl=%d bisect rpc_frac=%.4f measure=%d devs=%d margin=%d -> %s\n",
                        __func__, ngl, fm, ok_measure ? 1 : 0, ok_devs ? 1 : 0, ok_margin ? 1 : 0,
                        ok_margin ? "FEASIBLE" : "rejected");
                if (ok_margin) {
                    feas.push_back(fm);
                    f_hi = fm; // keep probing lower (local-first)
                } else if (fail < nd && analysis.devices[fail].is_rpc) {
                    f_hi = fm; // too much RPC: split must move lower
                } else {
                    f_lo = fm; // local overflow: split must move higher
                }
            }
        }
    }

    // grow the context from each feasible split and keep the one at the largest
    // context. scan ascending rpc_frac so among splits reaching the same maximum
    // context the smallest rpc_frac (least RPC offload) wins: RPC is used last.
    std::sort(feas.begin(), feas.end());
    int32_t best_ctx = -1;
    common_fit_plan best_plan;
    fit_measure_result best_m;
    for (float f : feas) {
        common_fit_plan plan;
        fit_measure_result m;
        const int32_t ctx = fit_grow_context(analysis, margins, params, mparams, cparams,
                probe, spec, f, loader_ngl, path_model, plan, m, ngl);
        if (ctx < 0) { continue; }
        if (best_ctx < 0 || ctx > best_ctx) {
            best_ctx = ctx;
            out_frac = f;
            best_plan = plan;
            best_m = m;
        }
    }
    if (best_ctx < 0) {
        return -1;
    }
    out_plan = best_plan;
    out_measure = best_m;
    return best_ctx;
}

// effective per-device margins: base margins plus every soft reservation that
// was not shed. device indices follow the analysis device order.
static std::vector<int64_t> fit_effective_margins(
        const std::vector<int64_t> & base,
        const std::vector<common_fit_reserve> & reserves,
        const std::vector<std::string> & shed_tags) {
    std::vector<int64_t> margins = base;
    for (const auto & r : reserves) {
        const bool shed = std::find(shed_tags.begin(), shed_tags.end(), r.tag) != shed_tags.end();
        if (!shed && r.device >= 0 && (size_t) r.device < margins.size()) {
            margins[r.device] += (int64_t) r.bytes;
        }
    }
    return margins;
}

// cumulative shed levels by ascending priority. level 0 sheds nothing; each
// later level additionally releases the next-lowest priority group, so the
// final level releases every reservation.
static std::vector<std::vector<std::string>> fit_shed_levels(
        const std::vector<common_fit_reserve> & reserves) {
    std::vector<int> prios;
    for (const auto & r : reserves) {
        if (std::find(prios.begin(), prios.end(), r.priority) == prios.end()) {
            prios.push_back(r.priority);
        }
    }
    std::sort(prios.begin(), prios.end());
    std::vector<std::vector<std::string>> levels;
    std::vector<std::string> shed;
    levels.push_back(shed);
    for (int p : prios) {
        for (const auto & r : reserves) {
            if (r.priority == p) {
                shed.push_back(r.tag);
            }
        }
        levels.push_back(shed);
    }
    return levels;
}

// solve for rpc-aware: dense decode-hot on local non-RPC devices, experts on a
// a candidate whose split set spans an RPC device offloads part of the model
// over RPC. this is true for both MoE (pool == RPC device) and dense mixes
// (pool stays the local main device, RPC is in split_devices).
static bool fit_split_has_rpc(const common_fit_analysis & analysis, const std::vector<int> & split_devices) {
    for (int d : split_devices) {
        if (d >= 0 && analysis.devices[d].is_rpc) return true;
    }
    return false;
}

// upper bound of the rpc_frac scan for a split: the fraction of the full model
// RPC can hold above its 2 GiB context reserve. device memory does not change
// with n_gpu_layers, so the same bound brackets every spill node.
static float fit_rpc_scan_hi(const common_fit_analysis & analysis, const std::vector<int> & split_devices) {
    int rpc_device = -1;
    for (int d : split_devices) {
        if (d >= 0 && analysis.devices[d].is_rpc) { rpc_device = d; break; }
    }
    GGML_ASSERT(rpc_device >= 0);
    const int64_t context_reserve = 2LL * 1024 * 1024 * 1024;
    const int64_t rpc_avail = analysis.devices[rpc_device].total_bytes - context_reserve;
    size_t total_model = 0;
    for (const auto & t : analysis.tensors) {
        total_model += t.bytes;
    }
    return total_model > 0 ? std::min(1.0f, (float) rpc_avail / (float) total_model) : 0.0f;
}

// pool that is never the main_gpu device. first fit wins; returns false when no
// candidate fits (caller falls back to the stock solver).
static bool fit_solve_rpc_aware(
        const common_fit_analysis & analysis,
        const std::vector<int64_t> & base_margins,
        const std::vector<common_fit_reserve> & reserves,
        const common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_probe & probe,
        const std::string & path_model,
        uint32_t loader_ngl,
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

    // dense placements: prefer local-only first so decode-hot layers stay off the
    // RPC link, then fall back to splitting across local+RPC only when local alone
    // cannot reach the minimum context. RPC is used last to avoid per-token network
    // lag on the dense path.
    std::vector<std::vector<int>> dense_splits;
    std::vector<int> all_local;
    for (const auto & d : analysis.devices) {
        if (!d.is_rpc && d.free_bytes > 0) {
            all_local.push_back(d.id);
        }
    }
    const int rpc_best = fit_best_rpc(analysis);
    if (!all_local.empty()) {
        dense_splits.push_back({ local_best });
        if (all_local.size() > 1) {
            dense_splits.push_back(all_local);
        }
    }
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

    // reservations are held while full offload is feasible and shed in priority
    // order (companions before model layers) when it is not. at each shed level
    // we re-run the full candidate suite with the released margins; the first
    // level where a full-offload candidate reaches the minimum context wins.
    // CPU spill is considered only after every reservation was shed, because
    // moving a companion off VRAM is always cheaper than spilling model layers.
    const auto shed_levels = fit_shed_levels(reserves);
    for (size_t li = 0; li < shed_levels.size(); li++) {
        const auto & shed_tags = shed_levels[li];
        const std::vector<int64_t> margins = fit_effective_margins(base_margins, reserves, shed_tags);

        // spill bookkeeping: if no full-offload candidate reaches the minimum
        // context, we reduce n_gpu_layers (CPU spill). spill on the widest split
        // still usable: an rpc-split spill (rocm+rpc+cpu) beats a local-only spill
        // (rocm+cpu) because RPC is faster than CPU. CPU is the last choice.
        bool have_spill = false;
        bool have_rpc_spill = false;
        fit_candidate_spec spill_spec;

        for (const auto & cand : candidates) {
            // rpc-split candidates split the full model across local + RPC via
            // tensor_split. rpc_frac feasibility is an interval: at low rpc_frac the
            // local device overflows (too much model) and at high rpc_frac the RPC
            // device overflows. We bracket that interval at the minimum context, then
            // grow the context from the minimum upward and keep the split that
            // maximizes it.
            // a candidate whose split set spans an RPC device is an rpc-split
            // candidate even for a dense model (where cand.pool stays the local
            // main device). route it through the rpc_frac scan so we pick a
            // feasible split instead of maximizing the RPC share and over-packing
            // the RPC device past its margin.
            const bool is_rpc_split = (cand.split_devices.size() > 1) &&
                    fit_split_has_rpc(analysis, cand.split_devices);

            if (!is_rpc_split) {
                // single-device placement: grow the context from the minimum upward
                common_fit_plan plan;
                fit_measure_result m;
                const int32_t ctx = fit_grow_context(analysis, margins, params, mparams, cparams,
                        probe, cand, -1.0f, loader_ngl, path_model, plan, m);
                if (ctx >= 0) {
                    out_plan = plan;
                    out_measure = m;
                    out_plan.shed_reserves = shed_tags;
                    LOG_DBG("%s: selected candidate %s with n_ctx=%d (min=%d)\n",
                            __func__, cand.label.c_str(), ctx, params.fit_strategy_min_ctx);
                    return true;
                }
                LOG_DBG("%s: candidate %s cannot fit even the minimum context, skipping\n",
                        __func__, cand.label.c_str());
                if (!have_spill) {
                    spill_spec = cand;
                    have_spill = true;
                }
                continue;
            }

            // bracket the feasible rpc_frac interval at the minimum context
            const float hi = fit_rpc_scan_hi(analysis, cand.split_devices);

            // coarse floor scan (step 0.05) to find feasible rpc_frac values
            std::vector<float> feas;
            for (float f = hi; f >= 0.0f; f -= 0.05f) {
                common_fit_plan plan = fit_plan_from_spec(analysis, cand, loader_ngl, (int32_t) params.fit_strategy_min_ctx, f);
                fit_measure_result m;
                const bool ok_measure = fit_measure_plan(probe, plan, mparams, cparams, path_model, m);
                const bool ok_devs    = ok_measure && fit_same_devices(m.devs, analysis);
                const bool ok_margin  = ok_devs && fit_fits_margins(m.memory, margins, analysis);
                LOG_DBG("%s: rpc_frac=%.4f measure=%d devs=%d margin=%d -> %s\n",
                        __func__, f, ok_measure ? 1 : 0, ok_devs ? 1 : 0, ok_margin ? 1 : 0,
                        ok_margin ? "FEASIBLE" : "rejected");
                if (ok_margin) {
                    feas.push_back(f);
                }
            }
            if (feas.empty()) {
                LOG_DBG("%s: candidate %s could not fit any rpc_frac at the minimum context, skipping\n",
                        __func__, cand.label.c_str());
                if (!have_rpc_spill) {
                    // this candidate overrides a local-only spill base recorded
                    // above, so RPC stays in use (rocm+rpc+cpu) during the spill
                    spill_spec = cand;
                    have_rpc_spill = true;
                    have_spill = true;
                }
                continue;
            }
            // scanned top-down: front = largest rpc_frac (max offload), back = smallest
            const float f_hi_off = feas.front();
            const float f_lo_off = feas.back();

            // pick the rpc_frac that maximizes the achievable context. with a layer
            // split the KV cache follows the weights, so the device holding the larger
            // layer share is the context bottleneck. a balanced split maximizes the
            // global context; a min-offload split leaves the local device holding most
            // layers and caps context far below the available headroom.
            // scan bottom-up so that among splits reaching the same maximum context
            // the smallest rpc_frac (least RPC offload) wins: RPC is used last to
            // avoid per-token network lag on the dense path.
            float best_ftry = -1.0f;
            int32_t best_ctx = -1;
            common_fit_plan best_plan;
            fit_measure_result best_m;
            for (float ftry = f_lo_off; ftry <= f_hi_off + 1e-6f; ftry += 0.05f) {
                common_fit_plan plan;
                fit_measure_result m;
                const int32_t ctx = fit_grow_context(analysis, margins, params, mparams, cparams,
                        probe, cand, ftry, loader_ngl, path_model, plan, m);
                if (ctx < 0) continue;
                if (best_ctx < 0 || ctx > best_ctx) {
                    best_ctx = ctx;
                    best_ftry = ftry;
                    best_plan = plan;
                    best_m = m;
                }
            }
            if (best_ctx >= 0) {
                out_plan = best_plan;
                out_measure = best_m;
                out_plan.shed_reserves = shed_tags;
                LOG_DBG("%s: selected candidate %s rpc_frac=%.4f n_ctx=%d (min=%d, feasible rpc_frac [%.4f,%.4f], max-context)\n",
                        __func__, cand.label.c_str(), best_ftry, best_ctx,
                        params.fit_strategy_min_ctx, f_lo_off, f_hi_off);
                return true;
            }
            LOG_DBG("%s: candidate %s could not grow the context at full offload, marking for CPU spill\n",
                    __func__, cand.label.c_str());
            if (!have_rpc_spill) {
                spill_spec = cand;
                have_rpc_spill = true;
                have_spill = true;
            }
        }

        // CPU-spill phase runs only after every shed level failed: even with all
        // reservations released no full-offload candidate reached min-ctx.
        // binary-search the largest n_gpu_layers (least spill) that still reaches it.
        // as n_gpu_layers decreases, GPU footprint shrinks and KV cache grows, so the
        // achievable context is monotonic non-decreasing -> standard upper-bound search.
        if (li + 1 == shed_levels.size() && have_spill) {
            const bool spec_is_rpc_split = (spill_spec.split_devices.size() > 1) &&
                    fit_split_has_rpc(analysis, spill_spec.split_devices);
            const int32_t min_ctx = (int32_t) (params.n_ctx > 0 ? params.n_ctx : params.fit_strategy_min_ctx);
            // rpc-split spills re-bracket the rpc_frac interval per n_gpu_layers
            // instead of reusing the full-offload ceiling: with layers on CPU the
            // RPC-maximizing split would overflow nothing and fill RPC first.
            const float scan_hi = spec_is_rpc_split ? fit_rpc_scan_hi(analysis, spill_spec.split_devices) : 0.0f;
            int32_t lo_ngl = 0;            // all layers on CPU -> max context
            int32_t hi_ngl = (int32_t) loader_ngl + 1; // full offload incl. output layer (failed min_ctx)
            int32_t best_ngl = -1;
            common_fit_plan best_plan;
            fit_measure_result best_m;
            float best_frac = -1.0f;
            while (lo_ngl <= hi_ngl) {
                const int32_t mid_ngl = lo_ngl + (hi_ngl - lo_ngl) / 2;
                common_fit_plan plan;
                fit_measure_result m;
                int32_t ctx;
                float frac = -1.0f;
                if (spec_is_rpc_split) {
                    ctx = fit_spill_split_at_ngl(analysis, margins, params, mparams, cparams,
                            probe, spill_spec, loader_ngl, path_model, mid_ngl, scan_hi,
                            plan, m, frac);
                } else {
                    ctx = fit_grow_context(analysis, margins, params, mparams, cparams,
                            probe, spill_spec, -1.0f, loader_ngl, path_model, plan, m, mid_ngl);
                }
                if (ctx >= min_ctx) {
                    best_ngl = mid_ngl;
                    best_plan = plan;
                    best_m = m;
                    best_frac = frac;
                    lo_ngl = mid_ngl + 1; // try keeping more layers on GPU
                } else {
                    hi_ngl = mid_ngl - 1;
                }
            }
            if (best_ngl >= 0) {
                out_plan = best_plan;
                out_measure = best_m;
                out_plan.shed_reserves = shed_tags;
                if (best_frac >= 0.0f) {
                    LOG_WRN("%s: spilling %u layers to CPU to reach min-ctx %d (n_gpu_layers=%d of %u, rpc_frac=%.4f)\n",
                            __func__, (uint32_t) (loader_ngl + 1 - best_ngl), min_ctx, best_ngl, loader_ngl + 1, best_frac);
                } else {
                    LOG_WRN("%s: spilling %u layers to CPU to reach min-ctx %d (n_gpu_layers=%d of %u)\n",
                            __func__, (uint32_t) (loader_ngl + 1 - best_ngl), min_ctx, best_ngl, loader_ngl + 1);
                }
                return true;
            }
        }
    }

    return false;
}

// solve for maxctx: two-anchor interpolation per placement, mirror fit.cpp:667-669.
// the floor is fit_strategy_min_ctx; an unreachable floor is a hard error.
static common_fit_plan fit_solve_maxctx(
        const common_fit_analysis & analysis,
        const std::vector<int64_t> & base_margins,
        const std::vector<common_fit_reserve> & reserves,
        const common_params & params,
        llama_model_params * mparams,
        llama_context_params * cparams,
        const common_fit_probe & probe,
        const std::string & path_model,
        uint32_t loader_ngl,
        fit_measure_result & out_measure) {
    if (params.fit_strategy_min_ctx <= 0) {
        throw common_params_fit_exception("maxctx strategy requires --fit-strategy-min-ctx");
    }
    const int32_t target_ctx = params.fit_strategy_min_ctx;

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

    // release reservations in priority order before failing: a companion moved
    // off VRAM is preferable to spilling or failing at the min-ctx floor.
    const auto shed_levels = fit_shed_levels(reserves);
    for (const auto & shed_tags : shed_levels) {
        const std::vector<int64_t> margins = fit_effective_margins(base_margins, reserves, shed_tags);

        for (const auto & placement : placements) {
            // grow the context from the minimum upward; this also tests the full
            // context first and uses it directly when it fits.
            common_fit_plan plan;
            fit_measure_result m;
            const int32_t ctx = fit_grow_context(analysis, margins, params, mparams, cparams,
                    probe, placement, -1.0f, loader_ngl, path_model, plan, m);
            if (ctx < 0) {
                LOG_DBG("%s: placement %s cannot fit even the minimum context, skipping\n",
                        __func__, placement.label.c_str());
                continue;
            }
            LOG_DBG("%s: placement %s selected context %" PRIi32 "\n", __func__, placement.label.c_str(), ctx);
            out_measure = m;
            plan.shed_reserves = shed_tags;
            return plan;
        }
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

    // The MoE cache session spans every split backend (local + RPC), so the
    // budget headroom is the total free VRAM across all split devices, not one
    // pool device. The scheduler divides the resulting grant across backends.
    int64_t remaining = 0;
    for (size_t id = 0; id < measure.memory.size() && id < margins.size(); id++) {
        if (id < plan.tensor_split.size() && plan.tensor_split[id] > 0.0f) {
            int64_t dev_rem = fit_device_remaining(measure.memory[id], margins[id]);
            if (dev_rem > 0) {
                remaining += dev_rem;
            }
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
        LOG_WRN("%s: no VRAM remaining for an MoE pool across split devices; pool disabled\n", __func__);
    } else {
        if (mc.mode_explicit && mc.mode == COMMON_MOE_CACHE_MODE_SOFT) {
            mc.mode = COMMON_MOE_CACHE_MODE_SOFT;
        } else {
            mc.mode = COMMON_MOE_CACHE_MODE_ON;
        }
        mc.budget_mib = grant / (1024 * 1024);
        mc.fit_selected = true;
        LOG_DBG("%s: MoE pool budget %zu MiB across split devices\n", __func__, grant / (1024 * 1024));
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
        const std::vector<int64_t> & margins,
        uint32_t loader_ngl,
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
        const int64_t used = (int64_t) fit_device_used(final_measure.memory[id]);
        // `free` is post-allocation free, so the headroom left after load is
        // `free` (committed backends) or `total - used` (lazy backends). Using
        // `free - used` double-counts the footprint and rejects valid placements.
        // No stats overhead is added: the device baseline is already in `free`.
        const int64_t rem = std::min((int64_t) final_measure.memory[id].free,
                                     (int64_t) final_measure.memory[id].total - used);
        const int64_t free_mib = (int64_t) final_measure.memory[id].free / (1024 * 1024);
        const int64_t used_mib = used / (1024 * 1024);
        const int64_t proj_free_mib = rem / (1024 * 1024);
        const int64_t target_mib = (id < margins.size()) ? (int64_t) margins[id] / (1024 * 1024) : 0;
        LOG_DBG("%s: device %s: probe free %" PRId64 " MiB, projected free after load %" PRId64 " MiB (used %" PRId64 "), fit-target %" PRId64 " MiB\n",
                __func__, ggml_backend_dev_name(analysis.devices[id].dev), free_mib, proj_free_mib, used_mib, target_mib);
        if (id >= margins.size() || rem < margins[id]) {
            throw std::runtime_error("final placement leaves less than the per-device margin on device " +
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

    // Expert tensors may land on any device that has a non-zero tensor_split
    // share (full-model split), not just the pool device. The pool device is
    // still allowed for backwards compatibility with single-pool placements.
    std::vector<ggml_backend_dev_t> allowed_expert;
    if (pool_dev != nullptr) {
        allowed_expert.push_back(pool_dev);
    }
    for (size_t i = 0; i < plan.tensor_split.size() && i < analysis.devices.size(); i++) {
        if (plan.tensor_split[i] > 0.0f) {
            allowed_expert.push_back(analysis.devices[i].dev);
        }
    }
    if (pool < 0) {
        allowed_expert.push_back(nullptr); // CPU pool
    }

    const size_t n_tensors = llama_model_get_tensor_devices(model.get(), nullptr, 0);
    std::vector<llama_tensor_device_info> tensor_infos(n_tensors);
    if (n_tensors > 0) {
        llama_model_get_tensor_devices(model.get(), tensor_infos.data(), n_tensors);
    }
    // Layers the plan keeps off the GPU (n_gpu_layers below full offload,
    // which is loader_ngl + 1) load on the host: the loader assigns the lowest
    // block indices to the CPU buffer. Expert tensors in those layers may
    // legitimately land on CPU.
    const int spill_layers = std::max((int) loader_ngl + 1 - plan.n_gpu_layers, 0);
    for (const auto & info : tensor_infos) {
        const common_fit_role role = fit_classify_role(info.name);
        const bool is_expert = role == COMMON_FIT_ROLE_EXPERT_UP ||
                               role == COMMON_FIT_ROLE_EXPERT_GATE ||
                               role == COMMON_FIT_ROLE_EXPERT_DOWN;
        const int il = fit_tensor_layer(info.name);
        const bool spilled_to_cpu = info.dev == nullptr && spill_layers > 0 && il >= 0 && il < spill_layers;
        // a CPU pool may land on the CPU buffer or on a device host buffer
        // (the CPU override path selects from buft_list_cpu, which can be host)
        const bool ok = is_expert
            ? std::find(allowed_expert.begin(), allowed_expert.end(), info.dev) != allowed_expert.end() || spilled_to_cpu
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
        common_fit_analysis analysis = fit_analyze(params, mparams, params.model.path, hp_ngl);

        // Device headroom is taken directly from the backend-reported `free`
        // (which already excludes the device baseline: framebuffer + HIP context
        // + display) and the exact model/context/compute sizes from metadata.
        // No fit-stats overhead is applied: adding it would double-count the
        // baseline and flip the placement between loads. The recorded stats
        // file is kept only for diagnostics.

        // margins are indexed by model device order; the user's explicit
        // fit-target values are honored per device (no RPC main_gpu swap).
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
            // fit cache: a hit substitutes the cached placement for the probe
            // suite. it is validated with a single probe; on any failure the
            // solver runs as usual (soft miss).
            common_fit_plan cache_plan;
            bool cache_ok = false;
            if (common_fit_cache_load(params, params.model.path, analysis, cache_plan)) {
                LOG_INF("%s: fit cache hit for %s, validating cached placement\n",
                        __func__, params.model.path.c_str());
                // an explicit --ctx-size is a hard requirement: never replay a
                // placement measured for a different context
                const bool ctx_mismatch = params.n_ctx > 0 && cache_plan.n_ctx != (int32_t) params.n_ctx;
                if (ctx_mismatch) {
                    LOG_WRN("%s: cached placement n_ctx=%d != requested n_ctx=%d, refitting\n",
                            __func__, cache_plan.n_ctx, params.n_ctx);
                }
                bool in_bounds = true;
                for (int id : cache_plan.role_devices) {
                    if (id >= (int) analysis.devices.size()) {
                        LOG_WRN("%s: cached plan references unknown device id %d, refitting\n",
                                __func__, id);
                        in_bounds = false;
                        break;
                    }
                }
                if (!ctx_mismatch && in_bounds &&
                        fit_measure_plan(probe, cache_plan, mparams, cparams, params.model.path, win_measure) &&
                        fit_same_devices(win_measure.devs, analysis)) {
                    // validate the cached placement against the margins that
                    // matched its own shed state: reservations the cached plan
                    // released are released here too
                    const std::vector<int64_t> cache_margins =
                        fit_effective_margins(margins, params.fit_params_reserves, cache_plan.shed_reserves);
                    if (fit_fits_margins(win_measure.memory, cache_margins, analysis)) {
                        final_plan = cache_plan;
                        have_plan  = true;
                        cache_ok   = true;
                    }
                }
                if (!cache_ok) {
                    LOG_WRN("%s: cached placement failed validation, refitting\n", __func__);
                }
            }
            if (!cache_ok) {
                if (params.fit_strategy == COMMON_FIT_STRATEGY_RPC_AWARE) {
                    have_plan = fit_solve_rpc_aware(
                            analysis, margins, params.fit_params_reserves, params, mparams, cparams, probe,
                            params.model.path, analysis.loader_ngl, final_plan, win_measure);
                    if (!have_plan) {
                        // exhaust: fall back to the stock solver. it cannot shed
                        // reservations, so its margins fold every one of them in
                        // as a hard floor (companions stay on VRAM).
                        LOG_WRN("%s: no rpc-aware candidate fit; falling back to the upstream fit\n", __func__);
                        std::vector<size_t> fallback_target = params.fit_params_target;
                        for (const auto & r : params.fit_params_reserves) {
                            if (r.device >= 0 && (size_t) r.device < fallback_target.size()) {
                                fallback_target[r.device] += r.bytes;
                            }
                        }
                        return common_fit_params(params.model.path.c_str(), mparams, cparams,
                                params.tensor_split, params.tensor_buft_overrides.data(), &params.moe_cache,
                                fallback_target.data(),
                                params.fit_strategy_min_ctx > 0 ? params.fit_strategy_min_ctx : params.fit_params_min_ctx,
                                GGML_LOG_LEVEL_INFO);
                    }
                } else {
                    final_plan = fit_solve_maxctx(
                            analysis, margins, params.fit_params_reserves, params, mparams, cparams, probe,
                            params.model.path, analysis.loader_ngl, win_measure);
                    have_plan = true;
                }
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
            if (!fit_fits_margins(win_measure.memory, margins, analysis)) {
                throw std::runtime_error("manual plan does not fit within the per-device margins");
            }
            have_plan = true;
        }

        if (!have_plan) {
            throw std::runtime_error("fit strategy produced no plan");
        }

        // the final placement is validated and serialized against the margins
        // that match its own shed state; reservations it released are released
        // here too
        const std::vector<int64_t> final_margins =
            fit_effective_margins(margins, params.fit_params_reserves, final_plan.shed_reserves);

        fit_serialize(final_plan, params, mparams, cparams, analysis, win_measure, final_margins);
        fit_validate(final_plan, analysis, final_margins, analysis.loader_ngl, mparams, cparams, probe, params.model.path);

        // publish the shed state so callers can move companions off VRAM
        params.fit_params_shed_tags = final_plan.shed_reserves;

        // save the successful placement to the disk cache (best-effort); on a
        // cache hit this refreshes the freshness timestamp
        common_fit_cache_save(params, params.model.path, analysis, final_plan);

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
    throw std::invalid_argument("unknown fit strategy: " + s);
}

bool common_activation_profile_sink(const common_activation_profile & profile, const std::string & path) {
    nlohmann::ordered_json row;
    row["version"]  = profile.version;
    row["model_id"] = profile.model_id;

    nlohmann::ordered_json counters = nlohmann::ordered_json::array();
    for (const auto & c : profile.counters) {
        nlohmann::ordered_json j;
        j["tensor_id"]  = c.tensor_id;
        j["role"]       = (int) c.role;
        j["n_tokens"]   = c.n_tokens;
        j["expert_set"] = c.expert_set;
        j["kv_touched"] = c.kv_touched;
        counters.push_back(j);
    }
    row["counters"] = counters;

    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        return false;
    }
    out << row.dump() << '\n';
    return (bool) out;
}

int llama_model_get_backend_device_id(const struct llama_model * model, int device_index) {
    if (model == nullptr || device_index < 0) {
        return -1;
    }

    const int32_t n_devices = llama_model_n_devices(model);
    if (device_index >= n_devices) {
        return -1;
    }

    ggml_backend_dev_t dev = llama_model_get_device(model, device_index);
    if (dev == nullptr) {
        return -1;
    }

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return -1;
    }

    const size_t backend_dev_count = ggml_backend_reg_dev_count(reg);
    for (size_t i = 0; i < backend_dev_count; i++) {
        if (ggml_backend_reg_dev_get(reg, i) == dev) {
            return (int)i;
        }
    }

    return -1;
}

ggml_backend_dev_t llama_model_get_device_by_backend_id(const struct llama_model * model, int backend_device_id) {
    if (model == nullptr || backend_device_id < 0) {
        return nullptr;
    }

    const int32_t n_devices = llama_model_n_devices(model);
    if (n_devices == 0) {
        return nullptr;
    }

    for (int32_t i = 0; i < n_devices; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model, i);
        if (dev == nullptr) {
            continue;
        }

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg == nullptr) {
            continue;
        }

        const size_t backend_dev_count = ggml_backend_reg_dev_count(reg);
        if ((size_t)backend_device_id < backend_dev_count) {
            if (ggml_backend_reg_dev_get(reg, backend_device_id) == dev) {
                return dev;
            }
        }
    }

    return nullptr;
}