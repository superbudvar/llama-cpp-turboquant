#include "fit-strategy.h"
#include "fit-cache.h"

#include "../src/llama-ext.h" // llama_model_get_moe_tensor_info, llama_model_get_tensor_devices (internal)
#include "common.h"


#include "log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <climits>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr uint32_t COMMON_FIT_SEARCH_POLICY_VERSION = 2;

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
        case COMMON_FIT_ROLE_LAYER_OUTPUT:return "layer_output";
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

// forward declaration
static std::vector<int64_t> fit_probe_ceiling(
        const common_fit_analysis & analysis,
        const common_params & params);

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

// Margins are excluded so MTP reserve-only refits can recheck exact footprints.
static std::string fit_measure_cache_key(
        const std::string & path_model,
        const llama_model_params & mparams,
        const llama_context_params & cparams,
        bool no_alloc) {
    std::ostringstream key;
    key << path_model << ':' << no_alloc << ':' << mparams.n_gpu_layers << ':'
        << mparams.split_mode << ':' << mparams.load_mode << ':' << mparams.tensor_read_lazy << ':'
        << mparams.main_gpu << ':' << mparams.vocab_only << ':' << mparams.check_tensors << ':'
        << mparams.use_extra_bufts << ':' << mparams.no_host << ':' << mparams.no_alloc << ':'
        << mparams.load_mtp << ':' << std::hexfloat;
    if (mparams.devices) {
        for (size_t i = 0; i < llama_max_devices(); i++) {
            key << reinterpret_cast<uintptr_t>(mparams.devices[i]) << ',';
            if (!mparams.devices[i]) break;
        }
    }
    key << ':';
    if (mparams.tensor_split) {
        for (size_t i = 0; i < llama_max_devices(); i++) key << mparams.tensor_split[i] << ',';
    }
    key << ':';
    if (mparams.tensor_buft_overrides) {
        for (const auto * item = mparams.tensor_buft_overrides; item->pattern; item++) {
            key << item->pattern << '=' << reinterpret_cast<uintptr_t>(item->buft) << ',';
        }
    }
    key << ':' << cparams.n_ctx << ':' << cparams.n_batch << ':' << cparams.n_ubatch << ':'
        << cparams.n_seq_max << ':' << cparams.n_rs_seq << ':' << cparams.gdn_replay << ':'
        << cparams.n_outputs_max << ':' << cparams.n_threads << ':' << cparams.n_threads_batch << ':'
        << cparams.ctx_type << ':' << cparams.rope_scaling_type << ':'
        << cparams.pooling_type << ':' << cparams.attention_type << ':' << cparams.flash_attn_type << ':'
        << cparams.rope_freq_base << ':' << cparams.rope_freq_scale << ':' << cparams.yarn_ext_factor << ':'
        << cparams.yarn_attn_factor << ':' << cparams.yarn_beta_fast << ':' << cparams.yarn_beta_slow << ':'
        << cparams.yarn_orig_ctx << ':' << cparams.defrag_thold << ':' << cparams.type_k << ':'
        << cparams.type_v << ':' << cparams.moe_cache_mode << ':' << cparams.moe_cache_budget_mib << ':'
        << cparams.embeddings << ':' << cparams.offload_kqv << ':' << cparams.no_perf << ':'
        << cparams.op_offload << ':' << cparams.swa_full << ':' << cparams.kv_unified << ':'
        << cparams.n_samplers << ':' << reinterpret_cast<uintptr_t>(cparams.samplers) << ':'
        << reinterpret_cast<uintptr_t>(cparams.ctx_other) << ':' << cparams.kv_stream_arena_mib;
    return key.str();
}

// measure a concrete plan through the probe seam. returns false when the
// context cannot be created at this placement (soft reject); hard errors and
// model-not-found rethrow.
static bool fit_measure_plan(
        const common_fit_probe & probe,
        const common_fit_plan & plan,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        const std::string & path_model,
        fit_measure_result & out,
        bool no_alloc = false,
        size_t * attempt_counter = nullptr,
        common_fit_search_session * search_session = nullptr) {
    const common_fit_measure_fn & measure = no_alloc && probe.measure_no_alloc
        ? probe.measure_no_alloc : probe.measure;
    if (!measure) {
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
    const std::string cache_key = search_session
        ? fit_measure_cache_key(path_model, cp, cc, no_alloc) : std::string();
    if (search_session) {
        const auto found = std::find_if(search_session->probe_cache.begin(), search_session->probe_cache.end(),
                [&](const auto & entry) { return entry.key == cache_key; });
        if (found != search_session->probe_cache.end()) {
            out.memory = found->memory;
            out.devs = found->devs;
            out.hp_ngl = found->hp_ngl;
            out.hp_nct = found->hp_nct;
            out.hp_nex = found->hp_nex;
            search_session->probe_cache_hits++;
            LOG_DBG("%s: reused exact probe for n_ctx=%u\n", __func__, cc.n_ctx);
            return true;
        }
    }
    try {
        if (attempt_counter) {
            (*attempt_counter)++;
        }
        out.memory = measure(path_model.c_str(), &cp, &cc,
                out.devs, out.hp_ngl, out.hp_nct, out.hp_nex, GGML_LOG_LEVEL_INFO);
    } catch (const std::runtime_error & e) {
        if (std::string(e.what()).find("failed to create llama_context") != std::string::npos) {
            LOG_DBG("%s: fit_measure_plan FAIL (no context) n_ctx=%u split0=%.3f\n",
                    __func__, cc.n_ctx, cp.tensor_split ? cp.tensor_split[0] : -1.0f);
            return false;
        }
        throw;
    }
    LOG_DBG("%s: fit_measure_plan OK n_ctx=%u split0=%.3f ndevs=%zu mode=%s\n",
            __func__, cc.n_ctx, cp.tensor_split ? cp.tensor_split[0] : -1.0f,
            out.devs.size(), no_alloc ? "no_alloc" : "alloc");
    if (search_session) {
        constexpr size_t probe_cache_limit = 256;
        if (search_session->probe_cache.size() >= probe_cache_limit) {
            search_session->probe_cache.erase(search_session->probe_cache.begin());
        }
        search_session->probe_cache.push_back({ cache_key, out.memory, out.devs, out.hp_ngl, out.hp_nct, out.hp_nex });
    }
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
        size_t * fail_dev = nullptr,
        int64_t * shortfall_bytes = nullptr,
        const std::vector<int64_t> * ceiling_free = nullptr,
        const std::vector<int64_t> * ceiling_free_warm = nullptr) {
    const size_t nd = analysis.devices.size();
    if (fail_dev) { *fail_dev = (size_t) -1; }
    if (shortfall_bytes) { *shortfall_bytes = 0; }
    LOG_DBG("%s: memory.size()=%zu nd+1=%zu margins.size()=%zu nd=%zu\n",
            __func__, memory.size(), nd + 1, margins.size(), nd);
    if (memory.size() != nd + 1 || margins.size() < nd) {
        LOG_DBG("%s: fit_fits_margins FAIL size guard\n", __func__);
        return false;
    }
    for (size_t id = 0; id < nd; id++) {
        const int64_t used = (int64_t) fit_device_used(memory[id]);
        // Use the cold-captured free VRAM, not the live value re-queried during
        // each no-allocation candidate probe. Context buffers can remain in the
        // backend pool between probes. Model bytes come from exact tensor
        // placement; context and compute bytes come from context initialization.
        // ceiling_free (when set) clamps the reported free to the measured ceiling
        // so that driver-level fluctuations above the measured ceiling don't cause
        // fits to overestimate usable memory.
        // ceiling_free_warm (when set) captures the true allocatable memory AFTER
        // model weights are loaded but before KV cache init, preventing the cold-only
        // ceiling from masking the ~800 MiB ROCm driver growth during init.
        int64_t free_cold = analysis.devices[id].free_bytes;
        if (ceiling_free && id < ceiling_free->size() && (*ceiling_free)[id] > 0) {
            free_cold = std::min(free_cold, (*ceiling_free)[id]);
        }
        if (ceiling_free_warm && id < ceiling_free_warm->size() && (*ceiling_free_warm)[id] > 0) {
            free_cold = std::min(free_cold, (*ceiling_free_warm)[id]);
        }
        const int64_t rem = free_cold - used;
        LOG_DBG("%s: dev[%zu] %s free_cold=%lld used=%lld (m%d c%d k%d) rem=%lld margin=%lld -> %s\n",
                __func__, id, ggml_backend_dev_name(analysis.devices[id].dev),
                (long long) free_cold, (long long) used,
                (int) memory[id].model, (int) memory[id].context, (int) memory[id].compute,
                (long long) rem, (long long) margins[id],
                (used > INT64_MAX || rem < margins[id]) ? "REJECT" : "ok");
        if (used > INT64_MAX ||
            rem < margins[id]) {
            LOG_DBG("%s: fit_fits_margins FAIL dev[%zu]\n", __func__, id);
            if (fail_dev) { *fail_dev = id; }
            // bytes the rejecting device is over budget by. the caller uses this
            // to size the next placement step instead of walking a blind grid.
            if (shortfall_bytes) { *shortfall_bytes = margins[id] - rem; }
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
    mparams_copy.load_mode = LLAMA_LOAD_MODE_MMAP;

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
    analysis.n_ctx_train = llama_model_n_ctx_train(model.get());

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
        int32_t ngl_override,
        const std::vector<int64_t> & ceiling_free,
        const std::vector<int64_t> & ceiling_free_warm,
        size_t * attempt_counter = nullptr,
        size_t attempt_limit = 0,
        common_fit_search_session * search_session = nullptr) {
    // an explicit --ctx-size is a hard requirement: fit model + exactly that
    // context and never grow or shrink it. the fit then only sheds model layers
    // (n_gpu_layers) to CPU/RPC if the full offload does not fit. when only
    // --fit-strategy-min-ctx is set we maximize context from the minimum upward
    // (the hi bound below is then the train context).
    //
    // fit policy (TurboQuant fork, B-second deferred to RPC-spill candidate):
    // 1. try full GPU offload at the largest ctx in [min_ctx, hi] that fits
    // 2. if even min_ctx does not fit at full offload, return -1 so the caller
    //    enters the CPU-spill phase (this function never spills)
    const int32_t lo_ctx = (int32_t) params.fit_strategy_min_ctx;
    int32_t hi = params.n_ctx > 0 ? (int32_t) params.n_ctx
        : 0;  // 0 = use the model's train context below
    // the original code used target_ctx = min_ctx as the floor of the binary
    // search. now we want lo = min_ctx and the upper bound = the largest ctx
    // we are willing to try. we pick that as either the explicit --ctx-size or
    // the model's n_ctx_train. the binary search below narrows it down.

    const auto fits = [&](const common_fit_plan & p, fit_measure_result & m) -> bool {
        if (attempt_counter && attempt_limit > 0 && *attempt_counter >= attempt_limit) {
            LOG_TRC("%s: grow fits n_ctx=%u measure=0 margins=0 -> FAIL (probe budget %zu/%zu)\n",
                    __func__, p.n_ctx, *attempt_counter, attempt_limit);
            return false;
        }
        const bool ok_m = fit_measure_plan(decode_probe, p, mparams, cparams, path_model, m, true,
                attempt_counter, search_session);
        const bool ok_devs = ok_m && fit_same_devices(m.devs, analysis);
        const bool ok_fm = ok_devs && fit_fits_margins(m.memory, margins, analysis, nullptr, nullptr, &ceiling_free, &ceiling_free_warm);
        LOG_TRC("%s: grow fits n_ctx=%u measure=%d margins=%d -> %s (probes=%zu/%zu)\n",
                __func__, p.n_ctx, ok_m ? 1 : 0, ok_fm ? 1 : 0, ok_fm ? "ok" : "FAIL",
                attempt_counter ? *attempt_counter : 0, attempt_limit);
        return ok_fm;
    };

    // step 1: measure at the largest ctx we want to consider. if it fits,
    // return immediately (no shrinkage needed). the upper bound is --ctx-size
    // when set, else the model's full train context.
    int32_t hp_nct_guess = 0;
    if (hi == 0) {
        // one quick measure at min-ctx to discover hp_nct (model's n_ctx_train)
        common_fit_plan plan_probe = fit_plan_from_spec(analysis, spec, loader_ngl, lo_ctx, rpc_frac_override, ngl_override);
        fit_measure_result m_probe;
        fits(plan_probe, m_probe);
        hp_nct_guess = m_probe.hp_nct > 0 ? (int32_t) m_probe.hp_nct : (int32_t) analysis.n_ctx_train;
        hi = hp_nct_guess > 0 ? hp_nct_guess : lo_ctx * 4;
    }
    if (hi <= lo_ctx) {
        // only one value to try. measure it and return.
        common_fit_plan plan_only = fit_plan_from_spec(analysis, spec, loader_ngl, hi, rpc_frac_override, ngl_override);
        fit_measure_result m_only;
        if (!fits(plan_only, m_only)) {
            return -1;
        }
        out_plan = plan_only;
        out_measure = m_only;
        return hi;
    }

    // step 2: try hi. if it fits, return. otherwise we need to shrink.
    common_fit_plan plan_hi = fit_plan_from_spec(analysis, spec, loader_ngl, hi, rpc_frac_override, ngl_override);
    fit_measure_result m_hi;
    if (fits(plan_hi, m_hi)) {
        out_plan = plan_hi;
        out_measure = m_hi;
        return hi;
    }

    // step 3: hi did not fit. check the hard floor (min-ctx): if even min-ctx
    // does not fit, the placement is infeasible at full GPU offload. the caller
    // will spill layers to CPU.
    common_fit_plan plan_lo = fit_plan_from_spec(analysis, spec, loader_ngl, lo_ctx, rpc_frac_override, ngl_override);
    fit_measure_result m_lo;
    if (!fits(plan_lo, m_lo)) {
        LOG_DBG("%s: even min-ctx=%d does not fit at full GPU offload, caller should spill to CPU\n",
                __func__, lo_ctx);
        return -1;
    }

    // step 4: shrink ctx. binary-search [lo_ctx, hi] for the largest ctx that
    // fits. lo_ctx already fits (step 3); hi does not (step 2). we look for
    // the boundary between fit and overflow.

    // binary search the largest context that still fits within margins.
    // invariant: lo fits, hi does not, lo < hi.
    int32_t lo = lo_ctx;
    int32_t best_ctx = lo_ctx;
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

    // fill headroom: after binary search finds the first-fit ctx, probe upward in
    // FILL_STEP increments to fill the margin headroom before committing. binary search
    // stopped at hi-lo <= 256, so hi is the smallest non-fitting ctx; the gap can be
    // ~50k tokens in typical cases (hi=262144, best_ctx=210688). bounded: at most
    // (hi - best_ctx) / 4096 + 1 extra probes, ~13 for the typical gap above.
    constexpr int32_t FILL_STEP = 4096;
    int32_t fill = best_ctx;
    while (fill + FILL_STEP <= hi) {
        int32_t trial = fill + FILL_STEP;
        trial -= trial % 256;
        if (trial <= fill) break;
        common_fit_plan pm = fit_plan_from_spec(analysis, spec, loader_ngl, trial, rpc_frac_override, ngl_override);
        fit_measure_result mm;
        if (fits(pm, mm)) {
            fill = trial;
            best_ctx = trial;
            best_plan = pm;
            best_m = mm;
        } else {
            break;
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
        float & out_frac,
        const std::vector<int64_t> & ceiling_free,
        const std::vector<int64_t> & ceiling_free_warm,
        size_t * attempt_counter = nullptr,
        size_t attempt_limit = 0,
        common_fit_search_session * search_session = nullptr,
        bool * out_budget_exhausted = nullptr) {
    out_frac = -1.0f;
    if (out_budget_exhausted) {
        *out_budget_exhausted = false;
    }
    bool budget_hit = false;
    const int32_t min_ctx = (int32_t) (params.n_ctx > 0 ? params.n_ctx : params.fit_strategy_min_ctx);
    const float step = 0.05f;
    const size_t nd = analysis.devices.size();

    // NOTE: fit_solve_rpc_aware is single-threaded; rpc_frac_cache_entry is safe.
    // The entry keeps the spec and margin hashes separately from the exact
    // key, so a neighbouring layer count can reuse the feasible window this
    // one found instead of scanning the whole range again.
    struct rpc_frac_cache_entry_t {
        uint64_t key;         // exact key: spec, ngl and margins
        uint64_t spec_hash;
        uint64_t margins_hash;
        float lo_frac;        // smallest feasible fraction
        float hi_frac;        // largest feasible fraction
        bool valid;
    };
    static rpc_frac_cache_entry_t rpc_frac_cache_entry[4] = {};

    // cache key: combine spec label hash, ngl, and margins hash
    uint64_t margins_hash = 1469598103934665603ull; // FNV offset basis
    for (int64_t m : margins) {
        margins_hash ^= (uint64_t)m;
        margins_hash *= 1097511628211ull; // FNV prime
    }
    uint64_t spec_hash = std::hash<std::string>{}(spec.label);
    uint64_t cache_key = spec_hash ^ ((uint64_t)ngl << 32) ^ margins_hash;

    // The feasible rpc_frac window moves slowly with the layer count, so a
    // window found at a neighbouring layer count is a good starting point.
    // Scanning the whole range at every layer count costs more probes than
    // the budget allows: the n_gpu_layers bisect needs several evaluations
    // and each one was paying for a full sweep plus a bisection.
    float hint_lo = 0.0f;
    float hint_hi = scan_hi;
    bool have_hint = false;
    for (int i = 0; i < 4; i++) {
        if (rpc_frac_cache_entry[i].valid &&
                rpc_frac_cache_entry[i].spec_hash == spec_hash &&
                rpc_frac_cache_entry[i].margins_hash == margins_hash) {
            hint_lo = std::max(0.0f, rpc_frac_cache_entry[i].lo_frac - 2.0f * step);
            hint_hi = std::min(scan_hi, rpc_frac_cache_entry[i].hi_frac + 2.0f * step);
            have_hint = true;
            LOG_DBG("%s: ngl=%d reusing window [%.4f,%.4f] from a neighbouring layer count\n",
                    __func__, ngl, hint_lo, hint_hi);
            break;
        }
    }

    struct row_t {
        float f;
        bool ok;
        size_t fail_dev;
    };
    std::vector<row_t> rows;
    std::vector<float> feas;
    bool cache_hit = false;

    // check cache before running the coarse scan
    for (int i = 0; i < 4; i++) {
        if (rpc_frac_cache_entry[i].valid && rpc_frac_cache_entry[i].key == cache_key) {
            LOG_DBG("%s: rpc_frac cache hit candidate=%s ngl=%d [%.4f, %.4f]\n",
                    __func__, spec.label.c_str(), ngl,
                    rpc_frac_cache_entry[i].lo_frac, rpc_frac_cache_entry[i].hi_frac);
            // reconstruct feas from cached bounds at step granularity; the binary search
            // below runs over these values so we skip ~20 coarse probes on cache hit.
            // false positives (non-feasible values in a contiguous interval) are benign:
            // the binary search tests each and discards infeasible ones, still arriving
            // at the correct best fraction.
            for (float f = rpc_frac_cache_entry[i].hi_frac; f >= rpc_frac_cache_entry[i].lo_frac - 1e-6f; f -= step) {                if (f < 0.0f) break;
                feas.push_back(f);
            }
            cache_hit = true;
            break;
        }
    }

    // coarse scan top-down, mirroring the full-offload bracket. rpc_frac
    // feasibility is an interval: low f overflows the local device, high f
    // overflows RPC.
    // The scan starts at the window a neighbouring layer count found, when
    // there is one, and widens to the full range only if nothing inside it
    // works. A full sweep at every layer count does not fit in the probe
    // budget: the n_gpu_layers bisect needs several evaluations.
    for (float f = have_hint ? hint_hi : scan_hi; f >= -step * 0.5f; f -= step) {
        if (f < 0.0f) { f = 0.0f; }
        if (attempt_counter && attempt_limit > 0 && *attempt_counter >= attempt_limit) {
            LOG_DBG("%s: ngl=%d scan stopped at rpc_frac=%.4f, probe budget %zu/%zu reached\n",
                    __func__, ngl, f, *attempt_counter, attempt_limit);
            budget_hit = true;
            break;
        }
        common_fit_plan plan = fit_plan_from_spec(analysis, spec, loader_ngl, min_ctx, f, ngl);
        fit_measure_result m;
        const bool ok_measure = fit_measure_plan(probe, plan, mparams, cparams, path_model, m, true,
                attempt_counter, search_session);
        const bool ok_devs    = ok_measure && fit_same_devices(m.devs, analysis);
        size_t fail = (size_t) -1;
        const bool ok_margin  = ok_devs && fit_fits_margins(m.memory, margins, analysis, &fail, nullptr, &ceiling_free, &ceiling_free_warm);
        LOG_DBG("%s: ngl=%d rpc_frac=%.4f measure=%d devs=%d margin=%d -> %s\n",
                __func__, ngl, f, ok_measure ? 1 : 0, ok_devs ? 1 : 0, ok_margin ? 1 : 0,
                ok_margin ? "FEASIBLE" : "rejected");
        rows.push_back({ f, ok_margin, fail });
        if (ok_margin) {
            feas.push_back(f);
        }
        if (f <= (have_hint ? hint_lo : 0.0f)) { break; }
    }

    // the hinted window held nothing: fall back to the full range before
    // concluding that this layer count has no feasible split.
    if (feas.empty() && have_hint) {
        LOG_DBG("%s: ngl=%d hinted window [%.4f,%.4f] held no feasible split, widening the scan\n",
                __func__, ngl, hint_lo, hint_hi);
        rows.clear();
        for (float f = scan_hi; f >= -step * 0.5f; f -= step) {
            if (f < 0.0f) { f = 0.0f; }
            if (attempt_counter && attempt_limit > 0 && *attempt_counter >= attempt_limit) {
                LOG_DBG("%s: ngl=%d scan stopped at rpc_frac=%.4f, probe budget %zu/%zu reached\n",
                        __func__, ngl, f, *attempt_counter, attempt_limit);
                budget_hit = true;
                break;
            }
            common_fit_plan plan = fit_plan_from_spec(analysis, spec, loader_ngl, min_ctx, f, ngl);
            fit_measure_result m;
            const bool ok_measure = fit_measure_plan(probe, plan, mparams, cparams, path_model, m, true,
                    attempt_counter, search_session);
            const bool ok_devs    = ok_measure && fit_same_devices(m.devs, analysis);
            size_t fail = (size_t) -1;
            const bool ok_margin  = ok_devs && fit_fits_margins(m.memory, margins, analysis, &fail, nullptr, &ceiling_free, &ceiling_free_warm);
            LOG_DBG("%s: ngl=%d rpc_frac=%.4f measure=%d devs=%d margin=%d -> %s\n",
                    __func__, ngl, f, ok_measure ? 1 : 0, ok_devs ? 1 : 0, ok_margin ? 1 : 0,
                    ok_margin ? "FEASIBLE" : "rejected");
            rows.push_back({ f, ok_margin, fail });
            if (ok_margin) {
                feas.push_back(f);
            }
            if (f <= 0.0f) { break; }
        }
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
        // the layer granularity is one layer over the GPU layer count (about 0.03 of
        // the fraction), so a precision far below that is wasted probes.
        if (have_lo && have_hi) {
            for (int it = 0; it < 8 && f_hi - f_lo > 1e-3f; it++) {
                if (attempt_counter && attempt_limit > 0 && *attempt_counter >= attempt_limit) {
                    LOG_DBG("%s: ngl=%d bisect stopped in [%.4f,%.4f], probe budget %zu/%zu reached\n",
                            __func__, ngl, f_lo, f_hi, *attempt_counter, attempt_limit);
                    budget_hit = true;
                    break;
                }
                const float fm = (f_lo + f_hi) * 0.5f;
                common_fit_plan plan = fit_plan_from_spec(analysis, spec, loader_ngl, min_ctx, fm, ngl);
                fit_measure_result m;
                const bool ok_measure = fit_measure_plan(probe, plan, mparams, cparams, path_model, m, true, attempt_counter);
                const bool ok_devs    = ok_measure && fit_same_devices(m.devs, analysis);
                size_t fail = (size_t) -1;
                const bool ok_margin  = ok_devs && fit_fits_margins(m.memory, margins, analysis, &fail, nullptr, &ceiling_free, &ceiling_free_warm);
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

    // populate the cache with the feasible interval for this (candidate, ngl, margins) key
    if (!cache_hit) {
        // lo_frac must be the smallest feasible fraction and hi_frac the
        // largest: the cache-hit reconstruction below scans from hi_frac down
        // to lo_frac. storing them the other way round made that loop a no-op,
        // so a cache hit produced an empty feasible set, fit_spill_split_at_ngl
        // returned -1 and the n_gpu_layers bisect rejected a layer count that
        // actually fits - which shows up as extra CPU spill.
        float lo_frac = 0.0f;
        float hi_frac = 0.0f;
        bool have_feas = !feas.empty();
        if (have_feas) {
            lo_frac = feas.front();
            hi_frac = feas.back();
        }
        // LRU eviction: find first invalid slot or oldest (index 0)
        int slot = 0;
        for (int i = 0; i < 4; i++) {
            if (!rpc_frac_cache_entry[i].valid) { slot = i; break; }
            if (i == 3) slot = 0; // all valid: evict oldest (slot 0)
        }
        rpc_frac_cache_entry[slot] = { cache_key, spec_hash, margins_hash, lo_frac, hi_frac, have_feas };
        LOG_DBG("%s: rpc_frac cache populate candidate=%s ngl=%d [%.4f, %.4f]%s\n",
                __func__, spec.label.c_str(), ngl, lo_frac, hi_frac,
                have_feas ? "" : " (empty)");
    }

    // grow the context from each feasible split and keep the one at the largest
    // context. scan ascending rpc_frac so among splits reaching the same maximum
    // context the smallest rpc_frac (least RPC offload) wins: RPC is used last.
    std::sort(feas.begin(), feas.end());
    int32_t best_ctx = -1;
    common_fit_plan best_plan;
    fit_measure_result best_m;
    for (float f : feas) {
        if (attempt_counter && attempt_limit > 0 && *attempt_counter >= attempt_limit) {
            LOG_DBG("%s: ngl=%d refine stopped at rpc_frac=%.4f, probe budget %zu/%zu reached\n",
                    __func__, ngl, f, *attempt_counter, attempt_limit);
            break;
        }
        common_fit_plan plan;
        fit_measure_result m;
        const int32_t ctx = fit_grow_context(analysis, margins, params, mparams, cparams,
                probe, spec, f, loader_ngl, path_model, plan, m, ngl, ceiling_free, ceiling_free_warm,
                attempt_counter, attempt_limit, search_session);
        if (ctx < 0) { continue; }
        if (best_ctx < 0 || ctx > best_ctx) {
            best_ctx = ctx;
            out_frac = f;
            best_plan = plan;
            best_m = m;
        }
    }
    if (best_ctx < 0) {
        // no feasible split. if the probe budget ran out, the scan never got
        // to answer the question: the caller must not read that as "this
        // layer count cannot work", because it then narrows the search to
        // fewer GPU layers and spills more of the model to the CPU.
        if (out_budget_exhausted) {
            *out_budget_exhausted = budget_hit;
        }
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

static int32_t fit_context_ceiling(const common_fit_analysis & analysis, const common_params & params) {
    return params.n_ctx > 0 ? (int32_t) params.n_ctx : (int32_t) analysis.n_ctx_train;
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

static float fit_rpc_preferred_fraction(
        const common_fit_analysis & analysis,
        const std::vector<int64_t> & margins,
        const std::vector<int> & split_devices,
        float hi) {
    int local = -1;
    int rpc = -1;
    for (int id : split_devices) {
        if (analysis.devices[id].is_rpc) rpc = id;
        else if (local < 0) local = id;
    }
    if (local < 0 || rpc < 0) return hi * 0.5f;
    const int64_t local_headroom = std::max<int64_t>(0, analysis.devices[local].free_bytes - margins[local]);
    const int64_t rpc_headroom = std::max<int64_t>(0, analysis.devices[rpc].free_bytes - margins[rpc]);
    const int64_t total_headroom = local_headroom + rpc_headroom;
    if (total_headroom <= 0) return hi * 0.5f;
    return std::min(hi, (float) rpc_headroom / (float) total_headroom);
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
        fit_measure_result & out_measure,
        int32_t ngl_hint,
        const std::vector<int64_t> & ceiling_free,
        const std::vector<int64_t> & ceiling_free_warm,
        const common_fit_plan * resume_plan,
        common_fit_search_session * search_session) {
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
    fprintf(stderr, "[FITDBG] rpc-aware entered nd=%zu local=%d loader_ngl=%u min_ctx=%d\n", analysis.devices.size(), local_best, (unsigned) loader_ngl, (int) params.fit_strategy_min_ctx); fflush(stderr);
    const int rpc_best = fit_best_rpc(analysis);
    fprintf(stderr, "[FITDBG] rpc-aware rpc_best=%d\n", rpc_best); fflush(stderr);

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
    fprintf(stderr, "[FITDBG] building candidates: %zu dense_splits, %zu pool_opts, moe.n_expert=%d\n", dense_splits.size(), pool_opts.size(), analysis.moe.n_expert); fflush(stderr);
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
    fprintf(stderr, "[FITDBG] %zu candidates, %zu shed levels\n", candidates.size(), shed_levels.size()); fflush(stderr);
    const uint64_t fit_key = common_fit_cache_key_hash(params, path_model, analysis);
    // the probe budget bounds searching against one margin configuration. the MTP
    // preflight raises fit_params_target between iterations by design, so each
    // iteration is a different problem. carrying the previous iteration's spend
    // left the next one unable to afford a single probe, so it returned the floor
    // it was handed instead of the context it could have reached. the incumbent
    // is still carried and re-validated below; only the spend is dropped.
    const bool fit_key_changed = search_session && search_session->fit_key != 0 &&
            search_session->fit_key != fit_key;
    size_t attempt_count = (search_session && !fit_key_changed) ? search_session->probes_used : 0;
    const size_t attempt_limit = params.fit_max_attempts;
    common_fit_plan session_resume_plan;
    const common_fit_plan * progress_plan = resume_plan;
    if (search_session && search_session->has_incumbent &&
            (!progress_plan || progress_plan->search.complete)) {
        session_resume_plan = search_session->incumbent;
        const auto shed_it = std::find(shed_levels.begin(), shed_levels.end(), session_resume_plan.shed_reserves);
        if (shed_it != shed_levels.end()) {
            const bool same_fit_policy = search_session->fit_key == fit_key &&
                    session_resume_plan.search.policy_version == COMMON_FIT_SEARCH_POLICY_VERSION;
            session_resume_plan.search.complete = false;
            session_resume_plan.search.shed_level = (int32_t) (shed_it - shed_levels.begin());
            if (!same_fit_policy || session_resume_plan.search.baseline_free.size() != analysis.devices.size()) {
                session_resume_plan.search.candidate_cursor = 0;
                session_resume_plan.search.rpc_stage = 0;
                session_resume_plan.search.rpc_frac_next = -1.0f;
                session_resume_plan.search.baseline_free.clear();
                for (const auto & d : analysis.devices) session_resume_plan.search.baseline_free.push_back(d.free_bytes);
            }
            session_resume_plan.search.policy_version = COMMON_FIT_SEARCH_POLICY_VERSION;
            progress_plan = &session_resume_plan;
        } else {
            progress_plan = nullptr;
        }
    }
    common_fit_plan policy_resume_plan;
    if (progress_plan && !progress_plan->search.complete &&
            progress_plan->search.policy_version != COMMON_FIT_SEARCH_POLICY_VERSION) {
        policy_resume_plan = *progress_plan;
        policy_resume_plan.search.policy_version = COMMON_FIT_SEARCH_POLICY_VERSION;
        policy_resume_plan.search.candidate_cursor = 0;
        policy_resume_plan.search.rpc_stage = 0;
        policy_resume_plan.search.rpc_frac_next = -1.0f;
        policy_resume_plan.search.baseline_free.clear();
        for (const auto & d : analysis.devices) policy_resume_plan.search.baseline_free.push_back(d.free_bytes);
        progress_plan = &policy_resume_plan;
    }
    const bool resume = progress_plan && !progress_plan->search.complete &&
            progress_plan->search.baseline_free.size() == analysis.devices.size() &&
            progress_plan->search.shed_level >= 0 &&
            (size_t) progress_plan->search.shed_level < shed_levels.size();
    bool search_complete = true;
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

        // track the best candidate across this shed level. when multiple candidates
        // reach the same max context, prefer the one with the widest split
        // (most devices in split_devices) so the placement uses available VRAM
        // across devices instead of packing one local device to the brim.
        int32_t best_ctx = -1;
        common_fit_plan best_plan;
        fit_measure_result best_m;
        size_t best_split_size = 0;
        std::string best_label;
        float best_rpc_frac = -1.0f;
        const int32_t context_ceiling = fit_context_ceiling(analysis, params);

        size_t first_candidate = 0;
        bool resume_rpc_stage = false;
        if (resume && li == (size_t) progress_plan->search.shed_level) {
            bool baseline_matches = true;
            for (size_t id = 0; id < analysis.devices.size(); id++) {
                const int64_t old_free = progress_plan->search.baseline_free[id];
                const int64_t new_free = analysis.devices[id].free_bytes;
                if (std::llabs(old_free - new_free) > 128LL * 1024 * 1024) {
                    baseline_matches = false;
                    break;
                }
            }
            if (baseline_matches) {
                first_candidate = std::min(candidates.size(), (size_t) std::max(0, progress_plan->search.candidate_cursor));
                resume_rpc_stage = progress_plan->search.rpc_stage == 2;
            }

            // the incumbent is a concrete placement, not part of the cursor: adopt
            // and re-validate it even when the cursor is not resumable. free memory
            // shifting between MTP preflight iterations is normal (the reserve
            // grows), and dropping a known-good plan there costs context: the next
            // iteration starts with no incumbent, spends its first probe on the
            // floor, and if the carried budget is already spent it retains that
            // floor instead of the context the previous iteration had found.
            best_ctx = progress_plan->n_ctx;
            best_plan = *progress_plan;
            best_split_size = (size_t) std::max(0, progress_plan->search.best_split_size);
            best_rpc_frac = progress_plan->search.best_rpc_frac;
            best_label = "cached incumbent";
            if (fit_measure_plan(probe, best_plan, mparams, cparams, path_model, best_m, true,
                        &attempt_count, search_session)) {
                if (!fit_same_devices(best_m.devs, analysis) ||
                        !fit_fits_margins(best_m.memory, margins, analysis, nullptr, nullptr, &ceiling_free, &ceiling_free_warm)) {
                    best_ctx = -1;
                }
            } else {
                best_ctx = -1;
            }
            if (best_ctx < 0) {
                first_candidate = 0;
                resume_rpc_stage = false;
            } else if (baseline_matches) {
                LOG_INF("%s: resuming search after candidate %zu with incumbent n_ctx=%d\n",
                        __func__, first_candidate, best_ctx);
            } else {
                LOG_INF("%s: device baseline shifted; keeping incumbent n_ctx=%d and restarting the search\n",
                        __func__, best_ctx);
            }
        }

        const auto retain_incumbent = [&](size_t candidate_cursor) {
            search_complete = false;
            out_plan = best_plan;
            out_plan.search.complete = false;
            out_plan.search.policy_version = COMMON_FIT_SEARCH_POLICY_VERSION;
            out_plan.search.shed_level = (int32_t) li;
            out_plan.search.candidate_cursor = (int32_t) candidate_cursor;
            out_plan.search.best_split_size = (int32_t) best_split_size;
            out_plan.search.best_rpc_frac = best_rpc_frac;
            out_plan.search.baseline_free.clear();
            for (const auto & d : analysis.devices) out_plan.search.baseline_free.push_back(d.free_bytes);
            out_measure = best_m;
        };

        for (size_t ci = first_candidate; ci < candidates.size(); ci++) {
            if (best_ctx >= 0 && attempt_limit > 0 && attempt_count >= attempt_limit) {
                retain_incumbent(ci);
                LOG_INF("%s: attempt budget reached (%zu measured probes); retaining incumbent n_ctx=%d\n",
                        __func__, attempt_count, best_ctx);
                break;
            }
            const auto & cand = candidates[ci];
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
                        probe, cand, -1.0f, loader_ngl, path_model, plan, m, ngl_hint, ceiling_free, ceiling_free_warm,
                        &attempt_count, attempt_limit, search_session);
                if (ctx >= 0) {
                    // prefer wider split on context tie. when a later candidate
                    // reaches the same context with more devices, overwrite.
                    const size_t split_size = cand.split_devices.size();
                    if (ctx > best_ctx || (ctx == best_ctx && split_size > best_split_size)) {
                        best_ctx = ctx;
                        best_split_size = split_size;
                        best_plan = plan;
                        best_m = m;
                        best_label = cand.label;
                        best_rpc_frac = -1.0f;
                    }
                } else {
                    LOG_DBG("%s: candidate %s cannot fit even the minimum context, skipping\n",
                            __func__, cand.label.c_str());
                    if (!have_spill) {
                        spill_spec = cand;
                        have_spill = true;
                    }
                }
                if (best_ctx >= context_ceiling && context_ceiling > 0) {
                    LOG_INF("%s: context ceiling reached by %s at n_ctx=%d; accepting this margin-valid plan without probing later candidates\n",
                            __func__, best_label.c_str(), best_ctx);
                    break;
                }
                if (best_ctx >= 0 && attempt_limit > 0 && attempt_count >= attempt_limit) {
                    search_complete = false;
                    out_plan = best_plan;
                    out_plan.search.complete = false;
                    out_plan.search.policy_version = COMMON_FIT_SEARCH_POLICY_VERSION;
                    out_plan.search.shed_level = (int32_t) li;
                    out_plan.search.candidate_cursor = (int32_t) (ci + 1);
                    out_plan.search.best_split_size = (int32_t) best_split_size;
                    out_plan.search.best_rpc_frac = best_rpc_frac;
                    out_plan.search.baseline_free.clear();
                    for (const auto & d : analysis.devices) out_plan.search.baseline_free.push_back(d.free_bytes);
                    out_measure = best_m;
                    LOG_INF("%s: attempt budget reached (%zu measured probes); retaining incumbent n_ctx=%d and resuming after candidate %zu\n",
                            __func__, attempt_count, best_ctx, ci + 1);
                    break;
                }
                continue;
            }

            const float hi = fit_rpc_scan_hi(analysis, cand.split_devices);
            const bool resume_this_rpc = resume_rpc_stage &&
                    li == (size_t) progress_plan->search.shed_level &&
                    ci == (size_t) std::max(0, progress_plan->search.candidate_cursor);
            // Start near the headroom balance and grow the first safe split before
            // spending probes on less promising fractions.
            const float preferred = fit_rpc_preferred_fraction(analysis, margins, cand.split_devices, hi);
            std::vector<float> fractions;
            for (float f = 0.0f; f < hi; f += 0.05f) fractions.push_back(f);
            if (fractions.empty() || std::fabs(fractions.back() - hi) > 1e-6f) fractions.push_back(hi);
            if (std::none_of(fractions.begin(), fractions.end(), [&](float f) { return std::fabs(f - preferred) < 1e-6f; })) {
                fractions.push_back(preferred);
            }
            std::stable_sort(fractions.begin(), fractions.end(), [&](float a, float b) {
                const float da = std::fabs(a - preferred);
                const float db = std::fabs(b - preferred);
                return da == db ? a < b : da < db;
            });

            size_t fraction_start = 0;
            if (resume_this_rpc) {
                const float next = progress_plan->search.rpc_frac_next;
                const auto next_it = std::find_if(fractions.begin(), fractions.end(),
                        [&](float f) { return std::fabs(f - next) < 1e-6f; });
                if (next_it == fractions.end()) {
                    LOG_WRN("%s: cached RPC split cursor is invalid; restarting this candidate\n", __func__);
                } else {
                    fraction_start = (size_t) (next_it - fractions.begin());
                }
            }

            float best_ftry = -1.0f;
            int32_t cand_best_ctx = -1;
            common_fit_plan cand_best_plan;
            fit_measure_result cand_best_m;
            bool found_floor_fit = false;
            // feasibility window. a local-device rejection raises frac_lo, an RPC
            // rejection lowers frac_hi, so grid points outside are skipped without
            // a probe. frac_target is the magnitude-directed guess for the next
            // probe: bytes over budget, divided by the total model bytes, is the
            // rpc_frac delta that moves them off the rejecting device.
            float frac_lo = resume_this_rpc ? progress_plan->search.rpc_frac_low : 0.0f;
            float frac_hi = resume_this_rpc ? progress_plan->search.rpc_frac_high : hi;
            float frac_target = -1.0f;
            int64_t total_model_bytes = 0;
            size_t frac_probes = 0;
            size_t fi = fraction_start;
            while (true) {
                if (best_ctx >= 0 && attempt_limit > 0 && attempt_count >= attempt_limit) {
                    retain_incumbent(ci);
                    out_plan.search.rpc_stage = 2;
                    out_plan.search.rpc_frac_next = fi < fractions.size() ? fractions[fi] : frac_target;
                    out_plan.search.rpc_frac_low = frac_lo;
                    out_plan.search.rpc_frac_high = frac_hi;
                    LOG_INF("%s: attempt budget reached (%zu measured probes); retaining incumbent n_ctx=%d\n",
                            __func__, attempt_count, best_ctx);
                    break;
                }

                // the directed guess goes first, then the next grid point inside
                // the window. Nothing inside the window is skipped: a fraction
                // that fits can reach more context than one that does not, so the
                // window only prunes what is known to be infeasible.
                float ftry;
                bool have_try = false;
                if (frac_target >= 0.0f && frac_target > frac_lo && frac_target < frac_hi) {
                    ftry = frac_target;
                    frac_target = -1.0f;
                    have_try = true;
                }
                while (fi < fractions.size()) {
                    const float f = fractions[fi];
                    if (f > frac_lo && f < frac_hi) {
                        ftry = f;
                        have_try = true;
                        break;
                    }
                    ++fi;
                }
                if (!have_try) {
                    break;
                }
                ++fi;
                ++frac_probes;

                common_fit_plan floor_plan = fit_plan_from_spec(analysis, cand, loader_ngl,
                        (int32_t) params.fit_strategy_min_ctx, ftry);
                fit_measure_result floor_measure;
                const bool ok_measure = fit_measure_plan(probe, floor_plan, mparams, cparams,
                        path_model, floor_measure, true, &attempt_count, search_session);
                const bool ok_devs = ok_measure && fit_same_devices(floor_measure.devs, analysis);
                size_t fail_dev = (size_t) -1;
                int64_t shortfall = 0;
                const bool ok_margin = ok_devs && fit_fits_margins(floor_measure.memory, margins, analysis,
                        &fail_dev, &shortfall, &ceiling_free, &ceiling_free_warm);
                if (total_model_bytes == 0) {
                    for (size_t id = 0; id < analysis.devices.size() && id < floor_measure.memory.size(); id++) {
                        total_model_bytes += (int64_t) floor_measure.memory[id].model;
                    }
                }
                LOG_TRC("%s: rpc_frac=%.4f min_ctx measure=%d devs=%d margin=%d -> %s (probe=%zu probes=%zu/%zu dev=%zu short=%lld window=[%.4f,%.4f] candidate=%s)\n",
                        __func__, ftry, ok_measure ? 1 : 0, ok_devs ? 1 : 0, ok_margin ? 1 : 0,
                        ok_margin ? "FEASIBLE" : "rejected", frac_probes, attempt_count,
                        attempt_limit, fail_dev, (long long) shortfall, frac_lo, frac_hi,
                        cand.label.c_str());
                if (!ok_margin) {
                    if (fail_dev >= analysis.devices.size()) {
                        continue;   // size guard or device mismatch, no direction to learn
                    }
                    // low rpc_frac overflows the local device, high overflows RPC
                    const bool rpc_overflow = analysis.devices[fail_dev].is_rpc;
                    if (rpc_overflow) {
                        frac_hi = std::min(frac_hi, ftry);
                    } else {
                        frac_lo = std::max(frac_lo, ftry);
                    }
                    if (shortfall > 0 && total_model_bytes > 0) {
                        const float delta = (float) ((double) shortfall / (double) total_model_bytes);
                        const float aimed = rpc_overflow ? ftry - delta : ftry + delta;
                        if (aimed > frac_lo && aimed < frac_hi) {
                            frac_target = aimed;
                        }
                    }
                    continue;
                }
                found_floor_fit = true;

                const int32_t min_ctx = (int32_t) params.fit_strategy_min_ctx;
                const size_t split_size = cand.split_devices.size();
                if (cand_best_ctx < 0 || min_ctx > cand_best_ctx ||
                        (min_ctx == cand_best_ctx && ftry < best_ftry)) {
                    cand_best_ctx = min_ctx;
                    best_ftry = ftry;
                    cand_best_plan = floor_plan;
                    cand_best_m = floor_measure;
                }
                if (min_ctx > best_ctx || (min_ctx == best_ctx && split_size > best_split_size) ||
                        (min_ctx == best_ctx && split_size == best_split_size && best_rpc_frac >= 0.0f && ftry < best_rpc_frac)) {
                    best_ctx = min_ctx;
                    best_split_size = split_size;
                    best_plan = floor_plan;
                    best_m = floor_measure;
                    best_label = cand.label;
                    best_rpc_frac = ftry;
                }

                common_fit_plan plan;
                fit_measure_result m;
                const size_t refine_limit = best_ctx >= 0 ? attempt_limit : 0;
                const int32_t ctx = fit_grow_context(analysis, margins, params, mparams, cparams,
                        probe, cand, ftry, loader_ngl, path_model, plan, m, -1, ceiling_free, ceiling_free_warm,
                        &attempt_count, refine_limit, search_session);
                if (ctx >= 0) {
                    if (ctx > cand_best_ctx || (ctx == cand_best_ctx && ftry < best_ftry)) {
                        cand_best_ctx = ctx;
                        best_ftry = ftry;
                        cand_best_plan = plan;
                        cand_best_m = m;
                    }
                    if (ctx > best_ctx || (ctx == best_ctx && split_size > best_split_size) ||
                            (ctx == best_ctx && split_size == best_split_size && best_rpc_frac >= 0.0f && ftry < best_rpc_frac)) {
                        best_ctx = ctx;
                        best_split_size = split_size;
                        best_plan = plan;
                        best_m = m;
                        best_label = cand.label;
                        best_rpc_frac = ftry;
                    }
                    if (context_ceiling > 0 && ctx >= context_ceiling) break;
                }
            }
            if (search_complete && best_ctx >= 0 && attempt_limit > 0 && attempt_count >= attempt_limit &&
                    !(context_ceiling > 0 && best_ctx >= context_ceiling)) {
                retain_incumbent(ci);
                out_plan.search.rpc_stage = 2;
                out_plan.search.rpc_frac_next = best_ftry >= 0.0f ? best_ftry : preferred;
                out_plan.search.rpc_frac_low = frac_lo;
                out_plan.search.rpc_frac_high = frac_hi;
            }
            if (!found_floor_fit && !resume_this_rpc) {
                LOG_DBG("%s: candidate %s could not fit any rpc_frac at the minimum context, skipping\n",
                        __func__, cand.label.c_str());
                if (!have_rpc_spill) {
                    spill_spec = cand;
                    have_rpc_spill = true;
                    have_spill = true;
                }
            }
            if (cand_best_ctx >= 0) {
                // candidate succeeded: merge into the outer best, prefer wider split on tie
                const size_t split_size = cand.split_devices.size();
                LOG_DBG("%s: candidate %s found n_ctx=%d rpc_frac=%.4f\n",
                        __func__, cand.label.c_str(), cand_best_ctx, best_ftry);
                if (cand_best_ctx > best_ctx ||
                        (cand_best_ctx == best_ctx && split_size > best_split_size)) {
                    best_ctx = cand_best_ctx;
                    best_split_size = split_size;
                    best_plan = cand_best_plan;
                    best_m = cand_best_m;
                    best_label = cand.label;
                    best_rpc_frac = best_ftry;
                }
            } else {
                LOG_DBG("%s: candidate %s could not grow the context at full offload, marking for CPU spill\n",
                    __func__, cand.label.c_str());
                if (!have_rpc_spill) {
                    spill_spec = cand;
                    have_rpc_spill = true;
                    have_spill = true;
                }
            }
            if (!search_complete) {
                LOG_INF("%s: attempt budget reached (%zu measured probes); retaining incumbent n_ctx=%d and resuming rpc_frac=%.4f\n",
                        __func__, attempt_count, best_ctx, out_plan.search.rpc_frac_next);
                break;
            }

            // Once a margin-valid candidate reaches the absolute context cap,
            // later candidates cannot improve the primary objective.
            if (best_ctx >= context_ceiling && context_ceiling > 0) {
                LOG_INF("%s: context ceiling reached by %s at n_ctx=%d; accepting this margin-valid plan without probing later candidates\n",
                        __func__, best_label.c_str(), best_ctx);
                break;
            }
        }

        // after trying all candidates, return the best one if any reached min_ctx.
        // among candidates tied at the max context, prefer the widest split
        // (most devices) so the placement uses available VRAM across devices
        // rather than packing one local device to the brim.
        if (best_ctx >= 0) {
            if (search_complete) {
                out_plan = best_plan;
                out_measure = best_m;
                out_plan.search.complete = true;
                out_plan.search.policy_version = COMMON_FIT_SEARCH_POLICY_VERSION;
                out_plan.search.rpc_stage = 0;
                out_plan.search.rpc_frac_next = -1.0f;
            }
            out_plan.shed_reserves = shed_tags;
            out_plan.projected_with_delta = !out_plan.delta_free_ema.empty();
            out_plan.projected_free.resize(analysis.devices.size());
            for (size_t id = 0; id < analysis.devices.size(); id++) {
                out_plan.projected_free[id] = analysis.devices[id].free_bytes - fit_device_used(out_measure.memory[id]);
            }
            LOG_INF("%s: selected candidate=%s n_ctx=%d rpc_frac=%.4f split_size=%zu (min=%d)\n",
                    __func__, best_label.c_str(), best_ctx, best_rpc_frac,
                    best_split_size, params.fit_strategy_min_ctx);
            if (search_session) {
                search_session->probes_used = attempt_count;
            }
            return true;
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
            // warm-start: if ngl_hint is available, bound hi_ngl to it to reduce binary
            // search iterations when the cached placement's layer count is still feasible
            const int32_t hi_ngl_bound = (ngl_hint >= 0 && ngl_hint <= (int32_t) loader_ngl)
                ? ngl_hint + 1 : (int32_t) loader_ngl + 1;
            int32_t hi_ngl = hi_ngl_bound; // full offload incl. output layer (failed min_ctx)
            int32_t best_ngl = -1;
            common_fit_plan best_plan;
            fit_measure_result best_m;
            float best_frac = -1.0f;
            while (lo_ngl <= hi_ngl) {
                if (attempt_limit > 0 && attempt_count >= attempt_limit) {
                    LOG_INF("%s: attempt budget reached (%zu measured probes) during the n_gpu_layers bisect\n",
                            __func__, attempt_count);
                    break;
                }
                const int32_t mid_ngl = lo_ngl + (hi_ngl - lo_ngl) / 2;
                common_fit_plan plan;
                fit_measure_result m;
                int32_t ctx;
                float frac = -1.0f;
                if (spec_is_rpc_split) {
                    bool exhausted = false;
                    ctx = fit_spill_split_at_ngl(analysis, margins, params, mparams, cparams,
                            probe, spill_spec, loader_ngl, path_model, mid_ngl, scan_hi,
                            plan, m, frac, ceiling_free, ceiling_free_warm, &attempt_count, attempt_limit, search_session,
                            &exhausted);
                    if (exhausted) {
                        // the probe budget ran out before this layer count could
                        // be judged. treat it as unknown, not as infeasible:
                        // narrowing the search here would drop GPU layers and
                        // spill more of the model to the CPU.
                        LOG_WRN("%s: probe budget reached while testing n_gpu_layers=%d, "                                "stopping the CPU-spill search with the best placement found so far\n",
                                __func__, mid_ngl);
                        break;
                    }
                } else {
                    ctx = fit_grow_context(analysis, margins, params, mparams, cparams,
                            probe, spill_spec, -1.0f, loader_ngl, path_model, plan, m, mid_ngl, ceiling_free,
                            ceiling_free_warm, &attempt_count, attempt_limit, search_session);
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
                out_plan.projected_with_delta = !out_plan.delta_free_ema.empty();
                out_plan.projected_free.resize(analysis.devices.size());
                for (size_t id = 0; id < analysis.devices.size(); id++) {
                    out_plan.projected_free[id] = analysis.devices[id].free_bytes - fit_device_used(out_measure.memory[id]);
                }
                if (best_frac >= 0.0f) {
                    LOG_WRN("%s: spilling %u layers to CPU to reach min-ctx %d (n_gpu_layers=%d of %u, rpc_frac=%.4f, probes=%zu/%zu)\n",
                            __func__, (uint32_t) (loader_ngl + 1 - best_ngl), min_ctx, best_ngl, loader_ngl + 1, best_frac, attempt_count, attempt_limit);
                } else {
                    LOG_WRN("%s: spilling %u layers to CPU to reach min-ctx %d (n_gpu_layers=%d of %u, probes=%zu/%zu)\n",
                            __func__, (uint32_t) (loader_ngl + 1 - best_ngl), min_ctx, best_ngl, loader_ngl + 1, attempt_count, attempt_limit);
                }
                // carry the budget across searches. the full-offload path writes
                // this on its own success return; without it a spilling solve
                // restarts the next search near zero probes.
                if (search_session) {
                    search_session->probes_used = attempt_count;
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
        fit_measure_result & out_measure,
        const std::vector<int64_t> & ceiling_free,
        const std::vector<int64_t> & ceiling_free_warm,
        size_t * attempt_counter = nullptr,
        size_t attempt_limit = 0) {
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
            if (attempt_counter && attempt_limit > 0 && *attempt_counter >= attempt_limit) {
                LOG_DBG("%s: placement %s not probed, probe budget %zu/%zu reached\n",
                        __func__, placement.label.c_str(), *attempt_counter, attempt_limit);
                break;
            }
            // grow the context from the minimum upward; this also tests the full
            // context first and uses it directly when it fits.
            common_fit_plan plan;
            fit_measure_result m;
            const int32_t ctx = fit_grow_context(analysis, margins, params, mparams, cparams,
                    probe, placement, -1.0f, loader_ngl, path_model, plan, m, -1, ceiling_free,
                    ceiling_free_warm, attempt_counter, attempt_limit);
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
        const std::string & path_model,
        const std::vector<int64_t> * ceiling_free_warm = nullptr) {
    // one final measure with the serialized placement
    fit_measure_result final_measure;
    if (!fit_measure_plan(probe, plan, mparams, cparams, path_model, final_measure, true)) {
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
        // ceiling_free_warm (when set) further constrains the headroom to the
        // measured warm ceiling, preventing ROCm driver overhead from masking the
        // true allocatable memory at context-init time.
        int64_t rem = std::min((int64_t) final_measure.memory[id].free,
                               (int64_t) final_measure.memory[id].total - used);
        if (ceiling_free_warm && id < ceiling_free_warm->size() && (*ceiling_free_warm)[id] > 0) {
            rem = std::min(rem, (*ceiling_free_warm)[id]);
        }
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
    rparams.load_mode = LLAMA_LOAD_MODE_MMAP;
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
        const common_fit_probe & probe,
        common_fit_plan * plan_out,
        common_fit_search_session * search_session) {
    const int64_t t0_us = llama_time_us();
    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_SUCCESS;
    try {
        if (params.fit_strategy == COMMON_FIT_STRATEGY_UPSTREAM) {
            uint32_t hp_ngl = 0;
            common_fit_analysis analysis = fit_analyze(params, mparams, params.model.path, hp_ngl);
            uint32_t hp_n_ctx_train = 0;
            uint32_t hp_n_expert = 0;
            std::vector<ggml_backend_dev_t> devs;
            devs.reserve(analysis.devices.size());
            for (const auto & d : analysis.devices) {
                devs.push_back(d.dev);
            }
            ggml_log_level log_level = params.verbosity >= LOG_LEVEL_DEBUG ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR;
            auto dmd = common_get_device_memory_data(
                    params.model.path.c_str(), mparams, cparams,
                    devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);
            if (plan_out) {
                plan_out->projected_free.resize(dmd.size());
                for (size_t i = 0; i < dmd.size(); i++) {
                    plan_out->projected_free[i] = dmd[i].free - dmd[i].model;
                }
            }
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
        // the context_reserve (2 GiB per device) is a hard floor for the margin:
        // fit-target can only increase the margin, never drop it below context_reserve.
        // the delta_free_ema (from cache) corrects systematic bias: actual_free tends
        // to be lower than projected_free when model+KV+compute are counted separately.
        // a negative EMA means we underestimated the footprint -> increase margin.
        const int64_t context_reserve = 2LL * 1024 * 1024 * 1024;
        std::vector<int64_t> margins;
        margins.reserve(analysis.devices.size());
        for (size_t id = 0; id < analysis.devices.size(); id++) {
            int64_t target = (id < params.fit_params_target.size()
                ? (int64_t) params.fit_params_target[id] : 0);
            int64_t margin = 0;
            if (params.fit_params_no_context_reserve && target > 0) {
                margin = target;
            } else {
                margin = std::max(target, context_reserve);
            }
            margins.push_back(margin);
            LOG_INF("%s: device[%zu]=%s fit-target=%lld MiB base-margin=%lld MiB\n",
                    __func__, id, ggml_backend_dev_name(analysis.devices[id].dev),
                    (long long) target / (1024 * 1024), (long long) margin / (1024 * 1024));
        }

        common_fit_plan final_plan;
        fit_measure_result win_measure;
        bool have_plan = false;
        // ceiling_free from an existing plan: clamps backend-reported free to the measured
        // ceiling so fluctuations above the ceiling don't cause fits to overestimate.
        std::vector<int64_t> ceiling_free;
        if (plan && !plan->ceiling_free.empty()) {
            ceiling_free = plan->ceiling_free;
            LOG_DBG("%s: using ceiling_free from provided plan (%zu devices)\n",
                    __func__, ceiling_free.size());
        }
        // Old cache entries may contain a warm ceiling from a previous build.
        std::vector<int64_t> ceiling_free_warm;
        if (plan && !plan->ceiling_free_warm.empty()) {
            ceiling_free_warm = plan->ceiling_free_warm;
            LOG_DBG("%s: using ceiling_free_warm from provided plan (%zu devices)\n",
                    __func__, ceiling_free_warm.size());
        }
        
        if (plan == nullptr) {
            // fit cache: a hit substitutes the cached placement for the probe
            // suite. it is validated with a single probe; on any failure the
            // solver runs as usual (soft miss).
            common_fit_plan cache_plan;
            bool cache_ok = false;
            bool resume_search = false;
            const bool cache_loaded = common_fit_cache_load(params, params.model.path, analysis, cache_plan);
            if (cache_loaded) {
                LOG_INF("%s: fit cache hit for %s, validating cached placement\n",
                        __func__, params.model.path.c_str());
                // apply learned delta EMA to margins before validation and solving
                if (!cache_plan.delta_free_ema.empty()) {
                    for (size_t id = 0; id < margins.size() && id < cache_plan.delta_free_ema.size(); id++) {
                        int64_t delta = cache_plan.delta_free_ema[id];
                        // delta is actual_free - proj_free: negative means fit overestimates
                        // free space, so margin should increase to compensate
                        int64_t new_margin = margins[id] - delta;
                        if (params.fit_params_no_context_reserve) {
                            margins[id] = std::max((int64_t)0, new_margin);
                        } else {
                            margins[id] = std::max(context_reserve, new_margin);
                        }
                        LOG_DBG("%s: device %zu: margin %lld - delta %lld (ema) = %lld\n",
                                __func__, id, (long long) margins[id] + delta, (long long) delta, (long long) margins[id]);
                    }
                }
                // extract ceiling from cache if available
                if (!cache_plan.ceiling_free.empty()) {
                    ceiling_free = cache_plan.ceiling_free;
                    LOG_DBG("%s: using ceiling_free from cache (%zu devices)\n",
                            __func__, ceiling_free.size());
                }
                // extract warm ceiling from cache if available
                if (!cache_plan.ceiling_free_warm.empty()) {
                    ceiling_free_warm = cache_plan.ceiling_free_warm;
                    LOG_DBG("%s: using ceiling_free_warm from cache (%zu devices)\n",
                            __func__, ceiling_free_warm.size());
                }
                // Older cache records may have only model_size_error. The current
                // producer stores the same observed free-space delta in both fields,
                // so applying both would double-count one measurement.
                if (cache_plan.delta_free_ema.empty() && !cache_plan.model_size_error.empty()) {
                    for (size_t id = 0; id < margins.size() && id < cache_plan.model_size_error.size(); id++) {
                        int64_t err = cache_plan.model_size_error[id];
                        // err > 0: fit overestimates free (actual_model > projected_model), need more margin
                        // err < 0: fit underestimates free, can reduce margin
                        // same sign convention as delta: margin -= err
                        int64_t prev = margins[id];
                        margins[id] -= err;
                        LOG_INF("%s: device %zu: margin %lld - model_size_error %lld = %lld (fallback correction from cache)\n",
                                __func__, id, (long long) prev, (long long) err, (long long) margins[id]);
                    }
                }
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
                        fit_measure_plan(probe, cache_plan, mparams, cparams, params.model.path, win_measure, true) &&
                        fit_same_devices(win_measure.devs, analysis)) {
                    // validate the cached placement against the margins that
                    // matched its own shed state: reservations the cached plan
                    // released are released here too
                    const std::vector<int64_t> cache_margins =
                        fit_effective_margins(margins, params.fit_params_reserves, cache_plan.shed_reserves);
                    if (fit_fits_margins(win_measure.memory, cache_margins, analysis, nullptr, nullptr, &ceiling_free, &ceiling_free_warm)) {
                        final_plan = cache_plan;
                        have_plan  = true;
                        cache_ok   = true;
                        if (!ceiling_free.empty()) {
                            const char * d0 = analysis.devices.size() > 0 ? ggml_backend_dev_name(analysis.devices[0].dev) : "dev0";
                            const char * d1 = analysis.devices.size() > 1 ? ggml_backend_dev_name(analysis.devices[1].dev) : "dev1";
                            LOG_INF("%s: cache hit loaded ceiling_free: %s=%.1f MiB %s=%.1f MiB\n",
                                    __func__, d0, ceiling_free[0] / (1024.0 * 1024.0),
                                    d1, ceiling_free.size() > 1 ? ceiling_free[1] / (1024.0 * 1024.0) : 0.0);
                        }
                        // cache hit but no ceiling data yet: probe now so the next
                        // load gets it without spending time on a redundant measure pass
                        if (params.fit_params_probe_ceiling && ceiling_free.empty()) {
                            ceiling_free = fit_probe_ceiling(analysis, params);
                            LOG_INF("%s: cache hit had no ceiling, probing now: %zu devices\n",
                                    __func__, ceiling_free.size());
                        }
                        if (!cache_plan.search.complete && params.fit_strategy == COMMON_FIT_STRATEGY_RPC_AWARE) {
                            resume_search = true;
                            cache_ok = false;
                            LOG_INF("%s: cached incumbent validated; continuing incomplete rpc-aware search\n", __func__);
                        }
                    }
                }
                if (!cache_ok) {
                    LOG_WRN("%s: cached placement failed validation, refitting\n", __func__);
                }
            } else {
                LOG_INF("%s: fit cache miss for %s; solving candidate placements\n",
                        __func__, params.model.path.c_str());
            }
            if (!cache_ok) {
                size_t maxctx_probes = 0;
                // run the ceiling probe if requested and no ceiling is available from cache or plan
                if (params.fit_params_probe_ceiling && ceiling_free.empty()) {
                    ceiling_free = fit_probe_ceiling(analysis, params);
                    LOG_DBG("%s: fit-probe-ceiling recorded: %zu devices\n", __func__, ceiling_free.size());
                }
                // carry the learned delta through to final_plan so it persists in the cache
                final_plan.delta_free_ema = cache_plan.delta_free_ema;
                final_plan.model_size_error = cache_plan.model_size_error;
                // carry the ceiling through as well so a subsequent refit doesn't clobber it
                if (!cache_plan.ceiling_free.empty()) {
                    final_plan.ceiling_free = cache_plan.ceiling_free;
                }
                if (params.fit_strategy == COMMON_FIT_STRATEGY_RPC_AWARE) {
                    const int32_t ngl_hint = cache_plan.n_gpu_layers >= 0 ? cache_plan.n_gpu_layers : -1;
                    if (ngl_hint >= 0) {
                        LOG_INF("%s: cache soft-miss, using n_gpu_layers=%d as ngl_hint to anchor binary search\n",
                                __func__, ngl_hint);
                    }
                    have_plan = fit_solve_rpc_aware(
                            analysis, margins, params.fit_params_reserves, params, mparams, cparams, probe,
                            params.model.path, analysis.loader_ngl, final_plan, win_measure, ngl_hint,
                            ceiling_free, ceiling_free_warm, resume_search ? &cache_plan : nullptr,
                            search_session);
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
                        return common_fit_params(
                                params,
                                params.model.path.c_str(), mparams, cparams,
                                params.tensor_split, params.tensor_buft_overrides.data(), &params.moe_cache,
                                fallback_target.data(),
                                params.fit_strategy_min_ctx > 0 ? params.fit_strategy_min_ctx : params.fit_params_min_ctx,
                                GGML_LOG_LEVEL_INFO, !params.fit_cache_defer_save);
                    }
                } else {
                    final_plan = fit_solve_maxctx(
                            analysis, margins, params.fit_params_reserves, params, mparams, cparams, probe,
                            params.model.path, analysis.loader_ngl, win_measure, ceiling_free,
                            ceiling_free_warm, &maxctx_probes, params.fit_max_attempts);
                    have_plan = true;
                }
            }
        } else {
            // manual plan: measure it once, a conflict is a hard error
            final_plan = *plan;
            const std::vector<int64_t> & cached_correction = !final_plan.delta_free_ema.empty()
                ? final_plan.delta_free_ema : final_plan.model_size_error;
            for (size_t id = 0; id < margins.size() && id < cached_correction.size(); id++) {
                const int64_t adjusted = margins[id] - cached_correction[id];
                margins[id] = params.fit_params_no_context_reserve
                    ? std::max<int64_t>(0, adjusted)
                    : std::max<int64_t>(context_reserve, adjusted);
            }
            for (int id : final_plan.role_devices) {
                if (id >= (int) analysis.devices.size()) {
                    throw std::runtime_error("manual plan references unknown device id " + std::to_string(id));
                }
            }
            if (!fit_measure_plan(probe, final_plan, mparams, cparams, params.model.path, win_measure, true)) {
                throw std::runtime_error("manual plan cannot create a context");
            }
            if (!fit_same_devices(win_measure.devs, analysis)) {
                throw std::runtime_error("manual plan changed the device set");
            }
            if (!fit_fits_margins(win_measure.memory, margins, analysis, nullptr, nullptr, &ceiling_free, &ceiling_free_warm)) {
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
        fit_validate(final_plan, analysis, final_margins, analysis.loader_ngl, mparams, cparams, probe, params.model.path,
                    ceiling_free_warm.empty() ? nullptr : &ceiling_free_warm);

        // publish the shed state so callers can move companions off VRAM
        params.fit_params_shed_tags = final_plan.shed_reserves;

        if (search_session) {
            final_plan.search.policy_version = COMMON_FIT_SEARCH_POLICY_VERSION;
            search_session->incumbent = final_plan;
            search_session->has_incumbent = true;
            search_session->fit_key = common_fit_cache_key_hash(params, params.model.path, analysis);
        }

        // record the ceiling so subsequent runs clamp free to the measured ceiling
        if (!ceiling_free.empty()) {
            final_plan.ceiling_free = ceiling_free;
        }
        if (!ceiling_free_warm.empty()) {
            final_plan.ceiling_free_warm = ceiling_free_warm;
        }

        // save the successful placement to the disk cache (best-effort); on a
        // cache hit this refreshes the freshness timestamp
        if (params.fit_cache_defer_save) {
            LOG_INF("%s: deferring intermediate fit cache save for %s\n",
                    __func__, params.model.path.c_str());
        } else {
            const bool cache_saved = common_fit_cache_save(params, params.model.path, analysis, final_plan);
            LOG_INF("%s: selected fit cache save %s for %s\n", __func__,
                    cache_saved ? "succeeded" : "failed", params.model.path.c_str());
        }

        LOG_TRC("%s: successfully fit params to free device memory\n", __func__);
        if (plan_out) {
            *plan_out = final_plan;
        }
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
        const common_fit_plan * plan,
        common_fit_plan * plan_out,
        common_fit_search_session * search_session) {
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
    probe.measure_no_alloc = [](const char * path_model,
                                const llama_model_params * mparams,
                                const llama_context_params * cparams,
                                std::vector<ggml_backend_dev_t> & devs,
                                uint32_t & hp_ngl,
                                uint32_t & hp_n_ctx_train,
                                uint32_t & hp_n_expert,
                                ggml_log_level log_level) {
        return common_get_device_memory_data_no_alloc(
            path_model, mparams, cparams, devs, hp_ngl, hp_n_ctx_train, hp_n_expert, log_level);
    };
    return common_fit_strategy_apply_impl(params, mparams, cparams, plan, probe, plan_out, search_session);
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
        j["layer"] = c.layer;
        j["device_id"] = c.device_id;
        j["is_rpc"] = c.is_rpc;
        j["activation_bytes"] = c.activation_bytes;
        j["n_calls"] = c.n_calls;
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

// Active OOM-threshold allocations can consume nearly all free VRAM and stall
// display workloads. Use the current free snapshot and explicit margins instead.
static std::vector<int64_t> fit_probe_ceiling(
        const common_fit_analysis & analysis,
        const common_params & /*params*/) {
    (void) analysis;
    LOG_WRN("%s: skipping active OOM allocation probes; fit uses current free memory and per-device margins\n",
            __func__);
    return {};
}

// Active warm OOM allocation probes are disabled for the same reason as the cold probe.
std::vector<int64_t> common_fit_probe_ceiling_warm(llama_model * model, const common_params & /*params*/) {
    (void) model;
    LOG_WRN("%s: skipping active warm OOM allocation probes\n", __func__);
    return {};
}

// Validate a user-provided list of RPC device IDs against the local id space.
// Contract per the Wave 2 plan: empty list is OK; per-id checks run in
// invalid-range -> non-local -> duplicate order; first failure wins.
// Duplicates downgrade OK to DUPLICATE but the function still returns;
// NON_LOCAL and INVALID are hard errors the caller must abort on.
fit_rpc_device_id_error fit_validate_rpc_device_ids(
    const std::vector<int> & device_ids,
    int local_device_count) {
    fit_rpc_device_id_error result = FIT_RPC_DEVICE_ID_OK;
    std::set<int> seen;
    for (size_t i = 0; i < device_ids.size(); i++) {
        const int id = device_ids[i];
        if (id < 0 || id == INT_MAX) {
            LOG_ERR("%s: device_ids[%zu]=%d is out of the [0, INT_MAX) range\n",
                    __func__, i, id);
            return FIT_RPC_DEVICE_ID_INVALID;
        }
        if (id >= local_device_count) {
            LOG_ERR("%s: device_ids[%zu]=%d is not local (local_device_count=%d)\n",
                    __func__, i, id, local_device_count);
            return FIT_RPC_DEVICE_ID_NON_LOCAL;
        }
        if (!seen.insert(id).second) {
            LOG_WRN("%s: device_ids[%zu]=%d is duplicated; proceeding with the unique set\n",
                    __func__, i, id);
            if (result == FIT_RPC_DEVICE_ID_OK) {
                result = FIT_RPC_DEVICE_ID_DUPLICATE;
            }
        }
    }
    return result;
}
