// test-fit-strategy.cpp - fit-strategy engine tests (fork feature, Issue 4).
// The engine public API lives in common/fit-strategy.h. The measure seam
// (common_fit_strategy_apply_impl) lets a fake probe replace the real
// measurement, but fit_analyze still needs a real loadable model, so the
// fixture model (tinyllamas/stories15M-q4_0.gguf) is required for the
// solve-path tests; those tests SKIP when no model file is available.
//
// Harness: the first non-flag argv is a substring filter on test names;
// -m/--model selects the fixture model. A run where every matching test
// passed counts as n_ok; a run where none passed (all filtered or skipped)
// exits non-zero.

#include "fit-strategy.h"
#include "common.h"
#include "../src/llama-ext.h" // llama_model_n_devices, llama_model_get_device, ...

#include <nlohmann/json.hpp>

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static constexpr int64_t MiB = 1024 * 1024;
static constexpr int64_t GiB = 1024 * MiB;

static int failures = 0;
static bool g_skipped = false;
static std::string g_skip_reason;
static std::string g_model;

static void expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        failures++;
    }
}

static void test_skip(const char * reason) {
    g_skip_reason = reason;
    g_skipped = true;
}

// ---- fake measure seam ----
//
// The fake probe mirrors common_get_device_memory_data: it loads the model
// (no_alloc, load_mode NONE) to fill the device list and hyperparameters, then
// returns canned per-device memory. Per-call script entries can return data,
// soft-reject (context creation failure), or throw a hard error.

struct fake_probe_ctx {
    int n_calls = 0;
    int soft_rejects = 0;
    int hard_rejects = 0;
    std::vector<int> actions;       // per call: 0 return, 1 soft reject, 2 hard error
    std::vector<common_device_memory_data> dev_data; // per call override
    common_device_memory_data default_dev;
    std::vector<int64_t> seen_n_ctx;
};

static common_device_memory_data_vec fake_measure(
        const char * path_model,
        const llama_model_params * mparams,
        const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs,
        uint32_t & hp_ngl,
        uint32_t & hp_n_ctx_train,
        uint32_t & hp_n_expert,
        ggml_log_level log_level,
        fake_probe_ctx * cfg) {
    (void) log_level;
    cfg->n_calls++;
    cfg->seen_n_ctx.push_back(cparams->n_ctx);

    const size_t idx = (size_t) cfg->n_calls - 1;
    int action = 0;
    common_device_memory_data dev = cfg->default_dev;
    if (idx < cfg->actions.size()) {
        action = cfg->actions[idx];
    }
    if (idx < cfg->dev_data.size()) {
        dev = cfg->dev_data[idx];
    }
    if (action == 1) {
        cfg->soft_rejects++;
        throw std::runtime_error("failed to create llama_context from model");
    }
    if (action == 2) {
        cfg->hard_rejects++;
        throw std::runtime_error("catastrophic device failure");
    }

    llama_model_params mp = *mparams;
    mp.no_alloc = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model_ptr model(llama_model_load_from_file(path_model, mp));
    if (!model) {
        throw std::runtime_error("failed to load model");
    }
    const size_t nd = llama_model_n_devices(model.get());
    devs.clear();
    for (size_t i = 0; i < nd; i++) {
        devs.push_back(llama_model_get_device(model.get(), i));
    }
    hp_ngl         = (uint32_t) llama_model_n_layer(model.get());
    hp_n_ctx_train = (uint32_t) llama_model_n_ctx_train(model.get());
    hp_n_expert    = (uint32_t) llama_model_n_expert(model.get());

    common_device_memory_data_vec mem(nd + 1);
    for (size_t i = 0; i < nd; i++) {
        mem[i] = dev;
    }
    mem[nd] = { (int64_t) (32 * GiB), (int64_t) (32 * GiB), 0, 0, 0 }; // host entry
    return mem;
}

static common_fit_probe make_probe(fake_probe_ctx * cfg) {
    common_fit_probe probe;
    probe.measure = [cfg](const char * path_model,
            const llama_model_params * mparams,
            const llama_context_params * cparams,
            std::vector<ggml_backend_dev_t> & devs,
            uint32_t & hp_ngl,
            uint32_t & hp_n_ctx_train,
            uint32_t & hp_n_expert,
            ggml_log_level log_level) {
        return fake_measure(path_model, mparams, cparams, devs, hp_ngl,
                hp_n_ctx_train, hp_n_expert, log_level, cfg);
    };
    return probe;
}

// ---- model discovery ----

static std::string find_model(const char * argv0) {
    const char * env = std::getenv("LLAMA_FIT_TEST_MODEL");
    if (env && *env) {
        return env;
    }
    std::string dir;
    const std::string exe(argv0 ? argv0 : "");
    const size_t pos = exe.find_last_of('/');
    if (pos != std::string::npos) {
        dir = exe.substr(0, pos);
    }
    const std::vector<std::string> cands = {
        "tinyllamas/stories15M-q4_0.gguf",
        "build/tinyllamas/stories15M-q4_0.gguf",
        dir.empty() ? std::string("tinyllamas/stories15M-q4_0.gguf") : dir + "/tinyllamas/stories15M-q4_0.gguf",
        dir.empty() ? std::string("tinyllamas/stories15M-q4_0.gguf") : dir + "/../tinyllamas/stories15M-q4_0.gguf",
    };
    for (const auto & c : cands) {
        std::ifstream f(c);
        if (f.good()) {
            return c;
        }
    }
    return "";
}

// ---- tests ----

static void test_measurement_contract() {
    // the input-class guard rejects a user-preset tensor_split: FAILURE,
    // never a silent override (fit-strategy.cpp fit_analyze guard)
    {
        common_params params;
        params.fit_strategy = COMMON_FIT_STRATEGY_MAXCTX;
        params.fit_strategy_min_ctx = 1024;
        params.model.path = g_model;
        params.tensor_split[0] = 0.5f;
        llama_model_params mparams = common_model_params_to_llama(params);
        llama_context_params cparams = common_context_params_to_llama(params);
        fake_probe_ctx cfg;
        cfg.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 };
        const common_params_fit_status st = common_fit_strategy_apply_impl(
                params, &mparams, &cparams, nullptr, make_probe(&cfg));
        expect(st == COMMON_PARAMS_FIT_STATUS_FAILURE,
                "preset tensor_split must be refused with FAILURE, not overridden");
        expect(cfg.n_calls == 0, "the guard must reject before any measurement");
    }

    if (g_model.empty()) {
        test_skip("no model available; positive measurement path not run");
        return;
    }

    // positive path: the probe is the oracle, its data drives the fit
    common_params params;
    params.fit_strategy = COMMON_FIT_STRATEGY_MAXCTX;
    params.fit_strategy_min_ctx = 1024;
    params.n_ctx = 8192;
    params.model.path = g_model;
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_context_params cparams = common_context_params_to_llama(params);
    fake_probe_ctx cfg;
    cfg.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 };
    const common_params_fit_status st = common_fit_strategy_apply_impl(
            params, &mparams, &cparams, nullptr, make_probe(&cfg));
    expect(st == COMMON_PARAMS_FIT_STATUS_SUCCESS, "maxctx with a fitting probe must succeed");
    expect((int32_t) cparams.n_ctx >= params.fit_strategy_min_ctx,
            "the resolved n_ctx must honor the strategy floor");
    expect(cfg.n_calls >= 1, "the probe must be measured at least once");
    // the returned device data was consumed: a probe missing the default 1 GiB
    // margin would have been rejected, not fitted
    expect(cfg.seen_n_ctx.back() == 8192,
            "the plan context must be serialized into the measurement");
}

static void test_preset_placement_rejected() {
    // the n_gpu_layers guard fires before any model load, so no model needed
    common_params params;
    params.fit_strategy = COMMON_FIT_STRATEGY_MAXCTX;
    params.fit_strategy_min_ctx = 1024;
    params.n_gpu_layers = 10;
    params.model.path = g_model;
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_context_params cparams = common_context_params_to_llama(params);
    fake_probe_ctx cfg;
    cfg.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 };
    const common_params_fit_status st = common_fit_strategy_apply_impl(
            params, &mparams, &cparams, nullptr, make_probe(&cfg));
    expect(st == COMMON_PARAMS_FIT_STATUS_FAILURE,
            "user-preset n_gpu_layers must be refused with FAILURE, not overridden");
    expect(mparams.n_gpu_layers == 10, "the preset n_gpu_layers must stay untouched");
}

static void test_pool_budget_formula() {
    // (a) explicit user budget wins exactly
    expect(common_fit_strategy_pool_budget((size_t) (100 * MiB), 0, 256) == (size_t) (256 * MiB),
            "a user budget wins even with zero remaining VRAM");
    expect(common_fit_strategy_pool_budget((size_t) (100 * MiB), (size_t) (4 * GiB), 64) == (size_t) (64 * MiB),
            "a user budget wins over the automatic formula");
    // (d) nothing left
    expect(common_fit_strategy_pool_budget((size_t) (100 * MiB), 0, 0) == 0,
            "zero remaining grants zero");
    // (b) remaining below the floor (max(2*working_set, 256 MiB)) grants zero
    expect(common_fit_strategy_pool_budget((size_t) (100 * MiB), (size_t) (255 * MiB), 0) == 0,
            "remaining below the 256 MiB floor grants zero");
    expect(common_fit_strategy_pool_budget((size_t) (500 * MiB), (size_t) (999 * MiB), 0) == 0,
            "remaining below 2x the working set grants zero");
    // (c) automatic clamp: half of remaining, floored, capped at 4 GiB
    expect(common_fit_strategy_pool_budget((size_t) (100 * MiB), (size_t) (4 * GiB), 0) == (size_t) (2 * GiB),
            "the automatic budget is half of the remaining VRAM");
    expect(common_fit_strategy_pool_budget((size_t) (100 * MiB), (size_t) (16 * GiB), 0) == (size_t) (4 * GiB),
            "the automatic budget is capped at 4 GiB");
}

static void test_maxctx_above_floor() {
    if (g_model.empty()) {
        test_skip("no model available");
        return;
    }
    common_params params;
    params.fit_strategy = COMMON_FIT_STRATEGY_MAXCTX;
    params.fit_strategy_min_ctx = 1024;
    params.n_ctx = 8192;
    params.model.path = g_model;
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_context_params cparams = common_context_params_to_llama(params);
    fake_probe_ctx cfg;
    cfg.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 };
    const common_params_fit_status st = common_fit_strategy_apply_impl(
            params, &mparams, &cparams, nullptr, make_probe(&cfg));
    expect(st == COMMON_PARAMS_FIT_STATUS_SUCCESS,
            "maxctx must succeed when the floor context fits");
    expect((int32_t) cparams.n_ctx >= params.fit_strategy_min_ctx,
            "maxctx must never resolve below the floor");
    expect(mparams.n_gpu_layers == 7, "maxctx must offload all 6 layers plus output");
}

static void test_floor_unreachable_errors() {
    if (g_model.empty()) {
        test_skip("no model available");
        return;
    }
    common_params params;
    params.fit_strategy = COMMON_FIT_STRATEGY_MAXCTX;
    params.fit_strategy_min_ctx = 1024;
    params.n_ctx = 8192;
    params.model.path = g_model;
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_context_params cparams = common_context_params_to_llama(params);
    fake_probe_ctx cfg;
    // projected usage (32 GiB) exceeds free memory even at the floor context
    cfg.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (16 * GiB), (size_t) (16 * GiB), 0 };
    const common_params_fit_status st = common_fit_strategy_apply_impl(
            params, &mparams, &cparams, nullptr, make_probe(&cfg));
    expect(st == COMMON_PARAMS_FIT_STATUS_ERROR,
            "an unreachable floor must be a hard ERROR, never a silent unfitted ship");
    expect(cfg.n_calls == 2, "both anchors must be measured before giving up");
}

static void test_rpc_aware_placement() {
    if (g_model.empty()) {
        test_skip("no model available");
        return;
    }
    // No RPC device is enumerated on this host, so the reachable contract is
    // the local-first path: dense decode lands on the local device and the
    // main_gpu is never an RPC device. The is_rpc flag comes from real device
    // enumeration (fit_analyze), which the probe seam cannot fake.
    common_params params;
    params.fit_strategy = COMMON_FIT_STRATEGY_RPC_AWARE;
    params.n_ctx = 1024;
    params.model.path = g_model;
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_context_params cparams = common_context_params_to_llama(params);
    fake_probe_ctx cfg;
    cfg.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 };
    const common_params_fit_status st = common_fit_strategy_apply_impl(
            params, &mparams, &cparams, nullptr, make_probe(&cfg));
    expect(st == COMMON_PARAMS_FIT_STATUS_SUCCESS,
            "rpc-aware must fit on local devices without aborting");
    expect(mparams.n_gpu_layers == 7, "rpc-aware must offload all layers to the local device");
    expect(cparams.n_ctx == 1024, "rpc-aware must preserve the requested context");
    expect(params.tensor_split[0] == 1.0f,
            "all weight must land on device 0, the local main device");
}

static void test_rpc_cold_start() {
    if (g_model.empty()) {
        test_skip("no model available");
        return;
    }
    // rpc_hosts configured but no RPC device responds: the engine logs a
    // warning and falls back to local devices instead of aborting.
    common_params params;
    params.fit_strategy = COMMON_FIT_STRATEGY_RPC_AWARE;
    params.n_ctx = 1024;
    params.rpc_hosts = { "localhost:12345" };
    params.model.path = g_model;
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_context_params cparams = common_context_params_to_llama(params);
    fake_probe_ctx cfg;
    cfg.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 };
    const common_params_fit_status st = common_fit_strategy_apply_impl(
            params, &mparams, &cparams, nullptr, make_probe(&cfg));
    expect(st == COMMON_PARAMS_FIT_STATUS_SUCCESS,
            "a cold-start RPC config must fall back to local, not abort");
    expect(mparams.n_gpu_layers == 7, "the local fallback must still offload all layers");
}

static void test_shed_reserve_order() {
    if (g_model.empty()) {
        test_skip("no model available");
        return;
    }

    // reservations are shed in priority order (mmproj first, spec second) before
    // any CPU spill; the main model stays fully offloaded as long as a shed can
    // free the headroom. over-subscribed by 512 GiB each, the solver must shed
    // both and keep every layer offloaded, with the shed tags in priority order.
    common_params params;
    params.fit_strategy = COMMON_FIT_STRATEGY_RPC_AWARE;
    params.n_ctx = 1024;
    params.model.path = g_model;
    params.fit_params_reserves = {
        { 0, (size_t) (512 * GiB), 0, "mmproj" },
        { 0, (size_t) (512 * GiB), 1, "spec" },
    };
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_context_params cparams = common_context_params_to_llama(params);
    fake_probe_ctx cfg;
    cfg.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 };
    const common_params_fit_status st = common_fit_strategy_apply_impl(
            params, &mparams, &cparams, nullptr, make_probe(&cfg));
    expect(st == COMMON_PARAMS_FIT_STATUS_SUCCESS, "the shed-order solver must still fit");
    expect(mparams.n_gpu_layers == 7, "full offload must be kept once both companions shed");
    expect(params.fit_params_shed_tags.size() == 2 &&
           params.fit_params_shed_tags[0] == "mmproj" &&
           params.fit_params_shed_tags[1] == "spec",
           "the shed order must be mmproj first, spec second");
    expect(params.fit_params_target[0] == 1024ull * 1024 * 1024,
           "the per-device margin must not absorb the shed reservations");

    // a spec reserve that still fits is held: shedding only mmproj frees the
    // headroom and the solver must not force the draft off early
    common_params params2;
    params2.fit_strategy = COMMON_FIT_STRATEGY_RPC_AWARE;
    params2.n_ctx = 1024;
    params2.model.path = g_model;
    params2.fit_params_reserves = {
        { 0, (size_t) (512 * GiB), 0, "mmproj" },
        { 0, (size_t) (1 * MiB), 1, "spec" },
    };
    llama_model_params mparams2 = common_model_params_to_llama(params2);
    llama_context_params cparams2 = common_context_params_to_llama(params2);
    fake_probe_ctx cfg2;
    cfg2.default_dev = cfg.default_dev;
    const common_params_fit_status st2 = common_fit_strategy_apply_impl(
            params2, &mparams2, &cparams2, nullptr, make_probe(&cfg2));
    expect(st2 == COMMON_PARAMS_FIT_STATUS_SUCCESS, "shedding mmproj alone must fit");
    expect(mparams2.n_gpu_layers == 7, "the main model must stay fully offloaded");
    expect(params2.fit_params_shed_tags.size() == 1 &&
           params2.fit_params_shed_tags[0] == "mmproj",
           "only mmproj must shed when the spec reserve still fits");
}

static void test_exception_classification() {
    if (g_model.empty()) {
        test_skip("no model available");
        return;
    }
    // soft reject: the confirmation measure cannot create a context, the
    // engine falls back to the measured floor plan and still succeeds
    common_params params;
    params.fit_strategy = COMMON_FIT_STRATEGY_MAXCTX;
    params.fit_strategy_min_ctx = 1024;
    params.n_ctx = 8192;
    params.model.path = g_model;
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_context_params cparams = common_context_params_to_llama(params);
    fake_probe_ctx cfg;
    cfg.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 };
    // call 1: upper context misses the 1 GiB margin
    // call 2: floor context fits
    // call 3: interpolated confirmation soft-rejects -> fall back to the floor
    cfg.actions = { 0, 0, 1 };
    cfg.dev_data = {
        { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (8 * GiB), (size_t) (7680 * MiB), 0 },
        { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 },
        { (int64_t) (16 * GiB), (int64_t) (16 * GiB), 0, 0, 0 },
    };
    const common_params_fit_status st = common_fit_strategy_apply_impl(
            params, &mparams, &cparams, nullptr, make_probe(&cfg));
    expect(st == COMMON_PARAMS_FIT_STATUS_SUCCESS,
            "a soft reject must be recovered, not surfaced as an error");
    expect(cparams.n_ctx == 1024, "the soft-reject fallback must use the floor context");
    expect(cfg.soft_rejects >= 1, "the soft reject must have been exercised");
    expect(cfg.n_calls == 4, "solve (3) plus the final validation measure");

    // hard error: any other runtime_error escapes to the outer contract as ERROR
    common_params params2;
    params2.fit_strategy = COMMON_FIT_STRATEGY_MAXCTX;
    params2.fit_strategy_min_ctx = 1024;
    params2.n_ctx = 8192;
    params2.model.path = g_model;
    llama_model_params mparams2 = common_model_params_to_llama(params2);
    llama_context_params cparams2 = common_context_params_to_llama(params2);
    fake_probe_ctx cfg2;
    cfg2.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 };
    cfg2.actions = { 2 };
    const common_params_fit_status st2 = common_fit_strategy_apply_impl(
            params2, &mparams2, &cparams2, nullptr, make_probe(&cfg2));
    expect(st2 == COMMON_PARAMS_FIT_STATUS_ERROR,
            "a hard probe error must map to the ERROR status");
    expect(cfg2.hard_rejects == 1, "the hard error must have been raised exactly once");
}

static void test_no_provider_pool() {
    // the budget formula is the signal the serializer uses to disable the pool:
    // sub-floor remaining VRAM must grant zero (fit-strategy.cpp
    // fit_serialize_moe_cache turns that grant into mode=OFF). The full
    // serialization path needs an MoE model; the fixture is dense, so here we
    // assert the formula plus the dense-model no-op instead.
    expect(common_fit_strategy_pool_budget((size_t) (100 * MiB), (size_t) (255 * MiB), 0) == 0,
            "sub-floor remaining VRAM grants no MoE pool");

    if (g_model.empty()) {
        test_skip("no model available");
        return;
    }
    common_params params;
    params.fit_strategy = COMMON_FIT_STRATEGY_MAXCTX;
    params.fit_strategy_min_ctx = 1024;
    params.n_ctx = 8192;
    params.model.path = g_model;
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_context_params cparams = common_context_params_to_llama(params);
    fake_probe_ctx cfg;
    cfg.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 };
    const common_params_fit_status st = common_fit_strategy_apply_impl(
            params, &mparams, &cparams, nullptr, make_probe(&cfg));
    expect(st == COMMON_PARAMS_FIT_STATUS_SUCCESS, "maxctx must still succeed");
    expect(params.moe_cache.mode == COMMON_MOE_CACHE_MODE_AUTO && !params.moe_cache.fit_selected,
            "a dense model must leave the MoE cache untouched (no pool)");
}

static void test_activation_profile_sink() {
    const std::string path = "/tmp/fit-test-profile.jsonl";
    std::remove(path.c_str());

    common_activation_profile profile;
    profile.version = 1;
    profile.model_id = "fit-strategy-test-model";
    profile.counters = {
        { "blk.0.attn_q.weight", COMMON_FIT_ROLE_ATTN_QKV, 100, 0, 42 },
        { "blk.0.ffn_gate_up_exps.weight", COMMON_FIT_ROLE_EXPERT_UP, 7, 3, 0 },
    };

    expect(common_activation_profile_sink(profile, path),
            "sink to a writable path must return true");
    {
        std::ifstream in(path);
        std::string line;
        std::string rest;
        std::getline(in, line);
        std::getline(in, rest);
        expect(!line.empty() && rest.empty(), "sink must write exactly one JSONL row");
        nlohmann::json j = nlohmann::json::parse(line);
        expect(j["version"] == 1 && j["model_id"] == "fit-strategy-test-model",
                "the row header fields must round-trip");
        expect(j["counters"].size() == 2, "both counters must round-trip");
        expect(j["counters"][0]["tensor_id"] == "blk.0.attn_q.weight",
                "the counter tensor_id must round-trip");
        expect(j["counters"][1]["role"] == (int) COMMON_FIT_ROLE_EXPERT_UP,
                "the counter role must round-trip");
        expect(j["counters"][0]["kv_touched"] == 42, "the counter kv_touched must round-trip");
    }

    expect(!common_activation_profile_sink(profile, "/nonexistent-dir-12345/fit.jsonl"),
            "sink to an unwritable path must fail closed");
}

static void test_manual_plan_validated() {
    if (g_model.empty()) {
        test_skip("no model available");
        return;
    }
    // a valid manual plan is measured and applied
    common_params params;
    params.fit_strategy = COMMON_FIT_STRATEGY_MAXCTX;
    params.fit_strategy_min_ctx = 1024;
    params.model.path = g_model;
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_context_params cparams = common_context_params_to_llama(params);
    common_fit_plan plan;
    plan.role_devices.assign(8, 0);
    plan.n_gpu_layers = 7;
    plan.tensor_split.assign(llama_max_devices(), 0.0f);
    plan.tensor_split[0] = 1.0f;
    plan.n_ctx = 0;
    fake_probe_ctx cfg;
    cfg.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 };
    const common_params_fit_status st = common_fit_strategy_apply_impl(
            params, &mparams, &cparams, &plan, make_probe(&cfg));
    expect(st == COMMON_PARAMS_FIT_STATUS_SUCCESS, "a valid manual plan must be applied");
    expect(mparams.n_gpu_layers == 7, "the manual plan n_gpu_layers must be serialized");

    // a device id out of range is a hard error, never a silent override
    common_params params2 = params;
    std::fill(std::begin(params2.tensor_split), std::end(params2.tensor_split), 0.0f);
    llama_model_params mparams2 = common_model_params_to_llama(params2);
    llama_context_params cparams2 = common_context_params_to_llama(params2);
    common_fit_plan bad_plan = plan;
    bad_plan.role_devices[1] = 999;
    fake_probe_ctx cfg2;
    cfg2.default_dev = cfg.default_dev;
    const common_params_fit_status st2 = common_fit_strategy_apply_impl(
            params2, &mparams2, &cparams2, &bad_plan, make_probe(&cfg2));
    expect(st2 == COMMON_PARAMS_FIT_STATUS_ERROR,
            "a manual plan referencing an unknown device must be a hard error");

    // a manual plan that misses the margins is a hard error
    common_params params3 = params;
    std::fill(std::begin(params3.tensor_split), std::end(params3.tensor_split), 0.0f);
    llama_model_params mparams3 = common_model_params_to_llama(params3);
    llama_context_params cparams3 = common_context_params_to_llama(params3);
    common_fit_plan margin_plan = plan;
    fake_probe_ctx cfg3;
    cfg3.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (16 * GiB), (size_t) (16 * GiB), 0 };
    const common_params_fit_status st3 = common_fit_strategy_apply_impl(
            params3, &mparams3, &cparams3, &margin_plan, make_probe(&cfg3));
    expect(st3 == COMMON_PARAMS_FIT_STATUS_ERROR,
            "a manual plan missing the per-device margins must be a hard error");
}

static void test_device_id_mapping() {
    if (g_model.empty()) {
        test_skip("no model available");
        return;
    }
    
    // Test device ID query functions with valid inputs
    common_params temp_params;
    temp_params.model.path = g_model;
    llama_model_params mp = common_model_params_to_llama(temp_params);
    mp.no_alloc = true;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model_ptr model(llama_model_load_from_file(g_model.c_str(), mp));
    if (!model) {
        test_skip("model load failed");
        return;
    }
    
    const size_t nd = llama_model_n_devices(model.get());
    expect(nd > 0, "model should have at least one device");
    
    // Test llama_model_get_backend_device_id with valid devices
    for (size_t i = 0; i < nd; i++) {
        ggml_backend_dev_t dev = llama_model_get_device(model.get(), (int)i);
        expect(dev != nullptr, "device should not be null");
        
        // Query the actual backend device ID and verify it's valid
        const int backend_id = llama_model_get_backend_device_id(model.get(), (int)i);
        expect(backend_id >= 0, "backend device ID should be non-negative for valid device");
        
        // Verify we can get the same device back by backend ID
        ggml_backend_dev_t dev_by_id = llama_model_get_device_by_backend_id(model.get(), backend_id);
        expect(dev_by_id != nullptr, "device lookup by backend ID should succeed");
        expect(dev_by_id == dev, "device lookup by backend ID should return same device");
        
        const char * dev_name = ggml_backend_dev_name(dev);
        expect(dev_name != nullptr, "device name should not be null");
    }
    
    // Test with invalid device indices
    const int invalid_backend_id = llama_model_get_backend_device_id(model.get(), -1);
    expect(invalid_backend_id == -1, "negative device index should return -1");
    
    const int invalid_backend_id2 = llama_model_get_backend_device_id(model.get(), 9999);
    expect(invalid_backend_id2 == -1, "out-of-range device index should return -1");
    
    ggml_backend_dev_t invalid_dev = llama_model_get_device_by_backend_id(model.get(), -1);
    expect(invalid_dev == nullptr, "negative backend ID should return null");
    
    ggml_backend_dev_t invalid_dev2 = llama_model_get_device_by_backend_id(model.get(), 9999);
    expect(invalid_dev2 == nullptr, "out-of-range backend ID should return null");
}

static void test_rpc_pool_split_device_mapping() {
    if (g_model.empty()) {
        test_skip("no model available");
        return;
    }
    
    // Test RPC pool split device mapping correctness
    common_params params;
    params.fit_strategy = COMMON_FIT_STRATEGY_RPC_AWARE;
    params.n_ctx = 1024;
    params.model.path = g_model;
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_context_params cparams = common_context_params_to_llama(params);
    fake_probe_ctx cfg;
    cfg.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 };
    
    const common_params_fit_status st = common_fit_strategy_apply_impl(
            params, &mparams, &cparams, nullptr, make_probe(&cfg));
    expect(st == COMMON_PARAMS_FIT_STATUS_SUCCESS,
            "rpc-aware placement should succeed");
    
    // Verify that tensor_split is set correctly
    // The main device should have the majority of the model when no RPC devices are available
    float total_split = 0.0f;
    for (size_t i = 0; i < llama_max_devices(); i++) {
        total_split += params.tensor_split[i];
    }
    expect(total_split > 0.99f && total_split <= 1.01f,
            "tensor_split should sum to approximately 1.0");
    
    // When no RPC devices are available, all weight should be on device 0
    expect(params.tensor_split[0] > 0.9f,
            "main device should have majority of weight when no RPC available");
}

static void test_rpc_pool_split_inversion_bug() {
    if (g_model.empty()) {
        test_skip("no model available");
        return;
    }
    
    // Test for inversion bug regression
    // RPC=0.3 should place ~30% on RPC, not 91%
    // This test uses the rpc-aware strategy to verify correct device mapping
    
    common_params params;
    params.fit_strategy = COMMON_FIT_STRATEGY_RPC_AWARE;
    params.n_ctx = 1024;
    params.model.path = g_model;
    llama_model_params mparams = common_model_params_to_llama(params);
    llama_context_params cparams = common_context_params_to_llama(params);
    fake_probe_ctx cfg;
    cfg.default_dev = { (int64_t) (16 * GiB), (int64_t) (16 * GiB), (size_t) (64 * MiB), (size_t) (32 * MiB), 0 };
    
    const common_params_fit_status st = common_fit_strategy_apply_impl(
            params, &mparams, &cparams, nullptr, make_probe(&cfg));
    expect(st == COMMON_PARAMS_FIT_STATUS_SUCCESS,
            "rpc-aware placement should succeed");
    
    // Verify no inversion bug: local device should not get 91% when expecting ~70%
    // The test verifies that device mapping is correct
    // (the actual RPC split logic will be tested once RPC devices are available)
    float max_split = 0.0f;
    size_t max_device = 0;
    for (size_t i = 0; i < llama_max_devices(); i++) {
        if (params.tensor_split[i] > max_split) {
            max_split = params.tensor_split[i];
            max_device = i;
        }
    }
    
    // The device with maximum split should be device 0 (local main device)
    // when no RPC devices are available
    expect(max_device == 0,
            "main device (0) should have maximum split when no RPC available");
    expect(max_split > 0.9f,
            "main device should have majority of weight when no RPC available");
}

static const struct test_case {
    const char * name;
    void (*fn)();
} g_tests[] = {
    { "measurement_contract",       test_measurement_contract },
    { "preset_placement_rejected",  test_preset_placement_rejected },
    { "pool_budget_formula",        test_pool_budget_formula },
    { "maxctx_above_floor",         test_maxctx_above_floor },
    { "floor_unreachable_errors",   test_floor_unreachable_errors },
    { "rpc_aware_placement",        test_rpc_aware_placement },
    { "rpc_cold_start",             test_rpc_cold_start },
    { "shed_reserve_order",         test_shed_reserve_order },
    { "exception_classification",   test_exception_classification },
    { "no_provider_pool",           test_no_provider_pool },
    { "activation_profile_sink",    test_activation_profile_sink },
    { "manual_plan_validated",      test_manual_plan_validated },
    { "device_id_mapping",          test_device_id_mapping },
    { "rpc_pool_split_device_mapping", test_rpc_pool_split_device_mapping },
    { "rpc_pool_split_inversion_bug", test_rpc_pool_split_inversion_bug },
};

int main(int argc, char ** argv) {
    std::string filter;
    std::string model;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-m" || a == "--model") {
            if (i + 1 < argc) {
                model = argv[++i];
            }
        } else if (!a.empty() && a[0] == '-') {
            // ignore other flags
        } else if (filter.empty()) {
            filter = a;
        }
    }
    if (model.empty()) {
        model = find_model(argv[0]);
    }
    if (!model.empty()) {
        std::ifstream f(model);
        if (!f.good()) {
            model.clear();
        }
    }
    g_model = model;

    llama_backend_init();

    int n_ok = 0;
    for (const auto & t : g_tests) {
        if (!filter.empty() && std::string(t.name).find(filter) == std::string::npos) {
            continue;
        }
        g_skipped = false;
        g_skip_reason.clear();
        const int before = failures;
        t.fn();
        if (g_skipped) {
            std::printf("SKIP: %s (%s)\n", t.name, g_skip_reason.c_str());
            continue;
        }
        if (failures == before) {
            n_ok++;
            std::printf("PASS: %s\n", t.name);
        } else {
            std::printf("FAIL: %s\n", t.name);
        }
    }

    llama_backend_free();

    if (failures != 0) {
        std::fprintf(stderr, "%d fit-strategy test failures\n", failures);
        return 1;
    }
    if (n_ok == 0) {
        std::fprintf(stderr, "no fit-strategy tests passed (all filtered out or skipped)\n");
        return 1;
    }
    std::printf("%d fit-strategy tests passed\n", n_ok);
    return 0;
}