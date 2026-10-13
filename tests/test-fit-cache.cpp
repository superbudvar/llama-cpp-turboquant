#include "common.h"
#include "fit-cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int g_failures = 0;

static void check(bool cond, const char * msg) {
    if (cond) {
        fprintf(stderr, "ok:   %s\n", msg);
    } else {
        fprintf(stderr, "FAIL: %s\n", msg);
        g_failures++;
    }
}

static common_fit_plan make_plan() {
    common_fit_plan plan;
    plan.role_devices   = {0, 0, 0, 0, 0, 0, 0, 0};
    plan.n_gpu_layers   = 99;
    plan.n_ctx          = 62144;
    plan.pool_grant_mib = 12345;
    plan.tensor_split.assign(llama_max_devices(), 0.0f);
    plan.tensor_split[0] = 1.0f;
    return plan;
}

static bool plan_equal(const common_fit_plan & a, const common_fit_plan & b) {
    return a.role_devices == b.role_devices &&
           a.n_gpu_layers == b.n_gpu_layers &&
           a.tensor_split == b.tensor_split &&
           a.n_ctx        == b.n_ctx &&
           a.pool_grant_mib == b.pool_grant_mib;
}

int main() {
    ggml_backend_load_all();
    const size_t n_dev = ggml_backend_dev_count();
    check(n_dev > 0, "at least one backend device is available");
    if (n_dev == 0) {
        return 1;
    }

    const std::string cache_dir = "/tmp/fit-cache-test";
    setenv("LLAMA_FIT_CACHE", cache_dir.c_str(), 1);

    const std::string model_path = cache_dir + "/model.gguf";
    {
        FILE * f = fopen(model_path.c_str(), "wb");
        if (f) {
            fputs("fake model", f);
            fclose(f);
        }
    }

    common_fit_analysis analysis;
    for (size_t i = 0; i < n_dev; i++) {
        common_fit_device dev;
        dev.id = (int) i;
        dev.dev = ggml_backend_dev_get(i);
        dev.is_rpc = std::string(ggml_backend_dev_name(dev.dev)).find("RPC") != std::string::npos;
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev.dev, &free, &total);
        dev.free_bytes = (int64_t) free;
        dev.total_bytes = (int64_t) total;
        analysis.devices.push_back(dev);
    }

    common_params params;
    params.fit_strategy       = COMMON_FIT_STRATEGY_RPC_AWARE;
    params.fit_strategy_min_ctx = 62144;

    common_fit_plan plan = make_plan();
    plan.projected_free.assign(n_dev, 0);
    plan.model_size_error.assign(n_dev, 0);

    const bool saved = common_fit_cache_save(params, model_path, analysis, plan);
    check(saved, "save succeeds");
    if (!saved) {
        return 1;
    }

    common_fit_plan loaded;
    const bool hit = common_fit_cache_load(params, model_path, analysis, loaded);
    check(hit, "identical config hits the cache");
    check(hit && plan_equal(plan, loaded), "loaded plan matches the saved plan");

    common_params params_other = params;
    params_other.fit_strategy = COMMON_FIT_STRATEGY_MAXCTX;
    common_fit_plan other_plan = make_plan();
    other_plan.delta_free_ema.assign(n_dev, 7 * 1024 * 1024);
    other_plan.projected_free.assign(n_dev, 0);
    other_plan.model_size_error.assign(n_dev, 0);
    check(common_fit_cache_save(params_other, model_path, analysis, other_plan),
            "save a second fit configuration for the same model");

    std::vector<int64_t> learned_delta(n_dev, -5 * 1024 * 1024);
    std::string refreshed_path;
    check(common_fit_cache_refresh_deltas(params, model_path, analysis, learned_delta, &refreshed_path),
            "refresh the exact fit-cache configuration after MODEL READY");
    check(!refreshed_path.empty(), "delta refresh returns the exact cache path");
    common_fit_plan refreshed_plan;
    check(common_fit_cache_load(params, model_path, analysis, refreshed_plan),
            "refreshed fit configuration remains loadable");
    check(refreshed_plan.delta_free_ema == learned_delta,
            "delta refresh patches the matching configuration");
    std::vector<int64_t> projected_free(n_dev, 100 * 1024 * 1024);
    std::vector<int64_t> model_size_error(n_dev, -12 * 1024 * 1024);
    check(common_fit_cache_save_with_model_size_error(params, model_path, refreshed_plan,
                    projected_free, model_size_error, {}, &refreshed_path),
            "post-load discrepancy update rewrites the same cache entry");
    common_fit_plan feedback_plan;
    check(common_fit_cache_load(params, model_path, analysis, feedback_plan),
            "cache remains loadable after the discrepancy update");
    check(feedback_plan.delta_free_ema == learned_delta &&
          feedback_plan.projected_free == projected_free &&
          feedback_plan.model_size_error == model_size_error,
            "discrepancy update preserves the cache key and refreshed values");
    common_fit_plan untouched_plan;
    check(common_fit_cache_load(params_other, model_path, analysis, untouched_plan),
            "other fit configuration remains loadable");
    check(untouched_plan.delta_free_ema == other_plan.delta_free_ema,
            "delta refresh does not modify another configuration's cache");

    common_params params_mismatch = params;
    params_mismatch.fit_strategy_min_ctx++;
    common_fit_plan loaded2;
    check(!common_fit_cache_load(params_mismatch, model_path, analysis, loaded2),
            "changed context floor misses the cache");

    {
        FILE * f = fopen(refreshed_path.c_str(), "r+b");
        if (f) {
            unsigned char b;
            if (fread(&b, 1, 1, f) == 1) {
                fseek(f, 0, SEEK_SET);
                b ^= 0xff;
                fwrite(&b, 1, 1, f);
            }
            fclose(f);
        }
    }
    common_fit_plan loaded3;
    check(!common_fit_cache_load(params, model_path, analysis, loaded3),
            "corrupted cache file is a miss, not a crash");

    common_params params_clear = params;
    params_clear.fit_cache_clear = true;
    common_fit_plan loaded4;
    check(!common_fit_cache_load(params_clear, model_path, analysis, loaded4),
            "--fit-cache-clear forces a miss");

    remove(model_path.c_str());
    fprintf(stderr, "test-fit-cache: %s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
