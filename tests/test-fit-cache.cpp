#include "common.h"
#include "fit-cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

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
    common_fit_device dev;
    dev.id = 0;
    dev.dev = ggml_backend_dev_get(0);
    dev.is_rpc = false;
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(dev.dev, &free, &total);
    dev.free_bytes  = (int64_t) free;
    dev.total_bytes = (int64_t) total;
    analysis.devices.push_back(dev);

    common_params params;
    params.fit_strategy       = COMMON_FIT_STRATEGY_RPC_AWARE;
    params.fit_strategy_min_ctx = 62144;

    common_fit_plan plan = make_plan();

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
    common_fit_plan loaded2;
    check(!common_fit_cache_load(params_other, model_path, analysis, loaded2),
            "changed fit strategy misses the cache");

    std::error_code ec;
    for (const auto & entry : fs::directory_iterator(cache_dir, ec)) {
        if (entry.path().extension() == ".bin") {
            const std::string p = entry.path().string();
            FILE * f = fopen(p.c_str(), "r+b");
            if (f) {
                unsigned char b;
                if (fread(&b, 1, 1, f) == 1) {
                    fseek(f, 0, SEEK_SET);
                    b ^= 0xff;
                    fwrite(&b, 1, 1, f);
                }
                fclose(f);
            }
            break;
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