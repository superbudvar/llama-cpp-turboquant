#include "common.h"
#include "fit-cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

static uint32_t read_u32_le(const std::vector<uint8_t> & bytes, size_t offset) {
    return (uint32_t) bytes[offset] | ((uint32_t) bytes[offset + 1] << 8) |
           ((uint32_t) bytes[offset + 2] << 16) | ((uint32_t) bytes[offset + 3] << 24);
}

static void write_u32_le(std::vector<uint8_t> & bytes, size_t offset, uint32_t value) {
    for (size_t i = 0; i < 4; i++) bytes[offset + i] = (uint8_t) (value >> (8 * i));
}

static bool write_bytes(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write((const char *) bytes.data(), (std::streamsize) bytes.size());
    return out.good();
}

static std::vector<uint8_t> read_bytes(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
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
    plan.search.complete = false;
    plan.search.policy_version = 1;
    plan.search.shed_level = 1;
    plan.search.candidate_cursor = 3;
    plan.search.best_split_size = 2;
    plan.search.best_rpc_frac = 0.35f;
    plan.search.rpc_stage = 2;
    plan.search.rpc_frac_next = 0.40f;
    plan.search.rpc_frac_low = 0.25f;
    plan.search.rpc_frac_high = 0.45f;
    for (const auto & dev : analysis.devices) plan.search.baseline_free.push_back(dev.free_bytes);

    const bool saved = common_fit_cache_save(params, model_path, analysis, plan);
    check(saved, "save succeeds");
    if (!saved) {
        return 1;
    }

    common_fit_plan loaded;
    const bool hit = common_fit_cache_load(params, model_path, analysis, loaded);
    check(hit, "identical config hits the cache");
    check(hit && plan_equal(plan, loaded), "loaded plan matches the saved plan");
    check(hit && !loaded.search.complete && loaded.search.shed_level == 1 &&
          loaded.search.candidate_cursor == 3 && loaded.search.best_split_size == 2 &&
          loaded.search.best_rpc_frac == plan.search.best_rpc_frac &&
          loaded.search.rpc_stage == 2 && loaded.search.rpc_frac_next == plan.search.rpc_frac_next &&
          loaded.search.rpc_frac_low == plan.search.rpc_frac_low &&
          loaded.search.rpc_frac_high == plan.search.rpc_frac_high &&
          loaded.search.policy_version == plan.search.policy_version &&
          loaded.search.baseline_free == plan.search.baseline_free,
          "incomplete search cursor and incumbent metadata survive cache round-trip");

    common_params params_external_mtp = params;
    params_external_mtp.skip_infile_mtp = true;
    common_fit_plan replacement_plan = make_plan();
    replacement_plan.n_ctx = 49152;
    check(common_fit_cache_save(params_external_mtp, model_path, analysis, replacement_plan),
            "save a fit that skips target in-file MTP tensors");
    common_fit_plan replacement_loaded;
    check(common_fit_cache_load(params_external_mtp, model_path, analysis, replacement_loaded) &&
          replacement_loaded.n_ctx == replacement_plan.n_ctx,
            "replacement-MTP cache configuration hits its own plan");
    common_fit_plan base_plan_still_loaded;
    check(common_fit_cache_load(params, model_path, analysis, base_plan_still_loaded) &&
          base_plan_still_loaded.n_ctx == plan.n_ctx,
            "in-file-MTP and external-MTP replacement plans use separate cache entries");

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
    check(common_fit_cache_update_feedback(params, model_path, analysis, learned_delta,
                    projected_free, model_size_error, {}),
            "post-load feedback updates the exact fit entry in one write");
    common_fit_plan feedback_plan;
    check(common_fit_cache_load(params, model_path, analysis, feedback_plan),
            "cache remains loadable after the discrepancy update");
    check(feedback_plan.delta_free_ema == learned_delta &&
          feedback_plan.projected_free == projected_free &&
          feedback_plan.model_size_error == model_size_error &&
          !feedback_plan.search.complete &&
          feedback_plan.search.candidate_cursor == plan.search.candidate_cursor &&
          feedback_plan.search.rpc_stage == plan.search.rpc_stage &&
          feedback_plan.search.rpc_frac_next == plan.search.rpc_frac_next &&
          feedback_plan.search.baseline_free == plan.search.baseline_free,
            "discrepancy update preserves the cache key, search cursor, and learned values");
    common_fit_plan untouched_plan;
    check(common_fit_cache_load(params_other, model_path, analysis, untouched_plan),
            "other fit configuration remains loadable");
    check(untouched_plan.delta_free_ema == other_plan.delta_free_ema,
            "delta refresh does not modify another configuration's cache");

    common_params params_migration = params;
    params_migration.fit_strategy_min_ctx += 17;
    common_fit_plan migration_plan = plan;
    migration_plan.delta_free_ema.assign(n_dev, -3 * 1024 * 1024);
    check(common_fit_cache_save(params_migration, model_path, analysis, migration_plan),
            "save a cache entry for legacy-format migration tests");
    std::string migration_path;
    check(common_fit_cache_refresh_deltas(params_migration, model_path, analysis,
                    migration_plan.delta_free_ema, &migration_path),
            "locate the migration fixture by its exact fit key");
    const std::vector<uint8_t> v9_bytes = read_bytes(migration_path);
    check(v9_bytes.size() >= 28, "read the v9 migration fixture");
    if (v9_bytes.size() >= 28) {
        std::vector<uint8_t> v8_bytes = v9_bytes;
        v8_bytes.resize(v8_bytes.size() - 4);
        write_u32_le(v8_bytes, 4, 8);
        write_u32_le(v8_bytes, 24, (uint32_t) (v8_bytes.size() - 28));
        check(write_bytes(migration_path, v8_bytes), "write a v8 fixture without the policy revision");
        common_fit_plan loaded_v8;
        check(common_fit_cache_load(params_migration, model_path, analysis, loaded_v8) &&
              loaded_v8.search.policy_version == 0 && loaded_v8.search.rpc_stage == 2,
              "v8 plans retain the split cursor and default the policy revision");

        std::vector<uint8_t> v7_bytes = v9_bytes;
        v7_bytes.resize(v7_bytes.size() - 20);
        write_u32_le(v7_bytes, 4, 7);
        write_u32_le(v7_bytes, 24, (uint32_t) (v7_bytes.size() - 28));
        check(write_bytes(migration_path, v7_bytes), "write a v7 fixture without RPC cursor fields");
        common_fit_plan loaded_v7;
        check(common_fit_cache_load(params_migration, model_path, analysis, loaded_v7) &&
              loaded_v7.search.candidate_cursor == migration_plan.search.candidate_cursor &&
              loaded_v7.search.rpc_stage == 0 && loaded_v7.search.rpc_frac_next == -1.0f,
              "v7 plans retain their incumbent and safely default the split cursor");

        std::vector<uint8_t> v6_bytes = v9_bytes;
        const size_t v7_progress_size = 24 + 8 * migration_plan.search.baseline_free.size();
        v6_bytes.resize(v6_bytes.size() - 20 - v7_progress_size);
        write_u32_le(v6_bytes, 4, 6);
        write_u32_le(v6_bytes, 24, (uint32_t) (v6_bytes.size() - 28));
        check(write_bytes(migration_path, v6_bytes), "write a v6 fixture without search progress");
        common_fit_plan loaded_v6;
        check(common_fit_cache_load(params_migration, model_path, analysis, loaded_v6) &&
              loaded_v6.delta_free_ema == migration_plan.delta_free_ema && loaded_v6.search.complete,
              "v6 plans preserve feedback and default to a complete search");
        check(common_fit_cache_save(params_migration, model_path, analysis, loaded_v6),
              "legacy v6 plans rewrite in the current format");
        check(read_u32_le(read_bytes(migration_path), 4) == 9,
              "legacy cache migration writes the current format version");
    }

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
