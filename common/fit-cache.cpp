#include "fit-cache.h"
#include "common.h"
#include "log.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr uint32_t CACHE_MAGIC   = 0x46495443; // "FITC"
constexpr uint32_t CACHE_VERSION = 6;
constexpr std::int64_t MAX_CACHE_AGE_S = 24 * 3600;

// parse guards: no plan can legitimately exceed these
constexpr size_t MAX_ROLES = 16;
constexpr size_t MAX_SPLIT = 64;

// SHA-256 (FIPS 180-4), compact single-shot implementation.
uint32_t g_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

uint32_t sha256_rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

void sha256_block(uint32_t h[8], const uint8_t * p) {
    uint32_t w[64];
    for (size_t i = 0; i < 16; i++) {
        w[i] = ((uint32_t) p[i * 4 + 0] << 24) |
               ((uint32_t) p[i * 4 + 1] << 16) |
               ((uint32_t) p[i * 4 + 2] <<  8) |
               ((uint32_t) p[i * 4 + 3] <<  0);
    }
    for (size_t i = 16; i < 64; i++) {
        const uint32_t s0 = sha256_rotr(w[i - 15], 7) ^ sha256_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = sha256_rotr(w[i - 2], 17)  ^ sha256_rotr(w[i - 2], 19)   ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

    for (size_t i = 0; i < 64; i++) {
        const uint32_t S1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = hh + S1 + ch + g_sha256_k[i] + w[i];
        const uint32_t S0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

std::vector<uint8_t> sha256_digest(const uint8_t * data, size_t len) {
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    };

    uint64_t bitlen = (uint64_t) len * 8;
    std::vector<uint8_t> buf(data, data + len);
    buf.push_back(0x80);
    while (buf.size() % 64 != 56) {
        buf.push_back(0x00);
    }
    for (int i = 7; i >= 0; i--) {
        buf.push_back((uint8_t) (bitlen >> (i * 8)));
    }

    for (size_t off = 0; off < buf.size(); off += 64) {
        sha256_block(h, buf.data() + off);
    }

    std::vector<uint8_t> out(32);
    for (size_t i = 0; i < 8; i++) {
        out[i * 4 + 0] = (uint8_t) (h[i] >> 24);
        out[i * 4 + 1] = (uint8_t) (h[i] >> 16);
        out[i * 4 + 2] = (uint8_t) (h[i] >>  8);
        out[i * 4 + 3] = (uint8_t) (h[i] >>  0);
    }
    return out;
}

std::string sha256_hex(const std::vector<uint8_t> & d) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(d.size() * 2);
    for (uint8_t b : d) {
        out.push_back(hex[b >> 4]);
        out.push_back(hex[b & 0x0f]);
    }
    return out;
}

std::string cache_root_dir() {
    const char * env = getenv("LLAMA_FIT_CACHE");
    if (env && *env) {
        return env;
    }
    const char * home = getenv("HOME");
    if (home && *home) {
        return std::string(home) + "/.cache/llama-cpp/fit-cache";
    }
    return ".cache/llama-cpp/fit-cache";
}

std::string cache_file_path(
        const common_params & params,
        const std::string & path_model,
        const common_fit_analysis & analysis) {
    const uint64_t key = common_fit_cache_key_hash(params, path_model, analysis);
    const std::vector<uint8_t> d = sha256_digest((const uint8_t *) &key, sizeof(key));
    return (fs::path(cache_root_dir()) / (sha256_hex(d) + ".bin")).string();
}

// little-endian serialization helpers
void app_u32(std::vector<uint8_t> & v, uint32_t x) {
    for (int i = 0; i < 4; i++) {
        v.push_back((uint8_t) (x >> (8 * i)));
    }
}

void app_i32(std::vector<uint8_t> & v, int32_t x) {
    app_u32(v, (uint32_t) x);
}

void app_u64(std::vector<uint8_t> & v, uint64_t x) {
    for (int i = 0; i < 8; i++) {
        v.push_back((uint8_t) (x >> (8 * i)));
    }
}

void app_i64(std::vector<uint8_t> & v, int64_t x) {
    app_u64(v, (uint64_t) x);
}

void app_f32(std::vector<uint8_t> & v, float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    app_u32(v, bits);
}

struct reader {
    const uint8_t * data;
    size_t size;
    size_t pos;

    bool u32(uint32_t & out) {
        if (pos + 4 > size) {
            return false;
        }
        out = ((uint32_t) data[pos + 0]) | ((uint32_t) data[pos + 1] << 8) |
              ((uint32_t) data[pos + 2] << 16) | ((uint32_t) data[pos + 3] << 24);
        pos += 4;
        return true;
    }

    bool u64(uint64_t & out) {
        uint32_t lo, hi;
        if (!u32(lo) || !u32(hi)) {
            return false;
        }
        out = ((uint64_t) lo) | ((uint64_t) hi << 32);
        return true;
    }

    bool i32(int32_t & out) {
        uint32_t u;
        if (!u32(u)) {
            return false;
        }
        std::memcpy(&out, &u, sizeof(out));
        return true;
    }

    bool i64(int64_t & out) {
        uint64_t u;
        if (!u64(u)) {
            return false;
        }
        std::memcpy(&out, &u, sizeof(out));
        return true;
    }

    bool f32(float & out) {
        uint32_t u;
        if (!u32(u)) {
            return false;
        }
        std::memcpy(&out, &u, sizeof(out));
        return true;
    }
};

int64_t file_mtime_sec(const std::string & path) {
    std::error_code ec;
    const auto ftp = fs::last_write_time(path, ec);
    if (ec) {
        return -1;
    }
    const auto sys = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftp - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
    return (int64_t) std::chrono::system_clock::to_time_t(sys);
}

int64_t file_size_bytes(const std::string & path) {
    std::error_code ec;
    const uintmax_t sz = fs::file_size(path, ec);
    if (ec) {
        return -1;
    }
    return (int64_t) sz;
}

const char * device_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:  return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:  return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accel";
        default:                            return "other";
    }
}

} // namespace

uint64_t common_fit_cache_key_hash(
        const common_params & params,
        const std::string & path_model,
        const common_fit_analysis & analysis) {
    std::string key;

key += "model=";
    key += path_model;
    key += "\n";
    key += "size=";
    key += std::to_string(file_size_bytes(path_model));
    key += "\n";
    key += "mtime=";
    key += std::to_string(file_mtime_sec(path_model));
    key += "\n";

    // device identity sorted by name so the key is independent of the
    // enumeration order; memory CAPACITY only, load is validated separately
    std::vector<std::string> devs;
    for (const auto & d : analysis.devices) {
        std::string s;
        s += std::string(ggml_backend_dev_name(d.dev));
        s += "|";
        s += device_type_name(ggml_backend_dev_type(d.dev));
        s += "|";
        s += std::to_string(d.total_bytes);
        s += "|";
        s += (d.is_rpc ? "rpc" : "local");
        devs.push_back(s);
    }
    std::sort(devs.begin(), devs.end());
    for (const auto & s : devs) {
        key += std::string("dev=") + s + "\n";
    }

    for (size_t t : params.fit_params_target) {
        key += "target=";
        key += std::to_string(t);
        key += "\n";
    }
    key += "shed=";
    key += std::to_string(params.fit_params_shed_tags.size());
    for (const auto & tag : params.fit_params_shed_tags) {
        key += ",";
        key += tag;
    }
    key += "\n";
    key += "reserve=";
    key += std::to_string(params.fit_params_reserves.size());
    for (const auto & r : params.fit_params_reserves) {
        key += ",";
        key += std::to_string(r.device);
        key += ":";
        key += std::to_string(r.bytes);
        key += ":";
        key += std::to_string(r.priority);
        key += ":";
        key += r.tag;
    }
    key += "\n";
    key += "strategy=";
    key += std::to_string((int) params.fit_strategy);
    key += "\n";
    key += "min_ctx=";
    key += std::to_string(params.fit_strategy_min_ctx);
    key += "\n";
    // an explicit --ctx-size is a hard requirement on the fit result; a cached
    // placement for a different context must not be reused
    key += "n_ctx=";
    key += std::to_string(params.n_ctx);
    key += "\n";
    // KV layout affects the projected and actual memory layout; a placement
    // measured for q8_0 KV is not safe to reuse for turbo4 KV (different
    // bytes per token) or for a different FA mode (different scratch size).
    // All fields that change the actual KV or compute footprint on the
    // device must be in the key; otherwise a fit may validate against a
    // memory budget that does not match the run that needs the cache.
    key += "kv=ctk=";
    key += std::to_string((int) params.cache_type_k);
    key += ",ctv=";
    key += std::to_string((int) params.cache_type_v);
    key += ",fa=";
    key += std::to_string((int) params.flash_attn_type);
    key += ",kvu=";
    key += std::to_string(params.kv_unified ? 1 : 0);
    key += ",kvsa=";
    key += std::to_string(params.kv_stream_arena_mib);
    key += ",nb=";
    key += std::to_string(params.n_batch);
    key += ",nub=";
    key += std::to_string(params.n_ubatch);
    key += ",np=";
    key += std::to_string(params.n_parallel);
    key += ",nkvo=";
    key += std::to_string(params.no_kv_offload ? 1 : 0);
    key += ",noopo=";
    key += std::to_string(params.no_op_offload ? 1 : 0);
    key += ",dctk=";
    key += std::to_string((int) params.speculative.draft.cache_type_k);
    key += ",dctv=";
    key += std::to_string((int) params.speculative.draft.cache_type_v);
    key += "\n";

    std::string rpc;
    for (const auto & h : params.rpc_hosts) {
        if (!rpc.empty()) {
            rpc += ",";
        }
        rpc += h;
    }
    key += "rpc=";
    key += rpc;
    key += "\n";

    // speculative decoding and a vision companion load after the main model fit
    // and consume leftover VRAM, so they must not reuse the same cached plan as
    // a run without them.
    key += "spec=";
    for (enum common_speculative_type t : params.speculative.types) {
        key += std::to_string((int) t);
        key += ",";
    }
    key += "\n";
    key += "spec_draft=";
    if (!params.speculative.draft.mparams.empty()) {
        key += params.speculative.draft.mparams.path;
        key += "|";
        key += std::to_string(file_size_bytes(params.speculative.draft.mparams.path));
        key += "|";
        key += std::to_string(file_mtime_sec(params.speculative.draft.mparams.path));
    } else {
        key += "none";
    }
    key += "\n";
    key += "mmproj=";
    if (!params.mmproj.empty() && !params.no_mmproj) {
        key += params.mmproj.path;
        key += "|";
        key += std::to_string(file_size_bytes(params.mmproj.path));
        key += "|";
        key += std::to_string(file_mtime_sec(params.mmproj.path));
    } else {
        key += "none";
    }
    key += "\n";

    const std::vector<uint8_t> d = sha256_digest((const uint8_t *) key.data(), key.size());
    uint64_t hash = 0;
    for (size_t i = 0; i < 8; i++) {
        hash = (hash << 8) | d[i];
    }
    return hash;
}

bool common_fit_cache_load(
        const common_params & params,
        const std::string & path_model,
        const common_fit_analysis & analysis,
        common_fit_plan & plan_out) {
    const std::string path = cache_file_path(params, path_model, analysis);
    const uint64_t key = common_fit_cache_key_hash(params, path_model, analysis);

    if (params.fit_cache_clear) {
        LOG_DBG("%s: --fit-cache-clear: removing %s\n", __func__, path.c_str());
        std::error_code ec;
        fs::remove(path, ec);
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (buf.size() < 32) {
        LOG_DBG("%s: cache file too small: %s\n", __func__, path.c_str());
        return false;
    }

    reader r{buf.data(), buf.size(), 0};
    uint32_t magic = 0, version = 0, body_size = 0;
    uint64_t key_hash = 0, timestamp = 0;
    if (!r.u32(magic) || !r.u32(version) || !r.u64(key_hash) || !r.u64(timestamp) || !r.u32(body_size)) {
        return false;
    }
    if (magic != CACHE_MAGIC || version != CACHE_VERSION) {
        LOG_DBG("%s: cache header mismatch: %s\n", __func__, path.c_str());
        return false;
    }
    if (key_hash != key) {
        LOG_DBG("%s: cache key mismatch: %s\n", __func__, path.c_str());
        return false;
    }
    const int64_t now = (int64_t) std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    if ((int64_t) timestamp <= 0 || now - (int64_t) timestamp > MAX_CACHE_AGE_S || (int64_t) timestamp > now) {
        LOG_DBG("%s: cache entry stale: %s\n", __func__, path.c_str());
        return false;
    }
    if (r.pos + body_size != buf.size()) {
        LOG_DBG("%s: cache body size mismatch: %s\n", __func__, path.c_str());
        return false;
    }

    int32_t n_roles = 0;
    if (!r.i32(n_roles) || n_roles < 0 || (size_t) n_roles > MAX_ROLES) {
        return false;
    }
    common_fit_plan plan;
    plan.role_devices.resize((size_t) n_roles);
    for (size_t i = 0; i < (size_t) n_roles; i++) {
        int32_t id = 0;
        if (!r.i32(id)) {
            return false;
        }
        plan.role_devices[i] = id;
    }
    if (!r.i32(plan.n_gpu_layers)) {
        return false;
    }
    int32_t split_len = 0;
    if (!r.i32(split_len) || split_len < 0 || (size_t) split_len > MAX_SPLIT) {
        return false;
    }
    plan.tensor_split.resize((size_t) split_len);
    for (size_t i = 0; i < (size_t) split_len; i++) {
        if (!r.f32(plan.tensor_split[i])) {
            return false;
        }
    }
    if (!r.i32(plan.n_ctx)) {
        return false;
    }
    uint64_t pool = 0;
    if (!r.u64(pool)) {
        return false;
    }
    plan.pool_grant_mib = (size_t) pool;

    int32_t n_shed = 0;
    if (!r.i32(n_shed) || n_shed < 0 || (size_t) n_shed > MAX_SPLIT) {
        return false;
    }
    plan.shed_reserves.reserve((size_t) n_shed);
    for (int32_t i = 0; i < n_shed; i++) {
        int32_t tag_len = 0;
        if (!r.i32(tag_len) || tag_len < 0 || tag_len > 63) {
            return false;
        }
        if (r.pos + (size_t) tag_len > r.size) {
            return false;
        }
        plan.shed_reserves.emplace_back((const char *) (r.data + r.pos), (size_t) tag_len);
        r.pos += (size_t) tag_len;
    }

    // v2+: per-device EMA of actual_free - projected_free (bytes).
    // negative = fit overestimates free space -> margin should be increased.
    int32_t n_delta = 0;
    if (!r.i32(n_delta) || n_delta < 0 || (size_t) n_delta > MAX_SPLIT) {
        return false;
    }
    plan.delta_free_ema.resize(n_delta > 0 ? (size_t) n_delta : 0);
    for (int32_t i = 0; i < n_delta; i++) {
        int64_t d = 0;
        if (!r.i64(d)) {
            return false;
        }
        plan.delta_free_ema[(size_t) i] = d;
    }

    // v3+: projected_free per device and whether delta was applied when projecting.
    // if version < 3, these default to empty/false (first run after upgrade is treated as blind).
    plan.projected_free.clear();
    plan.projected_with_delta = false;
    plan.ceiling_free.clear();
    if (version >= 3 && r.pos < buf.size()) {
        int32_t n_proj = 0;
        if (!r.i32(n_proj) || n_proj < 0 || (size_t) n_proj > MAX_SPLIT) {
            return false;
        }
        plan.projected_free.resize(n_proj > 0 ? (size_t) n_proj : 0);
        for (int32_t i = 0; i < n_proj; i++) {
            int64_t p = 0;
            if (!r.i64(p)) {
                return false;
            }
            plan.projected_free[(size_t) i] = p;
        }
        uint32_t flags = 0;
        if (!r.u32(flags)) {
            return false;
        }
        plan.projected_with_delta = (flags & 1) != 0;
    }

    // v4+: per-device ceiling free bytes (maximum observed free after successful load).
    // used to clamp backend-reported free when it fluctuates above the measured ceiling.
    if (version >= 4 && r.pos < buf.size()) {
        int32_t n_ceil = 0;
        if (!r.i32(n_ceil) || n_ceil < 0 || (size_t) n_ceil > MAX_SPLIT) {
            return false;
        }
        plan.ceiling_free.resize(n_ceil > 0 ? (size_t) n_ceil : 0);
        for (int32_t i = 0; i < n_ceil; i++) {
            int64_t c = 0;
            if (!r.i64(c)) {
                return false;
            }
            plan.ceiling_free[(size_t) i] = c;
        }
    }

    // v5+: per-device model_size_error (actual_model - projected_model)
    if (version >= 5 && r.pos < buf.size()) {
        int32_t n_err = 0;
        if (!r.i32(n_err) || n_err < 0 || (size_t) n_err > MAX_SPLIT) {
            return false;
        }
        plan.model_size_error.resize(n_err > 0 ? (size_t) n_err : 0);
        for (int32_t i = 0; i < n_err; i++) {
            int64_t e = 0;
            if (!r.i64(e)) {
                return false;
            }
            plan.model_size_error[(size_t) i] = e;
        }
    }

    // v6+: per-device warm ceiling free bytes (measured AFTER model weights loaded).
    // used to clamp margin to the true allocatable memory post-load.
    if (version >= 6 && r.pos < buf.size()) {
        int32_t n_warm = 0;
        if (!r.i32(n_warm) || n_warm < 0 || (size_t) n_warm > MAX_SPLIT) {
            return false;
        }
        plan.ceiling_free_warm.resize(n_warm > 0 ? (size_t) n_warm : 0);
        for (int32_t i = 0; i < n_warm; i++) {
            int64_t w = 0;
            if (!r.i64(w)) {
                return false;
            }
            plan.ceiling_free_warm[(size_t) i] = w;
        }
    }

    if (r.pos != buf.size()) {
        LOG_DBG("%s: cache body not fully consumed: %s\n", __func__, path.c_str());
        return false;
    }

    LOG_DBG("%s: loaded cached fit plan from %s\n", __func__, path.c_str());
    LOG_INF("%s: loaded cache: delta_ema=[%s] projected=[%s] ceiling=[%s] model_size_error=[%s] from %s\n",
            __func__,
            [&]() {
                std::string s;
                for (size_t i = 0; i < plan.delta_free_ema.size(); ++i) {
                    if (i > 0) s += ",";
                    s += std::to_string(plan.delta_free_ema[i] / 1024 / 1024);
                    s += "MiB";
                }
                return s;
            }().c_str(),
            [&]() {
                std::string s;
                for (size_t i = 0; i < plan.projected_free.size(); ++i) {
                    if (i > 0) s += ",";
                    s += std::to_string(plan.projected_free[i] / 1024 / 1024);
                    s += "MiB";
                }
                return s;
            }().c_str(),
            [&]() {
                std::string s;
                for (size_t i = 0; i < plan.ceiling_free.size(); ++i) {
                    if (i > 0) s += ",";
                    s += std::to_string(plan.ceiling_free[i] / 1024 / 1024);
                    s += "MiB";
                }
                return s;
            }().c_str(),
            [&]() {
                std::string s;
                for (size_t i = 0; i < plan.model_size_error.size(); ++i) {
                    if (i > 0) s += ",";
                    s += std::to_string(plan.model_size_error[i] / 1024 / 1024);
                    s += "MiB";
                }
                return s;
            }().c_str(),
            path.c_str());
    plan_out = std::move(plan);
    return true;
}

bool common_fit_cache_save(
        const common_params & params,
        const std::string & path_model,
        const common_fit_analysis & analysis,
        const common_fit_plan & plan,
        uint64_t override_key) {
    const uint64_t key = override_key ? override_key : common_fit_cache_key_hash(params, path_model, analysis);
    const std::string path = cache_file_path(params, path_model, analysis);

    std::vector<uint8_t> body;
    app_i32(body, (int32_t) plan.role_devices.size());
    for (int id : plan.role_devices) {
        app_i32(body, id);
    }
    app_i32(body, plan.n_gpu_layers);
    app_i32(body, (int32_t) plan.tensor_split.size());
    for (float f : plan.tensor_split) {
        app_f32(body, f);
    }
    app_i32(body, plan.n_ctx);
    app_u64(body, (uint64_t) plan.pool_grant_mib);
    app_i32(body, (int32_t) plan.shed_reserves.size());
    for (const auto & tag : plan.shed_reserves) {
        if (tag.size() > 63) {
            LOG_WRN("%s: shed tag too long, refusing to cache\n", __func__);
            return false;
        }
        app_i32(body, (int32_t) tag.size());
        body.insert(body.end(), tag.begin(), tag.end());
    }
    // v2+: per-device EMA of actual_free - projected_free
    app_i32(body, (int32_t) plan.delta_free_ema.size());
    for (int64_t d : plan.delta_free_ema) {
        app_i64(body, d);
    }
    // v3+: projected_free per device and whether delta was applied when projecting
    app_i32(body, (int32_t) plan.projected_free.size());
    for (int64_t p : plan.projected_free) {
        app_i64(body, p);
    }
    uint32_t flags = plan.projected_with_delta ? 1 : 0;
    app_u32(body, flags);
    // v4+: per-device ceiling free bytes
    app_i32(body, (int32_t) plan.ceiling_free.size());
    for (int64_t c : plan.ceiling_free) {
        app_i64(body, c);
    }
    // v5+: per-device model_size_error (actual_model - projected_model)
    app_i32(body, (int32_t) plan.model_size_error.size());
    for (int64_t e : plan.model_size_error) {
        app_i64(body, e);
    }
    // v6+: per-device warm ceiling free bytes (measured AFTER model weights loaded)
    app_i32(body, (int32_t) plan.ceiling_free_warm.size());
    for (int64_t w : plan.ceiling_free_warm) {
        app_i64(body, w);
    }

    std::vector<uint8_t> out;
    app_u32(out, CACHE_MAGIC);
    app_u32(out, CACHE_VERSION);
    app_u64(out, key);
    app_u64(out, (uint64_t) std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    app_u32(out, (uint32_t) body.size());
    out.insert(out.end(), body.begin(), body.end());

    std::error_code ec;
    fs::create_directories(cache_root_dir(), ec);
    if (ec) {
        LOG_WRN("%s: cannot create cache dir: %s\n", __func__, ec.message().c_str());
        return false;
    }

    const std::string tmp = path + ".tmp";
    {
        std::ofstream out_file(tmp, std::ios::binary | std::ios::trunc);
        if (!out_file.is_open()) {
            LOG_WRN("%s: cannot open cache file: %s\n", __func__, tmp.c_str());
            return false;
        }
        out_file.write((const char *) out.data(), (std::streamsize) out.size());
        out_file.flush();
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        LOG_WRN("%s: cannot move cache file into place: %s\n", __func__, ec.message().c_str());
        return false;
    }
    LOG_INF("%s: saved cache: delta_ema=[%s] projected=[%s] ceiling=[%s] model_size_error=[%s] to %s\n",
            __func__,
            [&]() {
                std::string s;
                for (size_t i = 0; i < plan.delta_free_ema.size(); ++i) {
                    if (i > 0) s += ",";
                    s += std::to_string(plan.delta_free_ema[i] / 1024 / 1024);
                    s += "MiB";
                }
                return s;
            }().c_str(),
            [&]() {
                std::string s;
                for (size_t i = 0; i < plan.projected_free.size(); ++i) {
                    if (i > 0) s += ",";
                    s += std::to_string(plan.projected_free[i] / 1024 / 1024);
                    s += "MiB";
                }
                return s;
            }().c_str(),
            [&]() {
                std::string s;
                for (size_t i = 0; i < plan.ceiling_free.size(); ++i) {
                    if (i > 0) s += ",";
                    s += std::to_string(plan.ceiling_free[i] / 1024 / 1024);
                    s += "MiB";
                }
                return s;
            }().c_str(),
            [&]() {
                std::string s;
                for (size_t i = 0; i < plan.model_size_error.size(); ++i) {
                    if (i > 0) s += ",";
                    s += std::to_string(plan.model_size_error[i] / 1024 / 1024);
                    s += "MiB";
                }
                return s;
            }().c_str(),
            path.c_str());
    return true;
}

bool common_fit_cache_refresh_deltas(
        const common_params & params,
        const std::string & path_model,
        const common_fit_analysis & analysis,
        const std::vector<int64_t> & new_delta_free_ema,
        std::string * cache_path_out) {
    {
        std::string ema_str;
        for (size_t i = 0; i < new_delta_free_ema.size(); ++i) {
            if (i > 0) ema_str += ",";
            ema_str += std::to_string(new_delta_free_ema[i] / 1024 / 1024) + "MiB";
        }
        LOG_INF("%s: ENTRY: new_delta_ema=[%s] path_model=%s\n", __func__, ema_str.c_str(), path_model.c_str());
    }
    // Free-memory values are not part of the key, so the reconstructed device
    // list from the measured model resolves the exact configuration entry.
    const std::string cache_path = cache_file_path(params, path_model, analysis);
    LOG_INF("fit_delta_refresh: found cache file: %s\n", cache_path.c_str());
    std::ifstream in(cache_path, std::ios::binary);
    if (!in.is_open()) {
        LOG_INF("fit_delta_refresh: cannot open cache file: %s\n", cache_path.c_str());
        return false;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    LOG_INF("fit_delta_refresh: read %zu bytes from cache file\n", buf.size());
    if (buf.size() < 64) {
        LOG_WRN("fit_delta_refresh: cache file too small: %s\n", cache_path.c_str());
        return false;
    }

    // Manually deserialize the cache to patch delta_free_ema without key validation.
    // The file format is: magic(4) + version(4) + key_hash(8) + timestamp(8) + body_size(4) + body
    struct reader { const uint8_t * p; size_t n; size_t pos; };
    auto get_u64 = [](reader & r) -> uint64_t {
        uint64_t v = 0;
        for (int i = 0; i < 8; i++) v |= ((uint64_t)r.p[r.pos++] << (8*i));
        return v;
    };
    auto get_u32 = [](reader & r) -> uint32_t {
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) v |= ((uint32_t)r.p[r.pos++] << (8*i));
        return v;
    };

    reader r{buf.data(), buf.size(), 0};
    (void)get_u32(r); // magic
    (void)get_u32(r); // version (v6 only)
    uint64_t stored_key = get_u64(r);
    uint64_t timestamp = get_u64(r);
    uint32_t body_size = get_u32(r);

    // Walk the old body to find delta_free_ema location.
    // We MUST read n_roles from the file (not use analysis.devices.size())
    // because n_roles=8 in the plan but analysis.devices=2.
    size_t pos = 28; // body starts at byte 28
    uint32_t n_roles = 0;
    for (int i = 0; i < 4; i++) n_roles |= ((uint32_t)buf[pos + i] << (8*i));
    pos += 4; // skip n_roles
    pos += 4 * n_roles; // role_devices
    pos += 4; // n_gpu_layers
    uint32_t split_len = 0;
    for (int i = 0; i < 4; i++) split_len |= ((uint32_t)buf[pos + i] << (8*i));
    pos += 4; // skip split_len
    pos += 4 * split_len; // tensor_split (float = 4 bytes)
    pos += 4; // n_ctx (app_i32 = 4 bytes, NOT 8)
    pos += 8; // pool_grant_mib
    uint32_t shed_count = 0;
    for (int i = 0; i < 4; i++) shed_count |= ((uint32_t)buf[pos + i] << (8*i));
    pos += 4; // shed_reserves_count
    for (uint32_t i = 0; i < shed_count; i++) {
        uint32_t taglen = 0;
        for (int j = 0; j < 4; j++) taglen |= ((uint32_t)buf[pos + j] << (8*j));
        pos += 4 + taglen;
    }
    // Now pos is at delta_free_ema count
    size_t delta_count_pos = pos;
    uint32_t n_delta = 0;
    for (int i = 0; i < 4; i++) n_delta |= ((uint32_t)buf[pos + i] << (8*i));
    pos += 4; // skip old delta_free_ema count
    pos += 8 * n_delta; // skip old delta_free_ema values (use file's count, not caller's)
    size_t proj_count_pos = pos; // projected_free starts here
    // v6: projected_free is per-device, read n_proj from file (NOT from new_delta_free_ema.size())
    uint32_t n_proj = 0;
    for (int i = 0; i < 4; i++) n_proj |= ((uint32_t)buf[proj_count_pos + i] << (8*i));
    pos += 4; // skip projected_free count
    pos += 8 * n_proj; // skip projected_free values
    pos += 4; // projected_with_delta
    // v6: ceiling_free (cold)
    uint32_t n_ceil = 0;
    for (int i = 0; i < 4; i++) n_ceil |= ((uint32_t)buf[pos + i] << (8*i));
    pos += 4 + 8 * n_ceil; // skip ceiling_free count + values
    // v6: model_size_error
    uint32_t n_mse = 0;
    for (int i = 0; i < 4; i++) n_mse |= ((uint32_t)buf[pos + i] << (8*i));
    pos += 4 + 8 * n_mse;
    // v6: ceiling_free_warm
    uint32_t n_warm = 0;
    for (int i = 0; i < 4; i++) n_warm |= ((uint32_t)buf[pos + i] << (8*i));
    pos += 4 + 8 * n_warm;
    size_t body_end = 28 + body_size;
    LOG_INF("fit_delta_refresh: walked body: delta_count_pos=%zu, body_end=%zu (file %zu bytes)\n",
            delta_count_pos, body_end, buf.size());

    // Log old delta values before overwriting
    {
        std::string old_s, new_s;
        for (size_t i = 0; i < new_delta_free_ema.size(); i++) {
            if (i > 0) { old_s += ","; new_s += ","; }
            int64_t old_val = 0;
            if (i < n_delta && delta_count_pos + 4 + i * 8 + 8 <= buf.size()) {
                for (int j = 0; j < 8; j++) old_val |= ((int64_t)buf[delta_count_pos + 4 + i * 8 + j] << (8 * j));
            }
            old_s += std::to_string(old_val / 1024 / 1024) + "MiB";
            new_s += std::to_string(new_delta_free_ema[i] / 1024 / 1024) + "MiB";
        }
        LOG_INF("%s: patching delta_free_ema at file offset %zu: old=[%s] new=[%s]\n",
                __func__, delta_count_pos, old_s.c_str(), new_s.c_str());
    }

    // Build new body = prefix + new_delta + the remaining body.
    std::vector<uint8_t> new_body;
    new_body.insert(new_body.end(), buf.begin() + 28, buf.begin() + delta_count_pos);
    app_i32(new_body, (int32_t)new_delta_free_ema.size());
    for (int64_t d : new_delta_free_ema) app_i64(new_body, d);
    const size_t after_delta = delta_count_pos + 4 + 8 * n_delta;
    new_body.insert(new_body.end(), buf.begin() + after_delta, buf.begin() + buf.size());

    // Build new file: header + new_body
    std::vector<uint8_t> out;
    for (int i = 0; i < 4; i++) out.push_back((uint8_t)((CACHE_MAGIC >> (8*i)) & 0xff));
    for (int i = 0; i < 4; i++) out.push_back((uint8_t)((CACHE_VERSION >> (8*i)) & 0xff));
    for (int i = 0; i < 8; i++) out.push_back((uint8_t)((stored_key >> (8*i)) & 0xff));
    for (int i = 0; i < 8; i++) out.push_back((uint8_t)((timestamp >> (8*i)) & 0xff));
    uint32_t new_body_size = (uint32_t)new_body.size();
    for (int i = 0; i < 4; i++) out.push_back((uint8_t)((new_body_size >> (8*i)) & 0xff));
    out.insert(out.end(), new_body.begin(), new_body.end());

    // Write back atomically
    std::error_code ec;
    fs::create_directories(cache_root_dir(), ec);
    std::string tmp = cache_path + ".tmp";
    std::ofstream fout(tmp, std::ios::binary);
    fout.write((char *)out.data(), out.size());
    fout.close();
    fs::rename(tmp, cache_path, ec);
    if (ec) {
        LOG_WRN("fit_delta_refresh: atomic rename failed: %s\n", ec.message().c_str());
        return false;
    }
    if (cache_path_out) *cache_path_out = cache_path;
    LOG_INF("fit_delta_refresh: refreshed delta_free_ema for %s (%zu devices): [%s] (%s)\n",
            path_model.c_str(), new_delta_free_ema.size(),
            [&]() {
                std::string s;
                for (size_t i = 0; i < new_delta_free_ema.size(); ++i) {
                    if (i > 0) s += ", ";
                    s += std::to_string(new_delta_free_ema[i]);
                }
                return s;
            }().c_str(),
            [&]() {
                std::string s;
                for (size_t i = 0; i < new_delta_free_ema.size(); ++i) {
                    if (i > 0) s += ", ";
                    s += std::to_string(new_delta_free_ema[i] / 1024 / 1024) + "MiB";
                }
                return s;
            }().c_str());
    return true;
}

bool common_fit_cache_save_with_model_size_error(
        const common_params & params,
        const std::string & path_model,
        const common_fit_plan & plan,
        const std::vector<int64_t> & projected_free,
        const std::vector<int64_t> & model_size_error,
        const std::vector<int64_t> & ceiling_free_warm,
        const std::string * cache_path_in) {
    (void) params; // the cache key is preserved from the selected entry
    (void) plan;
    // Use the exact entry that fit_delta_refresh updated. Never patch an
    // arbitrary cache entry when the selected plan could not be identified.
    std::string cache_path = cache_path_in ? *cache_path_in : std::string();
    if (cache_path.empty()) {
        LOG_WRN("%s: exact cache path is unavailable; refusing to update an arbitrary entry\n", __func__);
        return false;
    }
    LOG_INF("%s: updating feedback for model %s in %s\n",
            __func__, path_model.c_str(), cache_path.c_str());
    // Load existing plan via common_fit_cache_load using the file's embedded key.
    // To do this we reconstruct an analysis matching the saved state by reading
    // device identity from the cache key itself.  Fall back to directory scan +
    // full binary rebuild if key reconstruction fails.
    std::ifstream in(cache_path, std::ios::binary);
    if (!in.is_open()) {
        LOG_WRN("%s: cannot open cache file: %s\n", __func__, cache_path.c_str());
        return false;
    }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
    LOG_INF("%s: read %zu bytes from cache file\n", __func__, buf.size());
    if (buf.size() < 64) {
        LOG_WRN("%s: cache file too small\n", __func__);
        return false;
    }
    struct reader { const uint8_t * p; size_t n; size_t pos; };
    auto get_u64 = [](reader & r) -> uint64_t {
        uint64_t v = 0;
        for (int i = 0; i < 8; i++) v |= ((uint64_t)r.p[r.pos++] << (8*i));
        return v;
    };
    auto get_u32 = [](reader & r) -> uint32_t {
        uint32_t v = 0;
        for (int i = 0; i < 4; i++) v |= ((uint32_t)r.p[r.pos++] << (8*i));
        return v;
    };
    reader r{buf.data(), buf.size(), 0};
    (void)get_u32(r); // magic
    uint32_t version = get_u32(r);
    uint64_t stored_key = get_u64(r); // use stored key directly
    (void)get_u64(r); // timestamp
    uint32_t body_size = get_u32(r);
    (void)version; (void)body_size;

    // Directly deserialize the plan from the buffer we already read.
    // We bypass common_fit_cache_load because the stored key was computed at save time
    // with the frozen device state; a live analysis would produce a different key hash
    // since free_bytes changes between initial solve and MODEL READY.
    r = reader{buf.data(), buf.size(), 28}; // re-use existing r, skip 28-byte CTIF header
    common_fit_plan existing;
    int32_t n_roles = get_u32(r);
    if (n_roles < 0 || (size_t)n_roles > MAX_ROLES) return false;
    existing.role_devices.resize(n_roles);
    for (int32_t i = 0; i < n_roles; i++) existing.role_devices[i] = get_u32(r);
    existing.n_gpu_layers = get_u32(r);
    int32_t split_len = get_u32(r);
    if (split_len < 0 || (size_t)split_len > MAX_SPLIT) return false;
    existing.tensor_split.resize(split_len);
    for (int32_t i = 0; i < split_len; i++) {
        uint32_t u = get_u32(r); memcpy(&existing.tensor_split[i], &u, sizeof(float));
    }
    existing.n_ctx = get_u32(r);
    uint64_t pool = get_u64(r);
    existing.pool_grant_mib = (size_t)pool;
    int32_t n_shed = get_u32(r);
    if (n_shed < 0 || (size_t)n_shed > MAX_SPLIT) return false;
    existing.shed_reserves.reserve(n_shed);
    for (int32_t i = 0; i < n_shed; i++) {
        int32_t tag_len = get_u32(r);
        if (tag_len < 0 || tag_len > 63 || r.pos + tag_len > r.n) return false;
        existing.shed_reserves.emplace_back((const char *)(r.p + r.pos), tag_len);
        r.pos += tag_len;
    }
    int32_t n_delta = get_u32(r);
    if (n_delta < 0 || (size_t)n_delta > MAX_SPLIT) return false;
    existing.delta_free_ema.resize(n_delta > 0 ? n_delta : 0);
    for (int32_t i = 0; i < n_delta; i++) {
        int64_t d = get_u64(r); existing.delta_free_ema[i] = d;
    }
    int32_t n_proj = get_u32(r);
    if (n_proj < 0 || (size_t)n_proj > MAX_SPLIT) return false;
    existing.projected_free.resize(n_proj > 0 ? n_proj : 0);
    for (int32_t i = 0; i < n_proj; i++) {
        int64_t p = get_u64(r); existing.projected_free[i] = p;
    }
    uint32_t flags = get_u32(r);
    existing.projected_with_delta = (flags & 1) != 0;
    if (r.pos < buf.size()) {
        int32_t n_ceil = get_u32(r);
        if (n_ceil >= 0 && (size_t)n_ceil <= MAX_SPLIT) {
            existing.ceiling_free.resize(n_ceil > 0 ? n_ceil : 0);
            for (int32_t i = 0; i < n_ceil; i++) existing.ceiling_free[i] = get_u64(r);
        }
    }
    if (r.pos < buf.size()) {
        int32_t n_err = get_u32(r);
        if (n_err >= 0 && (size_t)n_err <= MAX_SPLIT) {
            existing.model_size_error.resize(n_err > 0 ? n_err : 0);
            for (int32_t i = 0; i < n_err; i++) existing.model_size_error[i] = get_u64(r);
        }
    }
    if (r.pos < buf.size()) {
        int32_t n_warm = get_u32(r);
        if (n_warm >= 0 && (size_t)n_warm <= MAX_SPLIT) {
            existing.ceiling_free_warm.resize(n_warm > 0 ? n_warm : 0);
            for (int32_t i = 0; i < n_warm; i++) existing.ceiling_free_warm[i] = get_u64(r);
        }
    }

    common_fit_plan updated = existing;
    updated.model_size_error  = model_size_error;
    updated.projected_free    = projected_free;
    updated.ceiling_free_warm = ceiling_free_warm;

    // Serialize the updated plan using the same format as common_fit_cache_save.
    std::vector<uint8_t> body;
    app_i32(body, (int32_t) updated.role_devices.size());
    for (int id : updated.role_devices) app_i32(body, id);
    app_i32(body, updated.n_gpu_layers);
    app_i32(body, (int32_t) updated.tensor_split.size());
    for (float f : updated.tensor_split) app_f32(body, f);
    app_i32(body, updated.n_ctx);
    app_u64(body, (uint64_t) updated.pool_grant_mib);
    app_i32(body, (int32_t) updated.shed_reserves.size());
    for (const auto & tag : updated.shed_reserves) {
        if (tag.size() > 63) return false;
        app_i32(body, (int32_t) tag.size());
        body.insert(body.end(), tag.begin(), tag.end());
    }
    app_i32(body, (int32_t) updated.delta_free_ema.size());
    for (int64_t d : updated.delta_free_ema) app_i64(body, d);
    app_i32(body, (int32_t) updated.projected_free.size());
    for (int64_t p : updated.projected_free) app_i64(body, p);
    app_u32(body, updated.projected_with_delta ? 1 : 0);
    app_i32(body, (int32_t) updated.ceiling_free.size());
    for (int64_t c : updated.ceiling_free) app_i64(body, c);
    app_i32(body, (int32_t) updated.model_size_error.size());
    for (int64_t e : updated.model_size_error) app_i64(body, e);
    app_i32(body, (int32_t) updated.ceiling_free_warm.size());
    for (int64_t w : updated.ceiling_free_warm) app_i64(body, w);

    std::vector<uint8_t> out;
    out.insert(out.end(), 'C'); out.insert(out.end(), 'T');
    out.insert(out.end(), 'I'); out.insert(out.end(), 'F');
    app_u32(out, (uint32_t) CACHE_VERSION);
    app_u64(out, stored_key);
    app_u64(out, (uint64_t) std::time(nullptr));
    app_u32(out, (uint32_t) body.size());
    out.insert(out.end(), body.begin(), body.end());

    // Write directly to cache_path (found via directory scan), bypassing
    // common_fit_cache_save key recomputation. This ensures we write to the
    // SAME file that fit_delta_refresh patched, keeping delta and mse together.
    try {
        std::string tmp = cache_path + ".tmp";
        std::ofstream of(tmp, std::ios::binary);
        if (!of.is_open()) {
            LOG_WRN("%s: cannot open temp file: %s\n", __func__, tmp.c_str());
            return false;
        }
        of.write((char *)out.data(), out.size());
        of.close();
        if (of.fail()) {
            LOG_WRN("%s: write failed: %s\n", __func__, tmp.c_str());
            return false;
        }
        std::error_code ec;
        fs::rename(tmp, cache_path, ec);
        if (ec) {
            LOG_WRN("%s: rename failed: %s\n", __func__, ec.message().c_str());
            return false;
        }
        LOG_INF("%s: direct-saved updated cache to %s\n", __func__, cache_path.c_str());
        return true;
    } catch (const std::exception & e) {
        LOG_WRN("%s: exception: %s\n", __func__, e.what());
        return false;
    }
}
