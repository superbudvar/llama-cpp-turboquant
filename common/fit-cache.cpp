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
constexpr uint32_t CACHE_VERSION = 10;
constexpr std::int64_t MAX_CACHE_AGE_S = 24 * 3600;

// Memory quantities in the key are quantised to this grid before hashing.
// The MTP preflight folds a measured reserve into fit_params_target, and the
// mmproj and spec reserves carry measured bytes, so the same configuration
// produced keys 1.5 MiB apart across consecutive loads and the cache never hit.
// Measured byte values are floored to the grid; everything else in the key stays
// exact. Two configurations whose targets differ by less than this share a key,
// which is the accepted tradeoff: the stored plan is re-validated against live
// free memory and margins on load, so a plan fitted under the larger target is
// simply conservative for the smaller one.
constexpr size_t CACHE_KEY_GRID_MIB = 24;

size_t quantize_mib(size_t bytes) {
    const size_t grid = CACHE_KEY_GRID_MIB * 1024 * 1024;
    return bytes / grid * grid;
}

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
        key += std::to_string(quantize_mib(t));
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
        key += std::to_string(quantize_mib(r.bytes));
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
    if (params.skip_infile_mtp) {
        key += "skip_infile_mtp=1\n";
    }
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

// Memory buckets that went into the key, so a miss can be attributed to a
// specific axis: same buckets but a different key means a non-quantised
// component moved, not the reserve granularity.
static std::string cache_key_buckets(const common_params & params) {
    std::string s;
    for (size_t t : params.fit_params_target) {
        s += std::to_string(quantize_mib(t) / (1024 * 1024));
        s += ",";
    }
    s += " reserves=";
    for (const auto & r : params.fit_params_reserves) {
        s += r.tag;
        s += ":";
        s += std::to_string(quantize_mib(r.bytes) / (1024 * 1024));
        s += ",";
    }
    return s;
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
        LOG_INF("%s: no cache file: %s (key %016llx, grid %zu MiB, target_mib=[%s])\n",
                __func__, path.c_str(), (unsigned long long) key, CACHE_KEY_GRID_MIB, cache_key_buckets(params).c_str());
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
    if (magic != CACHE_MAGIC || version < 6 || version > CACHE_VERSION) {
        LOG_INF("%s: cache header mismatch: %s\n", __func__, path.c_str());
        return false;
    }
    if (key_hash != key) {
        LOG_INF("%s: cache key mismatch: %s (want %016llx, have %016llx, grid %zu MiB, target_mib=[%s])\n",
                __func__, path.c_str(), (unsigned long long) key, (unsigned long long) key_hash,
                CACHE_KEY_GRID_MIB, cache_key_buckets(params).c_str());
        return false;
    }
    const int64_t now = (int64_t) std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    if ((int64_t) timestamp <= 0 || now - (int64_t) timestamp > MAX_CACHE_AGE_S || (int64_t) timestamp > now) {
        LOG_INF("%s: cache entry stale: %s (age %lld s, max %lld s)\n", __func__, path.c_str(),
                (long long) (now - (int64_t) timestamp), (long long) MAX_CACHE_AGE_S);
        return false;
    }
    if (r.pos + body_size != buf.size()) {
        LOG_INF("%s: cache body size mismatch: %s\n", __func__, path.c_str());
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

    if (version >= 7 && r.pos < buf.size()) {
        uint32_t search_complete = 1;
        int32_t n_baseline = 0;
        if (!r.u32(search_complete) || !r.i32(plan.search.shed_level) ||
                !r.i32(plan.search.candidate_cursor) || !r.i32(plan.search.best_split_size) ||
                !r.f32(plan.search.best_rpc_frac) || !r.i32(n_baseline) ||
                n_baseline < 0 || (size_t) n_baseline > MAX_SPLIT) {
            return false;
        }
        plan.search.complete = search_complete != 0;
        plan.search.baseline_free.resize(n_baseline > 0 ? (size_t) n_baseline : 0);
        for (int32_t i = 0; i < n_baseline; i++) {
            if (!r.i64(plan.search.baseline_free[(size_t) i])) {
                return false;
            }
        }
    }
    if (version >= 8 && r.pos < buf.size()) {
        if (!r.i32(plan.search.rpc_stage) || !r.f32(plan.search.rpc_frac_next) ||
                !r.f32(plan.search.rpc_frac_low) || !r.f32(plan.search.rpc_frac_high)) {
            return false;
        }
    }
    if (version >= 9 && r.pos < buf.size() && !r.u32(plan.search.policy_version)) {
        return false;
    }
    // v10+: the per-device margins the plan was solved with
    if (version >= 10 && r.pos < buf.size()) {
        int32_t n_margins = 0;
        if (!r.i32(n_margins) || n_margins < 0 || (size_t) n_margins > MAX_SPLIT) {
            return false;
        }
        plan.margins.resize(n_margins > 0 ? (size_t) n_margins : 0);
        for (int32_t i = 0; i < n_margins; i++) {
            if (!r.i64(plan.margins[(size_t) i])) {
                return false;
            }
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
    // v7+: resumable rpc-aware search state.
    app_u32(body, plan.search.complete ? 1 : 0);
    app_i32(body, plan.search.shed_level);
    app_i32(body, plan.search.candidate_cursor);
    app_i32(body, plan.search.best_split_size);
    app_f32(body, plan.search.best_rpc_frac);
    app_i32(body, (int32_t) plan.search.baseline_free.size());
    for (int64_t free : plan.search.baseline_free) {
        app_i64(body, free);
    }
    app_i32(body, plan.search.rpc_stage);
    app_f32(body, plan.search.rpc_frac_next);
    app_f32(body, plan.search.rpc_frac_low);
    app_f32(body, plan.search.rpc_frac_high);
    app_u32(body, plan.search.policy_version);
    // v10+: the per-device margins the plan was solved with
    app_i32(body, (int32_t) plan.margins.size());
    for (int64_t m : plan.margins) {
        app_i64(body, m);
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
    LOG_INF("%s: saved cache: delta_ema=[%s] projected=[%s] ceiling=[%s] model_size_error=[%s] to %s (grid %zu MiB, target_mib=[%s])\n",
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
            path.c_str(),
            CACHE_KEY_GRID_MIB,
            cache_key_buckets(params).c_str());
    return true;
}

bool common_fit_cache_refresh_deltas(
        const common_params & params,
        const std::string & path_model,
        const common_fit_analysis & analysis,
        const std::vector<int64_t> & new_delta_free_ema,
        std::string * cache_path_out) {
    common_fit_plan plan;
    if (!common_fit_cache_load(params, path_model, analysis, plan)) {
        return false;
    }
    plan.delta_free_ema = new_delta_free_ema;
    const bool saved = common_fit_cache_save(params, path_model, analysis, plan);
    if (saved && cache_path_out) {
        *cache_path_out = cache_file_path(params, path_model, analysis);
    }
    return saved;
}

bool common_fit_cache_update_feedback(
        const common_params & params,
        const std::string & path_model,
        const common_fit_analysis & analysis,
        const std::vector<int64_t> & delta_free_ema,
        const std::vector<int64_t> & projected_free,
        const std::vector<int64_t> & model_size_error,
        const std::vector<int64_t> & ceiling_free_warm) {
    common_fit_plan plan;
    if (!common_fit_cache_load(params, path_model, analysis, plan)) {
        LOG_WRN("%s: no exact fit-cache entry to update for %s\n", __func__, path_model.c_str());
        return false;
    }
    plan.delta_free_ema = delta_free_ema;
    plan.projected_free = projected_free;
    plan.model_size_error = model_size_error;
    plan.ceiling_free_warm = ceiling_free_warm;
    return common_fit_cache_save(params, path_model, analysis, plan);
}
