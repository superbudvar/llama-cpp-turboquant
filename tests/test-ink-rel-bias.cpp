// GGML_OP_INK_REL_BIAS against the unfused chain it replaces in src/models/inkling2.cpp
// (subtract the sinks, gather by distance, permute, add the mask, cast to F16), on the CPU backend.
// The indices and the mask follow llama_kv_cache::set_input_relative_pos and the causal mask: distances in
// [0, extent) index the table, everything else (beyond the extent, future cells, empty cells, cells of another
// sequence) indexes the padded zero row, and masked cells carry -inf. Positions run past 1M.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct cell {
    int64_t pos;   // -1: empty
    int     seq;
};

static int run_case(ggml_backend_t be, const char * name, int extent, int n_head, const std::vector<cell> & cells,
                    const std::vector<int64_t> & q_pos, int q_seq) {
    const int n_rel = extent + 1;
    const int n_kv  = (int) cells.size();
    const int T     = (int) q_pos.size();

    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * table = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_rel, n_head, T);
    ggml_tensor * sinks = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_head, T);
    ggml_tensor * idx   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, T);
    ggml_tensor * mask  = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, T);

    // the new op
    ggml_tensor * fused = ggml_ink_rel_bias(ctx, table, sinks, idx, mask, GGML_TYPE_F16);

    // the original chain
    ggml_tensor * t2 = ggml_sub(ctx, table, ggml_reshape_3d(ctx, sinks, 1, n_head, T));
    t2 = ggml_cont(ctx, ggml_permute(ctx, t2, 1, 0, 2, 3));
    ggml_tensor * kq_b = ggml_get_rows(ctx, t2, idx);
    kq_b = ggml_reshape_4d(ctx, kq_b, n_head, n_kv, T, 1);
    kq_b = ggml_cont(ctx, ggml_permute(ctx, kq_b, 2, 0, 1, 3));
    ggml_tensor * chain = ggml_cast(ctx, ggml_add(ctx, kq_b, ggml_cast(ctx, mask, GGML_TYPE_F32)), GGML_TYPE_F16);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fused);
    ggml_build_forward_expand(gf, chain);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u(-4.0f, 4.0f);
    std::vector<float> tv((size_t) n_rel*n_head*T);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < n_head; ++h) {
            for (int r = 0; r < n_rel; ++r) {
                tv[r + (size_t) n_rel*(h + (size_t) n_head*t)] = r == extent ? 0.0f : u(rng); // padded row is zero
            }
        }
    }
    std::vector<float> sv((size_t) n_head*T);
    for (auto & v : sv) {
        v = u(rng);
    }
    std::vector<int32_t>     iv((size_t) n_kv*T);
    std::vector<ggml_fp16_t> mv((size_t) n_kv*T);
    std::vector<bool>        masked((size_t) n_kv*T);
    for (int t = 0; t < T; ++t) {
        for (int j = 0; j < n_kv; ++j) {
            const cell & c = cells[j];
            const int64_t distance = c.pos < 0 ? -1 : q_pos[t] - c.pos;
            iv[(size_t) t*n_kv + j] = distance >= 0 && distance < extent ? (int32_t) distance : extent;
            const bool m = c.pos < 0 || c.seq != q_seq || c.pos > q_pos[t];
            masked[(size_t) t*n_kv + j] = m;
            mv[(size_t) t*n_kv + j] = ggml_fp32_to_fp16(m ? -INFINITY : 0.0f);
        }
    }
    ggml_backend_tensor_set(table, tv.data(), 0, ggml_nbytes(table));
    ggml_backend_tensor_set(sinks, sv.data(), 0, ggml_nbytes(sinks));
    ggml_backend_tensor_set(idx,   iv.data(), 0, ggml_nbytes(idx));
    ggml_backend_tensor_set(mask,  mv.data(), 0, ggml_nbytes(mask));

    ggml_backend_graph_compute(be, gf);

    std::vector<ggml_fp16_t> a(ggml_nelements(fused)), b(ggml_nelements(chain));
    ggml_backend_tensor_get(fused, a.data(), 0, ggml_nbytes(fused));
    ggml_backend_tensor_get(chain, b.data(), 0, ggml_nbytes(chain));

    int bad = 0;
    for (int h = 0; h < n_head; ++h) {
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < n_kv; ++j) {
                const size_t k = j + (size_t) n_kv*(t + (size_t) T*h);
                const float  x = ggml_fp16_to_fp32(a[k]);
                const bool   m = masked[(size_t) t*n_kv + j];
                if (memcmp(&a[k], &b[k], sizeof(ggml_fp16_t)) != 0 || (m != (std::isinf(x) && x < 0)) || std::isnan(x)) {
                    if (bad++ < 5) {
                        printf("  %s: mismatch j=%d t=%d h=%d fused=%f chain=%f masked=%d\n", name, j, t, h, x,
                               ggml_fp16_to_fp32(b[k]), (int) m);
                    }
                }
            }
        }
    }
    printf("%s: %s (%d cells x %d tokens x %d heads)\n", name, bad ? "FAIL" : "OK", n_kv, T, n_head);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return bad ? 1 : 0;
}

// flash attention: the per-head mask from GGML_OP_INK_REL_BIAS against the bias computed in the kernel, on one
// backend; both feed identical F16 mask values to the same kernels, so the outputs must match bit for bit
static int run_fa_case(ggml_backend_t be, const char * name, ggml_type type_kv, int T, int n_kv, int extent) {
    const int D = 128, H = 24, Hkv = 8, n_rel = extent + 1;
    ggml_init_params ip = { 64*ggml_tensor_overhead() + 2*ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q     = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, T, H, 1);
    ggml_tensor * k     = ggml_new_tensor_4d(ctx, type_kv, D, n_kv, Hkv, 1);
    ggml_tensor * v     = ggml_new_tensor_4d(ctx, type_kv, D, n_kv, Hkv, 1);
    ggml_tensor * mask  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, T, 1, 1);
    ggml_tensor * table = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_rel, H, T);
    ggml_tensor * sinks = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
    ggml_tensor * idx   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, T);
    ggml_tensor * kpos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);
    ggml_tensor * qpos  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);

    ggml_tensor * mask_h = ggml_ink_rel_bias(ctx, table, sinks, idx, mask, GGML_TYPE_F16);
    ggml_tensor * fa_a = ggml_flash_attn_ext(ctx, q, k, v, mask_h, 1.0f/sqrtf(D), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa_a, GGML_PREC_F32);
    ggml_tensor * fa_b = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f/sqrtf(D), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(fa_b, GGML_PREC_F32);
    ggml_flash_attn_ext_set_rel_bias(fa_b, table, sinks, kpos, qpos, extent, 0);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, fa_a);
    ggml_build_forward_expand(gf, fa_b);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    auto fill_f32 = [&](ggml_tensor * t, float s) {
        std::vector<float> d(ggml_nelements(t));
        for (auto & x : d) { x = s*u(rng); }
        ggml_backend_tensor_set(t, d.data(), 0, ggml_nbytes(t));
    };
    auto fill_q = [&](ggml_tensor * t) {
        std::vector<float> d(ggml_nelements(t));
        for (auto & x : d) { x = u(rng); }
        std::vector<uint8_t> qd(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, d.data(), qd.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        ggml_backend_tensor_set(t, qd.data(), 0, ggml_nbytes(t));
    };
    fill_f32(q, 1.0f);
    fill_q(k);
    fill_q(v);
    std::vector<float> tv((size_t) n_rel*H*T);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            for (int r = 0; r < n_rel; ++r) {
                tv[r + (size_t) n_rel*(h + (size_t) H*t)] = r == extent ? 0.0f : 3.0f*u(rng);
            }
        }
    }
    ggml_backend_tensor_set(table, tv.data(), 0, ggml_nbytes(table));
    fill_f32(sinks, 3.0f);
    const int32_t base = 1048000, last = base + n_kv - 9;
    std::vector<int32_t> kp(n_kv), qp(T), iv((size_t) n_kv*T);
    std::vector<ggml_fp16_t> mv((size_t) n_kv*T);
    for (int j = 0; j < n_kv; ++j) {
        kp[j] = base + j > last || j % 97 == 13 ? -1 : base + j;
    }
    for (int t = 0; t < T; ++t) {
        qp[t] = last - (T - 1 - t);
        for (int j = 0; j < n_kv; ++j) {
            const int64_t d = kp[j] < 0 ? -1 : (int64_t) qp[t] - kp[j];
            iv[(size_t) t*n_kv + j] = d >= 0 && d < extent ? (int32_t) d : extent;
            mv[(size_t) t*n_kv + j] = ggml_fp32_to_fp16(kp[j] < 0 || kp[j] > qp[t] ? -INFINITY : 0.0f);
        }
    }
    ggml_backend_tensor_set(kpos, kp.data(), 0, ggml_nbytes(kpos));
    ggml_backend_tensor_set(qpos, qp.data(), 0, ggml_nbytes(qpos));
    ggml_backend_tensor_set(idx,  iv.data(), 0, ggml_nbytes(idx));
    ggml_backend_tensor_set(mask, mv.data(), 0, ggml_nbytes(mask));

    ggml_backend_graph_compute(be, gf);

    std::vector<float> a(ggml_nelements(fa_a)), b(ggml_nelements(fa_b));
    ggml_backend_tensor_get(fa_a, a.data(), 0, ggml_nbytes(fa_a));
    ggml_backend_tensor_get(fa_b, b.data(), 0, ggml_nbytes(fa_b));
    // the per-head mask of the reference path disables the GQA optimization, so the two paths can run different
    // flash attention kernels and sum in a different order: compare with a tolerance instead of bit for bit
    const double tol = 2e-3;
    int bad = 0;
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double diff = std::fabs(a[i] - b[i]);
        max_diff = std::max(max_diff, diff);
        if (std::isnan(a[i]) || std::isnan(b[i]) || diff > tol) {
            bad++;
        }
    }
    printf("%s [%s]: %s (%d of %zu differ, max |diff| %.3g)\n", name, ggml_backend_name(be), bad ? "FAIL" : "OK", bad, a.size(), max_diff);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return bad ? 1 : 0;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!be) {
        fprintf(stderr, "no CPU backend\n");
        return 1;
    }

    int fails = 0;
    for (const int64_t base : { (int64_t) 0, (int64_t) 1048000 }) {
        for (const int extent : { 1024, 8 }) {
            // cells: a run of the sequence ending at the queries, plus the edge cases
            std::vector<cell> cells;
            const int64_t q0 = base + 3*extent;
            for (int64_t p = q0 - extent - 3; p <= q0 + 2; ++p) {
                cells.push_back({ p, 0 });             // distances extent+3 .. -2 around the first query
            }
            cells.push_back({ -1, 0 });                 // empty cell
            cells.push_back({ q0 - 1, 1 });             // another sequence
            cells.push_back({ q0 - extent, 0 });        // exactly extent away
            cells.push_back({ q0 - extent + 1, 0 });    // extent - 1 away
            cells.push_back({ base, 0 });               // far in the past
            cells.push_back({ q0 + 100, 0 });           // future
            const std::vector<int64_t> q_pos = { q0, q0 + 1, q0 + 2 };
            char name[96];
            snprintf(name, sizeof(name), "rel_bias base=%lld extent=%d", (long long) base, extent);
            fails += run_case(be, name, extent, 24, cells, q_pos, 0);
        }
    }

    // flash attention parity, on every backend
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_t b = ggml_backend_dev_init(dev, nullptr);
        if (!b) {
            continue;
        }
        for (ggml_type t : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_F16 }) {
            for (int T : { 1, 7, 64 }) {
                char name[96];
                snprintf(name, sizeof(name), "fa_rel_bias %s T=%d", ggml_type_name(t), T);
                fails += run_fa_case(b, name, t, T, 2048, 1024);
            }
        }
        ggml_backend_free(b);
    }

    ggml_backend_free(be);
    printf("%s\n", fails ? "FAILED" : "PASSED");
    return fails ? 1 : 0;
}
